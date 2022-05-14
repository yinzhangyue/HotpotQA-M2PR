from transformers import RobertaPreTrainedModel, RobertaModel
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import ipdb

########## RobertaRetriever ##########
class ARobertaRetriever(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # Self-define layer
        self.single_document_classifier_layer = nn.Linear(config.hidden_size, 2)
        self.double_document_classifier_layer = nn.Linear(config.hidden_size, 2)
        # Kendall_Multi-Task_Learning_Using_CVPR_2018_paper
        self.log_loss_coeff = nn.Parameter(torch.zeros(2, requires_grad=True))

        self.init_weights()

    def forward(
        self,
        question_ids=None,
        document_ids=None,
        document_labels=None,
        gold_doc_pair=None,
        question_attention_mask=None,
        document_attention_mask=None,
        doc_num=None,
        doc_length=None,
    ):
        r"""
        question_ids: (Batch Size, Question Length)
        document_ids: (Batch Size, Document Number, Document Length)
        question_attention_mask: (Batch Size, Question Length)
        document_attention_mask: (Batch Size, Document Number, Document Length)
        document_labels: binary number indicate whether the document is related (Batch Size, Document Number)
        gold_doc_pair: index of two related document (Batch Size, 2)
        doc_num: document number in each batch (Batch Size)
        doc_length: document length (Batch Size, Document Number)
        """
        batch_size = len(document_ids)
        device = document_ids.device
        # Loss
        total_rank_loss = torch.tensor(0, requires_grad=True).to(device)
        total_document_pair_loss = torch.tensor(0, requires_grad=True).to(device)
        total_retriever_loss = torch.tensor(0, requires_grad=True).to(device)

        seleced_document_pair_list = torch.zeros((batch_size, 2)).to(device)
        candidate_document_pair_list = torch.zeros((3, 2)).to(device)

        for b in range(batch_size):
            if doc_num[b] <= 2:
                seleced_document_pair_list[b] = torch.tensor((0, 1)).to(device)
                continue

            question_length = len(question_ids[b])
            # Record documents' probability
            document_probability = torch.zeros((doc_num[b], 2)).to(device)
            double_document_probability = torch.zeros((3, 2)).to(device)
            double_document_label = torch.zeros((3)).to(device)
            for i in range(doc_num[b]):
                question_document_ids = torch.cat((question_ids[b], document_ids[b][i][: doc_length[b][i]]))  # (Question Length + Document Length)
                question_document_attention_mask = torch.cat(
                    (
                        question_attention_mask[b],
                        document_attention_mask[b][i][: doc_length[b][i]],
                    )
                )  # (Question Length + Document Length)

                assert question_document_ids.shape == question_document_attention_mask.shape, "Attention mask should be the same dimension of ids"

                # Input Length <= 512
                if len(question_document_ids) > 512:
                    question_document_ids = question_document_ids[:512]
                if len(question_document_attention_mask) > 512:
                    question_document_attention_mask = question_document_attention_mask[:512]
                # Rank Model
                stage_one_outputs = self.roberta(
                    input_ids=question_document_ids.unsqueeze(0),
                    attention_mask=question_document_attention_mask.unsqueeze(0),
                    return_dict=True,
                )
                document_probability[i] = self.single_document_classifier_layer(torch.squeeze(stage_one_outputs[0])[0])  # (2)
            # Rank Loss
            rank_loss_function = CrossEntropyLoss()
            rank_loss = rank_loss_function(document_probability.T.unsqueeze(0), document_labels[b].unsqueeze(0))  # (1, 2, Document Number) (1, Document Number)
            total_rank_loss += rank_loss
            # Select 3 documents
            _, seleced_documents = torch.topk(F.softmax(document_probability, dim=-1)[:, 1], 3)

            # Select document pair
            for ii in range(3):
                for jj in range(ii + 1, 3):
                    if (seleced_documents[ii] == gold_doc_pair[b][0] and seleced_documents[jj] == gold_doc_pair[b][1]) or (seleced_documents[ii] == gold_doc_pair[b][1] and seleced_documents[jj] == gold_doc_pair[b][0]):
                        double_document_label[ii + jj - 1] = 1
                    candidate_document_pair_list[ii + jj - 1] = torch.tensor(
                        (
                            seleced_documents[ii],
                            seleced_documents[jj],
                        )
                    )

                    question_double_document_ids = torch.cat(
                        (
                            question_ids[b],
                            document_ids[b][ii][: doc_length[b][ii]],
                            document_ids[b][jj][: doc_length[b][jj]],
                        ),
                        dim=-1,
                    )  # (Question Length + Document1 Length + Document2 Length)
                    question_double_document_attention_mask = torch.cat(
                        (
                            question_attention_mask[b],
                            document_attention_mask[b][ii][: doc_length[b][ii]],
                            document_attention_mask[b][jj][: doc_length[b][jj]],
                        ),
                        dim=-1,
                    )  # (Question Length + Document1 Length + Document2 Length)
                    assert question_double_document_ids.shape == question_double_document_attention_mask.shape, "Attention mask should be the same dimension of ids"
                    # Input Length <= 512
                    if len(question_double_document_ids) > 512:
                        question_double_document_ids = question_double_document_ids[:512]
                    if len(question_double_document_attention_mask) > 512:
                        question_double_document_attention_mask = question_double_document_attention_mask[:512]

                    # Double Document Selection Model
                    stage_two_outputs = self.roberta(
                        input_ids=question_double_document_ids.unsqueeze(0),
                        attention_mask=question_double_document_attention_mask.unsqueeze(0),
                        return_dict=True,
                    )
                    double_document_probability[ii + jj - 1] = self.double_document_classifier_layer(torch.squeeze(stage_two_outputs[0])[0])  # (2)

            # Document Pair Loss
            document_pair_loss_function = CrossEntropyLoss()
            document_pair_loss = document_pair_loss_function(
                double_document_probability.T.unsqueeze(0),
                double_document_label.unsqueeze(0).long(),
            )  # (1, 2, 3) (1, 3)
            total_document_pair_loss += document_pair_loss
            # Select document pair
            seleced_document_pair = torch.argmax(F.softmax(double_document_probability, dim=-1)[:, 1])
            # ipdb.set_trace()
            seleced_document_pair_list[b] = candidate_document_pair_list[seleced_document_pair]
            # Retriever Loss
            retriever_loss = torch.exp(-self.log_loss_coeff[0]) * rank_loss + self.log_loss_coeff[0] + torch.exp(-self.log_loss_coeff[1]) * document_pair_loss + self.log_loss_coeff[1]
            # Navie Retriever Loss
            # retriever_loss = rank_loss + document_pair_loss
            # print(
            #     "Prediction:{}, Label:{}".format(
            #         seleced_document_pair_list[b], gold_doc_pair[b]
            #     )
            # )
            # print(rank_loss, document_pair_loss)
            total_retriever_loss += retriever_loss
        return {
            "loss": total_retriever_loss,
            "selected_pair": seleced_document_pair_list,
        }
