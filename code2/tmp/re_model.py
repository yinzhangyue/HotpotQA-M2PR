from transformers.models.roberta import RobertaPreTrainedModel, RobertaModel
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from collections import Counter


class ARobertaForDocselection(RobertaPreTrainedModel):
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
        doc_num=None,
        doc_length=None,
        question_length=None,
        document_labels=None,
    ):
        r"""
        question_ids: (Batch Size, Question Length)
        document_ids: (Batch Size, Document Number, Document Length)
        doc_num: document number in each batch (Batch Size)
        doc_length: document length (Batch Size, Document Number)
        question_length: question length (Batch Size)
        document_labels: binary number indicate whether the document is related (Batch Size, Document Number)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        B, Doc_size, L_d = document_ids.size()
        L_q = question_ids.size(-1)
        device = question_ids.device

        question_ids = question_ids.unsqueeze(1)
        question_attention_mask = question_attention_mask.unsqueeze(1)

        question_ids_input = question_ids.expand(B, Doc_size, L_q)
        question_attention_mask_input = question_attention_mask.expand(B, Doc_size, L_q)

        # 输入ids
        total_input = torch.cat([question_ids_input, document_ids], dim=-1).reshape(B * Doc_size, -1)
        # 输入attention_mask
        total_attention_mask = torch.cat([question_attention_mask_input, document_attention_mask], dim=-1).reshape(B * Doc_size, -1)

        selection1_outputs = self.roberta(
            total_input,
            attention_mask=total_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        L_total = selection1_outputs[0].size(1)
        # 先过Linear，再选取位置
        _selection1_outputs = self.document_outputs(selection1_outputs[0])
        doc_selection = _selection1_outputs.reshape(B, Doc_size, L_total, -1)
        E = selection1_outputs[0].size(-1)  # Embedding大小
        # 取[DOC]的位置
        doc_selection_output = torch.zeros(B, Doc_size, 2).to(device)
        for b in range(B):
            # size: B*D*2
            doc_selection_output[b] = doc_selection[b, :, DOC_index[b].item(), :]
        Ltotal = None
        Ldoc = 0
        if document_labels is not None:
            document_loss = CrossEntropyLoss()
            cemask = torch.Tensor([1e10, -1e10])
            cemask.to(device)
            # mask padding sentence
            for b in range(B):
                # sentencemask
                doc_selection_output[b, doc_num[b] : -1] = cemask
            doc_selection_output_p = doc_selection_output.permute(0, 2, 1)  # B*2*D
            # cal Loss
            Ldoc = document_loss(doc_selection_output_p, document_labels)
            Ltotal = Ldoc
        # 取index为1的位置
        doc_sort_input = F.softmax(doc_selection_output, dim=-1)[:, :, -1]  # BD
        _, doc_select = torch.topk(doc_sort_input, 3, dim=-1)
        # 概率为1最高的3个doc的index
        doc_select = doc_select[:, :3]  # B*3

        # 初步选出来的3个二元组
        doc_pair = np.zeros([B, 3, 2])
        for b in range(B):
            pairs = []
            for ii in range(3):
                for jj in range(ii + 1, 3):
                    pairs.append([doc_select[b, ii].item(), doc_select[b, jj].item()])
            doc_pair[b] = np.array(pairs)

        pairlabels = None
        if gold_doc_pair is not None:
            pairlabels = torch.zeros([B, 3], device=device)  #####phase2label
            for b in range(B):
                for ii in range(3):
                    if Counter(doc_pair[b, ii]) == Counter(gold_doc_pair[b].detach().cpu().numpy()):
                        pairlabels[b, ii] = 1

        phase_2_question_ids = question_ids.expand(B, 3, L_q)
        doc_pair = torch.Tensor(doc_pair).long().to(device)

        pair_list = []
        pair_length_list = []
        for b in range(B):
            for ii in range(3):
                # 2*L_d
                first = doc_pair[b, ii][0].item()
                second = doc_pair[b, ii][1].item()
                first_document_ids = document_ids[b][first][: doc_length[b][first]]
                second_document_ids = document_ids[b][second][: doc_length[b][second]]

                pair = torch.cat((first_document_ids, second_document_ids), dim=-1)
                pair_list.append(pair)
                pair_length_list.append(len(pair))
        max_length = max(pair_length_list)
        # phase_2_document_ids
        phase_2_document_ids = torch.ones([B, 3, max_length]).to(device)
        for b in range(B):
            for ii in range(3):
                phase_2_document_ids[b, ii][: len(pair_list[ii])] = pair_list[ii]

        pair_list = []
        pair_length_list = []
        for b in range(B):
            for ii in range(3):
                # 2*L_d
                first = doc_pair[b, ii][0].item()
                second = doc_pair[b, ii][1].item()
                first_document_attention_mask = document_attention_mask[b][first][: doc_length[b][first]]
                second_document_attention_mask = document_attention_mask[b][second][: doc_length[b][second]]
                pair = torch.cat(
                    (first_document_attention_mask, second_document_attention_mask),
                    dim=-1,
                )
                pair_list.append(pair)
                pair_length_list.append(len(pair))
        max_length = max(pair_length_list)
        # phase_2_document_attention_mask
        phase_2_document_attention_mask = torch.ones([B, 3, max_length]).to(device)
        for b in range(B):
            for ii in range(3):
                phase_2_document_attention_mask[b, ii][: len(pair_list[ii])] = pair_list[ii]

        phase_2_question_attention_mask_input = question_attention_mask.expand(B, 3, L_q)

        phase_2_total_input = torch.cat([phase_2_question_ids, phase_2_document_ids], dim=-1)
        phase_2_total_attention_mask = torch.cat(
            [phase_2_question_attention_mask_input, phase_2_document_attention_mask],
            dim=-1,
        )
        if phase_2_total_input.size(-1) > 1024:
            phase_2_total_input = phase_2_total_input[:, :, :1024]
            phase_2_total_attention_mask = phase_2_total_attention_mask[:, :, :1024]
        selection2_outputs = self.roberta(
            input_ids=phase_2_total_input.long().reshape(B * 3, -1),
            attention_mask=phase_2_total_attention_mask.reshape(B * 3, -1),
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        phase2_output = selection2_outputs[0].reshape(B, 3, -1, E)
        phase2_cls_output = phase2_output[:, :, 0, :]
        phase2_pair_selection = self.phase2_outputs(phase2_cls_output)  # B*3*2

        phase2_pair_selection_soft = F.softmax(phase2_pair_selection, dim=-1)
        phase2_pair_output = phase2_pair_selection_soft[:, :, -1]  # B*3
        phase2_pair = torch.argmax(phase2_pair_output, dim=-1)  # B*1

        selected_pair = torch.zeros([B, 2], device=device)
        for b in range(B):
            selected_pair[b] = doc_pair[b, phase2_pair[b].item()]

        if Ltotal is not None:
            phase2loss = CrossEntropyLoss()
            phase2_pair_selection_p = phase2_pair_selection.permute(0, 2, 1)  # B*2*3
            Lphase2 = phase2loss(phase2_pair_selection_p, pairlabels.long())

            Ltotal = Ltotal + Lphase2

        return {
            "loss": Ltotal,
            "selected_pair": selected_pair,
        }
