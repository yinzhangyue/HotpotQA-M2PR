from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bert import (
    BertPreTrainedModel,
    BertModel,
    BertLayer,
    BertTokenizerFast,
)
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.albert import (
    AlbertConfig,
    AlbertTokenizer,
    AlbertModel,
    AlbertForQuestionAnswering,
    AlbertPreTrainedModel,
)
from transformers.models.roberta import RobertaPreTrainedModel, RobertaModel
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import json
import ipdb
from collections import Counter
from fastNLP.core.metrics import MetricBase, seq_len_to_mask
from fastNLP.core.losses import LossBase
from torch.nn import CosineSimilarity
import fastNLP


from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig


class ARobertaForDocselection(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # self.use_sentencetrans=True
        # self.use_answertrans=True

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.document_outputs = nn.Linear(config.hidden_size, 2)
        self.phase2_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sentence_index_start=None,
        sentence_index_end=None,
        sentence_labels=None,
        answer_type=None,
        sentence_num=None,
        question_ids=None,
        document_ids=None,
        document_labels=None,
        gold_doc_pair=None,
        question_attention_mask=None,
        document_attention_mask=None,
        doc_num=None,
        DOC_index=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        sentence_index:每句话对应的<e>的index位置以及本sentence结束的位置.B*sentencenumber*2
        sentece_labels:每句话是否是sp.
        question_ids:B*L_q
        document_ids:B*D*L_d
        gold_doc_pair:B*2
        doc_num:B

        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        B, Doc_size, L_d = document_ids.size()
        L_q = question_ids.size(-1)
        device = question_ids.device

        question_ids = question_ids.unsqueeze(1)
        question_attention_mask = question_attention_mask.unsqueeze(1)

        question_ids_input = question_ids.expand(B, Doc_size, L_q)
        question_attention_mask_input = question_attention_mask.expand(B, Doc_size, L_q)

        # 输入ids
        total_input = torch.cat([question_ids_input, document_ids], dim=-1).reshape(
            B * Doc_size, -1
        )
        # 输入attention_mask
        total_attention_mask = torch.cat(
            [question_attention_mask_input, document_attention_mask], dim=-1
        ).reshape(B * Doc_size, -1)
        # 截断
        if total_input.size(-1) > 512:
            total_input = total_input[:, :512]
            total_attention_mask = total_attention_mask[:, :512]

        length_total_input = total_input.size(0)
        selection1_outputs_list = []
        for i in range(length_total_input):
            selection1_outputs = self.roberta(
                total_input[i].unsqueeze(0),
                attention_mask=total_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            selection1_outputs_list.append(selection1_outputs)
        selection1_outputs = torch.tensor(selection1_outputs_list).to(device)

        L_total = selection1_outputs[0].size(1)
        # 先过Linear，再选取位置
        _selection1_outputs = self.document_outputs(selection1_outputs[0])
        doc_selection = _selection1_outputs.reshape(B, Doc_size, L_total, -1)

        # 取[DOC]的位置
        doc_selection_output = torch.index_select(doc_selection, 2, DOC_index).squeeze(
            2
        )  # size: B*D*2
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
        _, doc_select = torch.topk(doc_sort_input, 2, dim=-1)
        # 概率为1最高的2个doc的index
        selected_pair = doc_select  # B*2

        # print(gold_doc_pair, selected_pair)
        return {
            "loss": Ltotal,
            "selected_pair": selected_pair,
            "input_ids": input_ids,
        }
