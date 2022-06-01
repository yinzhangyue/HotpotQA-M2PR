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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import Tensor


import os
import numpy as np
import json

import fastNLP
from fastNLP.core.metrics import MetricBase, seq_len_to_mask
from fastNLP.core.losses import LossBase

import ipdb
import math
from collections import Counter

from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig

from model import *
from seqmixing import upcoverAttention


class ARobertaForDocselection(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_sentencetrans = False
        self.use_answertrans = False
        self.SqeMix = False
        self.SqeMix_sentence_and_answer = False

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.document_outputs = nn.Linear(config.hidden_size, 2)
        self.sentence_outputs = nn.Linear(config.hidden_size, 2)
        self.answer_typeout = nn.Linear(config.hidden_size, 3)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.roberta_config = RobertaConfig.from_pretrained("roberta-base")
        self.roberta_config.hidden_size = config.hidden_size
        self.roberta_config.intermediate_size = config.intermediate_size
        self.roberta_config.num_attention_heads = config.num_attention_heads
        # SeqMix
        if self.SqeMix is True:
            for i, layer in enumerate(self.roberta.encoder.layer):
                layer.attention.self = upcoverAttention(config)

        self.robertencoder = RobertaEncoder(self.roberta_config)
        # SqeMix_sentence_and_answer
        if self.SqeMix_sentence_and_answer is True:
            for i, layer in enumerate(self.robertencoder.layer):
                layer.attention.self = upcoverAttention(config)

        # 和Roberta一样参数
        if self.use_sentencetrans is True:
            self.sentence_transformer = self.robertencoder
        if self.use_answertrans is True:
            self.answer_transformer = self.robertencoder

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
        doc_length=None,
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
        # 截断
        if total_input.size(-1) > 512:
            total_input = total_input[:, :512]
            total_attention_mask = total_attention_mask[:, :512]

        selection1_outputs = self.roberta(total_input, attention_mask=total_attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,)
        L_total = selection1_outputs[0].size(1)
        # 先过Linear，再选取位置
        _selection1_outputs = self.document_outputs(selection1_outputs[0])
        doc_selection = _selection1_outputs.reshape(B, Doc_size, L_total, -1)
        E = selection1_outputs[0].size(-1)  # Embedding大小
        # 取[DOC]的位置
        doc_selection_output = torch.index_select(doc_selection, 2, DOC_index).squeeze(2)  # size: B*D*2
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
                        # ipdb.set_trace()

        # 从docpair里选对应文章出来
        phase_2_question_ids = question_ids.expand(B, 3, L_q)
        doc_pair = torch.Tensor(doc_pair).long().to(device)

        # ipdb.set_trace()

        phase_2_document_ids = torch.ones([B, 3, L_d * 2]).to(device)
        for b in range(B):
            for ii in range(3):
                # 2*L_d
                first = doc_pair[b, ii][0].item()
                second = doc_pair[b, ii][1].item()
                first_document_ids = document_ids[b][first][: doc_length[b][first]]
                second_document_ids = document_ids[b][second][: doc_length[b][second]]

                pair = torch.cat((first_document_ids, second_document_ids), dim=-1)
                # ipdb.set_trace()
                phase_2_document_ids[b, ii][: len(pair)] = pair

        phase_2_document_attention_mask = torch.zeros([B, 3, L_d * 2]).to(device)
        for b in range(B):
            for ii in range(3):
                # 2*L_d
                first = doc_pair[b, ii][0].item()
                second = doc_pair[b, ii][1].item()
                first_document_attention_mask = document_attention_mask[b][first][: doc_length[b][first]]
                second_document_attention_mask = document_attention_mask[b][second][: doc_length[b][second]]
                pair = torch.cat((first_document_attention_mask, second_document_attention_mask), dim=-1,)
                phase_2_document_attention_mask[b, ii][: len(pair)] = pair
        phase_2_question_attention_mask_input = question_attention_mask.expand(B, 3, L_q)

        phase_2_total_input = torch.cat([phase_2_question_ids, phase_2_document_ids], dim=-1)
        phase_2_total_attention_mask = torch.cat([phase_2_question_attention_mask_input, phase_2_document_attention_mask], dim=-1,)
        if phase_2_total_input.size(-1) > 512:
            phase_2_total_input = phase_2_total_input[:, :, :512]
            phase_2_total_attention_mask = phase_2_total_attention_mask[:, :, :512]
        selection2_outputs = self.roberta(input_ids=phase_2_total_input.long().reshape(B * 3, -1), attention_mask=phase_2_total_attention_mask.reshape(B * 3, -1), token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,)
        # ipdb.set_trace()
        phase2_output = selection2_outputs[0].reshape(B, 3, -1, E)
        phase2_cls_output = phase2_output[:, :, 0, :]
        phase2_pair_selection = self.phase2_outputs(phase2_cls_output)  # B*3*2

        phase2_pair_selection_soft = F.softmax(phase2_pair_selection, dim=-1)
        phase2_pair_output = phase2_pair_selection_soft[:, :, -1]  # B*3
        phase2_pair = torch.argmax(phase2_pair_output, dim=-1)  # B*1
        # 最终选出的二元组
        selected_pair = torch.zeros([B, 2], device=device)
        for b in range(B):
            selected_pair[b] = doc_pair[b, phase2_pair[b].item()]

        if Ltotal is not None:
            phase2loss = CrossEntropyLoss()
            phase2_pair_selection_p = phase2_pair_selection.permute(0, 2, 1)  # B*2*3
            Lphase2 = phase2loss(phase2_pair_selection_p, pairlabels.long())

            Ltotal = Ltotal + Lphase2
        # print(gold_doc_pair, selected_pair)

        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,)

        sequence_output = outputs[0]

        B, L, E = sequence_output.size()
        device = sequence_output.device
        S = sentence_index_start.size(1)  ###最大句子数量
        sentence_output = torch.zeros([B, S, E], device=device)
        for b in range(B):
            sentence_output[b] = torch.index_select(sequence_output[b], 0, sentence_index_start[b])
        if self.use_sentencetrans:
            sentence_attention_mask = self.get_extended_attention_mask(seq_len_to_mask(sentence_num), sentence_output.size(), device)
            sentence_outputs = self.sentence_transformer(sentence_output, attention_mask=sentence_attention_mask, head_mask=head_mask, encoder_attention_mask=None, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,)
            sentence_select = self.sentence_outputs(sentence_outputs[0])  ### B*S*3
        else:
            sentence_select = self.sentence_outputs(sentence_output)
        # sentence_select=self.sentence_outputs(sentence_output)
        Lsentence = None
        if sentence_labels is not None:
            sentence_loss = CrossEntropyLoss()

            sentencemask = torch.Tensor([1e10, -1e10])
            sentencemask.to(device)
            for b in range(B):
                sentence_select[b, sentence_num[b] : -1] = sentencemask
                ### mask sentence

            sentence_select_p = sentence_select.permute(0, 2, 1)  ### B*3*S
            ### sentencemask
            Lsentence = sentence_loss(sentence_select_p, sentence_labels)
        sentence_select = torch.argmax(sentence_select, dim=-1)  ### BS
        _sentence_select = torch.nonzero(sentence_select)  ### 这一步应该转化为index
        if self.use_answertrans:

            answerspanmask = torch.zeros_like(sequence_output[:, :, 0])  ### BL
            for nonzero in _sentence_select:
                batch = nonzero[0]
                start, end = (
                    sentence_index_start[batch, nonzero[1]],
                    sentence_index_end[batch, nonzero[1]],
                )

                answerspanmask[batch, start:end] = 1

            extended_answerspanmask = self.get_extended_attention_mask(answerspanmask, sequence_output.size(), device)
            # answerspanout=sequence_output
            answerspanout = self.answer_transformer(sequence_output, attention_mask=extended_answerspanmask, head_mask=head_mask, encoder_attention_mask=None, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,)[0]
        else:
            answerspanout = sequence_output
        output_answer_type = self.answer_typeout(answerspanout[:, 0, :])  # B*3

        Ltype = None
        if answer_type is not None:
            typeloss = CrossEntropyLoss()

            Ltype = typeloss(output_answer_type, answer_type)
        ##
        logits = self.qa_outputs(answerspanout)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        Lspan = None
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            Lspan = (start_loss + end_loss) / 2

            total_loss = 2 * Lsentence + Lspan + Ltype
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return {
            "loss": total_loss + Ltotal,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
            "sentence_predictions": sentence_select,
            "type_logits": output_answer_type,
            "input_ids": input_ids,
            "selected_pair": selected_pair,
            "input_ids": input_ids,
        }
