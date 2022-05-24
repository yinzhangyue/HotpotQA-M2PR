from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.albert import AlbertModel, AlbertPreTrainedModel
from transformers.models.roberta import RobertaPreTrainedModel, RobertaModel
from transformers import DebertaPreTrainedModel, DebertaV2PreTrainedModel
from transformers import DebertaModel, DebertaV2Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import Tensor


import os
import numpy as np
import json
import ipdb

import fastNLP
from fastNLP.core.metrics import MetricBase, seq_len_to_mask
from fastNLP.core.losses import LossBase

from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig

from model import *
from seqmixing import upcoverAttention
class ARobertaForSP(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.encoder_k = RobertaModel(config, add_pooling_layer=False)
                
        self.encoder_q = RobertaModel(config, add_pooling_layer=False)
        self.init_weights()
        for para in self.encoder_k.parameters():
            para.requires_grad=False


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
        document_labels=None,
        sentence_index_start=None,
        sentence_index_end=None,
        sentence_labels=None,
        answer_type=None,
        sentence_num=None,
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
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs_q = self.encoder_q(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs_k = self.encoder_k(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        question_output = outputs_q[0][:, 0]
        sequence_output = outputs_k[0]

        B, L, E = sequence_output.size()
        device = sequence_output.device
        S = sentence_index_start.size(1)  ###最大句子数量
        sentence_output = torch.zeros([B, S, E], device=device)
        for b in range(B):
            sentence_output[b] = torch.index_select(
                sequence_output[b], 0, sentence_index_start[b]
            )
        # sentence_select = self.sentence_outputs(sentence_output)
        t = 0.1
        sim = torch.zeros([B, S], device=device)
        for b in range(B):
        	for s in range(S):
        		sim[b][s] = torch.dot(question_output[b], sentence_output[b][s])
        sim = torch.exp(sim)/t

        Lsentence = - torch.log(torch.sum(sentence_labels * sim) / torch.sum(sim) / torch.sum(sentence_labels)).unsqueeze(0)

        m = 0.9
        self.encoder_k.parameters() = m * self.encoder_k.parameters() + (1 - m) * self.encoder_q.parameters()

        if not return_dict:
            return Lsentence
        return {
            "loss": Lsentence,
            "hidden_states": outputs_q.hidden_states,
            "attentions": outputs_q.attentions,
            "input_ids": input_ids,
        }