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
import torch
import torch.nn as nn

from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from transformers import RobertaModel, RobertaForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaEncoder
from transformers.models.albert.modeling_albert import AlbertTransformer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutput,
)
from fastNLP.core.metrics import MetricBase, seq_len_to_mask
from fastNLP.core.losses import LossBase
from torch.nn import CosineSimilarity
import fastNLP
from torch import Tensor
import math

from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig

from model import *


class GuoMaskingRobertaEncoder(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        # self.mask_weights = nn.Parameter(torch.randn(config.num_hidden_layers))
        self.mask_weights = nn.Parameter(torch.ones(config.num_hidden_layers) * 0.55)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None

        bsz, seq_len, hidden_dim = hidden_states.shape
        attention_probs = torch.zeros(
            [bsz, self.config.num_attention_heads, seq_len, seq_len],
            device=hidden_states.device,
        )

        for i, layer_module in enumerate(self.layer):
            mask = self.mask_weights[i] * attention_probs
            # mask = 0.55 * attention_probs
            if attention_mask is not None:
                mask = mask + attention_mask

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                raise NotImplementedError
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions=True,
                )

            hidden_states = layer_outputs[0]
            attention_probs = layer_outputs[1]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


###################
class GuoMaskingAlbertEncoder(AlbertTransformer):
    def __init__(self, config):
        super().__init__(config)
        # self.mask_weights = nn.Parameter(torch.randn(config.num_hidden_layers))
        self.mask_weights = nn.Parameter(torch.ones(config.num_hidden_layers) * 0.55)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        head_mask = (
            [None] * self.config.num_hidden_layers if head_mask is None else head_mask
        )
        bsz, seq_len, hidden_dim = hidden_states.shape
        attention_probs = torch.zeros(
            [bsz, self.config.num_attention_heads, seq_len, seq_len],
            device=hidden_states.device,
        )

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            # ipdb.set_trace()
            if i > 0:
                mask = self.mask_weights[i] * attention_probs[0]
            else:
                mask = self.mask_weights[i] * attention_probs
            # mask = 0.55 * attention_probs
            if attention_mask is not None:
                mask = mask + attention_mask
            layers_per_group = int(
                self.config.num_hidden_layers / self.config.num_hidden_groups
            )

            # Index of the hidden group
            group_idx = int(
                i / (self.config.num_hidden_layers / self.config.num_hidden_groups)
            )

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                mask,
                head_mask[
                    group_idx * layers_per_group : (group_idx + 1) * layers_per_group
                ],
                output_attentions=True,
                output_hidden_states=output_hidden_states,
            )
            hidden_states = layer_group_output[0]
            attention_probs = layer_group_output[1]
            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class Guomasking_ARobertaForQuestionAnswering(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_sentencetrans = True
        self.use_answertrans = True

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.document_outputs = nn.Linear(config.hidden_size, 2)
        self.sentence_outputs = nn.Linear(config.hidden_size, 2)
        self.answer_typeout = nn.Linear(config.hidden_size, 3)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.roberta_config = RobertaConfig.from_pretrained("roberta-base")
        self.roberta_config.hidden_size = config.hidden_size
        self.roberta_config.intermediate_size = config.intermediate_size
        self.roberta_config.num_attention_heads = config.num_attention_heads
        self.roberta_config.num_attention_heads = config.num_attention_heads
        self.sentence_transformer = RobertaEncoder(self.roberta_config)
        self.answer_transformer = RobertaEncoder(self.roberta_config)
        self.init_weights()
        self.roberta.encoder = GuoMaskingRobertaEncoder(config)

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

        outputs = self.roberta(
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

        sequence_output = outputs[0]

        B, L, E = sequence_output.size()
        device = sequence_output.device
        S = sentence_index_start.size(1)  ###最大句子数量
        sentence_output = torch.zeros([B, S, E], device=device)
        for b in range(B):

            sentence_output[b] = torch.index_select(
                sequence_output[b], 0, sentence_index_start[b]
            )
        if self.use_sentencetrans:
            sentence_attention_mask = self.get_extended_attention_mask(
                seq_len_to_mask(sentence_num), sentence_output.size(), device
            )
            sentence_outputs = self.sentence_transformer(
                sentence_output,
                attention_mask=sentence_attention_mask,
                head_mask=head_mask,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sentence_select = self.sentence_outputs(sentence_outputs[0])  #####B*S*3
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
                #######mask sentence

            sentence_select_p = sentence_select.permute(0, 2, 1)  #####B*3*S
            ##########sentencemask
            Lsentence = sentence_loss(sentence_select_p, sentence_labels)
        sentence_select = torch.argmax(sentence_select, dim=-1)  ##BS
        _sentence_select = torch.nonzero(sentence_select)  #########这一步应该转化为index
        if self.use_answertrans:

            answerspanmask = torch.zeros_like(sequence_output[:, :, 0])  ######BL
            for nonzero in _sentence_select:
                batch = nonzero[0]
                start, end = (
                    sentence_index_start[batch, nonzero[1]],
                    sentence_index_end[batch, nonzero[1]],
                )

                answerspanmask[batch, start:end] = 1

            extended_answerspanmask = self.get_extended_attention_mask(
                answerspanmask, sequence_output.size(), device
            )
            # import ipdb
            # ipdb.set_trace()
            # answerspanout=sequence_output###########暂时替代
            answerspanout = self.answer_transformer(
                sequence_output,
                attention_mask=extended_answerspanmask,
                head_mask=head_mask,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[
                0
            ]  ##BLE
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
            "loss": total_loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
            "sentence_predictions": sentence_select,
            "type_logits": output_answer_type,
            "input_ids": input_ids,
        }


class Guomasking_AAlbertForQuestionAnswering(AlbertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_sentencetrans = False
        self.use_answertrans = False

        self.albert = AlbertModel(config, add_pooling_layer=False)
        self.document_outputs = nn.Linear(config.hidden_size, 2)
        self.sentence_outputs = nn.Linear(config.hidden_size, 2)
        self.answer_typeout = nn.Linear(config.hidden_size, 3)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.roberta_config = RobertaConfig.from_pretrained("roberta-base")
        self.roberta_config.hidden_size = config.hidden_size
        self.roberta_config.intermediate_size = config.intermediate_size
        self.roberta_config.num_attention_heads = config.num_attention_heads
        self.roberta_config.num_hidden_layers = 2
        self.sentence_transformer = RobertaEncoder(self.roberta_config)
        self.answer_transformer = RobertaEncoder(self.roberta_config)
        self.init_weights()
        self.albert.encoder = GuoMaskingAlbertEncoder(config)

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

        outputs = self.albert(
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

        sequence_output = outputs[0]

        B, L, E = sequence_output.size()
        device = sequence_output.device
        S = sentence_index_start.size(1)  ###最大句子数量
        sentence_output = torch.zeros([B, S, E], device=device)
        for b in range(B):

            sentence_output[b] = torch.index_select(
                sequence_output[b], 0, sentence_index_start[b]
            )
        if self.use_sentencetrans:
            sentence_attention_mask = self.get_extended_attention_mask(
                seq_len_to_mask(sentence_num), sentence_output.size(), device
            )
            sentence_outputs = self.sentence_transformer(
                sentence_output,
                attention_mask=sentence_attention_mask,
                head_mask=head_mask,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sentence_select = self.sentence_outputs(sentence_outputs[0])  #####B*S*3
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
                #######mask sentence

            sentence_select_p = sentence_select.permute(0, 2, 1)  #####B*3*S
            ##########sentencemask
            Lsentence = sentence_loss(sentence_select_p, sentence_labels)
        sentence_select = torch.argmax(sentence_select, dim=-1)  ##BS
        _sentence_select = torch.nonzero(sentence_select)  #########这一步应该转化为index
        if self.use_answertrans:

            answerspanmask = torch.zeros_like(sequence_output[:, :, 0])  ######BL
            for nonzero in _sentence_select:
                batch = nonzero[0]
                start, end = (
                    sentence_index_start[batch, nonzero[1]],
                    sentence_index_end[batch, nonzero[1]],
                )

                answerspanmask[batch, start:end] = 1

            extended_answerspanmask = self.get_extended_attention_mask(
                answerspanmask, sequence_output.size(), device
            )
            # import ipdb
            # ipdb.set_trace()
            # answerspanout=sequence_output###########暂时替代
            answerspanout = self.answer_transformer(
                sequence_output,
                attention_mask=extended_answerspanmask,
                head_mask=head_mask,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[
                0
            ]  ##BLE
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
            "loss": total_loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
            "sentence_predictions": sentence_select,
            "type_logits": output_answer_type,
            "input_ids": input_ids,
        }
