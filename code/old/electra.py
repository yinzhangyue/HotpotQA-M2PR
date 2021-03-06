from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import json
import ipdb
import fastNLP
from fastNLP.core.metrics import seq_len_to_mask


from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaConfig
from transformers.models.electra import (
    ElectraPreTrainedModel,
    ElectraConfig,
    ElectraModel,
)


class AElectraForQuestionAnswering(ElectraPreTrainedModel):
    config_class = ElectraConfig
    base_model_prefix = "electra"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_sentencetrans = False
        self.use_answertrans = False

        self.electra = ElectraModel(config)
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
        sentence_index:??????????????????<e>???index???????????????sentence???????????????.B*sentencenumber*2
        sentece_labels:??????????????????sp.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.electra(
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
        S = sentence_index_start.size(1)  ###??????????????????
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
        _sentence_select = torch.nonzero(sentence_select)  #########????????????????????????index
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
            # answerspanout=sequence_output###########????????????
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
