from transformers.models.roberta import RobertaPreTrainedModel, RobertaModel
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import json
from collections import Counter
import fastNLP
import ipdb


class ARobertaForTeacherforcingDocselection(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.document_outputs = nn.Linear(config.hidden_size, 2)
        self.phase2_outputs = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(
        self,
        teacher_ids=None,
        labels_T=None,
        teacher_token_type_ids=None,
        teacher_attention_mask=None,
        DOC_index=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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
        # ipdb.set_trace()

        selection1_outputs = self.roberta(
            teacher_ids,
            attention_mask=teacher_attention_mask,
            token_type_ids=teacher_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        doc_selection = self.document_outputs(selection1_outputs[0])
        B = doc_selection.size(0)
        device = doc_selection.device
        # B*2
        doc_selection_output = torch.zeros(B, 2).to(device)
        for b in range(B):
            doc_selection_output[b] = doc_selection[b][DOC_index[b]]

        # ipdb.set_trace()

        L = None
        if labels_T is not None:
            document_loss = CrossEntropyLoss()
            L = document_loss(doc_selection_output, labels_T)

        doc_out = F.softmax(doc_selection_output, dim=-1)
        _, doc_out = torch.topk(doc_out, 1, dim=-1)  ##B

        return {"loss": L, "doc_out": doc_out}
