#!/usr/bin/env python
# coding: utf-8
"""
Source of code: https://github.com/gilmoright/CommandClassifier
"""
from simpletransformers.config.model_args import ModelArgs
from dataclasses import asdict, dataclass, field, fields


@dataclass
class MyMultiLabelClassificationArgs(ModelArgs):
    """
    Model args for a MultiLabelClassificationModel
    """

    model_class: str = "MyMultiLabelClassificationModel"
    sliding_window: bool = False
    stride: float = 0.8
    threshold: float = 0.5
    tie_value: int = 1
    labels_list: list = field(default_factory=list)
    labels_map: dict = field(default_factory=dict)
    lazy_loading: bool = False
    special_tokens_list: list = field(default_factory=list)
    # num_sublabels_per_biglabel: list = field(default_factory=list)


import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    BertModel,
    BertPreTrainedModel
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_


class MyAttentionOutput(torch.nn.Module):
    def __init__(self, hidden_size, num_sublabels_per_biglabel, num_labels, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MyAttentionOutput, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_sublabels_per_biglabel = num_sublabels_per_biglabel
        self.seqvec_to_query_linear = torch.nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.output_classifier = torch.nn.Linear(hidden_size * 2, max(num_sublabels_per_biglabel), **factory_kwargs)

    def forward(self, outputs, pooled_output):
        pooled_output = pooled_output.view(outputs[0].shape[0], -1, self.hidden_size)
        seqvec_for_every_label = pooled_output.repeat([1, len(self.num_sublabels_per_biglabel), 1])
        seqvec_for_every_label_transformed = self.seqvec_to_query_linear(seqvec_for_every_label)
        seqvec_for_every_label_transformed += seqvec_for_every_label
        att_output = torch.nn.functional._scaled_dot_product_attention(seqvec_for_every_label_transformed, outputs[0],
                                                                       outputs[0])
        concated = torch.concat([att_output[0], seqvec_for_every_label], dim=-1)
        outputs_for_labelxclasses = self.output_classifier(concated)
        logits = torch.zeros((outputs_for_labelxclasses.shape[0], self.num_labels),
                             device=outputs_for_labelxclasses.device)
        shift = 0
        assert len(outputs_for_labelxclasses.shape) == 3 and outputs_for_labelxclasses.shape[1] == len(
            self.num_sublabels_per_biglabel)
        for i, num_sublabels in enumerate(self.num_sublabels_per_biglabel):
            logits[:, shift:shift + num_sublabels] = outputs_for_labelxclasses[:, i, :num_sublabels]
            shift += num_sublabels
        assert shift == self.num_labels
        return logits


class MyBertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Bert model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None, num_sublabels_per_biglabel=[], add_attention_for_labels=False,
                 device=None, dtype=None):
        super(MyBertForMultiLabelSequenceClassification, self).__init__(config)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_labels = config.num_labels
        # print(config)
        self.num_sublabels_per_biglabel = num_sublabels_per_biglabel
        self.add_attention_for_labels = add_attention_for_labels

        if self.add_attention_for_labels:
            # self.bert = BertModel(config, add_pooling_layer = False)
            self.bert = BertModel(config, add_pooling_layer=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.myattentionoutput = MyAttentionOutput(config.hidden_size, num_sublabels_per_biglabel,
                                                       config.num_labels)
            # self.label_query_weights = torch.nn.Parameter(torch.empty((len(num_sublabels_per_biglabel), config.hidden_size), **factory_kwargs))  # batch?
            # self.output_classifier_weights = torch.nn.Parameter(torch.empty((config.hidden_size * 2, max(num_sublabels_per_biglabel)), **factory_kwargs))  # batch?
            # self.label_query_weights = torch.nn.Parameter(data=torch.Tensor(len(num_sublabels_per_biglabel), config.hidden_size), requires_grad=True)  # batch?
            # self.output_classifier_weights = torch.nn.Parameter(data=torch.Tensor(config.hidden_size * 2, max(num_sublabels_per_biglabel)), requires_grad=True)  # batch?
            # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        else:
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pos_weight = pos_weight

        self.init_weights()

    def _reset_parameters(self):
        if self.add_attention_for_labels:
            xavier_uniform_(self.label_query_weights)
            xavier_uniform_(self.output_classifier_weights)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        if self.add_attention_for_labels:
            """
            att_output = torch.nn.functional._scaled_dot_product_attention(self.label_query_weights, outputs[0], outputs[0])
            seqvec_for_every_label = pooled_output.repeat([len(self.num_sublabels_per_biglabel),1])
            concated = torch.concat([att_output[0], seqvec_for_every_label], dim=-1)
            outputs_for_labelxclasses = torch.bmm(concated, self.output_classifier_weights)
            logits = torch.zeros(self.num_labels)
            shift = 0
            assert len(outputs_for_labelxclasses.shape)==2, outputs_for_labelxclasses.shape[0]==len(self.num_sublabels_per_biglabel)
            for i, num_sublabels in enumerate(self.num_sublabels_per_biglabel):
                logits[shift:shift + num_sublabels] = outputs_for_labelxclasses[i,:num_sublabels]
                shift += num_sublabels
            assert shift == self.num_labels
            """
            logits = self.myattentionoutput(outputs, pooled_output)
        else:
            logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[
                              2:
                              ]  # add hidden states and attention if they are here
        if labels is not None:
            losses = []
            shift = 0
            logits = logits.view(-1, self.num_labels)
            labels = labels.float()
            labels = labels.view(-1, self.num_labels)
            pos_weight = None
            for num_sublabels in self.num_sublabels_per_biglabel:
                if self.pos_weight is not None:
                    pos_weight = self.pos_weight[shift:shift + num_sublabels]
                loss_fct = CrossEntropyLoss(weight=pos_weight)
                biglabel_logits = logits[:, shift:shift + num_sublabels]
                biglabel_labels = labels[:, shift:shift + num_sublabels]
                loss = loss_fct(biglabel_logits, biglabel_labels)
                losses.append(loss)
                shift += num_sublabels
            assert shift == self.num_labels
            loss = sum(losses) / len(losses)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
