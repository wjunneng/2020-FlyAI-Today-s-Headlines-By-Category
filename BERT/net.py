# -*- coding: utf-8 -*-
from pytorch_transformers import BertModel, BertPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch


# ########################################## BERTOrigin

# class Net(BertPreTrainedModel):
#     def __init__(self, config):
#         super(Net, self).__init__(config)
#         self.num_labels = config.num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.init_weights()
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         """
#         Args:
#             input_ids: 词对应的 id
#             token_type_ids: 区分句子，0 为第一句，1表示第二句
#             attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
#         """
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
#         # pooled_output: [batch_size, dim=768]
#         pooled_output = self.dropout(pooled_output)
#
#         logits = self.classifier(pooled_output)
#         # logits: [batch_size, output_dim=2]
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits

# ########################################## BERTATT

class Net(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    """

    def __init__(self, config):
        super(Net, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.W_w = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.u_w = nn.Parameter(torch.Tensor(config.hidden_size, 1))

        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
            input_ids: 词对应的 id
            token_type_ids: 区分句子，0 为第一句，1表示第二句
            attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
        """
        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)

        # encoded_layers: [batch_size, seq_len, bert_dim=768]
        encoded_layers = self.dropout(encoded_layers)

        # score: [batch_size, seq_len, bert_dim]
        score = torch.tanh(torch.matmul(encoded_layers, self.W_w))

        # attention_weights: [batch_size, seq_len, 1]
        attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)

        # scored_x : [batch_size, seq_len, bert_dim]
        scored_x = encoded_layers * attention_weights

        # feat: [batch_size, bert_dim=768]
        feat = torch.sum(scored_x, dim=1)

        # logits: [batch_size, output_dim]
        logits = self.classifier(feat)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

# ########################################## BERTHAN

# class Net(BertPreTrainedModel):
#
#     def __init__(self, config):
#         super(Net, self).__init__(config=config)
#         self.num_labels = config.num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.init_weights()
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         """
#         Args:
#             input_ids: 词对应的 id
#             token_type_ids: 区分句子，0 为第一句，1表示第二句
#             attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
#         """
#         # flat_input_ids: [batch_size * sentence_num, seq_len]
#         flat_input_ids = input_ids.view(-1, input_ids.size(-1))
#
#         # flat_token_type_ids: [batch_size * sentence_num, seq_len]
#         flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
#
#         # flat_attention_mask: [batch_size * sentence_num, seq_len]
#         flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
#
#         # pooled_output: [batch_size * sentence_num, bert_dim]
#         _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask)
#
#         pooled_output = self.dropout(pooled_output)
#
#         # pooled_output: [batch_size, sentence_num, bert_dim]  sentence_num = 1
#         pooled_output = pooled_output.view(input_ids.size(0), 1, pooled_output.size(-1))
#         # pooled_output: [batch_size, bert_dim, sentence_num]
#         pooled_output = pooled_output.permute(0, 2, 1)
#
#         # pooled_output: [batch_size, bert_dim]
#         pooled_output = F.max_pool1d(pooled_output, pooled_output.size()[2]).squeeze(2)
#
#         # logits: [batch_size, num_labels]
#         logits = self.classifier(pooled_output)
#
#         reshape_logits = logits.view(-1, self.num_labels)
#
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(reshape_logits, labels.view(-1))
#             return loss
#         else:
#             return logits
