"""BERT + Softmax for named entity recognition"""

import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel, gelu
from torch.nn import CrossEntropyLoss

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BERTSoftmax(BertPreTrainedModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Args:
        config: a BertConfig class instance with the configuration to build a new model.
        num_labels: the number of classes for the classifier. Default = 2.
    
    Inputs:
        input_ids: a torch.LongTensor of shape (batch_size, seq_length)
            with the word token indices in the vocabulary.
        token_type_ids: an optional torch.LongTensor of shape (batch_size, seq_length) with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token.
        attention_mask: an optional torch.LongTensor of shape (batch_size, seq_length) with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        labels: labels for the classification output: torch.LongTensor of shape (batch_size, seq_length)
            with indices selected in [0, ..., num_labels].

    Returns:
        if labels is not None:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if labels is None:
            Outputs the classification labels indices of shape (batch_size, seq_length).
    """

    def __init__(self, config, num_labels, ent_num_lables=20):
        super(BERTSoftmax, self).__init__(config)
        self.num_labels = num_labels  # the number of relation labels
        ent_label_dim = 128
        self.ent_label_embeddings = nn.Embedding(ent_num_lables, ent_label_dim)
        self.ent_emb_layer_norm = BertLayerNorm(ent_label_dim * 2, eps=1e-12)

        self.fused_layer_norm_1 = BertLayerNorm(config.hidden_size * 3, eps=1e-12)
        self.fused_layer_norm_2 = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.fused_layer = nn.Linear(config.hidden_size * 3, config.hidden_size)
        # features_size = ent_label_dim * 2 + config.hidden_size * 3
        self.classifier = nn.Linear(ent_label_dim * 2 + config.hidden_size, num_labels)

        self.init_weights()

        self.loss_fct = CrossEntropyLoss()

    # def init_bert_weights(self, module):
    #     """ Initialize the weights.
    #     """
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #     elif isinstance(module, BertLayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     if isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_()

    def init_weights(self):
        self.apply(self.init_bert_weights)

        self.ent_label_embeddings.weight.data.normal_(mean=0.0, std=0.2)
        self.ent_emb_layer_norm.weight.data.fill_(1.0)
        self.ent_emb_layer_norm.bias.data.zero_()

        self.fused_layer_norm_1.weight.data.fill_(1.0)
        self.fused_layer_norm_1.bias.data.zero_()

        self.fused_layer_norm_2.weight.data.fill_(1.0)
        self.fused_layer_norm_2.bias.data.zero_()

        self.fused_layer.weight.data.normal_(mean=0.0, std=0.2)
        self.fused_layer.bias.data.zero_()

        self.classifier.weight.data.normal_(mean=0.0, std=0.2)
        self.classifier.bias.data.zero_()

    def get_ent_features(self, seq_output, ent_masks):
        """
        Args:
            seq_output: (batch_size, seq_length, hidden_size)
            ent_mentions: (batch_size, 2, ent_label_dim), 2: 2 mentions, ent_label_dim: `start` and `end` indices
        """
        # shape: (batch_size, seq_length, hidden_size)
        # ent_masks_expand = ent_masks.unsqueeze(-1).expand_as(seq_output).half()
        ent_masks_expand = ent_masks.unsqueeze(-1).expand_as(seq_output).float()
        # shape: (batch_size, 1)
        # ent_masks_sum = ent_masks.sum(dim=1).unsqueeze(1).half()
        ent_masks_sum = ent_masks.sum(dim=1).unsqueeze(1).float()
        ones = torch.ones_like(ent_masks_sum)
        ent_masks_sum = torch.where(ent_masks_sum > 0, ent_masks_sum, ones)
        # shape: (batch_size, hidden_size)
        ent_features = seq_output.mul(ent_masks_expand).sum(dim=1).div(ent_masks_sum)
        return ent_features

    def forward(self, batch_data, labels=None):
        ent_labels = batch_data['ent_labels']
        e1_masks = batch_data['e1_masks']
        e2_masks = batch_data['e2_masks']
        sents = batch_data['sents']

        masks = batch_data['sents'].gt(0)

        batch_size = sents.size(0)
        # shape: (batch_size, ent_label_dim*2)
        ent_label_features = self.ent_label_embeddings(ent_labels).view(batch_size, -1)
        ent_label_features = self.ent_emb_layer_norm(ent_label_features)

        # shape: (batch_size, seq_length, hidden_size), (batch_size, hidden_size)
        seq_output, pooled_output = self.bert(sents, attention_mask=masks, output_all_encoded_layers=False)

        # shape: (batch_size, hidden_size)
        e1_features = self.get_ent_features(seq_output, e1_masks)
        e2_features = self.get_ent_features(seq_output, e2_masks)

        # shape: (batch_size, hidden_size*3)
        fused_features = torch.cat((e1_features, e2_features, pooled_output), dim=1)
        fused_features = self.fused_layer_norm_1(fused_features)
        fused_features = self.dropout(fused_features)
        # shape: (batch_size, hidden_size)
        fused_features = self.fused_layer(fused_features)
        fused_features = gelu(fused_features)
        fused_features = self.fused_layer_norm_2(fused_features)

        all_features = torch.cat((ent_label_features, fused_features), dim=1)

        # shape: (batch_size, features_size)
        all_features = self.dropout(all_features)

        logits = self.classifier(all_features)
        label_indices = logits.argmax(dim=1)

        if labels is None:
            return label_indices
        else:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return label_indices, loss
