# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class BiLSTM_ATT(nn.Module):
    def __init__(self, vocab_size, num_ent_tags):
        super(BiLSTM_ATT, self).__init__()
        self.batch_size = Config.batch_size
        self.sequence_len = Config.max_sequence_len
        self.vocab_size = vocab_size
        self.embedding_dim = Config.ent_emb_dim
        self.hidden_dim = 256
        self.num_ent_tags = num_ent_tags
        self.ent_emb_dim = Config.ent_emb_dim

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.ent_label_embeddings = nn.Embedding(num_ent_tags, Config.ent_emb_dim)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2,
                            num_layers=2, bidirectional=True)

        self.dropout_emb = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)
        self.dropout = nn.Dropout(0.5)

        self.att_weight = nn.Parameter(torch.randn(self.batch_size, 1, self.hidden_dim))
        self.classifier = nn.Linear(self.hidden_dim, self.sequence_len).to(Config.device)
        self.criterion_loss = nn.CrossEntropyLoss()

    def attention(self, H):
        M = F.tanh(H)
        a = F.softmax(torch.bmm(self.att_weight, M), 2)
        a = torch.transpose(a, 1, 2)
        return torch.bmm(H, a)

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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        embeds = self.word_embeds(input_ids)
        embeds = torch.transpose(embeds, 0, 1)
        lstm_out, (h_n, h_c) = self.lstm(embeds)  # seq_len, batch, num_directions * hidden_size
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = self.dropout_lstm(lstm_out)
        att_out = F.tanh(self.attention(lstm_out)).view(self.batch_size, -1)
        all_features = self.dropout(att_out)
        logits = self.classifier(all_features)
        if labels is None:
            _, label_indices = logits.max(dim=1)
            return label_indices
        else:
            # import ipdb
            # ipdb.set_trace()
            # TODO ERROR
            loss = self.criterion_loss(logits.view(-1, input_ids.shape[1]), labels.view(-1))
            return loss
