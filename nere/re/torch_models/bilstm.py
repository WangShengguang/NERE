# coding:utf8
import torch
import torch.nn as nn

from config import Config


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_ent_tags, num_rel_tags):
        super(BiLSTM, self).__init__()
        self.batch_size = Config.batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = Config.ent_emb_dim
        self.hidden_dim = 256
        self.num_rel_tags = num_rel_tags
        self.num_ent_tags = num_ent_tags
        self.ent_emb_dim = Config.ent_emb_dim

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.ent_label_embeddings = nn.Embedding(num_ent_tags, Config.ent_emb_dim)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2,
                            num_layers=2, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.num_rel_tags)

        self.dropout_emb = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(Config.ent_emb_dim * 2 + self.hidden_dim * 3, num_rel_tags)
        self.criterion_loss = nn.CrossEntropyLoss(size_average=True)

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

    def forward(self, batch_data, rel_labels=None):
        sents = batch_data["sents"]
        e1_masks = batch_data['e1_masks']
        e2_masks = batch_data['e2_masks']
        ent_labels = batch_data["ent_labels"]
        # self.hidden = self.init_hidden_lstm()
        batch_size = sents.size(0)
        ent_label_features = self.ent_label_embeddings(ent_labels).view(batch_size, -1)

        embeds = self.word_embeds(sents)
        embeds = torch.transpose(embeds, 0, 1)

        lstm_out, (h_n, h_c) = self.lstm(embeds)  # (seq_len, batch, num_directions * hidden_size)

        lstm_out = torch.transpose(lstm_out, 0, 1)
        # shape: (batch_size, hidden_size)
        e1_features = self.get_ent_features(lstm_out, e1_masks)
        e2_features = self.get_ent_features(lstm_out, e2_masks)
        lstm_out = self.dropout_lstm(lstm_out)
        # import ipdb
        # ipdb.set_trace()
        all_features = torch.cat((ent_label_features, lstm_out[:, -1, :], e1_features, e2_features), dim=1)
        all_features = self.dropout(all_features)

        logits = self.classifier(all_features)
        if rel_labels is None:
            _, label_indices = logits.max(dim=1)
            return label_indices
        else:
            loss = self.criterion_loss(logits, rel_labels)
            return loss
