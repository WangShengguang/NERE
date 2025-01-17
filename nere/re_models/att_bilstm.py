import torch
import torch.nn as nn
import torch.nn.functional as F


class ATT_BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_ent_tags, num_rel_tags, ent_emb_dim, batch_size, sequence_len):
        super(ATT_BiLSTM, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = ent_emb_dim
        self.hidden_dim = 256
        self.num_rel_tags = num_rel_tags
        self.num_ent_tags = num_ent_tags

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.ent_label_embeddings = nn.Embedding(num_ent_tags, ent_emb_dim)

        self.bi_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2,
                               num_layers=2, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.num_rel_tags)

        self.dropout_emb = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)
        self.dropout = nn.Dropout(0.5)

        self.att_weight = nn.Parameter(torch.randn(self.batch_size, sequence_len, self.embedding_dim))

        self.classifier = nn.Linear(ent_emb_dim * 2 + self.hidden_dim * 3, self.num_rel_tags)
        self.criterion_loss = nn.CrossEntropyLoss(size_average=True)

    def attention(self, H):
        M = F.tanh(H)
        a = F.softmax(torch.bmm(self.att_weight, M.permute(0, 2, 1)), 2)
        return torch.bmm(a, H)

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
        ent_label_features = self.ent_label_embeddings(ent_labels).view(self.batch_size, -1)

        # embeds = torch.cat((self.word_embeds(sents), self.pos1_embeds(pos1), self.pos2_embeds(pos2)), 2)
        embeds = self.word_embeds(sents)
        att_out = self.attention(embeds)

        lstm_out, (h_n, h_c) = self.bi_lstm(att_out)  # sequence_len, batch_size, hidden_dim

        # _lstm_out = torch.transpose(lstm_out, 0, 1)  # batch_size, sequence_len, hidden_dim
        # shape: (batch_size, hidden_size)
        e1_features = self.get_ent_features(lstm_out, e1_masks)
        e2_features = self.get_ent_features(lstm_out, e2_masks)

        # lstm_out = self.dropout_lstm(lstm_out[-1])  # sequence_len, hidden_dim, batch_size

        all_features = torch.cat((ent_label_features, lstm_out[:, -1, :], e1_features, e2_features), dim=1)
        all_features = self.dropout(all_features)

        logits = self.classifier(all_features)
        label_indices = logits.argmax(dim=1)
        if rel_labels is None:
            return label_indices
        else:
            loss = self.criterion_loss(logits, rel_labels)
            return label_indices, loss
