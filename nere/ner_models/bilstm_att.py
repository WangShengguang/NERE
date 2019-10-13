import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_ATT(nn.Module):
    def __init__(self, vocab_size, num_ent_tags, ent_emb_dim, batch_size, sequence_len):
        super(BiLSTM_ATT, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = ent_emb_dim
        self.hidden_dim = 256
        self.num_ent_tags = num_ent_tags

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.ent_label_embeddings = nn.Embedding(num_ent_tags, ent_emb_dim)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2,
                            num_layers=2, bidirectional=True)

        self.dropout_emb = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)
        self.dropout = nn.Dropout(0.5)

        self.att_weight = nn.Parameter(torch.randn(self.batch_size, sequence_len, self.hidden_dim))
        # todo
        # https://discuss.pytorch.org/t/define-the-number-of-in-feature-in-nn-linear-dynamically/31185/2
        # sequence_feature_len = 100
        # self.adaptive_max_pool1d = nn.AdaptiveMaxPool1d(sequence_feature_len)
        self.classifier = nn.Linear(self.hidden_dim, self.num_ent_tags)
        self.criterion_loss = nn.CrossEntropyLoss()

    def attention(self, H):
        """
        :param H: batch_size, hidden_dim, sequence_len
        :return:  batch_size, hidden_dim, sequence_len
        """
        M = F.tanh(H)
        a = F.softmax(torch.bmm(self.att_weight, M), 2)  # batch_size,sequence_len,sequence_len
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
        # embeds = torch.transpose(embeds, 0, 1)
        lstm_out, (h_n, h_c) = self.lstm(embeds)  # seq_len, batch, num_directions * hidden_size
        # lstm_out = lstm_out.permute(1, 2, 0)
        lstm_out = self.dropout_lstm(lstm_out)  # batch_size, hidden_dim, sequence_len
        att_out = F.tanh(self.attention(lstm_out))
        att_out = torch.transpose(att_out, 1, 2)
        all_features = self.dropout_att(att_out)
        logits = self.classifier(all_features)
        label_indices = logits.argmax(dim=1)
        if labels is None:
            return label_indices
        else:
            loss = self.criterion_loss(logits.view(-1, self.num_ent_tags), labels.view(-1))
            return label_indices, loss
