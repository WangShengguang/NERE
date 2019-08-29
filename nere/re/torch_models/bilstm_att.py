# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

from nere.config import Config


class BiLSTM_ATT(nn.Module):
    def __init__(self, vocab_size, num_ent_tags, num_rel_tags):
        super(BiLSTM_ATT, self).__init__()
        self.batch_size = Config.batch_size

        self.vocab_size = vocab_size
        self.embedding_dim = Config.ent_emb_dim

        self.hidden_dim = 256
        self.num_rel_tags = num_rel_tags

        self.num_ent_tags = num_ent_tags
        self.ent_emb_dim = Config.ent_emb_dim

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.pos1_embeds = nn.Embedding(Config.max_sequence_len, self.embedding_dim)
        self.pos2_embeds = nn.Embedding(Config.max_sequence_len, self.embedding_dim)
        self.relation_embeds = nn.Embedding(self.num_rel_tags, self.hidden_dim)

        self.lstm = nn.LSTM(input_size=self.embedding_dim + self.ent_emb_dim * 2, hidden_size=self.hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.num_rel_tags)

        self.dropout_emb = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)

        # self.hidden = self.init_hidden()

        self.att_weight = nn.Parameter(torch.randn(self.batch_size, 1, self.hidden_dim))
        self.relation_bias = nn.Parameter(torch.randn(self.batch_size, self.num_rel_tags, 1))
        self.criterion_loss = nn.CrossEntropyLoss(size_average=True)

    # def init_hidden(self):
    #     return torch.randn(2, self.batch_size, self.hidden_dim // 2).to(Config.device)

    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2).to(Config.device),
                torch.randn(2, self.batch_size, self.hidden_dim // 2).to(Config.device))

    def attention(self, H):
        M = F.tanh(H)
        a = F.softmax(torch.bmm(self.att_weight, M), 2)
        a = torch.transpose(a, 1, 2)
        return torch.bmm(H, a)

    def forward(self, batch_data, rel_labels=None):
        sents = batch_data["sents"]
        pos1 = batch_data['pos1']
        pos2 = batch_data['pos2']
        self.hidden = self.init_hidden_lstm()

        embeds = torch.cat((self.word_embeds(sents), self.pos1_embeds(pos1), self.pos2_embeds(pos2)), 2)

        embeds = torch.transpose(embeds, 0, 1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)

        lstm_out = self.dropout_lstm(lstm_out)
        att_out = F.tanh(self.attention(lstm_out))
        # att_out = self.dropout_att(att_out)

        relation = torch.tensor([i for i in range(self.num_rel_tags)], dtype=torch.long).repeat(self.batch_size, 1)

        relation = self.relation_embeds(relation)

        res = torch.add(torch.bmm(relation, att_out), self.relation_bias)

        res = F.softmax(res, 1)
        logits = res.view(self.batch_size, -1)
        if rel_labels is None:
            _, label_indices = logits.max(dim=1)
            return label_indices
        else:
            loss = self.criterion_loss(logits, rel_labels)
            return loss
