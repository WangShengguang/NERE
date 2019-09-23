import torch
import torch.nn.functional as F
from keras.utils import to_categorical
from torch import nn

from config import Config


class ACNN(nn.Module):
    def __init__(self, vocab_size, num_ent_tags, num_rel_tags, ent_emb_dim, max_len):
        super(ACNN, self).__init__()
        slide_window = 3
        pos_embed_size = ent_emb_dim
        num_filters = 3

        self.embedding_size = ent_emb_dim
        self.num_rel_tags = num_rel_tags

        self.d = self.embedding_size + 2 * pos_embed_size
        self.p = (slide_window - 1) // 2
        self.x_embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.rel_embedding = nn.Embedding(num_rel_tags, ent_emb_dim)
        self.rel_weight = nn.Parameter(torch.randn(num_rel_tags, num_filters))
        self.dist_embedding = nn.Embedding(max_len, pos_embed_size)
        self.dropout = nn.Dropout(0.6)
        self.conv = nn.Conv2d(1, num_filters, (slide_window, self.d), (1, self.d), (self.p, 0), bias=True)
        self.U = nn.Parameter(torch.randn(num_filters, num_rel_tags))
        self.batch_y_zeros = torch.zeros(Config.batch_size, self.num_rel_tags).to(Config.device)
        self.margin = 1.0

    def loss(self, wo_norm, all_distance, in_y):
        y_one_hot = self.batch_y_zeros.scatter(1, in_y.reshape([-1, 1]), 1)
        masking_y = torch.mul(y_one_hot, 10000)
        neg_y = torch.argmin(torch.add(all_distance, masking_y), dim=1)  # bz,
        neg_y_one_hot = self.batch_y_zeros.scatter(1, neg_y.reshape([-1, 1]), 1)
        _neg_y = torch.mm(neg_y_one_hot, self.rel_weight)
        neg_distance = torch.norm(wo_norm - F.normalize(_neg_y, dim=1), dim=1)
        _y = torch.mm(y_one_hot, self.rel_weight)
        pos_distance = torch.norm(wo_norm - F.normalize(_y, dim=1), dim=1)
        loss = torch.mean(self.margin + pos_distance - neg_distance)
        return loss

    def input_attention(self, batch_data):
        sents = batch_data["sents"]  # [bz,n]
        ent_labels = batch_data["ent_labels"]
        e1, e2 = ent_labels.unbind(dim=1)  # [bz]
        d1 = batch_data["pos1"]  # [bz,n]
        d2 = batch_data["pos2"]  # [bz,n]
        x_emb = self.x_embedding(sents)  # (bs, n, dw)
        e1_emb = self.x_embedding(e1.unsqueeze(-1))
        e2_emb = self.x_embedding(e2.unsqueeze(-1))
        dist1_emb = self.dist_embedding(d1)
        dist2_emb = self.dist_embedding(d2)
        x_cat = torch.cat((x_emb, dist1_emb, dist2_emb), 2)  # .reshape([Config.batch_size, sents.shape[1], self.d, 1])
        ine1_aw = F.softmax(torch.bmm(x_emb, e1_emb.transpose(2, 1)), 1)  # (bs, n, 1)
        ine2_aw = F.softmax(torch.bmm(x_emb, e2_emb.transpose(2, 1)), 1)
        in_aw = (ine1_aw + ine2_aw) / 2
        R = torch.mul(x_cat, in_aw)
        return R

    def attentive_pooling(self, R_star):
        """
        :param R_star: bz, num_filters,n
        :return:  (batch_size,num_filters)
        """
        RU = torch.matmul(R_star.transpose(2, 1), self.U)  # (bs, n, num_rel_tags)
        G = torch.matmul(RU, self.rel_weight)  # (bs, n, num_filters)
        AP = F.softmax(G, dim=1)
        RA = torch.mul(R_star, AP.transpose(2, 1))  # (bz,num_filters,num_filters)
        wo = nn.MaxPool1d(G.shape[1], stride=1)(RA).squeeze(-1)  # G.shape[1]==n
        return wo

    def forward(self, batch_data, rel_label=None):
        R = self.input_attention(batch_data)
        R_star = self.conv(R.unsqueeze(1)).squeeze(-1)  # (bs, num_filters, n)
        R_star = torch.tanh(R_star)
        wo = self.attentive_pooling(R_star)
        wo_norm = F.normalize(wo)  # (bctch_size, num_filters)  in_y (bctch_size, num_classes)
        wo_norm_tile = wo_norm.unsqueeze(1).repeat(1, self.num_rel_tags, 1)  # (bs, num_classes, num_filters)
        all_distance = torch.norm(wo_norm_tile - F.normalize(self.rel_weight, dim=1), dim=2)
        # all_distance = torch.norm(wo_norm_tile - rel_label_embed, 2, 2)  # (bs, num_classes)
        if rel_label is not None:
            loss = self.loss(wo_norm, all_distance, rel_label)
            return loss
        else:
            predict = torch.argmin(all_distance, dim=1)
            # predict = torch.argmax(all_distance, dim=1)
            return predict
