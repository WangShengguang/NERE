import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_ent_tags, ent_emb_dim, batch_size):
        super(BiLSTM, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = ent_emb_dim
        self.hidden_dim = 256
        self.num_ent_tags = num_ent_tags

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.ent_label_embeddings = nn.Embedding(num_ent_tags, ent_emb_dim)

        self.bi_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2,
                               num_layers=2, bidirectional=True)
        self.classifier = nn.Linear(self.hidden_dim, self.num_ent_tags)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.cel_loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        embeds = self.word_embeds(input_ids)  # batch_size,sequence_len,dim
        lstm_out, (h_n, h_c) = self.bi_lstm(embeds)  # (seq_len, batch_size, hide_dim)
        output = self.classifier(lstm_out)  # (batch_size, seq_len, num_ent_tags)
        # output = self.softmax(output) #若模型使用CrossEntropyLoss这个loss函数，则不应该在最后一层再使用softmax进行激活
        pred = torch.argmax(output, dim=-1)
        if labels is None:  # predict
            return pred
        else:  # train
            # https://blog.csdn.net/jiangpeng59/article/details/79583292
            loss = self.cel_loss(output.permute(0, 2, 1), labels)
            return pred, loss
