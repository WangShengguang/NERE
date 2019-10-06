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

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2,
                            num_layers=2, bidirectional=True)
        self.classifier = nn.Linear(self.hidden_dim, self.num_ent_tags)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        embeds = self.word_embeds(input_ids)
        embeds = torch.transpose(embeds, 0, 1)

        lstm_out, (h_n, h_c) = self.lstm(embeds)  # (seq_len, batch_size, hide_dim)
        lstm_out = torch.transpose(lstm_out, 0, 1)  # (batch_size, seq_len, hide_dim)
        output = self.classifier(lstm_out)  # (batch_size, seq_len, num_ent_tags)

        if labels is None:
            output = nn.Softmax(dim=-1)(output)
            pred = torch.argmax(output, dim=-1)
            return pred
        else:
            # https://blog.csdn.net/jiangpeng59/article/details/79583292
            # _labels = to_categorical(labels.tolist(), num_classes=self.num_ent_tags)
            # target = torch.from_numpy(_labels).to(Config.device)
            # output = torch.clamp(output, 1e-7, 1 - 1e-7)
            # loss = - torch.sum(target * torch.log(output))
            loss = nn.CrossEntropyLoss()(output.permute(0, 2, 1), labels)
            return loss
