import keras.backend as K
from keras import Input
from keras.layers import (Dense, Embedding, Bidirectional, LSTM)
from keras.models import Model

from nere.config import Config

sequence_len = Config.max_sequence_len
ent_embedding_dim = Config.ent_emb_dim
word_embedding_dim = Config.ent_emb_dim


def get_bilstm(vocab_size, num_ent_tags, num_rel_tags):
    # 模型结构：词嵌入-双向GRU-Attention-全连接
    ent1 = Input(shape=())
    ent2 = Input(shape=())
    sentence = Input(shape=(sequence_len,))
    ent_emb_layer = Embedding(num_ent_tags, ent_embedding_dim)
    embed_ent1 = ent_emb_layer(ent1)
    embed_ent2 = ent_emb_layer(ent2)
    embed_sentence = Embedding(vocab_size, word_embedding_dim)(sentence)
    #
    bilstm_sentence = Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1))(embed_sentence)
    feature = K.concatenate([embed_ent1, embed_ent2, bilstm_sentence])

    x = Bidirectional(LSTM(256 // 2, dropout=0.2, recurrent_dropout=0.1))(feature)
    output = Dense(num_rel_tags, activation='softmax')(x)
    model = Model(inputs=(ent1, ent2, sentence), outputs=output)
    return model


def get_bilstm_crf(vocab_size, num_ent_tags, num_rel_tags):
    pass
