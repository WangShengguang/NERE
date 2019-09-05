from keras.layers import (Dense, Embedding, Bidirectional, LSTM, TimeDistributed)
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras_contrib.layers import CRF

from nere.config import Config

embedding_dim = Config.ent_emb_dim


def get_bilstm(vocab_size, num_classes):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=["accuracy"])
    return model


def get_bilstm_crf(vocab_size, num_classes):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))  # Random embedding
    model.add(Bidirectional(LSTM(256 // 2, return_sequences=True)))
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
    # crf = CRF(num_classes, sparse_target=True)
    crf = CRF(num_classes)
    model.add(crf)
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    return model
