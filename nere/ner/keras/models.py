import numpy as np
from keras import Sequential
from keras.layers import (Dense, Embedding, Bidirectional, LSTM, TimeDistributed)


def get_bi_lstm(vocab_size, embedding_dim, num_classes, embedding_matrix=None):
    if embedding_matrix is not None:
        weights = np.asarray([embedding_matrix])
        trainable = False
    else:
        weights = None
        trainable = True
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, weights=weights, trainable=trainable))
    model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
    # model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1)))
    # model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
    # model.add(Dense(num_classes, activation='sigmoid'))
    return model
