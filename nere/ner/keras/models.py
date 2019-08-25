import numpy as np
from keras import Model, Sequential
from keras.layers import (Dense, Input, Embedding, Dropout, Flatten, Convolution1D, MaxPool1D, concatenate,
                          Bidirectional, LSTM)


def get_text_cnn(vocab_size, max_sequence_len, embedding_dim, num_classes, embedding_matrix=None):
    # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
    main_input = Input(shape=(max_sequence_len,))
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(vocab_size, embedding_dim, input_length=max_sequence_len,
                         weights=np.asarray([embedding_matrix]), trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn2 = Convolution1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn3 = Convolution1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    output = Dense(num_classes, activation='sigmoid')(drop)
    model = Model(inputs=main_input, outputs=output)
    return model


def get_bi_lstm(vocab_size, max_sequence_len, embedding_dim, num_classes, embedding_matrix=None):
    if embedding_matrix is not None:
        weights = np.asarray([embedding_matrix])
        trainable = False
    else:
        weights = None
        trainable = True
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_len,
                        weights=weights,
                        trainable=trainable))
    model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.1)))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model
