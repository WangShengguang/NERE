import gc
import logging
import os

import keras
import keras_metrics
import matplotlib.pyplot as plt
from keras.models import load_model

from nere.re.data_helper import DataHelper
from .models import get_bi_lstm
from .config import Config

models_func = {"bilstm": get_bi_lstm,

               }


def train(model_name):
    logging.info("***keras train start, model_name : {}".format(model_name))
    data_helper = DataHelper()
    model_path = os.path.join(Config.keras_ckpt_dir, "{}.hdf5".format(model_name))
    if os.path.exists(model_path):
        # 载入预训练model
        model = load_model(model_path, custom_objects={})
        logging.info("\n*** keras load model :{}".format(model_path))
    else:
        assert model_name in models_func, "{} is not exist ".format(model_name)
        get_model = models_func[model_name]
        model = get_model(vocab_size=len(data_helper.tokenizer.vocab),
                          max_sequence_len=Config.max_len,
                          embedding_dim=Config.embedding_dim,
                          num_classes=Config.max_len,
                          embedding_matrix=None)
        model.compile(
            loss=keras.losses.binary_crossentropy,  # 'binary_crossentropy',categorical_crossentropy
            optimizer='adam',
            metrics=['accuracy'],
            # metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()],
        )
        model.summary()

    x_train, y_train = data_helper.get_samples(data_type="train", sample_type="ner")
    x_valid, y_valid = data_helper.get_samples(data_type="val", sample_type="ner")

    del data_helper
    gc.collect()
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=8,
                        epochs=10,
                        verbose=1,
                        validation_data=(x_valid, y_valid),
                        )

    model.save(model_path)
    # plot_history(history)
    logging.info("save model : {}".format(model_path))


def plot_history(history):
    plt.subplot(211)
    plt.title("accuracy")
    plt.plot(history.history["acc"], color="r", label="train")
    plt.plot(history.history["val_acc"], color="b", label="val")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title("loss")
    plt.plot(history.history["loss"], color="r", label="train")
    plt.plot(history.history["val_loss"], color="b", label="val")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()
