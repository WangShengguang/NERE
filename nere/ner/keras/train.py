import gc
import logging
import os

import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.models import load_model
from keras.utils.np_utils import to_categorical

from nere.config import Config
from nere.data_helper import DataHelper
from nere.ner.keras.models import get_bi_lstm

models_func = {"bilstm": get_bi_lstm,

               }


def train(model_name):
    logging.info("***keras train start, model_name : {}".format(model_name))
    data_helper = DataHelper()
    embedding_dim = 128
    num_classes = len(data_helper.ent_tag2id)
    model_path = os.path.join(Config.keras_ckpt_dir, "{}.hdf5".format(model_name))
    if os.path.exists(model_path):
        # 载入预训练model
        model = load_model(model_path, custom_objects={})
        logging.info("\n*** keras load model :{}".format(model_path))
    else:
        assert model_name in models_func, "{} is not exist ".format(model_name)
        get_model = models_func[model_name]
        model = get_model(vocab_size=len(data_helper.tokenizer.vocab),
                          embedding_dim=embedding_dim,
                          num_classes=num_classes,
                          embedding_matrix=None)
        model.compile(
            optimizer='adam',
            loss=categorical_crossentropy, metrics=["accuracy"],
        )
        model.summary()

    # multilabel_binarizer = MultiLabelBinarizer()
    # multilabel_binarizer.fit([list(data_helper.ent_tag2id.values())])

    train_data = data_helper.get_samples(data_type="train")
    x_train = train_data["sents"]
    y_train = to_categorical(train_data["ent_tags"], num_classes=num_classes)
    # time_distributed_1 to have 3 dimensions
    valid_data = data_helper.get_samples(data_type="val")
    x_valid = valid_data["sents"]
    y_valid = to_categorical(valid_data["ent_tags"], num_classes=num_classes)

    del data_helper
    gc.collect()
    callbacks = [
        ReduceLROnPlateau(),
        EarlyStopping(patience=4),
        ModelCheckpoint(filepath=model_path, save_best_only=True)
    ]
    class_weight = None
    history = model.fit(x=x_train,
                        y=y_train,
                        class_weight=class_weight,
                        batch_size=Config.batch_size,
                        epochs=Config.max_epoch_nums,
                        verbose=1,
                        validation_data=(x_valid, y_valid),
                        callbacks=callbacks
                        )
    # model.save(model_path)
    # plot_history(history)
    logging.info("Done. save model : {}".format(model_path))


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
