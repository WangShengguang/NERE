import gc
import logging
import os

import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from config import Config
from nere.data_helper import DataHelper
from nere.evaluator import Evaluator


class Trainer(object):
    def __init__(self, task, model_name, mode="train"):
        self.task = task
        self.model_name = model_name
        self.mode = mode
        self.model_path = os.path.join(Config.keras_ckpt_dir, "{}.hdf5".format(model_name))
        self.data_helper = DataHelper()
        self._init()

    def _init(self):
        if self.task == "ner":
            self.num_classes = len(self.data_helper.ent_tag2id)
        elif self.task == "re":
            self.num_classes = len(self.data_helper.rel_label2id)

    def get_data(self):
        train_data = self.data_helper.get_samples(task=self.task, data_type="train")
        valid_data = self.data_helper.get_samples(task=self.task, data_type="val")
        if self.task == "ner":
            x_train, y_train = train_data["sents"], train_data["ent_tags"]
            x_valid, y_valid = valid_data["sents"], valid_data["ent_tags"]
            # if self.model_name == "bilstm":  # time_distributed_1 to have 3 dimensions
            y_train = to_categorical(y_train, num_classes=self.num_classes)
            y_valid = to_categorical(y_valid, num_classes=self.num_classes)
        elif self.task == "re":
            x_train = train_data["sents"], train_data["ent_tags"]
            y_train = train_data["re_labels"]
            x_valid = valid_data["sents"], valid_data["ent_tags"]
            y_valid = valid_data["re_labels"]
        else:
            raise ValueError(self.task)
        del self.data_helper
        gc.collect()
        return (x_train, y_train), (x_valid, y_valid)

    def get_model(self):
        vocab_size = len(self.data_helper.tokenizer.vocab)
        num_ent_tags = len(self.data_helper.ent_tag2id)
        num_rel_tags = len(self.data_helper.rel_label2id)
        if Config.load_pretrain and os.path.isfile(self.model_path):
            # 载入预训练model
            model = load_model(self.model_path, custom_objects={"CRF": CRF,
                                                                "crf_loss": crf_loss,
                                                                "crf_viterbi_accuracy": crf_viterbi_accuracy})
            logging.info("\n*** keras load model :{}".format(self.model_path))
        elif self.task == "ner":
            from nere.ner.keras_models import get_bilstm, get_bilstm_crf
            get_model = {"bilstm": get_bilstm, "bilstm_crf": get_bilstm_crf}[self.model_name]
            model = get_model(vocab_size=vocab_size, num_classes=self.num_classes)
        # elif self.task == "re":
        #     from nere.re.keras_models import get_bilstm, get_bilstm_crf
        #     get_model = {"bilstm": get_bilstm, "bilstm_crf": get_bilstm_crf}[self.model_name]
        #     model = get_model(vocab_size=vocab_size, num_ent_tags=num_ent_tags, num_rel_tags=num_rel_tags)
        else:
            raise ValueError(self.model_name)
        model.summary()
        return model

    def run(self):
        model = self.get_model()
        if self.mode == "test":
            acc, precision, recall, f1 = Evaluator(framework="keras", task="ner", data_type="test").test(model=model)
            _test_log = "acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                acc, precision, recall, f1)
            logging.info(_test_log)
            print(_test_log)
            return
        logging.info("***keras train start, model_name : {}".format(self.model_name))
        (x_train, y_train), (x_valid, y_valid) = self.get_data()
        callbacks = [
            ReduceLROnPlateau(),
            EarlyStopping(patience=Config.patience_num),
            ModelCheckpoint(filepath=self.model_path, save_best_only=True)
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
        # plot_history(history)
        logging.info("Done. save model : {}".format(self.model_path))


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
