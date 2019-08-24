import logging
import os

from nere.utils.gpu_selector import get_available_gpu
from nere.utils.logger import logging_config


def train_ner():
    model_name = "BERTCRF"
    from nere.ner.torchs.trainer import Trainer
    Trainer(model_name=model_name).run()


def train_re():
    model_name = "BERTMultitask"
    from nere.re.torchs.trainer import Trainer
    Trainer(model_name=model_name).run()


def train_joint():
    ner_model = "BERTCRF"
    re_model = "BERTMultitask"
    from nere.joint.trainer import Trainer
    Trainer(ner_model, re_model).run()


def evaluate():
    model_name = "BERTCRF"
    pass


def main():
    # train_ner()
    # evaluate()
    # train_re()
    train_joint()


if __name__ == '__main__':
    logging_config("dev.log", stream_log=True)
    available_gpu = get_available_gpu(num_gpu=1)
    logging.info("use GPU {} ...".format(available_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu
    main()
