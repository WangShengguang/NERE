import logging
import os

from nere.utils.gpu_selector import get_available_gpu
from nere.utils.logger import logging_config


def torch_train():
    task = "ner"
    logging_config("torch_{}.log".format(task))
    ner_model = "BERTCRF"
    re_model = "BERTMultitask"
    if task == "ner":
        from nere.torch_trainer import Trainer
        Trainer(model_name=ner_model, task=task).run()
    elif task == "re":
        from nere.torch_trainer import Trainer
        Trainer(model_name=re_model, task=task).run()
    elif task == "joint":
        # from nere.joint.trainer import Trainer
        from nere.torch_trainer import JoinTrainer
        JoinTrainer(ner_model, re_model).run()


def keras_train(model_name=None):
    task = "ner"
    logging_config("keras_{}.log".format(task))
    ner_model = "bilstm"
    re_model = "bilstm"
    if task == "ner":
        from nere.ner.keras.train import train
        train(model_name=ner_model)


def main():
    torch_train()


if __name__ == '__main__':
    logging_config("dev.log", stream_log=True)
    available_gpu = get_available_gpu(num_gpu=1)
    logging.info("use GPU {} ...".format(available_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu
    main()
