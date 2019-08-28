import os

from nere.utils.gpu_selector import get_available_gpu
from nere.utils.logger import logging_config


def torch_run(mode="train"):
    task = "ner"
    ner_model = "BERTCRF"
    re_model = "BERTMultitask"
    if task == "ner":
        from nere.torch_trainer import Trainer
        Trainer(model_name=ner_model, task=task, mode=mode).run()
    elif task == "re":
        from nere.torch_trainer import Trainer
        Trainer(model_name=re_model, task=task, mode=mode).run()
    elif task == "joint":
        # from nere.joint.trainer import Trainer
        from nere.torch_trainer import JoinTrainer
        JoinTrainer(ner_model, re_model, mode=mode).run()


def keras_run():
    task = "ner"
    ner_model = "bilstm"
    re_model = "bilstm"
    if task == "ner":
        from nere.keras_trainer import Trainer
        Trainer(model_name=ner_model, task="ner", mode="evaluate").run()


def main():
    # torch_run(mode="train")
    # torch_run(mode="evaluate")
    keras_run()

if __name__ == '__main__':
    logging_config("dev.log", stream_log=True)
    available_gpu = get_available_gpu(num_gpu=1)
    print("* using GPU: {} ".format(available_gpu))  # config前不可logging，否则config失效
    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu
    main()
