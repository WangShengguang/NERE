import os
import random

import numpy as np
import torch

__all__ = ["Config"]

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)

model_ckpt_dir = os.path.join(root_dir, "model_ckpt")

# random seed
rand_seed = 1234
torch.cuda.manual_seed_all(rand_seed)
random.seed(rand_seed)
np.random.seed(rand_seed)


class DataConfig(object):
    """
        数据和模型所在文件夹
    """
    data_dir = os.path.join(root_dir, "data")
    ner_data_dir = os.path.join(data_dir, "ner")
    re_data_dir = os.path.join(data_dir, "re")
    # checkpoint
    tf_ckpt_dir = os.path.join(model_ckpt_dir, "tf")
    torch_ckpt_dir = os.path.join(model_ckpt_dir, "torch")
    keras_ckpt_dir = os.path.join(model_ckpt_dir, "keras")
    # pretrain model
    bert_pretrained_dir = os.path.join(model_ckpt_dir, "bert-base-chinese-pytorch")
    # sample generate


class TorchConfig(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_nums = torch.cuda.device_count()
    multi_gpu = False


class TrainConfig(object):
    # sample data
    max_len = 400
    batch_size = 16
    # train params
    learning_rate = 1e-5
    weight_decay = 0.01
    clip_grad = 2
    # early stop
    max_epoch_nums = 20
    min_epoch_nums = 5
    patience = 0.02
    patience_num = 3
    # model save & load
    load_pretrain = True  # 断点续训
    max_to_keep = 10
    save_step = 200


class BertConfig(object):
    gradient_accumulation_steps = 1
    full_finetuning = True


class EvaluateConfig(object):
    load_model_mode = "max_step"


class Config(DataConfig, TorchConfig, BertConfig, TrainConfig, EvaluateConfig):
    pass


for _dir in [DataConfig.torch_ckpt_dir, DataConfig.keras_ckpt_dir, DataConfig.tf_ckpt_dir]:
    os.makedirs(_dir, exist_ok=True)
