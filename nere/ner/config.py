__all__ = ["Config"]

import torch

from nere.config import DataConfig


class TrainConfig(DataConfig):
    sequence_len = 512  # (h,r,t)
    batch_size = 16

    epoch_nums = 20
    max_epoch_nums = 20
    min_epoch_nums = 5

    save_step = 200
    max_to_keep = 10
    full_finetuning = True
    learning_rate = 1e-5
    weight_decay = 0.01
    clip_grad = 2
    patience = 0.02
    patience_num = 3


class BertConfig(object):
    gradient_accumulation_steps = 1


class TorchConfig(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_nums = torch.cuda.device_count()
    multi_gpu = False
    torch.cuda.manual_seed_all(1234)


class Evaluate(TrainConfig):
    load_model_mode = "max_step"


class Config(Evaluate, TorchConfig, BertConfig):
    pass
