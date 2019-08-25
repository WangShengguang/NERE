__all__ = ["Config"]

from nere.config import DataConfig, TorchConfig


class TrainConfig(DataConfig):
    # sample
    sequence_len = 512
    batch_size = 16
    embedding_dim = 128
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
    save_dir = "ner"


class BertConfig(object):
    gradient_accumulation_steps = 1
    full_finetuning = True


class Evaluate(TrainConfig):
    load_model_mode = "max_step"


class Config(Evaluate, TorchConfig, BertConfig):
    pass