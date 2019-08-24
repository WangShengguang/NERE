import os

import torch

__all__ = ["DataConfig"]

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)


class DataConfig(object):
    """
        数据和模型所在文件夹
    """
    data_dir = os.path.join(root_dir, "data")
    ner_data_dir = os.path.join(data_dir, "ner")
    re_data_dir = os.path.join(data_dir, "re")
    # checkpoint
    model_ckpt_dir = os.path.join(root_dir, "model_ckpt")
    torch_ckpt_dir = os.path.join(model_ckpt_dir, "torch")
    keras_ckpt_dir = os.path.join(model_ckpt_dir, "keras")
    # pretrain model
    bert_pretrained_dir = os.path.join(model_ckpt_dir, "bert-base-chinese-pytorch")

    for _dir in [torch_ckpt_dir, keras_ckpt_dir]:
        os.makedirs(_dir, exist_ok=True)


class TorchConfig(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_nums = torch.cuda.device_count()
    multi_gpu = False
    torch.cuda.manual_seed_all(1234)
