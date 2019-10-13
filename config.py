import os
import random

import numpy as np
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = cur_dir
data_dir = os.path.join(root_dir, "data")
output_dir = os.path.join(root_dir, "output")

# random seed
rand_seed = 1234
torch.cuda.manual_seed_all(rand_seed)
torch.manual_seed(rand_seed)
random.seed(rand_seed)
np.random.seed(rand_seed)

tmp_suffix = "-join_data"


class PreprocessConfig(object):
    annotation_data_dir = os.path.join(data_dir, "raw_data", "annotation")
    predefined_file = os.path.join(annotation_data_dir, 'predefined_lables.txt')
    # output as  model input
    # preprocess output & model input
    # ner_data_dir = os.path.join(data_dir, "ner")# old 各自
    # re_data_dir = os.path.join(data_dir, "re")
    ner_data_dir = os.path.join(data_dir, "joint", "ner")  # 新生成joint
    re_data_dir = os.path.join(data_dir, "joint", "re")
    # pretrain model
    bert_pretrained_dir = os.path.join(data_dir, "bert-base-chinese-pytorch")
    bert_config_path = os.path.join(bert_pretrained_dir, 'bert_config.json')


class DataConfig(PreprocessConfig):
    """
        数据和模型所在文件夹
    """
    # output
    # tf_ckpt_dir = os.path.join(output_dir, "tf_ckpt")
    torch_ckpt_dir = os.path.join(output_dir, "torch_ckpt") + tmp_suffix
    keras_ckpt_dir = os.path.join(output_dir, "keras_ckpt") + tmp_suffix
    for _dir in [torch_ckpt_dir, keras_ckpt_dir]:
        os.makedirs(_dir, exist_ok=True)


class TrainConfig(object):
    # sample data
    max_sequence_len = 402
    batch_size = 16  # joint memory out  # default 16
    test_batch_size = batch_size  # default 8
    # train params
    ent_emb_dim = 128
    rel_emb_dim = 768
    learning_rate = 1e-5
    weight_decay = 0.01
    clip_grad = 2
    # early stop
    max_epoch_nums = 30
    min_epoch_nums = 3
    # patience = 0.01
    patience_num = 3
    # model save & load
    load_pretrain = True  # 断点续训
    max_to_keep = 1
    check_step = 100


class EvaluateConfig(object):
    # load_model_mode = "max_step"
    load_model_mode = "min_loss"


class BertConfig(object):
    gradient_accumulation_steps = 1
    full_finetuning = True


class TorchConfig(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    gpu_nums = torch.cuda.device_count()
    multi_gpu = True


class Config(DataConfig, TorchConfig, BertConfig, TrainConfig, EvaluateConfig):
    pass


# custom config
class KGGConfig(object):
    def __init__(self, data_set):
        self.kgg_data_dir = os.path.join(data_dir, "kgg", data_set)  # inupt data
        kgg_out_dir = os.path.join(output_dir, "kgg", data_set)  # output data
        os.makedirs(kgg_out_dir, exist_ok=True)
        self.cases_triples_txt = os.path.join(kgg_out_dir, "cases_triples_txt.txt")
        self.cases_triples_json = os.path.join(kgg_out_dir, "cases_triples_result.json")  # 案由对应triple
        self.entity2id_path = os.path.join(kgg_out_dir, 'entity2id.txt')
        self.relation2id_path = os.path.join(kgg_out_dir, 'relation2id.txt')
        self.train_triple_file = os.path.join(kgg_out_dir, 'train.txt')
        # KE train file
        self.train2id_file = os.path.join(kgg_out_dir, "train2id.txt")
        self.valid2id_file = os.path.join(kgg_out_dir, "valid2id.txt")
        self.test2id_file = os.path.join(kgg_out_dir, "test2id.txt")
