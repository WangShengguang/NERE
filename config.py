import os
import random

import numpy as np
import torch

__all__ = ["Config"]

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = cur_dir

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
    bert_config_path = os.path.join(bert_pretrained_dir, 'bert_config.json')


class KGGConfig(object):
    kgg_data_dir = os.path.join(DataConfig.data_dir, "kgg")
    kgg_cate_file = os.path.join(kgg_data_dir, "cate_data", "001.txt")
    #
    kgg_raw_data_dir = os.path.join(kgg_data_dir, "data")  # 待抽取数据
    kgg_out_data_dir = os.path.join(kgg_data_dir, "out_data")
    # result file
    result_file = os.path.join(kgg_data_dir, "result.txt")
    triples_result_file = os.path.join(kgg_data_dir, "triples_result.txt")
    # lawdata for ke
    lawdata_dir = os.path.join(kgg_data_dir, "lawdata_new")
    out_entity_vocab = os.path.join(lawdata_dir, 'entity2id.txt')
    out_relation_vocab = os.path.join(lawdata_dir, 'relation2id.txt')
    out_triple_file = os.path.join(lawdata_dir, 'train.txt')
    # KE train file
    train2id_file = os.path.join(lawdata_dir, "train2id.txt")
    valid2id_file = os.path.join(lawdata_dir, "valid2id.txt")
    test2id_file = os.path.join(lawdata_dir, "test2id.txt")


class TrainConfig(object):
    # sample data
    max_sequence_len = 400
    batch_size = 8
    # train params
    ent_emb_dim = 128
    rel_emb_dim = 768
    learning_rate = 1e-5
    weight_decay = 0.01
    clip_grad = 2
    # early stop
    max_epoch_nums = 50
    min_epoch_nums = 3
    patience = 0.01
    patience_num = 3
    # model save & load
    load_pretrain = True  # 断点续训
    max_to_keep = 10
    check_step = 100


class EvaluateConfig(object):
    load_model_mode = "max_step"
    # load_model_mode = "min_loss"


class BertConfig(object):
    gradient_accumulation_steps = 1
    full_finetuning = True


class TorchConfig(object):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    gpu_nums = torch.cuda.device_count()
    multi_gpu = False


class TfConfig(object):
    """
        TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
        TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
        TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
        TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


class Config(DataConfig, TorchConfig, BertConfig, TrainConfig, EvaluateConfig, KGGConfig):
    pass


for _dir in [DataConfig.torch_ckpt_dir, DataConfig.keras_ckpt_dir, DataConfig.tf_ckpt_dir,
             KGGConfig.lawdata_dir]:
    os.makedirs(_dir, exist_ok=True)
