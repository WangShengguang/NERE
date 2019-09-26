import os
import random

import numpy as np
import torch

__all__ = ["Config"]

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = cur_dir
data_dir = os.path.join(root_dir, "data")
output_dir = os.path.join(root_dir, "output")

# random seed
rand_seed = 1234
torch.cuda.manual_seed_all(rand_seed)
random.seed(rand_seed)
np.random.seed(rand_seed)


class DataConfig(object):
    """
        数据和模型所在文件夹
    """
    data_dir = data_dir
    # input
    ner_data_dir = os.path.join(data_dir, "ner")
    re_data_dir = os.path.join(data_dir, "re")
    # output
    tf_ckpt_dir = os.path.join(output_dir, "tf_ckpt")
    torch_ckpt_dir = os.path.join(output_dir, "torch_ckpt")
    keras_ckpt_dir = os.path.join(output_dir, "keras_ckpt")
    for _dir in [torch_ckpt_dir, keras_ckpt_dir, tf_ckpt_dir]:
        os.makedirs(_dir, exist_ok=True)
    # pretrain model
    bert_pretrained_dir = os.path.join(data_dir, "bert-base-chinese-pytorch")
    bert_config_path = os.path.join(bert_pretrained_dir, 'bert_config.json')


class KGGConfig(object):
    # kgg_cate_file = os.path.join(kgg_data_dir, "cate_data", "001.txt")
    #
    # kgg_raw_data_dir = os.path.join(kgg_data_dir, "data")  # 待抽取数据 # 裁判文书来源->ner->re->triple->ke->rank
    # kgg_raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(root_dir)), "traffic500")  # 待抽取数据
    # out file dir
    kgg_out_data_dir = os.path.join(output_dir, "{data_set}", "kgg")
    # result file
    cases_triples_result_json_file_tmpl = os.path.join(kgg_out_data_dir, "cases_triples_result.json")  # 案由对应triple
    # lawdata for ke
    entity2id_path_tmpl = os.path.join(kgg_out_data_dir, 'entity2id.txt')
    relation2id_path_tmpl = os.path.join(kgg_out_data_dir, 'relation2id.txt')
    train_triple_file_tmpl = os.path.join(kgg_out_data_dir, 'train.txt')
    # KE train file
    train2id_file_tmpl = os.path.join(kgg_out_data_dir, "train2id.txt")
    valid2id_file_tmpl = os.path.join(kgg_out_data_dir, "valid2id.txt")
    test2id_file_tmpl = os.path.join(kgg_out_data_dir, "test2id.txt")


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
