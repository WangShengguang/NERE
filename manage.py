import gc
import os
import sys

from nere.utils.gpu_selector import get_available_gpu
from nere.utils.hparams import Hparams
from nere.utils.logger import logging_config
from nere.utils.tools import GetTime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def torch_run(task, model_name, mode):
    if model_name == "all":
        run_all(task, mode)
        return
    logging_config("{}_{}_{}_torch.log".format(task, model_name, mode))
    if task == "ner":
        from nere.torch_trainer import Trainer
        Trainer(model_name=model_name, task=task, mode=mode).run()
    elif task == "re":
        from nere.torch_trainer import Trainer
        Trainer(model_name=model_name, task=task, mode=mode).run()


def keras_run(task, model_name, mode):
    logging_config("{}_{}_{}_keras.log".format(task, model_name, mode))
    from nere.keras_trainer import Trainer
    Trainer(task="ner", model_name=model_name, mode=mode).run()


def join_run(mode):
    ner_model = "BERTCRF"
    re_model = "BERTMultitask"
    logging_config("joint_{}.log".format(mode))
    from nere.torch_trainer import JoinTrainer
    if mode == "train":
        JoinTrainer(task="joint", ner_model=ner_model, re_model=re_model, mode=mode).run()
    # test
    JoinTrainer(task="joint", ner_model=ner_model, re_model=re_model, mode="test").run()


def run_all(task, mode):
    """
    训练或测试 所有 NER/RE 模型
    :param task: ner,re
    """
    if task == "ner":
        logging_config(f"ner_all_{mode}.log")
        # torch
        from nere.torch_trainer import Trainer
        for model_name in torch_ner_models:
            with GetTime(prefix=f"torch model:{task} {model_name}"):
                Trainer(model_name=model_name, task=task, mode=mode).run()
            gc.collect()
        # keras
        from nere.keras_trainer import Trainer
        for model_name in Keras_ner_models:
            with GetTime(prefix=f"keras model:{task} {model_name}"):
                Trainer(task="ner", model_name=model_name, mode=mode).run()
            gc.collect()
    elif task == "re":
        logging_config(f"re_all_{mode}.log")
        from nere.torch_trainer import Trainer
        for model_name in RE_models:
            with GetTime(prefix=f"torch model: {task} {model_name}"):
                Trainer(model_name=model_name, task=task, mode=mode).run()
            gc.collect()


def kgg(data_set):
    logging_config("kgg_{}.log".format(data_set))
    from nere.kgg.kgg import KGG2KE
    KGG2KE(data_set=data_set).run()


def data_prepare(task):
    logging_config("{}_data_prepare.log".format(task))
    # from nere.data_preparation.prepare_ner import create_ner_data  #标注有误，绝不可用之；使用现有标注好的数据
    # from nere.data_preparation.prepare_re import create_re_data
    from nere.data_preparation.prepare_joint import create_joint_data
    if task == "joint":
        # create_ner_data()
        # print("\n\n")
        # create_re_data()
        # print("\n\n")
        create_joint_data()
    # elif task == "ner":
    #     create_ner_data()
    # elif task == "re":
    #     create_re_data()
    else:
        raise ValueError(task)


Keras_ner_models = ["bilstm", "bilstm_crf"]
torch_ner_models = ["BERTCRF", "BERTSoftmax", "BiLSTM"]  # BiLSTM_ATT
NER_models = torch_ner_models + Keras_ner_models
RE_models = ["BERTMultitask", "BERTSoftmax", "BiLSTM_ATT", "ACNN", "BiLSTM"]


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --log_level
        --allow_gpus, --cpu_only
    '''
    parser = Hparams().parser
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个
    group.add_argument('--ner', type=str, choices=NER_models + ["all"], help="Named Entity Recognition，实体识别")
    group.add_argument('--re', type=str, choices=RE_models + ["all"], help="Relation Extraction，关系抽取")
    group.add_argument('--joint', action="store_true", help="联合训练，load pretrain 的模式")
    parser.add_argument('--mode', type=str, choices=["train", "test"],
                        required=bool({"--ner", "--re", "--joint"} & set(sys.argv)),
                        help="模型训练or测试")
    group.add_argument('--kgg', action="store_true", help="generation of law knowledge graph")
    parser.add_argument('--dataset', type=str, choices=["lawdata_new", "traffic_all", "traffic_500"],
                        required="--kgg" in sys.argv, help="数据集")
    group.add_argument('--data_prepare', choices=["ner", "re", "joint"], help="数据预处理")
    # parse args
    args = parser.parse_args()
    # mode = "train" if args.train else "test"
    mode = args.mode
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("CPU only ...")
    else:
        available_gpu = get_available_gpu(num_gpu=1, allow_gpus=args.allow_gpus)  # default allow_gpus 0,1,2,3
        os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu
        print("* using GPU: {} ".format(available_gpu))  # config前不可logging，否则config失效
    # set_process_name(args.process_name)  # 设置进程名
    if args.ner:
        if args.ner in Keras_ner_models:  # Keras
            keras_run(task="ner", model_name=args.ner, mode=mode)
        else:  # Pytorch
            torch_run(task="ner", model_name=args.ner, mode=mode)
    elif args.re:
        torch_run(task="re", model_name=args.re, mode=mode)
    elif args.joint:
        join_run(mode=mode)
    elif args.kgg:
        kgg(data_set=args.dataset)
    elif args.data_prepare:
        data_prepare(args.data_prepare)


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python3 manage.py --data_prepare joint 
        python3 manage.py --ner BERTCRF --mode train   
        python3 manage.py --re BERTMultitask --mode train  
        python3 manage.py --joint --mode train  
        nohup python3 manage.py --kgg --dataset traffic_all &>kgg.out&
    """

    main()
