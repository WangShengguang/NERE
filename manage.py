import gc
import os
import sys

from nere.utils.gpu_selector import get_available_gpu
from nere.utils.hparams import Hparams
from nere.utils.logger import logging_config
from nere.utils.tools import Debug


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
    logging_config(f"joint_{mode}.log")
    from nere.torch_trainer import JoinTrainer
    for transe_rate in [0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.0001]:
        for ner_loss_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            re_loss_rate = 1 - ner_loss_rate - transe_rate
            if re_loss_rate <= 0:
                continue
            if mode == "train":
                JoinTrainer(task="joint", ner_model=ner_model, re_model=re_model, mode=mode,
                            ner_loss_rate=ner_loss_rate, re_loss_rate=re_loss_rate, transe_rate=transe_rate).run()
            # test
            JoinTrainer(task="joint", ner_model=ner_model, re_model=re_model, mode="test",
                        ner_loss_rate=ner_loss_rate, re_loss_rate=re_loss_rate, transe_rate=transe_rate).run()
            gc.collect()


def run_all(task, mode):
    """
    训练或测试 所有 NER/RE 模型
    :param task: ner,re
    """
    if task == "ner":
        logging_config(f"ner_all_{mode}.log")
        # torch
        from nere.torch_trainer import Trainer
        for model_name in ["BiLSTM_ATT", "BERTCRF", "BERTSoftmax"]:
            with Debug(prefix=f"torch model:{task} {model_name}"):
                Trainer(model_name=model_name, task=task, mode=mode).run()
            gc.collect()
        # keras
        from nere.keras_trainer import Trainer
        for model_name in ["bilstm", "bilstm_crf"]:
            with Debug(prefix=f"keras model:{task} {model_name}"):
                Trainer(task="ner", model_name=model_name, mode=mode).run()
            gc.collect()
    elif task == "re":
        logging_config(f"re_all_{mode}.log")
        from nere.torch_trainer import Trainer
        for model_name in ["BiLSTM_ATT", "ACNN", "BiLSTM", "BERTSoftmax", "BERTMultitask"]:
            with Debug(prefix=f"torch model: {task} {model_name}"):
                Trainer(model_name=model_name, task=task, mode=mode).run()
            gc.collect()


def kgg():
    logging_config("kgg.log")
    from nere.kgg.kgg import create_lawdata
    create_lawdata()


Keras_ner_models = ["bilstm", "bilstm_crf"]
NER_models = ["BERTCRF", "BERTSoftmax", "BiLSTM", "BiLSTM_ATT", "all"] + Keras_ner_models
RE_models = ["BERTSoftmax", "BERTMultitask", "BiLSTM_ATT", "ACNN", "BiLSTM", "all"]


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --log_level
        --allow_gpus, --cpu_only
    '''
    parser = Hparams().parser
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个
    # mode_group = parser.add_mutually_exclusive_group(
    #     required=bool(set(["--ner", "--re", "--joint"]) & set(sys.argv)))  # 一组互斥参数,且至少需要互斥参数中的一个
    # mode_group.add_argument('--train', action="store_true", help="模型训练")
    # mode_group.add_argument('--test', action="store_true", help="模型测试")
    # 函数名参数
    group.add_argument('--ner', type=str, choices=NER_models, help="Named Entity Recognition，实体识别")
    group.add_argument('--re', type=str, choices=RE_models, help="Relation Extraction，关系抽取")
    group.add_argument('--joint', action="store_true", help="联合训练，load pretrain 的模式")
    parser.add_argument('--mode', type=str, choices=["train", "test"],
                        required="--ner" in sys.argv or "--re" in sys.argv or "--joint" in sys.argv,
                        # required==True if x
                        help="模型训练or测试")
    group.add_argument('--kgg', action="store_true", help="generation of law knowledge graph")

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
        kgg()


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python manage.py --ner BERTCRF --mode train  
        python manage.py --re BERTMultitask --mode train  
        python manage.py --joint --mode test  
    """

    main()
