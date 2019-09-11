import gc
import os

from nere.utils.gpu_selector import get_available_gpu
from nere.utils.hparams import Hparams, set_process_name
from nere.utils.logger import logging_config


def torch_run(task, model_name, mode):
    logging_config("{}_{}_{}_torch.log".format(task, model_name, mode))
    if task == "ner":
        from nere.torch_trainer import Trainer
        Trainer(model_name=model_name, task=task, mode=mode).run()
    elif task == "re":
        from nere.torch_trainer import Trainer
        Trainer(model_name=model_name, task=task, mode=mode).run()


def join_run():
    ner_model = "BERTCRF"
    re_model = "BERTMultitask"
    for ner_loss_rate, re_loss_rate, transe_rate in [(0.1, 0.89, 0.01),
                                                     (0.1, 0.85, 0.05),
                                                     (0.2, 0.75, 0.05),
                                                     (0.3, 0.65, 0.05),
                                                     (0.4, 0.55, 0.05)]:
        assert sum([ner_loss_rate, re_loss_rate, transe_rate]) == 1.0, print(ner_loss_rate, re_loss_rate, transe_rate)
        model_name = "joint_{:.5}{}_{:.5}{}_{:.5}TransE".format(ner_loss_rate, ner_model, re_loss_rate, re_model,
                                                                transe_rate)
        for mode in ["train", "test"]:
            logging_config("{}_{}_torch.log".format(model_name, mode))
            from nere.torch_trainer import JoinTrainer
            JoinTrainer(task="joint", ner_model=ner_model, re_model=re_model, mode=mode,
                        ner_loss_rate=ner_loss_rate, re_loss_rate=re_loss_rate, transe_rate=transe_rate).run()
            gc.collect()


def run_all(task):
    import logging, traceback
    if task == "ner":
        logging_config("ner_all.log")
        from nere.keras_trainer import Trainer
        for model_name in ["bilstm", "bilstm_crf"]:
            gc.collect()
            try:
                Trainer(task="ner", model_name=model_name, mode="train").run()
            except:
                logging.info("keras model: {}".format(model_name))
                logging.error(traceback.format_exc())
        # keras
        from nere.torch_trainer import Trainer
        for model_name in ["BERTCRF", "BERTSoftmax", "BiLSTM_ATT"]:
            try:
                Trainer(model_name=model_name, task=task, mode="train").run()
            except:
                logging.info("torch model:{}".format(model_name))
                logging.error(traceback.format_exc())
    elif task == "re":
        logging_config("re_all.log")
        from nere.torch_trainer import Trainer
        for model_name in ["BiLSTM_ATT", "ACNN", "BiLSTM", "BERTSoftmax", "BERTMultitask", ]:
            gc.collect()
            try:
                Trainer(model_name=model_name, task=task, mode="train").run()
            except:
                logging.info(model_name)
                logging.error(traceback.format_exc())


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --log_level
        --allow_gpus, --cpu_only
    '''
    parser = Hparams().parser
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个
    # 函数名参数
    group.add_argument('--ner', type=str,
                       choices=["BERTCRF", "BERTSoftmax", "BiLSTM_ATT"],  # model name
                       help="Named Entity Recognition，实体识别")
    group.add_argument('--re', type=str,
                       choices=["BERTSoftmax", "BERTMultitask", "BiLSTM_ATT", "ACNN", "BiLSTM"],  # model name
                       help="Relation Extraction，关系抽取")
    group.add_argument('--joint', type=str,
                       choices=["joint", "respective"],  # model name
                       help="联合训练，load pretrain 的模式")
    group.add_argument('--all', type=str, choices=["ner", "re"])
    parser.add_argument('--mode', type=str,
                        choices=["train", "test"],
                        required=True,
                        help="模型训练or测试")
    # parse args
    args = parser.parse_args()
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("CPU only ...")
    else:
        available_gpu = get_available_gpu(num_gpu=1, allow_gpus=args.allow_gpus)  # default allow_gpus 0,1,2,3
        os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu
        print("* using GPU: {} ".format(available_gpu))  # config前不可logging，否则config失效
    # set_process_name(args.process_name)  # 设置进程名
    if args.ner:
        torch_run(task="ner", model_name=args.ner, mode=args.mode)
    elif args.re:
        torch_run(task="re", model_name=args.re, mode=args.mode)
    elif args.joint:
        join_run()
    elif args.all:
        run_all(task=args.all)


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python manage_torch.py  --ner BERTCRF --mode train  
        python manage_torch.py  --joint respective --mode test  
        nohup python manage_torch.py --joint respective --mode train --process_name joint &
        nohup python manage_torch.py  --all re --process_name re_all &  
    """

    main()
