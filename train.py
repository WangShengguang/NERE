import os

from nere.utils.gpu_selector import get_available_gpu
from nere.utils.hparams import Hparams
from nere.utils.logger import logging_config


def torch_train(task, model_name=None):
    logging_config("torch_{}.log".format(task))
    ner_model = "BERTCRF"
    re_model = "BERTMultitask"
    if task == "ner":
        from nere.ner.torchs.trainer import Trainer
        Trainer(model_name=ner_model).run()
    elif task == "re":
        logging_config("re.log")
        from nere.re.torchs.trainer import Trainer
        Trainer(model_name=re_model).run()
    elif task == "joint":
        logging_config("joint.log")
        from nere.joint.trainer import Trainer
        Trainer(ner_model, re_model).run()
    elif task == "joint_ner":
        from nere.joint.ner_trainer import JointNerTrainer
        JointNerTrainer(ner_model).run()


def keras_train(task, model_name=None):
    logging_config("keras_{}.log".format(task))
    ner_model = "bilstm"
    re_model = "bilstm"
    if task == "ner":
        from nere.ner.keras.train import train
        train(model_name=ner_model)


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --level
        --gpu
    '''
    parser = Hparams().parser
    # 函数名参数
    parser.add_argument('--torch', type=str,
                        choices=["ner", "re", "joint", "joint_ner"],
                        help="模型训练")
    parser.add_argument('--keras', type=str,
                        choices=["ner"],
                        help="模型训练")
    parser.add_argument('--model', type=str,
                        choices=["BERTCRF", "BERTMultitask"],
                        help="模型训练")
    parser.add_argument('--test', action="store_true", default=False, help="测试")
    # parse args
    args = parser.parse_args()
    available_gpu = get_available_gpu(num_gpu=1, allow_gpus=args.allow_gpus)  # default 0,1,2,3
    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu
    print("* use GPU: {} ".format(available_gpu))  # config前不可logging，否则config失效
    #
    if args.torch:
        torch_train(args.torch)
    elif args.keras:
        keras_train(args.keras)


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python train.py  --torch ner # 执行 ner()
        nohup python train.py --torch ner  &>ner.nohup.out&
    """

    main()
