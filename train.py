import os

from nere.utils.gpu_selector import get_available_gpu
from nere.utils.hparams import ArgCallback, Hparams
from nere.utils.logger import logging_config


def train_ner():
    logging_config("ner.log")
    model_name = "BERTCRF"
    from nere.ner.torchs.trainer import Trainer
    Trainer(model_name=model_name).run()


def train_re():
    logging_config("re.log")
    model_name = "BERTMultitask"
    from nere.re.torchs.trainer import Trainer
    Trainer(model_name=model_name).run()


def train_joint():
    logging_config("joint.log")
    ner_model = "BERTCRF"
    re_model = "BERTMultitask"
    from nere.joint.trainer import Trainer
    Trainer(ner_model, re_model).run()


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --level
        --gpu
    '''
    parser = Hparams().parser
    # 函数名参数
    parser.add_argument('--train', type=str,
                        choices=["ner", "re", "joint"],
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
    if args.train == "ner":
        train_ner()
    elif args.train == "re":
        train_re()
    elif args.train == "joint":
        train_joint()
    else:
        # 不需要接收参数的函数可放在此处执行;日志：函数名.log
        ArgCallback(vars(args), __name__)  # 自动寻找并调用此模块(manage.py) 中与参数名同名的函数


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python train.py --ner # 执行 ner()
        nohup python train.py --re  &>re.nohup.out&
    """

    main()
