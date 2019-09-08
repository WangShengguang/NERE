import os

from nere.utils.gpu_selector import get_available_gpu
from nere.utils.hparams import Hparams, set_process_name
from nere.utils.logger import logging_config


def keras_run(task, model_name, mode):
    logging_config("{}_{}_{}_keras.log".format(task, model_name, mode))
    from nere.keras_trainer import Trainer
    Trainer(task="ner", model_name=model_name, mode=mode).run()


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --log_level
        --allow_gpus
    '''
    parser = Hparams().parser
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个
    # 函数名参数
    group.add_argument('--ner', type=str,
                       choices=["bilstm", "bilstm_crf"],  # model name
                       help="Named Entity Recognition，实体识别")
    group.add_argument('--re', type=str,
                       choices=[],  # model name
                       help="Relation Extraction，关系抽取")
    group.add_argument('--joint', action="store_true", default=False, help="联合训练")

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
    set_process_name(args.process_name)  # 设置进程名
    if args.ner:
        keras_run(task="ner", model_name=args.ner, mode=args.mode)
    elif args.re:
        keras_run(task="re", model_name=args.re, mode=args.mode)
    elif args.joint:
        keras_run(task="joint", model_name=args.joint, mode=args.mode)


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        nohup python manage_keras.py  --ner bilstm_crf --mode train  --process_name k_bc&
        python manage_keras.py  --ner BERTCRF --mode test  
    """

    main()
