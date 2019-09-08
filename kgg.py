import os

from nere.utils.gpu_selector import get_available_gpu
from nere.utils.hparams import Hparams, set_process_name


def kgg_test():
    from nere.lkg.kgg import KGG
    import time, json
    case_file = 'data/001.txt'
    # case_file = 'new_data/441 (248).txt'
    ner_model = "BERTCRF"
    re_model = "BERTMultitask"
    kgg = KGG(ner_model, re_model)

    # kgg.parse(case_file)
    start_time = time.time()
    err_file = []
    long_sentence = 0
    i = 0
    result = {}
    triples = set()
    for file in os.listdir('nere/lkg/new_data'):
        case_file = os.path.join('new_data', file)
        i += 1
        print(case_file)
        # if len(result) > 2:
        #     break
        try:
            flag, case_id, res = kgg.parse(case_file)
            if flag:
                result[case_id] = list(res)
                for tep in res:
                    triples.add(tep)
            else:
                err_file.append(file)
        except:
            err_file.append(file)
            long_sentence += 1
            print("句子长度超过512：", file)
    end_time = time.time()
    print('running time: {}'.format(end_time - start_time))
    print("case num: {}".format(i))
    print('Average running time: {}'.format((end_time - start_time) / i))
    print("不存在案情事实的文件有{}个".format(len(err_file)))
    print(err_file)
    print("句子长度超过512：{}".format(long_sentence))
    with open("result.txt", 'w', encoding='utf-8') as fw:
        json.dump(result, fw, ensure_ascii=False, indent=4)
    # triples = list(set(triples))
    triples = list(triples)
    with open("triples_result.txt", 'w', encoding='utf-8') as fw:
        for item in triples:
            if len(item) < 3:
                continue
            fw.write(item[0] + '\t' + item[1] + '\t' + item[2] + '\n')

    # start_time = time.time()
    # num = 10
    # for i in range(num):
    #     kgg.parse(case_file)
    # end_time = time.time()
    # print('Average running time: {}'.format((end_time-start_time)/num))


def kgg():
    ner_model = "BERTCRF"
    re_model = "BERTMultitask"
    # KGG(ner_model, re_model)


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --log_level
        --allow_gpus, --cpu_only
    '''
    parser = Hparams().parser
    group = parser.add_mutually_exclusive_group(required=False)  # 一组互斥参数,且至少需要互斥参数中的一个
    # 函数名参数
    group.add_argument('--ner', type=str,
                       default="BERTCRF",
                       choices=["BERTCRF", "BERTSoftmax", "BiLSTM_ATT"],  # model name
                       help="Named Entity Recognition，实体识别")
    group.add_argument('--re', type=str,
                       default="BERTMultitask",
                       choices=["BERTSoftmax", "BERTMultitask", "BiLSTM_ATT", "ACNN", "BiLSTM"],  # model name
                       help="Relation Extraction，关系抽取")
    group.add_argument('--joint', action="store_true", help="联合训练，load pretrain 的模式")

    group.add_argument('--mode', type=str,
                       default="run",
                       choices=["run", "test"],  # model name
                       help="")
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
    if args.mode == "run":
        kgg()
    elif args.mode == "test":
        kgg_test()
    else:
        raise ValueError(args.mode)


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python kgg.py  --ner BERTCRF  
        python kgg.py  --joint respective  
        nohup python kgg.py --joint respective --process_name J &
    """

    main()
