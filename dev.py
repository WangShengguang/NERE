from nere.utils.logger import logging_config


def preprocess():
    """数据预处理，建立字典，提取词向量等"""
    pass


def train_ner():
    model_name = "BERTCRF"
    from nere.ner.torch_models.trainer import Trainer
    Trainer(model_name=model_name,save_dir="ner").run()


def train_re():
    model_name = "BERTMultitask"
    from nere.re.torch_models.trainer import Trainer
    Trainer(model_name=model_name,save_dir="re").run()


def evaluate():
    model_name = "BERTCRF"
    pass


def main():
    logging_config("dev.log", stream_log=True, relative_path=".")
    preprocess()  # 构建词典
    train_ner()
    # evaluate()
    train_re()


if __name__ == '__main__':
    main()
