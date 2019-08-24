import logging

import torch

from nere.re.config import Config
from nere.re.data_helper import DataHelper
from nere.re.torch_models.models import BERTMultitask, BERTSoftmax
from nere.torch_utils import Trainer as BaseTrainer
from .evaluator import Evaluator


class Trainer(BaseTrainer):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__(model_name, save_dir=Config.save_dir)

    def get_model(self):
        self.data_helper = DataHelper()
        if self.model_name == 'BERTSoftmax':
            model = BERTSoftmax.from_pretrained(Config.bert_pretrained_dir,
                                                num_labels=len(self.data_helper.rel_label2id))
        elif self.model_name == 'BERTMultitask':
            model = BERTMultitask.from_pretrained(Config.bert_pretrained_dir,
                                                  num_labels=len(self.data_helper.rel_label2id))
        else:
            raise ValueError("Unknown model, must be one of 'BERTSoftmax'/'BERTMultitask'")
        return model

    def train_step(self, batch_data):
        # since all data are indices, we convert them to torch LongTensors
        batch_data["ent_labels"] = torch.tensor(batch_data["ent_labels"], dtype=torch.long).to(Config.device)
        batch_data["e1_masks"] = torch.tensor(batch_data["e1_masks"], dtype=torch.long).to(Config.device)
        batch_data["e2_masks"] = torch.tensor(batch_data["e2_masks"], dtype=torch.long).to(Config.device)
        batch_data["sents"] = torch.tensor(batch_data["sents"], dtype=torch.long).to(Config.device)
        batch_data["fake_rel_labels"] = torch.tensor(batch_data["fake_rel_labels"], dtype=torch.long).to(Config.device)
        batch_rel_labels = torch.tensor(batch_data["rel_labels"], dtype=torch.long).to(Config.device)

        loss = self.model(batch_data, batch_rel_labels)
        return loss

    def run(self):
        self.model = self.get_model()
        self.init_model()
        logging.info("RE start train {}...".format(self.model_name))
        last_epoch_num = 0
        for batch_data in self.data_helper.batch_iter(data_type="train",
                                                      batch_size=Config.batch_size,
                                                      epoch_nums=Config.epoch_nums):
            loss = self.train_step(batch_data)
            logging.info("* global_step:{} loss: {:.4f}".format(self.global_step, loss))
            self.backfoward(loss)
            self.scheduler.step(epoch=self.data_helper.epoch_num)  # 更新学习率
            if self.data_helper.epoch_num > last_epoch_num:
                last_epoch_num = self.data_helper.epoch_num
            acc, precision, recall, f1 = Evaluator(model=self.model).test()
            print("acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(acc, precision, recall, f1))
            if self.global_step % Config.save_step == 0:
                model_path = self.saver.save(self.model)
                logging.info("**save to model_path: {}".format(model_path))
