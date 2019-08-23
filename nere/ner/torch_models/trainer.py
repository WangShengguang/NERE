import logging

import torch

from nere.ner.config import Config
from nere.ner.data_helper import DataHelper
from nere.ner.torch_models.models import BERTCRF, BERTSoftmax
from nere.torch_utils import Trainer as BeseTrainer


class Trainer(BeseTrainer):
    def __init__(self, model_name, save_dir="re"):
        super().__init__(model_name, save_dir)

    def get_model(self):
        self.data_helper = DataHelper()
        num_labels = len(self.data_helper.tag2id)
        if self.model_name == 'BERTSoftmax':
            model = BERTSoftmax.from_pretrained(Config.bert_pretrained_dir, num_labels=num_labels)
        elif self.model_name == 'BERTCRF':
            model = BERTCRF.from_pretrained(Config.bert_pretrained_dir, num_labels=num_labels)
        else:
            raise ValueError("Unknown model, must be one of 'BERTSoftmax'/'BERTCRF'")
        return model

    def train_step(self, x_batch, batch_rel_labels):
        x_batch = torch.tensor(x_batch, dtype=torch.long).to(Config.device)
        batch_rel_labels = torch.tensor(batch_rel_labels, dtype=torch.long).to(Config.device)
        loss = self.model(x_batch, labels=batch_rel_labels)
        return loss

    def run(self):
        self.model = self.get_model()
        self.init_model()
        logging.info("NER start train {}...".format(self.model_name))
        for x_batch, batch_rel_labels in self.data_helper.batch_iter(data_type="train",
                                                                     batch_size=Config.batch_size,
                                                                     epoch_nums=Config.epoch_nums):
            loss = self.train_step(x_batch, batch_rel_labels)
            loss_val = self.backfoward(loss)
            logging.info("**global_step:{} loss: {:.6f}".format(self.global_step, loss_val))
            self.scheduler.step(epoch=self.data_helper.epoch_num)  # 更新学习率

            if self.global_step % Config.save_step:
                self.saver.save(self.model)
