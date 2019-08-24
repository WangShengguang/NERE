import logging
import os
from pathlib import Path

import torch

from nere.ner.config import Config
from nere.ner.data_helper import DataHelper
from nere.ner.torchs.evaluator import Evaluator
from nere.ner.torchs.models import BERTCRF, BERTSoftmax
from nere.torch_utils import Trainer as BeseTrainer


class Trainer(BeseTrainer):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        model_dir = os.path.join(Config.torch_ckpt_dir, Config.save_dir)
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, self.model_name + ".bin")

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
        loss_val = self.backfoward(loss)
        return loss

    def run(self):
        self.model = self.get_model()
        if Config.load_pretrain and Path(self.model_path).is_file():
            self.model.load_state_dict(torch.load(self.model_path))  # 断点续训
        self.init_model()
        logging.info("NER start train {}...".format(self.model_name))
        last_epoch_num = 0
        patience_counter = 0
        best_val_f1 = 0
        for x_batch, batch_rel_labels in self.data_helper.batch_iter(data_type="train",
                                                                     batch_size=Config.batch_size,
                                                                     epoch_nums=Config.max_epoch_nums):
            loss = self.train_step(x_batch, batch_rel_labels)
            logging.info("* global_step:{} loss: {:.4f}".format(self.global_step, loss))
            epoch_num = self.data_helper.epoch_num
            self.scheduler.step(epoch=epoch_num)  # 更新学习率
            if epoch_num > last_epoch_num:  # 新的epoch
                last_epoch_num = epoch_num
                acc, precision, recall, f1 = Evaluator(model=self.model).test()
                logging.info("acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                    acc, precision, recall, f1))
                if f1 - best_val_f1 > 0:
                    torch.save(self.model.state_dict(), self.model_path)
                    logging.info("** - Found new best F1 ,save to model_path: {}".format(self.model_path))
                    best_val_f1 = f1
                    if f1 - best_val_f1 < Config.patience:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                else:
                    patience_counter += 1

            # Early stopping and logging best f1
            if (patience_counter >= Config.patience_num and epoch_num > Config.min_epoch_nums) \
                    or epoch_num == Config.max_epoch_nums:
                logging.info("Best val f1: {:05.2f}".format(best_val_f1))
                break
