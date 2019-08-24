import logging
import os
from pathlib import Path
import torch

from nere.re.config import Config
from nere.re.data_helper import DataHelper
from nere.re.torchs.models import BERTMultitask, BERTSoftmax
from nere.torch_utils import Trainer as BaseTrainer
from .evaluator import Evaluator


class Trainer(BaseTrainer):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        model_dir = os.path.join(Config.torch_ckpt_dir, Config.save_dir)
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, self.model_name + ".bin")

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
        if Config.load_pretrain and Path(self.model_path).is_file():
            self.model.load_state_dict(torch.load(self.model_path))  # 断点续训
        self.init_model()
        logging.info("RE start train {}...".format(self.model_name))
        last_epoch_num = 0
        patience_counter = 0
        best_val_f1 = 0
        for batch_data in self.data_helper.batch_iter(data_type="train",
                                                      batch_size=Config.batch_size,
                                                      epoch_nums=Config.max_epoch_nums):
            loss = self.train_step(batch_data)
            epoch_num = self.data_helper.epoch_num
            logging.info("* global_step:{} loss: {:.4f}".format(self.global_step, loss))
            self.backfoward(loss)
            self.scheduler.step(epoch=epoch_num)  # 更新学习率
            if epoch_num > last_epoch_num:
                last_epoch_num = epoch_num
                acc, precision, recall, f1 = Evaluator(model=self.model).test()
                logging.info("acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                    acc, precision, recall, f1))
                if f1 - best_val_f1 > 0:
                    re_path = torch.save(self.model.state_dict(), self.model_path)
                    logging.info("** - Found new best F1 ,save to model_path: {}".format(re_path))
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
