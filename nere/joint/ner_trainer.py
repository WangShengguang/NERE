import logging
import os
from pathlib import Path

import torch

from nere.joint.config import Config
from nere.ner.torchs.evaluator import Evaluator as NerEvaluator
from nere.ner.torchs.models import BERTSoftmax, BERTCRF
from nere.re.data_helper import DataHelper
from nere.model_urils.torch_utils import Trainer as BaseTrainer


class JointNerTrainer(BaseTrainer):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        model_dir = os.path.join(Config.torch_ckpt_dir, Config.save_dir, "ner")  # different
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, self.model_name + ".bin")

    def get_model(self):
        self.data_helper = DataHelper()
        num_labels = len(self.data_helper.entity_tag2id)
        if self.model_name == 'BERTSoftmax':
            model = BERTSoftmax.from_pretrained(Config.bert_pretrained_dir, num_labels=num_labels)
        elif self.model_name == 'BERTCRF':
            model = BERTCRF.from_pretrained(Config.bert_pretrained_dir, num_labels=num_labels)
        else:
            raise ValueError("Unknown model, must be one of 'BERTSoftmax'/'BERTCRF'")
        return model

    def train_step(self, batch_data):
        batch_data["sents"] = torch.tensor(batch_data["sents"], dtype=torch.long).to(Config.device)
        batch_data["sents_tags"] = torch.tensor(batch_data["sents_tags"], dtype=torch.long).to(Config.device)
        batch_masks = batch_data["sents"].gt(0)
        loss = self.model(batch_data["sents"], token_type_ids=None,
                          attention_mask=batch_masks, labels=batch_data["sents_tags"])
        self.backfoward(loss)
        return loss

    def run(self):
        self.model = self.get_model()
        if Config.load_pretrain and Path(self.model_path).is_file():
            self.model.load_state_dict(torch.load(self.model_path))  # 断点续训
            logging.info("load model from {}...".format(self.model_path))
        self.init_model()
        logging.info("Joint NER only start train {}...".format(self.model_name))
        last_epoch_num = 0
        patience_counter = 0
        best_val_f1 = 0
        for batch_data in self.data_helper.batch_iter(data_type="train",
                                                      batch_size=Config.batch_size,
                                                      epoch_nums=Config.max_epoch_nums):
            loss = self.train_step(batch_data)
            logging.info("* global_step:{} loss: {:.4f}".format(self.global_step, loss))
            epoch_num = self.data_helper.epoch_num
            self.scheduler.step(epoch=epoch_num)  # 更新学习率
            if epoch_num > last_epoch_num:  # 新的epoch
                last_epoch_num = epoch_num
                with torch.no_grad():  # 适用于测试阶段，不需要反向传播
                    acc, precision, recall, f1 = NerEvaluator(model=self.model).test()
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
