import logging
import os
from pathlib import Path

import torch

from nere.joint.config import Config
from nere.joint.evaluator import Evaluator
from nere.joint.models import JointNerRe
from nere.re.data_helper import DataHelper
from nere.torch_utils import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, ner_model, re_model):
        super().__init__()
        self.ner_model = ner_model
        self.re_model = re_model
        ckpt_dir = os.path.join(Config.torch_ckpt_dir, Config.save_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ner_path = os.path.join(ckpt_dir, self.ner_model + ".bin")
        self.re_path = os.path.join(ckpt_dir, self.re_model + ".bin")
        self.joint_path = os.path.join(ckpt_dir, "joint{}{}.bin".format(self.ner_model, self.re_model))

    def get_model(self):
        self.data_helper = DataHelper()
        num_ner_tags = len(self.data_helper.entity_tag2id)
        num_re_tags = len(self.data_helper.rel_label2id)
        model = JointNerRe.from_pretrained(Config.bert_pretrained_dir,
                                           ner_model=self.ner_model, re_model=self.re_model,
                                           num_ner_labels=num_ner_tags, num_re_labels=num_re_tags)
        return model

    def train_step(self, batch_data, is_train=True):
        batch_data["ent_labels"] = torch.tensor(batch_data["ent_labels"], dtype=torch.long).to(Config.device)
        batch_data["e1_masks"] = torch.tensor(batch_data["e1_masks"], dtype=torch.long).to(Config.device)
        batch_data["e2_masks"] = torch.tensor(batch_data["e2_masks"], dtype=torch.long).to(Config.device)
        batch_data["sents"] = torch.tensor(batch_data["sents"], dtype=torch.long).to(Config.device)
        batch_data["fake_rel_labels"] = torch.tensor(batch_data["fake_rel_labels"], dtype=torch.long).to(
            Config.device)
        batch_data["sents_tags"] = torch.tensor(batch_data["sents_tags"], dtype=torch.long).to(Config.device)
        batch_data["rel_labels"] = torch.tensor(batch_data["rel_labels"], dtype=torch.long).to(Config.device)
        if is_train:
            loss = self.model(batch_data, is_train=is_train)
            self.backfoward(loss)
            return loss
        else:
            ner_logits, re_logits = self.model(batch_data, is_train=is_train)
            return ner_logits, re_logits

    def run(self):
        self.model = self.get_model()
        if Config.load_pretrain and Path(self.joint_path).is_file():
            self.model.load_state_dict(torch.load(self.joint_path))  # 断点续训
        self.init_model()
        logging.info("NER&RE start train {}+{}...".format(self.ner_model, self.re_model))
        last_epoch_num = 0
        best_val_f1 = 0
        patience_counter = 0
        for batch_data in self.data_helper.batch_iter(data_type="train",
                                                      batch_size=Config.batch_size,
                                                      epoch_nums=Config.max_epoch_nums):

            loss = self.train_step(batch_data, is_train=True)
            epoch_num = self.data_helper.epoch_num
            self.scheduler.step(epoch=epoch_num)  # 更新学习率
            logging.info("* global_step:{} loss: {:.4f}".format(self.global_step, loss))
            if epoch_num > last_epoch_num:
                last_epoch_num = epoch_num
                metrics = Evaluator(model=self.model).test()
                logging.info("*NER acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(*metrics["NER"]))
                logging.info("* RE acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(*metrics["RE"]))
                f1 = (metrics["NER"][-1] + metrics["RE"][-1]) / 2
                if f1 - best_val_f1 > 0:
                    # model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    torch.save(self.model.ner.state_dict(), self.ner_path)
                    torch.save(self.model.re.state_dict(), self.re_path)
                    torch.save(self.model.state_dict(), self.joint_path)  # Only save the model it-self
                    logging.info("** - Found new best F1 ,save to model_path: {}".format(self.joint_path))
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
