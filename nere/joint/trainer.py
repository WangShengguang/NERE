import logging

import torch

from nere.joint.config import Config
from nere.joint.evaluator import Evaluator
from nere.joint.models import JointNerRe
from nere.re.data_helper import DataHelper
from nere.torch_utils import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, ner_model, re_model):
        self.ner_model = ner_model
        self.re_model = re_model
        super().__init__("Joint" + ner_model + re_model, save_dir=Config.save_dir)

    def get_model(self):
        self.data_helper = DataHelper()
        num_ner_tags = len(self.data_helper.entity_tag2id)
        num_re_tags = len(self.data_helper.rel_label2id)
        model = JointNerRe.from_pretrained(Config.bert_pretrained_dir,
                                           ner_model=self.ner_model, re_model=self.re_model,
                                           num_ner_labels=num_ner_tags, num_re_labels=num_re_tags)

        return model

    def train_step(self, batch_data, is_train=True):
        # NER
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
            self.scheduler.step(epoch=self.data_helper.epoch_num)  # 更新学习率
            return loss
        else:
            ner_logits, re_logits = self.model(batch_data, is_train=is_train)
            return ner_logits, re_logits

    def run(self):
        self.model = self.get_model()
        self.init_model()
        logging.info("NER&RE start train {}+{}...".format(self.ner_model, self.re_model))
        last_epoch_num = 0
        for batch_data in self.data_helper.batch_iter(data_type="train",
                                                      batch_size=Config.batch_size,
                                                      epoch_nums=Config.epoch_nums):

            loss = self.train_step(batch_data, is_train=True)

            logging.info("* global_step:{} loss: {:.4f}".format(self.global_step, loss))
            if self.data_helper.epoch_num > last_epoch_num:
                last_epoch_num = self.data_helper.epoch_num
                metrics = Evaluator(model=self.model).test()
                acc, precision, recall, f1 = metrics["NER"]
                logging.info("*NER acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                    acc, precision, recall, f1))
                acc, precision, recall, f1 = metrics["RE"]
                logging.info("* RE acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                    acc, precision, recall, f1))
            if self.global_step % Config.save_step == 0:
                ner_model_path = torch.save(self.model.ner.state_dict(), self.ner_model + ".bin")
                re_model_path = torch.save(self.model.re.state_dict(), self.re_model + ".bin")
                logging.info("**save to ner_model_path: {},re_model_path: {}".format(ner_model_path, re_model_path))
