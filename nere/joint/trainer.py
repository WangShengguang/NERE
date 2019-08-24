import logging

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from nere.joint.config import Config
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

    def init_model(self):
        self.model.to(Config.device)  # without this there is no error, but it runs in CPU (instead of GPU).
        self.model.eval()  # declaring to the system that we're only doing 'forward' calculations
        self.optimizer = Adam(self.model.parameters(), lr=Config.learning_rate)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))

    def train_step(self, batch_data, is_train=True):
        # NER
        # RE
        batch_data["ent_labels"] = torch.tensor(batch_data["ent_labels"], dtype=torch.long).to(Config.device)
        batch_data["e1_masks"] = torch.tensor(batch_data["e1_masks"], dtype=torch.long).to(Config.device)
        batch_data["e2_masks"] = torch.tensor(batch_data["e2_masks"], dtype=torch.long).to(Config.device)
        batch_data["sents"] = torch.tensor(batch_data["sents"], dtype=torch.long).to(Config.device)
        batch_data["fake_rel_labels"] = torch.tensor(batch_data["fake_rel_labels"], dtype=torch.long).to(Config.device)
        batch_data["sents_tags"] = torch.tensor(batch_data["sents_tags"], dtype=torch.long).to(Config.device)
        batch_data["rel_labels"] = torch.tensor(batch_data["rel_labels"], dtype=torch.long).to(Config.device)
        if is_train:
            loss = self.model(batch_data, is_train)
            self.backfoward(loss)
            self.scheduler.step(epoch=self.data_helper.epoch_num)  # 更新学习率
            return loss
        else:
            ner_logits, re_logits = self.model(batch_data, is_train)
            return ner_logits, re_logits

    def run(self):
        self.model = self.get_model()
        self.init_model()
        logging.info("NER&RE start train {}+{}...".format(self.ner_model, self.re_model))
        for batch_data in self.data_helper.batch_iter(data_type="train",
                                                      batch_size=Config.batch_size,
                                                      epoch_nums=Config.epoch_nums):
            loss = self.train_step(batch_data, is_train=True)
            logging.info("**global_step:{} loss: {:.6f}".format(self.global_step, loss))
            if self.global_step % Config.save_step == 0:
                model_path = self.saver.save(self.model)
                logging.info("**save to model_path: {}".format(model_path))
