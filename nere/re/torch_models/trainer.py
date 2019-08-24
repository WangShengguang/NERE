import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from nere.re.config import Config
from nere.re.data_helper import DataHelper
from nere.re.torch_models.models import BERTMultitask, BERTSoftmax
from nere.torch_utils import Trainer as BaseTrainer


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
        for batch_data in self.data_helper.batch_iter(data_type="train",
                                                      batch_size=Config.batch_size,
                                                      epoch_nums=Config.epoch_nums):
            loss = self.train_step(batch_data)
            logging.info("**global_step:{} loss: {:.6f}".format(self.global_step, loss))
            self.backfoward(loss)
            self.scheduler.step(epoch=self.data_helper.epoch_num)  # 更新学习率
            if self.global_step % Config.save_step == 0:
                model_path = self.saver.save(self.model)
                logging.info("**save to model_path: {}".format(model_path))


class Trainer_old(object):
    def __init__(self, model_name, config: Config = None):
        self.model_name = model_name
        self.model = self.get_model()
        self.init_model()
        self.global_step = 0

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

    def init_model(self):
        self.model.to(Config.device)
        if Config.gpu_nums > 1 and Config.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

        if Config.full_finetuning:
            pass  # TODO 参考源代码含义
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        exclude_params = ['cls.predictions.bias', 'cls.predictions.transform.dense.weight',
                          'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight',
                          'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight',
                          'cls.seq_relationship.weight', 'cls.seq_relationship.bias']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if
                        not any(nd in n for nd in no_decay) and n not in exclude_params],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if
                        any(nd in n for nd in no_decay) and n not in exclude_params],
             'weight_decay_rate': 0.0}
        ]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=Config.learning_rate)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))

    def train_step(self, x_batch, y_batch):
        # since all data are indices, we convert them to torch LongTensors
        x_batch["ent_labels"] = torch.tensor(x_batch["ent_labels"], dtype=torch.long).to(Config.device)
        x_batch["e1_masks"] = torch.tensor(x_batch["e1_masks"], dtype=torch.long).to(Config.device)
        x_batch["e2_masks"] = torch.tensor(x_batch["e2_masks"], dtype=torch.long).to(Config.device)
        x_batch["sents"] = torch.tensor(x_batch["sents"], dtype=torch.long).to(Config.device)
        x_batch["fake_rel_labels"] = torch.tensor(x_batch["fake_rel_labels"], dtype=torch.long).to(Config.device)
        batch_rel_labels = torch.tensor(y_batch, dtype=torch.long).to(Config.device)

        loss = self.model(x_batch, batch_rel_labels)
        return loss

    def backfoward(self, loss):
        if Config.gpu_nums > 1 and Config.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        # compute gradients of all variables wrt loss
        loss.backward()
        self.global_step += 1
        # logging.info("**global_step:{} loss: {}".format(self.global_step, loss))
        if self.global_step % Config.gradient_accumulation_steps == 0:
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=Config.clip_grad)
            # performs updates using calculated gradients
            self.optimizer.step()
            # clear previous gradients
            self.optimizer.zero_grad()

    def run(self):
        for x_batch, batch_rel_labels in self.data_helper.batch_iter(data_type="train",
                                                                     batch_size=Config.batch_size,
                                                                     epoch_nums=Config.epoch_nums):
            loss = self.train_step(x_batch, batch_rel_labels)
            logging.info("**global_step:{} loss: {}".format(self.global_step, loss))
            self.backfoward(loss)
            self.scheduler.step(epoch=self.data_helper.epoch_num)  # 更新学习率
