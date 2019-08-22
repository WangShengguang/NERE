import logging

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn

from nere.ner.config import Config
from nere.ner.data_helper import DataHelper
from nere.ner.torch_models.models import BERTCRF, BERTSoftmax


class Trainer(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.data_helper = DataHelper()

    def get_model(self):
        if self.model_name == 'BERTSoftmax':
            model = BERTSoftmax.from_pretrained(Config.bert_pretrained_dir, num_labels=len(self.data_helper.tag2id))
        elif self.model_name == 'BERTCRF':
            model = BERTCRF.from_pretrained(Config.bert_pretrained_dir, num_labels=len(self.data_helper.tag2id))
        else:
            raise ValueError("Unknown model, must be one of 'BERTSoftmax'/'BERTCRF'")
        return model

    def run(self):
        model = self.get_model()
        if Config.gpu_nums > 1 and Config.multi_gpu:
            model = torch.nn.DataParallel(model)
        if Config.full_finetuning:
            pass  # TODO 参考源代码含义
        param_optimizer = list(model.named_parameters())
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
        optimizer = Adam(optimizer_grouped_parameters, lr=Config.learning_rate)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
        step = 0
        for x_batch, y_batch in self.data_helper.batch_iter(data_type="train",
                                                            batch_size=Config.batch_size,
                                                            epoch_nums=Config.epoch_nums):

            x_batch = torch.tensor(x_batch).to(torch.int64)
            loss = model(input_ids=x_batch, token_type_ids=None, attention_mask=x_batch.gt(0),
                         labels=torch.tensor(y_batch).to(torch.int64))

            if Config.gpu_nums > 1 and Config.multi_gpu:
                loss = loss.mean()  # mean() to average on multi-gpu

            if Config.gradient_accumulation_steps > 1:
                loss = loss / Config.gradient_accumulation_steps

            # compute gradients of all variables wrt loss
            loss.backward()
            logging.info("** loss: {}".format(loss))
            step += 1
            # if step % Config.save_step == 0:
            #     torch.save()
            if (step + 1) % Config.gradient_accumulation_steps == 0:
                # gradient clipping
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=Config.clip_grad)
                # performs updates using calculated gradients
                optimizer.step()
                # clear previous gradients
                optimizer.zero_grad()
            scheduler.step(epoch=self.data_helper.epoch_num)  # 更新学习率
