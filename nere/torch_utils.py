import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from nere.re.config import Config


class Trainer(object):
    def __init__(self, model_name, save_dir="."):
        self.model_name = model_name
        self.saver = Saver(model_name, save_dir=save_dir, mode=Config.save_mode)
        self.global_step = 0

    def get_model(self):
        return NotImplemented

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

    def forward(self, x_batch, y_batch):
        loss = self.model(x_batch, y_batch)
        return loss

    def backfoward(self, loss):
        if Config.gpu_nums > 1 and Config.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        # compute gradients of all variables wrt loss
        loss.backward()
        self.global_step += 1

        if self.global_step % Config.gradient_accumulation_steps == 0:
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=Config.clip_grad)
            # performs updates using calculated gradients
            self.optimizer.step()
            # clear previous gradients
            self.optimizer.zero_grad()
        return loss


class Saver(object):
    def __init__(self, model_name, save_dir=".", mode=Config.save_mode):
        self.model_name = model_name
        self.mode = mode
        model_dir = os.path.join(Config.torch_ckpt_dir, save_dir)
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(Config.torch_ckpt_dir, self.model_name + ".bin")

    def save(self, model):
        """  https://blog.csdn.net/u011276025/article/details/78507950
        :param mode: full_model or only params
        :return:
        """
        if self.mode == "full_model":
            torch.save(model, self.model_path)
        else:
            torch.save(model.state_dict(), self.model_path)
        return self.model_path

    def load(self, model=None):
        if self.mode == "full_model":
            model = torch.load(self.model_path)
        else:
            model.load_state_dict(torch.load(self.model_path))
        return model
