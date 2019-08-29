import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from nere.config import Config


class Trainer(object):
    def __init__(self):
        self.global_step = 0
        self.no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.exclude_params = not_used_params

    def init_model(self, model):
        model.to(Config.device)  # without this there is no error, but it runs in CPU (instead of GPU).
        if Config.gpu_nums > 1 and Config.multi_gpu:
            model = torch.nn.DataParallel(model)

        if Config.full_finetuning:
            pass  # TODO 参考源代码含义
        param_optimizer = list(model.named_parameters())

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if
                        not any(nd in n for nd in self.no_decay) and n not in self.exclude_params],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if
                        any(nd in n for nd in self.no_decay) and n not in self.exclude_params],
             'weight_decay_rate': 0.0}
        ]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=Config.learning_rate)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))

    def backfoward(self, loss):
        if Config.gpu_nums > 1 and Config.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        # compute gradients of all variables wrt loss
        loss.backward()

        if self.global_step % Config.gradient_accumulation_steps == 0:
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=Config.clip_grad)
            # performs updates using calculated gradients
            self.optimizer.step()
            # clear previous gradients
            self.optimizer.zero_grad()
        return loss


not_used_params = ['cls.predictions.bias', 'cls.predictions.transform.dense.weight',
                   'cls.predictions.transform.dense.bias',
                   'cls.predictions.transform.LayerNorm.weight',
                   'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight',
                   'cls.seq_relationship.weight', 'cls.seq_relationship.bias']
