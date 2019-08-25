import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from nere.re.config import Config


class Trainer(object):
    def __init__(self):
        self.global_step = 0
        self.no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.exclude_params = re_BERTMultitask_not_used_params + joint_not_used_params

    def get_model(self):
        raise NotImplemented

    def init_model(self):
        self.model.to(Config.device)  # without this there is no error, but it runs in CPU (instead of GPU).
        if Config.gpu_nums > 1 and Config.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

        if Config.full_finetuning:
            pass  # TODO 参考源代码含义
        param_optimizer = list(self.model.named_parameters())

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


re_BERTMultitask_not_used_params = ['cls.predictions.bias', 'cls.predictions.transform.dense.weight',
                                    'cls.predictions.transform.dense.bias',
                                    'cls.predictions.transform.LayerNorm.weight',
                                    'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight',
                                    'cls.seq_relationship.weight', 'cls.seq_relationship.bias']

joint_not_used_params = ['bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight',
                         'bert.embeddings.token_type_embeddings.weight', 'bert.embeddings.LayerNorm.weight',
                         'bert.embeddings.LayerNorm.bias',
                         'bert.encoder.layer.0.attention.self.query.weight',
                         'bert.encoder.layer.0.attention.self.query.bias',
                         'bert.encoder.layer.0.attention.self.key.weight',
                         'bert.encoder.layer.0.attention.self.key.bias',
                         'bert.encoder.layer.0.attention.self.value.weight',
                         'bert.encoder.layer.0.attention.self.value.bias',
                         'bert.encoder.layer.0.attention.output.dense.weight',
                         'bert.encoder.layer.0.attention.output.dense.bias',
                         'bert.encoder.layer.0.attention.output.LayerNorm.weight',
                         'bert.encoder.layer.0.attention.output.LayerNorm.bias',
                         'bert.encoder.layer.0.intermediate.dense.weight',
                         'bert.encoder.layer.0.intermediate.dense.bias',
                         'bert.encoder.layer.0.output.dense.weight', 'bert.encoder.layer.0.output.dense.bias',
                         'bert.encoder.layer.0.output.LayerNorm.weight', 'bert.encoder.layer.0.output.LayerNorm.bias',
                         'bert.encoder.layer.1.attention.self.query.weight',
                         'bert.encoder.layer.1.attention.self.query.bias',
                         'bert.encoder.layer.1.attention.self.key.weight',
                         'bert.encoder.layer.1.attention.self.key.bias',
                         'bert.encoder.layer.1.attention.self.value.weight',
                         'bert.encoder.layer.1.attention.self.value.bias',
                         'bert.encoder.layer.1.attention.output.dense.weight',
                         'bert.encoder.layer.1.attention.output.dense.bias',
                         'bert.encoder.layer.1.attention.output.LayerNorm.weight',
                         'bert.encoder.layer.1.attention.output.LayerNorm.bias',
                         'bert.encoder.layer.1.intermediate.dense.weight',
                         'bert.encoder.layer.1.intermediate.dense.bias',
                         'bert.encoder.layer.1.output.dense.weight', 'bert.encoder.layer.1.output.dense.bias',
                         'bert.encoder.layer.1.output.LayerNorm.weight', 'bert.encoder.layer.1.output.LayerNorm.bias',
                         'bert.encoder.layer.2.attention.self.query.weight',
                         'bert.encoder.layer.2.attention.self.query.bias',
                         'bert.encoder.layer.2.attention.self.key.weight',
                         'bert.encoder.layer.2.attention.self.key.bias',
                         'bert.encoder.layer.2.attention.self.value.weight',
                         'bert.encoder.layer.2.attention.self.value.bias',
                         'bert.encoder.layer.2.attention.output.dense.weight',
                         'bert.encoder.layer.2.attention.output.dense.bias',
                         'bert.encoder.layer.2.attention.output.LayerNorm.weight',
                         'bert.encoder.layer.2.attention.output.LayerNorm.bias',
                         'bert.encoder.layer.2.intermediate.dense.weight',
                         'bert.encoder.layer.2.intermediate.dense.bias',
                         'bert.encoder.layer.2.output.dense.weight', 'bert.encoder.layer.2.output.dense.bias',
                         'bert.encoder.layer.2.output.LayerNorm.weight', 'bert.encoder.layer.2.output.LayerNorm.bias',
                         'bert.encoder.layer.3.attention.self.query.weight',
                         'bert.encoder.layer.3.attention.self.query.bias',
                         'bert.encoder.layer.3.attention.self.key.weight',
                         'bert.encoder.layer.3.attention.self.key.bias',
                         'bert.encoder.layer.3.attention.self.value.weight',
                         'bert.encoder.layer.3.attention.self.value.bias',
                         'bert.encoder.layer.3.attention.output.dense.weight',
                         'bert.encoder.layer.3.attention.output.dense.bias',
                         'bert.encoder.layer.3.attention.output.LayerNorm.weight',
                         'bert.encoder.layer.3.attention.output.LayerNorm.bias',
                         'bert.encoder.layer.3.intermediate.dense.weight',
                         'bert.encoder.layer.3.intermediate.dense.bias',
                         'bert.encoder.layer.3.output.dense.weight', 'bert.encoder.layer.3.output.dense.bias',
                         'bert.encoder.layer.3.output.LayerNorm.weight', 'bert.encoder.layer.3.output.LayerNorm.bias',
                         'bert.encoder.layer.4.attention.self.query.weight',
                         'bert.encoder.layer.4.attention.self.query.bias',
                         'bert.encoder.layer.4.attention.self.key.weight',
                         'bert.encoder.layer.4.attention.self.key.bias',
                         'bert.encoder.layer.4.attention.self.value.weight',
                         'bert.encoder.layer.4.attention.self.value.bias',
                         'bert.encoder.layer.4.attention.output.dense.weight',
                         'bert.encoder.layer.4.attention.output.dense.bias',
                         'bert.encoder.layer.4.attention.output.LayerNorm.weight',
                         'bert.encoder.layer.4.attention.output.LayerNorm.bias',
                         'bert.encoder.layer.4.intermediate.dense.weight',
                         'bert.encoder.layer.4.intermediate.dense.bias',
                         'bert.encoder.layer.4.output.dense.weight', 'bert.encoder.layer.4.output.dense.bias',
                         'bert.encoder.layer.4.output.LayerNorm.weight', 'bert.encoder.layer.4.output.LayerNorm.bias',
                         'bert.encoder.layer.5.attention.self.query.weight',
                         'bert.encoder.layer.5.attention.self.query.bias',
                         'bert.encoder.layer.5.attention.self.key.weight',
                         'bert.encoder.layer.5.attention.self.key.bias',
                         'bert.encoder.layer.5.attention.self.value.weight',
                         'bert.encoder.layer.5.attention.self.value.bias',
                         'bert.encoder.layer.5.attention.output.dense.weight',
                         'bert.encoder.layer.5.attention.output.dense.bias',
                         'bert.encoder.layer.5.attention.output.LayerNorm.weight',
                         'bert.encoder.layer.5.attention.output.LayerNorm.bias',
                         'bert.encoder.layer.5.intermediate.dense.weight',
                         'bert.encoder.layer.5.intermediate.dense.bias',
                         'bert.encoder.layer.5.output.dense.weight', 'bert.encoder.layer.5.output.dense.bias',
                         'bert.encoder.layer.5.output.LayerNorm.weight', 'bert.encoder.layer.5.output.LayerNorm.bias',
                         'bert.encoder.layer.6.attention.self.query.weight',
                         'bert.encoder.layer.6.attention.self.query.bias',
                         'bert.encoder.layer.6.attention.self.key.weight',
                         'bert.encoder.layer.6.attention.self.key.bias',
                         'bert.encoder.layer.6.attention.self.value.weight',
                         'bert.encoder.layer.6.attention.self.value.bias',
                         'bert.encoder.layer.6.attention.output.dense.weight',
                         'bert.encoder.layer.6.attention.output.dense.bias',
                         'bert.encoder.layer.6.attention.output.LayerNorm.weight',
                         'bert.encoder.layer.6.attention.output.LayerNorm.bias',
                         'bert.encoder.layer.6.intermediate.dense.weight',
                         'bert.encoder.layer.6.intermediate.dense.bias',
                         'bert.encoder.layer.6.output.dense.weight', 'bert.encoder.layer.6.output.dense.bias',
                         'bert.encoder.layer.6.output.LayerNorm.weight', 'bert.encoder.layer.6.output.LayerNorm.bias',
                         'bert.encoder.layer.7.attention.self.query.weight',
                         'bert.encoder.layer.7.attention.self.query.bias',
                         'bert.encoder.layer.7.attention.self.key.weight',
                         'bert.encoder.layer.7.attention.self.key.bias',
                         'bert.encoder.layer.7.attention.self.value.weight',
                         'bert.encoder.layer.7.attention.self.value.bias',
                         'bert.encoder.layer.7.attention.output.dense.weight',
                         'bert.encoder.layer.7.attention.output.dense.bias',
                         'bert.encoder.layer.7.attention.output.LayerNorm.weight',
                         'bert.encoder.layer.7.attention.output.LayerNorm.bias',
                         'bert.encoder.layer.7.intermediate.dense.weight',
                         'bert.encoder.layer.7.intermediate.dense.bias',
                         'bert.encoder.layer.7.output.dense.weight', 'bert.encoder.layer.7.output.dense.bias',
                         'bert.encoder.layer.7.output.LayerNorm.weight', 'bert.encoder.layer.7.output.LayerNorm.bias',
                         'bert.encoder.layer.8.attention.self.query.weight',
                         'bert.encoder.layer.8.attention.self.query.bias',
                         'bert.encoder.layer.8.attention.self.key.weight',
                         'bert.encoder.layer.8.attention.self.key.bias',
                         'bert.encoder.layer.8.attention.self.value.weight',
                         'bert.encoder.layer.8.attention.self.value.bias',
                         'bert.encoder.layer.8.attention.output.dense.weight',
                         'bert.encoder.layer.8.attention.output.dense.bias',
                         'bert.encoder.layer.8.attention.output.LayerNorm.weight',
                         'bert.encoder.layer.8.attention.output.LayerNorm.bias',
                         'bert.encoder.layer.8.intermediate.dense.weight',
                         'bert.encoder.layer.8.intermediate.dense.bias',
                         'bert.encoder.layer.8.output.dense.weight', 'bert.encoder.layer.8.output.dense.bias',
                         'bert.encoder.layer.8.output.LayerNorm.weight', 'bert.encoder.layer.8.output.LayerNorm.bias',
                         'bert.encoder.layer.9.attention.self.query.weight',
                         'bert.encoder.layer.9.attention.self.query.bias',
                         'bert.encoder.layer.9.attention.self.key.weight',
                         'bert.encoder.layer.9.attention.self.key.bias',
                         'bert.encoder.layer.9.attention.self.value.weight',
                         'bert.encoder.layer.9.attention.self.value.bias',
                         'bert.encoder.layer.9.attention.output.dense.weight',
                         'bert.encoder.layer.9.attention.output.dense.bias',
                         'bert.encoder.layer.9.attention.output.LayerNorm.weight',
                         'bert.encoder.layer.9.attention.output.LayerNorm.bias',
                         'bert.encoder.layer.9.intermediate.dense.weight',
                         'bert.encoder.layer.9.intermediate.dense.bias',
                         'bert.encoder.layer.9.output.dense.weight', 'bert.encoder.layer.9.output.dense.bias',
                         'bert.encoder.layer.9.output.LayerNorm.weight', 'bert.encoder.layer.9.output.LayerNorm.bias',
                         'bert.encoder.layer.10.attention.self.query.weight',
                         'bert.encoder.layer.10.attention.self.query.bias',
                         'bert.encoder.layer.10.attention.self.key.weight',
                         'bert.encoder.layer.10.attention.self.key.bias',
                         'bert.encoder.layer.10.attention.self.value.weight',
                         'bert.encoder.layer.10.attention.self.value.bias',
                         'bert.encoder.layer.10.attention.output.dense.weight',
                         'bert.encoder.layer.10.attention.output.dense.bias',
                         'bert.encoder.layer.10.attention.output.LayerNorm.weight',
                         'bert.encoder.layer.10.attention.output.LayerNorm.bias',
                         'bert.encoder.layer.10.intermediate.dense.weight',
                         'bert.encoder.layer.10.intermediate.dense.bias',
                         'bert.encoder.layer.10.output.dense.weight', 'bert.encoder.layer.10.output.dense.bias',
                         'bert.encoder.layer.10.output.LayerNorm.weight',
                         'bert.encoder.layer.10.output.LayerNorm.bias',
                         'bert.encoder.layer.11.attention.self.query.weight',
                         'bert.encoder.layer.11.attention.self.query.bias',
                         'bert.encoder.layer.11.attention.self.key.weight',
                         'bert.encoder.layer.11.attention.self.key.bias',
                         'bert.encoder.layer.11.attention.self.value.weight',
                         'bert.encoder.layer.11.attention.self.value.bias',
                         'bert.encoder.layer.11.attention.output.dense.weight',
                         'bert.encoder.layer.11.attention.output.dense.bias',
                         'bert.encoder.layer.11.attention.output.LayerNorm.weight',
                         'bert.encoder.layer.11.attention.output.LayerNorm.bias',
                         'bert.encoder.layer.11.intermediate.dense.weight',
                         'bert.encoder.layer.11.intermediate.dense.bias',
                         'bert.encoder.layer.11.output.dense.weight', 'bert.encoder.layer.11.output.dense.bias',
                         'bert.encoder.layer.11.output.LayerNorm.weight',
                         'bert.encoder.layer.11.output.LayerNorm.bias',
                         'bert.pooler.dense.weight', 'bert.pooler.dense.bias']
