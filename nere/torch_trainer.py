import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange

from config import Config
from nere.data_helper import DataHelper
from nere.evaluator import Evaluator


class BaseTrainer(object):
    def __init__(self):
        self.global_step = 0

    def init_model(self, model):
        model.to(Config.device)  # without this there is no error, but it runs in CPU (instead of GPU).
        if Config.gpu_nums > 1 and Config.multi_gpu:
            model = torch.nn.DataParallel(model)

        if Config.full_finetuning:
            pass  # TODO 参考源代码含义
        param_optimizer = list(model.named_parameters())
        self.no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.exclude_params = ['cls.predictions.bias', 'cls.predictions.transform.dense.weight',
                               'cls.predictions.transform.dense.bias',
                               'cls.predictions.transform.LayerNorm.weight',
                               'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight',
                               'cls.seq_relationship.weight', 'cls.seq_relationship.bias']

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
        # model, self.optimizer = amp.initialize(model, self.optimizer, opt_level="O1")  # 这里是“欧一”，不是“零一”
        return model

    def backfoward(self, loss, model):
        if Config.gpu_nums > 1 and Config.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        # https://zhuanlan.zhihu.com/p/79887894
        # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #     scaled_loss.backward()
        # compute gradients of all variables wrt loss
        loss.backward(retain_graph=True)

        if self.global_step % Config.gradient_accumulation_steps == 0:
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=Config.clip_grad)
            # performs updates using calculated gradients
            self.optimizer.step()
            # clear previous gradients
            self.optimizer.zero_grad()
        return loss


class Trainer(BaseTrainer):
    def __init__(self, model_name, task):
        super().__init__()
        self.model_name = model_name
        self.task = task
        self.model_dir = os.path.join(Config.torch_ckpt_dir, task)
        self.model_path = os.path.join(self.model_dir, model_name + ".bin")
        os.makedirs(self.model_dir, exist_ok=True)
        # evaluate
        self.evaluator = Evaluator(task, model_name, framework="torch", load_model=False)
        self.best_val_f1 = 0
        self.best_loss = 100
        self.patience_counter = 0
        #
        self.data_helper = DataHelper()
        self.fixed_seq_len = None

    def get_re_model(self):
        from nere.re_models import BERTMultitask, BERTSoftmax, BiLSTM_ATT, ACNN, BiLSTM
        vocab_size = len(self.data_helper.tokenizer.vocab)
        num_ent_tags = len(self.data_helper.ent_tag2id)
        num_rel_tags = len(self.data_helper.rel_label2id)
        if self.model_name == 'BERTSoftmax':
            model = BERTSoftmax.from_pretrained(Config.bert_pretrained_dir, num_labels=num_rel_tags)
        elif self.model_name == 'BERTMultitask':
            model = BERTMultitask.from_pretrained(Config.bert_pretrained_dir, num_labels=num_rel_tags)
        elif self.model_name == "BiLSTM_ATT":
            model = BiLSTM_ATT(vocab_size, num_ent_tags, num_rel_tags, Config.ent_emb_dim, Config.batch_size)
        elif self.model_name == "BiLSTM":
            model = BiLSTM(vocab_size, num_ent_tags, num_rel_tags)
        elif self.model_name == "ACNN":
            model = ACNN(vocab_size, num_ent_tags, num_rel_tags, Config.ent_emb_dim,
                         Config.max_sequence_len)
        else:
            raise ValueError("Unknown RE model {}".format(self.model_name))
        return model

    def get_ner_model(self):
        from nere.ner_models import BERTCRF, BiLSTM, BERTSoftmax, BiLSTM_ATT

        num_ent_tags = len(self.data_helper.ent_tag2id)
        vocab_size = len(self.data_helper.tokenizer.vocab)

        if self.model_name == 'BERTCRF':
            model = BERTCRF.from_pretrained(Config.bert_pretrained_dir, num_labels=num_ent_tags)
        elif self.model_name == 'BERTSoftmax':
            model = BERTSoftmax.from_pretrained(Config.bert_pretrained_dir, num_labels=num_ent_tags)
        elif self.model_name == "BiLSTM":
            # self.fixed_seq_len = Config.max_sequence_len
            model = BiLSTM(vocab_size, num_ent_tags, Config.ent_emb_dim, Config.batch_size)
        elif self.model_name == "BiLSTM_ATT":
            self.fixed_seq_len = Config.max_sequence_len
            model = BiLSTM_ATT(vocab_size, num_ent_tags, Config.ent_emb_dim, Config.batch_size, self.fixed_seq_len)
        else:
            raise ValueError("Unknown NER model {}".format(self.model_name))
        return model

    def get_model(self):
        if self.task == "ner":
            model = self.get_ner_model()
        elif self.task == "re":
            model = self.get_re_model()
        else:
            raise ValueError(self.task)
        if Config.load_pretrain and Path(self.model_path).is_file():
            model.load_state_dict(torch.load(self.model_path))  # 断点续训
            logging.info("* load model from {}".format(self.model_path))
        model = self.init_model(model)
        return model

    def save_best_loss_model(self, loss, model):
        if loss <= self.best_loss:
            torch.save(model.state_dict(), self.model_path)
            self.best_loss = loss
            _log = "loss: {:.3f}, save to :{}".format(loss, self.model_path)
            logging.info(_log)

    def evaluate_save(self, model):
        # with torch.no_grad():  # 适用于测试阶段，不需要反向传播
        self.evaluator.set_model(model=model, fixed_seq_len=self.fixed_seq_len)
        acc, precision, recall, f1 = self.evaluator.test(data_type="valid")
        if f1 > self.best_val_f1:
            torch.save(model.state_dict(), self.model_path)
            logging.info("** - Found new best F1 ,save to model_path: {}".format(self.model_path))
            # if f1 - self.best_val_f1 < Config.patience:
            #     self.patience_counter += 1
            # else:
            #     self.patience_counter = 0
            # self.best_val_f1 = f1
        else:
            self.patience_counter += 1
        return acc, precision, recall, f1

    def train_step(self, batch_data, model):
        if self.task == "ner":
            pred, loss = model(input_ids=batch_data["sents"], attention_mask=batch_data["sents"].gt(0),
                               labels=batch_data["ent_tags"])
        elif self.task == "re":
            pred, loss = model(batch_data, batch_data["rel_labels"])
        else:
            raise ValueError(self.task)
        return pred, loss

    def run(self, mode):
        model = self.get_model()
        if mode == "test":
            self.evaluator.set_model(model=model, fixed_seq_len=self.fixed_seq_len)
            acc, precision, recall, f1 = self.evaluator.test(data_type="test")
            _test_log = "* model: {} {}, test acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(
                self.task, self.model_name, acc, precision, recall, f1)
            logging.info(_test_log)
            print(_test_log)
            return
        logging.info("{}-{} start train , epoch_nums:{}...".format(self.task, self.model_name, Config.max_epoch_nums))
        for epoch_num in trange(1, Config.max_epoch_nums + 1,
                                desc="{} {} train epoch num".format(self.task, self.model_name)):
            model.train()
            for batch_data in self.data_helper.batch_iter(self.task, data_type="train", batch_size=Config.batch_size,
                                                          re_type="torch", fixed_seq_len=self.fixed_seq_len):
                try:
                    if self.task == "ner":
                        pred, loss = model(input_ids=batch_data["sents"], attention_mask=batch_data["sents"].gt(0),
                                           labels=batch_data["ent_tags"])
                        acc, precision, recall, f1 = self.evaluator.evaluate_ner(
                            batch_y_ent_ids=batch_data["ent_tags"].tolist(), batch_pred_ent_ids=pred.tolist())
                    else:  # self.task == "re":
                        pred, loss = model(batch_data, batch_data["rel_labels"])
                        acc, precision, recall, f1 = self.evaluator.get_re_metrics(
                            y_true=batch_data["rel_labels"].tolist(), y_pred=pred.tolist())
                except Exception as e:
                    logging.error(e)
                    import ipdb,traceback
                    traceback.print_exc()
                    ipdb.set_trace()
                    continue
                self.backfoward(loss, model)
                self.global_step += 1
                self.scheduler.step(epoch=epoch_num)  # 更新学习率
                logging.info("train {} {} epoch_num: {}, global_step:{} loss: {:.3f}, "
                             "acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(
                    self.task, self.model_name, epoch_num, self.global_step, loss.item(), acc, precision, recall, f1))
                # if self.global_step % Config.check_step == 0:
                #     logging.info("* global_step:{} loss: {:.3f}".format(self.global_step, loss.item()))
                # print("* global_step:{} loss: {:.3f}".format(self.global_step, loss.item()))
                # self.save_best_loss_model(loss)
            acc, precision, recall, f1 = self.evaluate_save(model)
            logging.info("valid {} {} epoch_num: {}, acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(
                self.task, self.model_name, epoch_num, acc, precision, recall, f1))
            logging.info("epoch_num: {} end .\n".format(epoch_num))
            # Early stopping and logging best f1
            if self.patience_counter >= Config.patience_num and epoch_num > Config.min_epoch_nums:
                logging.info("{}, Best val f1: {:.3f} best loss:{:.3f}".format(
                    self.model_name, self.best_val_f1, self.best_loss))
                break
