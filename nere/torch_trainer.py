import gc
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

    def backfoward(self, loss):
        if Config.gpu_nums > 1 and Config.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        # compute gradients of all variables wrt loss
        loss.backward(retain_graph=True)

        if self.global_step % Config.gradient_accumulation_steps == 0:
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=Config.clip_grad)
            # performs updates using calculated gradients
            self.optimizer.step()
            # clear previous gradients
            self.optimizer.zero_grad()
        return loss


class Trainer(BaseTrainer):
    def __init__(self, model_name, task, mode="train"):
        super().__init__()
        self.model_name = model_name
        self.task = task
        self.mode = mode  # train evaluate
        self.model_dir = os.path.join(Config.torch_ckpt_dir, task)
        self.model_path = os.path.join(self.model_dir, model_name + ".bin")
        os.makedirs(self.model_dir, exist_ok=True)
        # evaluate
        self.evaluator = Evaluator(task, model_name, framework="torch", load_model=False)
        self.best_val_f1 = 0
        self.best_loss = 100000
        self.patience_counter = Config.patience_num
        #
        self.data_helper = DataHelper()

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
            model = BiLSTM_ATT(vocab_size, num_ent_tags, num_rel_tags)
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
            model = BiLSTM(vocab_size, num_ent_tags)
        elif self.model_name == "BiLSTM_ATT":
            model = BiLSTM_ATT(vocab_size, num_ent_tags)
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
        self.init_model(model)
        return model

    def save_best_loss_model(self, loss):
        if loss <= self.best_loss:
            torch.save(self.model.state_dict(), self.model_path)
            self.best_loss = loss
            _log = "loss: {:.4f}, save to :{}".format(loss, self.model_path)
            logging.info(_log)

    def evaluate_save(self):
        # with torch.no_grad():  # 适用于测试阶段，不需要反向传播
        self.evaluator.set_model(model=self.model)
        acc, precision, recall, f1 = self.evaluator.test(data_type="val")
        logging.info("acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
            acc, precision, recall, f1))
        if f1 > self.best_val_f1:
            torch.save(self.model.state_dict(), self.model_path)
            logging.info("** - Found new best F1 ,save to model_path: {}".format(self.model_path))
            if f1 - self.best_val_f1 < Config.patience:
                self.patience_counter += 1
            else:
                self.patience_counter = 0
            self.best_val_f1 = f1
        else:
            self.patience_counter += 1

    def train_step(self, batch_data):
        if self.task == "ner":
            loss = self.model(input_ids=batch_data["sents"], attention_mask=batch_data["sents"].gt(0),
                              labels=batch_data["ent_tags"])
        elif self.task == "re":
            loss = self.model(batch_data, batch_data["rel_labels"])
        else:
            raise ValueError(self.task)
        return loss

    def run(self):
        self.model = self.get_model()
        if self.mode == "test":
            self.evaluator.set_model(model=self.model)
            acc, precision, recall, f1 = self.evaluator.test(data_type="test")
            _test_log = "* model: {} {}, test acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                self.task, self.model_name, acc, precision, recall, f1)
            logging.info(_test_log)
            print(_test_log)
            return
        logging.info("{}-{} start train , epoch_nums:{}...".format(self.task, self.model_name, Config.max_epoch_nums))
        for epoch_num in trange(1, Config.max_epoch_nums + 1,
                                desc="{} {} train epoch num".format(self.task, self.model_name)):
            self.model.train()
            for batch_data in self.data_helper.batch_iter(self.task, data_type="train", batch_size=Config.batch_size,
                                                          re_type="torch"):
                loss = self.train_step(batch_data)
                self.backfoward(loss)
                self.global_step += 1
                self.scheduler.step(epoch=epoch_num)  # 更新学习率
                if self.global_step % Config.check_step == 0:
                    logging.info("* global_step:{} loss: {:.4f}".format(self.global_step, loss.item()))
                    # print("* global_step:{} loss: {:.4f}".format(self.global_step, loss.item()))
                    # self.save_best_loss_model(loss)
            self.evaluate_save()
            logging.info("epoch_num: {} end .".format(epoch_num))
            # Early stopping and logging best f1
            if self.patience_counter >= Config.patience_num and epoch_num > Config.min_epoch_nums:
                logging.info("{}, Best val f1: {:.4f} best loss:{:.4f}".format(self.model_name, self.best_val_f1,
                                                                               self.best_loss))
                break


class JoinTrainer(Trainer):
    def __init__(self, task, ner_model, re_model, mode="train",
                 ner_loss_rate=0.15, re_loss_rate=0.8, transe_rate=0.05):
        self.model_name = "joint_{:.5}{}_{:.5}{}_{:.5}TransE".format(
            ner_loss_rate, ner_model, re_loss_rate, re_model, transe_rate)
        self.model_name = "joint_{}_{}".format(ner_model, re_model)
        super().__init__(model_name=self.model_name, task=task)
        self.ner_model = ner_model
        self.re_model = re_model
        self.mode = mode
        # join rate
        self.ner_loss_rate = ner_loss_rate
        self.re_loss_rate = re_loss_rate
        self.transe_rate = transe_rate
        self.ner_path = os.path.join(self.model_dir, "{}_ner.bin".format(self.model_name))
        self.re_path = os.path.join(self.model_dir, "{}_re.bin".format(self.model_name))
        self.joint_path = os.path.join(self.model_dir, self.model_name + ".bin")
        self.best_val_f1_dict = {"NER": 0, "RE": 0, "Joint": {"Joint": 0, "NER": 0, "RE": 0}}

    def get_model(self):
        from nere.joint_models import JointNerRe
        self.data_helper = DataHelper()
        num_ner_tags = len(self.data_helper.ent_tag2id)
        num_re_tags = len(self.data_helper.rel_label2id)
        # model = JointNerRe(ner_model=self.ner_model, re_model=self.re_model,
        #                      num_ner_labels=num_ner_tags, num_re_labels=num_re_tags,
        #                      ner_loss_rate=0.1, re_loss_rate=0.8, transe_rate=0.1)
        model = JointNerRe(num_ner_labels=num_ner_tags, num_re_labels=num_re_tags)
        if self.task == "joint":
            if Config.load_pretrain and Path(self.joint_path).is_file():
                model.load_state_dict(torch.load(self.joint_path))  # 断点续训
                logging.info("load model from {}".format(self.joint_path))
        # 分别load
        else:  # respective
            if Config.load_pretrain and Path(self.ner_path).is_file():
                model.ner.load_state_dict(torch.load(self.ner_path))  # 断点续训
                model.re.load_state_dict(torch.load(self.re_path))  # 断点续训
                logging.info("load model from ner_path:{}, re_path:{}".format(self.ner_path, self.re_path))
        self.init_model(model)
        return model

    def evaluate_save(self):
        self.evaluator.set_model(model=self.model)
        metrics = self.evaluator.test(data_type="val")
        logging.info("*model:{} valid, NER acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
            self.model_name, *metrics["NER"]))
        logging.info("*model:{} valid,  RE acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
            self.model_name, *metrics["RE"]))
        ner_f1 = metrics["NER"][-1]
        re_f1 = metrics["RE"][-1]
        ave_f1 = (ner_f1 + re_f1) / 2
        if ave_f1 > self.best_val_f1_dict["Joint"]["Joint"]:
            # model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(self.model.state_dict(), self.joint_path)  # Only save the model it-self
            logging.info("** - Found new best NER&RE F1,ave_f1:{:.4f},ner_f1:{:.4f},re_f1:{:.4f}"
                         " ,save to model_path: {}".format(ave_f1, ner_f1, re_f1, self.joint_path))
            if ave_f1 - self.best_val_f1_dict["Joint"]["Joint"] < Config.patience:
                self.patience_counter += 1
            else:
                self.patience_counter = 0
            self.best_val_f1_dict["Joint"] = {"Joint": ave_f1, "NER": ner_f1, "RE": re_f1}
        else:
            self.patience_counter += 1
        if ner_f1 > self.best_val_f1_dict["NER"]:
            self.best_val_f1_dict["NER"] = ner_f1
            torch.save(self.model.ner.state_dict(), self.ner_path)  # Only save the model it-self
            logging.info("** - Found new best NER F1: {:.4f} ,save to model_path: {}".format(
                ner_f1, self.ner_path))
        if re_f1 > self.best_val_f1_dict["RE"]:
            self.best_val_f1_dict["RE"] = re_f1
            torch.save(self.model.re.state_dict(), self.re_path)  # Only save the model it-self
            logging.info("** - Found new best RE F1:{:.4f} ,save to model_path: {}".format(
                re_f1, self.re_path))

    def train_step(self, batch_data):
        loss = self.model(batch_data, is_train=True)
        return loss

    def run(self):
        _log_str = "* {} {} start ...".format(self.model_name, self.mode)
        logging.info(_log_str)
        print(_log_str)
        self.model = self.get_model()
        if self.mode == "test":
            self.evaluator.set_model(model=self.model)
            metrics = self.evaluator.test(data_type="test")
            _ner_log = "*{} test NER acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(self.model_name,
                                                                                                        *metrics["NER"])
            _re_log = "*{} test RE acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(self.model_name,
                                                                                                      *metrics["RE"])
            logging.info(_ner_log)
            logging.info(_re_log)
            print(_ner_log)
            print(_re_log)
            return
        for epoch_num in trange(1, Config.max_epoch_nums + 2,
                                desc="{} {} train epoch num".format(self.task, self.model_name)):
            self.model.train()
            for batch_data in self.data_helper.batch_iter(self.task, data_type="train",
                                                          batch_size=Config.batch_size, re_type="torch"):
                try:
                    loss = self.train_step(batch_data)
                except Exception as e:
                    # import traceback
                    # traceback.print_exc()
                    logging.error(e)
                    gc.collect()
                    continue
                self.backfoward(loss)
                self.global_step += 1
                self.scheduler.step(epoch=epoch_num)  # 更新学习率
                if self.global_step % Config.check_step == 0:
                    logging.info("* global_step:{} loss: {:.4f}".format(self.global_step, loss.item()))
                    # self.save_best_loss_model(loss)
            # self.save_best_loss_model(loss)
            self.evaluate_save()
            logging.info("epoch_num: {} end .".format(epoch_num))
            # Early stopping and logging best f1
            if self.patience_counter >= Config.patience_num and epoch_num > Config.min_epoch_nums:
                logging.info("{}, Best val f1: {:.4f} best loss:{:.4f}".format(self.model_name, self.best_val_f1,
                                                                               self.best_loss))
                break
