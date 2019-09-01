import logging
import os
import traceback
from pathlib import Path

import torch
from tqdm import trange

from nere.config import Config
from nere.data_helper import DataHelper
from nere.evaluator import Evaluator
from nere.model_utils.torch_utils import Trainer as BaseTrainer


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
        self.evaluator = Evaluator(framework="torch", data_type="val")
        self.best_val_f1 = 0
        self.patience_counter = Config.patience_num
        #
        self.data_helper = DataHelper()

    def get_re_model(self):
        from nere.re.torch_models import BERTMultitask, BERTSoftmax, BiLSTM_ATT
        vocab_size = len(self.data_helper.tokenizer.vocab)
        num_ent_tags = len(self.data_helper.ent_tag2id)
        num_rel_tags = len(self.data_helper.rel_label2id)
        if self.model_name == 'BERTSoftmax':
            model = BERTSoftmax.from_pretrained(Config.bert_pretrained_dir, num_labels=num_rel_tags)
        elif self.model_name == 'BERTMultitask':
            model = BERTMultitask.from_pretrained(Config.bert_pretrained_dir, num_labels=num_rel_tags)
        elif self.model_name == "bilstm_att":
            model = BiLSTM_ATT(vocab_size, num_ent_tags, num_rel_tags)
        else:
            raise ValueError("Unknown model, must be one of 'BERTSoftmax'/'BERTMultitask'")
        return model

    def get_ner_model(self):
        from nere.ner.torch_models import BERTCRF, BERTSoftmax
        num_tags = len(self.data_helper.ent_tag2id)
        if self.model_name == 'BERTCRF':
            model = BERTCRF.from_pretrained(Config.bert_pretrained_dir, num_labels=num_tags)
        elif self.model_name == 'BERTSoftmax':
            model = BERTSoftmax.from_pretrained(Config.bert_pretrained_dir, num_labels=num_tags)
        else:
            raise ValueError("Unknown model, must be one of 'BERTSoftmax'/'BERTMultitask'")
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

    def evaluate(self):
        # with torch.no_grad():  # 适用于测试阶段，不需要反向传播
        acc, precision, recall, f1 = self.evaluator.test(model=self.model, task=self.task)
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
            acc, precision, recall, f1 = Evaluator(framework="torch", data_type="test").test(model=self.model,
                                                                                             task=self.task)
            _test_log = "* test acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                acc, precision, recall, f1)
            logging.info(_test_log)
            print(_test_log)
            return
        logging.info("{}-{} start train , epoch_nums:{}...".format(self.task, self.model_name, Config.max_epoch_nums))
        train_data = self.data_helper.get_joint_data(data_type="train")
        for epoch_num in trange(Config.max_epoch_nums, desc="train epoch num"):
            for batch_data in self.data_helper.batch_iter(train_data, batch_size=Config.batch_size, re_type="torch"):
                loss = self.train_step(batch_data)
                logging.info("* global_step:{} loss: {:.4f}".format(self.global_step, loss.item()))
                self.backfoward(loss)
                self.global_step += 1
                self.scheduler.step(epoch=epoch_num)  # 更新学习率
                if self.global_step % Config.save_step == 0:
                    self.evaluate()
            self.evaluate()
            logging.info("epoch_num: {} end .".format(epoch_num))
            # Early stopping and logging best f1
            if (self.patience_counter >= Config.patience_num and epoch_num > Config.min_epoch_nums) \
                    or epoch_num == Config.max_epoch_nums:
                logging.info("{}-{}, Best val f1: {:05.2f}".format(self.task, self.model_name, self.best_val_f1))
                break


from nere.joint_models import nnJointNerRe


class JoinTrainer(Trainer):
    def __init__(self, task, ner_model, re_model, mode="train"):
        super().__init__(model_name="joint" + ner_model + re_model, task=task)
        self.ner_model = ner_model
        self.re_model = re_model
        self.mode = mode
        self.ner_path = os.path.join(self.model_dir, "joint_ner_{}.bin".format(self.ner_model))
        self.re_path = os.path.join(self.model_dir, "joint_re_{}.bin".format(self.re_model))
        self.joint_path = os.path.join(self.model_dir, "joint{}{}.bin".format(self.ner_model, self.re_model))
        self.best_val_f1_dict = {"NER": 0, "RE": 0, "Joint": {"Joint": 0, "NER": 0, "RE": 0}}

    def get_model(self):
        self.data_helper = DataHelper()
        num_ner_tags = len(self.data_helper.ent_tag2id)
        num_re_tags = len(self.data_helper.rel_label2id)
        # model = JointNerRe.from_pretrained(Config.bert_pretrained_dir,
        #                                    ner_model=self.ner_model, re_model=self.re_model,
        #                                    num_ner_labels=num_ner_tags, num_re_labels=num_re_tags)
        model = nnJointNerRe(ner_model=self.ner_model, re_model=self.re_model,
                             num_ner_labels=num_ner_tags, num_re_labels=num_re_tags)
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

    def evaluate(self):
        metrics = self.evaluator.test(model=self.model, task=self.task)
        logging.info("* NER acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(*metrics["NER"]))
        logging.info("* RE acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(*metrics["RE"]))
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
        self.model = self.get_model()
        if self.mode == "test":
            metrics = Evaluator(framework="torch", data_type="test").test(model=self.model, task=self.task)
            _ner_log = "* NER acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(*metrics["NER"])
            _re_log = "* RE acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(*metrics["RE"])
            logging.info(_ner_log)
            logging.info(_re_log)
            print(_ner_log)
            print(_re_log)
            return
        train_data = self.data_helper.get_joint_data(data_type="train")
        for epoch_num in trange(Config.max_epoch_nums, desc="train epoch num"):
            for batch_data in self.data_helper.batch_iter(train_data, batch_size=Config.batch_size, re_type="torch"):
                try:
                    loss = self.train_step(batch_data)
                except:
                    logging.error(traceback.format_exc())
                    continue
                logging.info("* global_step:{} loss: {:.4f}".format(self.global_step, loss.item()))
                self.backfoward(loss)
                self.global_step += 1
                self.scheduler.step(epoch=epoch_num)  # 更新学习率
                if self.global_step % Config.save_step == 0:
                    self.evaluate()
            self.evaluate()
            logging.info("epoch_num: {} end .".format(epoch_num))
            # Early stopping and logging best f1
            if (self.patience_counter >= Config.patience_num and epoch_num > Config.min_epoch_nums) \
                    or epoch_num == Config.max_epoch_nums:
                logging.info("{}-{}, Best val f1: {}".format(self.task, self.model_name, self.best_val_f1_dict))
                break
