import gc
import logging
import os
from pathlib import Path

import torch
from tqdm import trange

from config import Config
from nere.data_helper import DataHelper
from nere.torch_trainer import Trainer


class JoinTrainer(Trainer):
    def __init__(self, load_mode, ner_model, re_model,
                 fix_loss_rate=False,
                 ner_loss_rate=0.15, re_loss_rate=0.8, transe_rate=0.05):
        self.fixed_rate = fix_loss_rate
        if fix_loss_rate:
            self.model_name = "joint_{:.5}{}_{:.5}{}_{:.5}TransE".format(
                ner_loss_rate, ner_model, re_loss_rate, re_model, transe_rate)
            # join rate
            self.ner_loss_rate = ner_loss_rate
            self.re_loss_rate = re_loss_rate
            self.transe_rate = transe_rate
        else:
            self.model_name = "joint_{}_{}".format(ner_model, re_model)
        super().__init__(model_name=self.model_name, task="joint")
        self.load_mode = load_mode
        self.ner_model = ner_model
        self.re_model = re_model
        self.ner_path = os.path.join(self.model_dir, self.model_name, "{}_ner.bin".format(self.model_name))
        self.re_path = os.path.join(self.model_dir, self.model_name, "{}_re.bin".format(self.model_name))
        self.joint_path = os.path.join(self.model_dir, self.model_name, self.model_name + ".bin")
        os.makedirs(os.path.dirname(self.joint_path), exist_ok=True)
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
        if self.load_mode == "join":
            if Config.load_pretrain and Path(self.joint_path).is_file():
                model.load_state_dict(torch.load(self.joint_path))  # 断点续训
                logging.info("load model from {}".format(self.joint_path))
        # 分别load
        else:  # respective
            if Config.load_pretrain and Path(self.ner_path).is_file():
                model.ner.load_state_dict(torch.load(self.ner_path))  # 断点续训
                model.re.load_state_dict(torch.load(self.re_path))  # 断点续训
                logging.info("load model from ner_path:{}, re_path:{}".format(self.ner_path, self.re_path))
        model = self.init_model(model)
        return model

    def evaluate_save(self, model):
        self.evaluator.set_model(model=model)
        metrics = self.evaluator.test(data_type="valid")
        logging.info("*model:{} valid, NER acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
            self.model_name, *metrics["NER"]))
        logging.info("*model:{} valid,  RE acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
            self.model_name, *metrics["RE"]))
        ner_f1 = metrics["NER"][-1]
        re_f1 = metrics["RE"][-1]
        ave_f1 = (ner_f1 + re_f1) / 2
        if ave_f1 > self.best_val_f1_dict["Joint"]["Joint"]:
            # model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model.state_dict(), self.joint_path)  # Only save the model it-self
            logging.info("** - Found new best NER&RE F1,ave_f1:{:.4f},ner_f1:{:.4f},re_f1:{:.4f}"
                         " ,save to model_path: {}".format(ave_f1, ner_f1, re_f1, self.joint_path))
            # if ave_f1 - self.best_val_f1_dict["Joint"]["Joint"] < Config.patience:
            #     self.patience_counter += 1
            # else:
            #     self.patience_counter = 0
            self.best_val_f1_dict["Joint"] = {"Joint": ave_f1, "NER": ner_f1, "RE": re_f1}
        else:
            self.patience_counter += 1
        if ner_f1 > self.best_val_f1_dict["NER"]:
            self.best_val_f1_dict["NER"] = ner_f1
            torch.save(model.ner.state_dict(), self.ner_path)  # Only save the model it-self
            logging.info("** - Found new best NER F1: {:.4f} ,save to model_path: {}".format(
                ner_f1, self.ner_path))
        if re_f1 > self.best_val_f1_dict["RE"]:
            self.best_val_f1_dict["RE"] = re_f1
            torch.save(model.re.state_dict(), self.re_path)  # Only save the model it-self
            logging.info("** - Found new best RE F1:{:.4f} ,save to model_path: {}".format(
                re_f1, self.re_path))

    def run(self, mode):
        _log_str = "* {} {} start ...".format(self.model_name, mode)
        logging.info(_log_str)
        print(_log_str)
        model = self.get_model()
        if mode == "test":
            self.evaluator.set_model(model=model)
            metrics = self.evaluator.test(data_type="test")
            _ner_log = "*{} test NER acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                self.model_name, *metrics["NER"])
            _re_log = "*{} test RE acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                self.model_name, *metrics["RE"])
            logging.info(_ner_log)
            logging.info(_re_log)
            print(_ner_log)
            print(_re_log)
            return
        for epoch_num in trange(1, Config.max_epoch_nums + 2,
                                desc="{} {} train epoch num".format(self.task, self.model_name)):
            model.train()
            for batch_data in self.data_helper.batch_iter(self.task, data_type="train",
                                                          batch_size=Config.batch_size // 2,  # TODO CUDA out of memory.
                                                          re_type="torch"):
                try:
                    ((ner_pred, re_pred),
                     (joint_loss, ner_loss, re_loss, transe_loss),
                     (ner_loss_rate, re_loss_rate, trane_loss_rate)) = model(batch_data, is_train=True)
                except Exception as e:
                    logging.error(e)
                    gc.collect()
                    continue
                logging.info("\n---------------------------------------------------------")
                if self.fixed_rate:
                    loss = self.ner_loss_rate * ner_loss + self.re_loss_rate * re_loss + self.transe_rate * transe_loss
                else:
                    logging.info("not fixed rate, model.ner_loss_rate: {:.4f}, model.re_loss_rate: {:.4f}, "
                                 "model.transe_loss_rate: {:.4f}".format(
                        ner_loss_rate.item(), re_loss_rate.item(), trane_loss_rate.item()))
                    loss = joint_loss
                self.backfoward(loss, model)
                self.global_step += 1
                self.scheduler.step(epoch=epoch_num)  # 更新学习率
                # log
                acc, precision, recall, f1 = self.evaluator.evaluate_ner(
                    batch_y_ent_ids=batch_data["ent_tags"].tolist(), batch_pred_ent_ids=ner_pred.tolist())
                logging.info("{} joint train NER global_step:{} loss: {:.4f}, "
                             "acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                    self.model_name, self.global_step, ner_loss.item(), acc, precision, recall, f1))
                acc, precision, recall, f1 = self.evaluator.get_re_metrics(
                    y_true=batch_data["rel_labels"].tolist(), y_pred=re_pred.tolist())
                logging.info("{} joint train RE global_step:{} loss: {:.4f}, "
                             "acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(
                    self.model_name, self.global_step, re_loss.item(), acc, precision, recall, f1))
                logging.info("* joint global_step:{}, ner_loss: {:.4f}, re_loss: {:.4f}, transe_loss: {:.4f}，"
                             "joint_loss: {:.4f}".format(
                    self.global_step, ner_loss.item(), re_loss.item(), transe_loss.item(), loss.item()))
                # if self.global_step % 10 == 0:  # Config.check_step == 0:
                #     logging.info(_log_str)
                # print(_log_str)
                # self.save_best_loss_model(loss)
            # self.save_best_loss_model(loss)
            self.evaluate_save(model)
            logging.info("epoch_num: {} end .".format(epoch_num))
            # Early stopping and logging best f1
            if self.patience_counter >= Config.patience_num and epoch_num > Config.min_epoch_nums:
                logging.info("{}, Best val f1: {:.4f} best loss:{:.4f}".format(
                    self.model_name, self.best_val_f1, self.best_loss))
                break
