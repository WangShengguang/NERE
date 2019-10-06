import torch
from torch import nn

from config import Config
from nere.ner_models import BERTSoftmax as NerBERTSoftmax, BERTCRF as NerBERTCRF
from nere.re_models import BERTSoftmax as ReBERTSoftmax, BERTMultitask as ReBERTMultitask


class JointNerRe(nn.Module):

    def __init__(self, num_ner_labels, num_re_labels):
        super(JointNerRe, self).__init__()
        self.ner_loss_rate = nn.Parameter(torch.rand(1))
        self.re_loss_rate = nn.Parameter(torch.rand(1))
        self.trane_loss_rate = nn.Parameter(torch.rand(1))
        # self.linear_joint_loss = nn.Linear(3, 1)
        self.ner = NerBERTCRF.from_pretrained(Config.bert_pretrained_dir, num_ner_labels)
        self.re = ReBERTMultitask.from_pretrained(Config.bert_pretrained_dir, num_re_labels, mode="joint")

    def forward(self, batch_data, is_train=True, mode="joint"):
        """
        :param batch_data:
        :param is_train:
        :param mode: joint,ner,re
        :return:
        """
        if is_train:  # 训练
            if mode == "ner":
                batch_masks = batch_data["sents"].gt(0)
                ner_loss = self.ner(batch_data["sents"], token_type_ids=None,
                                    attention_mask=batch_masks, labels=batch_data["ent_tags"])
                return ner_loss
            elif mode == "re":
                re_loss = self.re(batch_data, labels=batch_data["rel_labels"])
                return re_loss
            else:  # joint
                batch_masks = batch_data["sents"].gt(0)
                ner_loss = self.ner(batch_data["sents"], token_type_ids=None,
                                    attention_mask=batch_masks, labels=batch_data["ent_tags"])
                re_loss, transe_loss = self.re(batch_data, labels=batch_data["rel_labels"])
                joint_loss = self.ner_loss_rate * ner_loss + self.re_loss_rate * re_loss + self.trane_loss_rate * transe_loss
                return joint_loss, ner_loss, re_loss, transe_loss
        else:  # eval
            if mode == "ner":
                ner_logits = self.ner(batch_data["sents"])
                return ner_logits
            elif mode == "re":
                re_logits = self.re(batch_data)
                return re_logits
            else:  # joint
                ner_logits = self.ner(batch_data["sents"])
                re_logits = self.re(batch_data)
                return ner_logits, re_logits


class _JointNerRe(nn.Module):

    def __init__(self, ner_model, re_model, num_ner_labels, num_re_labels,
                 ner_loss_rate, re_loss_rate, transe_rate=1e-5):
        super(_JointNerRe, self).__init__()
        self.ner_loss_rate = ner_loss_rate
        self.re_loss_rate = re_loss_rate
        self.transe_rate = transe_rate / re_loss_rate
        if ner_model == "BERTSoftmax":
            self.ner = NerBERTSoftmax.from_pretrained(Config.bert_pretrained_dir, num_ner_labels)
        elif ner_model == "BERTCRF":
            self.ner = NerBERTCRF.from_pretrained(Config.bert_pretrained_dir, num_ner_labels)
        else:
            raise ValueError(ner_model)
        if re_model == "BERTSoftmax":
            self.re = ReBERTSoftmax.from_pretrained(Config.bert_pretrained_dir, num_re_labels)
        elif re_model == "BERTMultitask":
            self.re = ReBERTMultitask.from_pretrained(Config.bert_pretrained_dir, num_re_labels)
        else:
            raise ValueError(ner_model)

    def forward(self, batch_data, is_train=True, mode="joint"):
        """
        :param batch_data:
        :param is_train:
        :param mode: joint,ner,re
        :return:
        """
        if is_train:
            if mode == "ner":
                batch_masks = batch_data["sents"].gt(0)
                ner_loss = self.ner(batch_data["sents"], token_type_ids=None,
                                    attention_mask=batch_masks, labels=batch_data["ent_tags"])
                return ner_loss
            elif mode == "re":
                re_loss = self.re(batch_data, labels=batch_data["rel_labels"])
                return re_loss
            else:  # joint
                batch_masks = batch_data["sents"].gt(0)
                ner_loss = self.ner(batch_data["sents"], token_type_ids=None,
                                    attention_mask=batch_masks, labels=batch_data["ent_tags"])
                re_loss = self.re(batch_data, labels=batch_data["rel_labels"])
                loss = self.ner_loss_rate * ner_loss + self.re_loss_rate * re_loss
                return loss
        else:  # eval
            if mode == "ner":
                ner_logits = self.ner(batch_data["sents"])
                return ner_logits
            elif mode == "re":
                re_logits = self.re(batch_data)
                return re_logits
            else:  # joint
                ner_logits = self.ner(batch_data["sents"])
                re_logits = self.re(batch_data)
                return ner_logits, re_logits