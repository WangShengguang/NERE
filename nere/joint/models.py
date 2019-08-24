"""BERT + Softmax for named entity recognition"""

import torch
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

from nere.joint.config import Config
from nere.ner.torch_models.models import BERTSoftmax as NerBERTSoftmax, BERTCRF as NerBERTCRF
from nere.re.torch_models.models import BERTSoftmax as ReBERTSoftmax, BERTMultitask as ReBERTMultitask


class JointNerRe(BertPreTrainedModel):

    def __init__(self, config, ner_model, re_model, num_ner_labels, num_re_labels):
        super(JointNerRe, self).__init__(config)
        if ner_model == "BERTSoftmax":
            self.ner = NerBERTSoftmax(config, num_ner_labels)
        elif ner_model == "BERTCRF":
            self.ner = NerBERTCRF(config, num_ner_labels)
        if re_model == "BERTSoftmax":
            self.re = ReBERTSoftmax(config, num_re_labels)
        elif re_model == "BERTMultitask":
            self.re = ReBERTMultitask(config, num_re_labels)

    def forward(self, batch_data, is_train=True):
        if is_train:
            ner_loss = self.ner(batch_data["sents"], labels=batch_data["sents_tags"])
            re_loss = self.re(batch_data, batch_data["rel_labels"])
            return ner_loss + re_loss
        else:
            ner_logits = self.ner(batch_data["sents"])
            re_logits = self.re(batch_data)
            return ner_logits, re_logits
