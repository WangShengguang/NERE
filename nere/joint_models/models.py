from pytorch_pretrained_bert.modeling import BertPreTrainedModel

from nere.ner_models import BERTSoftmax as NerBERTSoftmax, BERTCRF as NerBERTCRF
from nere.re_models import BERTSoftmax as ReBERTSoftmax, BERTMultitask as ReBERTMultitask


class JointNerRe(BertPreTrainedModel):

    def __init__(self, config, ner_model, re_model, num_ner_labels, num_re_labels,
                 ner_loss_rate, re_loss_rate, transe_rate=1e-5):
        super(JointNerRe, self).__init__(config)
        self.ner_loss_rate = ner_loss_rate
        self.re_loss_rate = re_loss_rate
        self.transe_rate = transe_rate / re_loss_rate
        if ner_model == "BERTSoftmax":
            self.ner = NerBERTSoftmax(config, num_ner_labels)
        elif ner_model == "BERTCRF":
            self.ner = NerBERTCRF(config, num_ner_labels)
        else:
            raise ValueError(ner_model)
        if re_model == "BERTSoftmax":
            self.re = ReBERTSoftmax(config, num_re_labels)
        elif re_model == "BERTMultitask":
            self.re = ReBERTMultitask(config, num_re_labels, coefficient=self.transe_rate)
        else:
            raise ValueError(ner_model)

    def forward(self, batch_data, is_train=True):
        if is_train:
            batch_masks = batch_data["sents"].gt(0)
            ner_loss = self.ner(batch_data["sents"], token_type_ids=None,
                                attention_mask=batch_masks, labels=batch_data["ent_tags"])
            re_loss = self.re(batch_data, labels=batch_data["rel_labels"])
            loss = self.ner_loss_rate * ner_loss + self.re_loss_rate * re_loss
            return loss
        else:
            ner_logits = self.ner(batch_data["sents"])
            re_logits = self.re(batch_data)
            return ner_logits, re_logits
