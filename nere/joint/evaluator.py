import torch
from pytorch_pretrained_bert import BertTokenizer

from nere.ner.config import Config
from nere.ner.torchs.evaluator import Evaluator as NerEvaluator
from nere.re.data_helper import DataHelper
from nere.re.torchs.evaluator import Evaluator as ReEvaluator


class Predictor(object):
    def __init__(self, model=None, model_name=None):
        self.model_name = model_name
        self.data_helper = DataHelper()
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)

    def load_model(self):
        if self.model:
            self.model.eval()  # set model to evaluation mode  declaring to the system that we're only doing 'forward' calculations


class Evaluator(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ner_evaluator = NerEvaluator()
        self.re_evaluator = ReEvaluator()

    def test(self):
        ner_pred_tags, ner_true_tags = [], []
        re_pred_tags, re_true_tags = [], []
        for batch_data in self.data_helper.batch_iter(data_type="val",
                                                      batch_size=Config.batch_size,
                                                      epoch_nums=1):
            batch_data["ent_labels"] = torch.tensor(batch_data["ent_labels"], dtype=torch.long).to(Config.device)
            batch_data["e1_masks"] = torch.tensor(batch_data["e1_masks"], dtype=torch.long).to(Config.device)
            batch_data["e2_masks"] = torch.tensor(batch_data["e2_masks"], dtype=torch.long).to(Config.device)
            batch_data["sents"] = torch.tensor(batch_data["sents"], dtype=torch.long).to(Config.device)
            batch_data["fake_rel_labels"] = torch.tensor(batch_data["fake_rel_labels"], dtype=torch.long).to(
                Config.device)
            batch_data["sents_tags"] = torch.tensor(batch_data["sents_tags"], dtype=torch.long).to(Config.device)
            batch_data["rel_labels"] = torch.tensor(batch_data["rel_labels"], dtype=torch.long).to(Config.device)
            with torch.no_grad():  # 适用于测试阶段，不需要反向传播
                ner_logits, re_logits = self.model(batch_data, is_train=False)  # shape: (batch_size, seq_length)
            ner_pred_tags.extend(ner_logits.tolist())
            ner_true_tags.extend(batch_data["sents_tags"].tolist())
            re_pred_tags.extend(re_logits.tolist())
            re_true_tags.extend(batch_data["rel_labels"].tolist())
        assert len(ner_pred_tags) == len(ner_true_tags) == len(re_pred_tags) == len(re_true_tags)
        metrics = {"NER": [], "RE": []}
        acc, precision, recall, f1 = self.ner_evaluator.evaluate(ner_true_tags, ner_pred_tags)
        metrics["NER"] = acc, precision, recall, f1
        acc, precision, recall, f1 = self.re_evaluator.metrics.get_metrics_1d(re_true_tags, re_pred_tags)
        metrics["RE"] = acc, precision, recall, f1
        return metrics
