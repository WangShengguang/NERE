import torch
from pytorch_pretrained_bert import BertTokenizer

from nere.metrics import MutilabelMetrics
from nere.re.config import Config
from nere.re.data_helper import DataHelper


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
        self.metrics = MutilabelMetrics(list(self.data_helper.rel_label2id.keys()))

    def test(self):
        pred_tags = []
        true_tags = []
        for batch_data in self.data_helper.batch_iter(data_type="val",
                                                      batch_size=Config.batch_size,
                                                      epoch_nums=1):
            batch_data["ent_labels"] = torch.tensor(batch_data["ent_labels"], dtype=torch.long).to(Config.device)
            batch_data["e1_masks"] = torch.tensor(batch_data["e1_masks"], dtype=torch.long).to(Config.device)
            batch_data["e2_masks"] = torch.tensor(batch_data["e2_masks"], dtype=torch.long).to(Config.device)
            batch_data["sents"] = torch.tensor(batch_data["sents"], dtype=torch.long).to(Config.device)
            batch_data["fake_rel_labels"] = torch.tensor(batch_data["fake_rel_labels"], dtype=torch.long).to(
                Config.device)
            batch_data["rel_labels"] = torch.tensor(batch_data["rel_labels"], dtype=torch.long).to(Config.device)
            batch_pred_ids = self.model(batch_data)  # shape: (batch_size, 1)
            pred_tags.extend(batch_pred_ids.tolist())
            true_tags.extend(batch_data["rel_labels"].tolist())
        assert len(pred_tags) == len(true_tags)
        acc, precision, recall, f1 = self.metrics.get_metrics_1d(true_tags, pred_tags)
        return acc, precision, recall, f1