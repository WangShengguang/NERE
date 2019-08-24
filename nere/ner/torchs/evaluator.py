import torch
from pytorch_pretrained_bert import BertTokenizer
from sklearn.metrics import accuracy_score

from nere.metrics import MutilabelMetrics
from nere.ner.config import Config
from nere.ner.data_helper import DataHelper, get_entities


class Predictor(object):
    def __init__(self, model=None, model_name=None):
        self.model_name = model_name
        self.data_helper = DataHelper()
        self.model = model  # if model else self.load_model()
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)

    def load_model(self):
        if self.model:
            self.model.eval()  # set model to evaluation mode  declaring to the system that we're only doing 'forward' calculations

    def cellect_entities(self, tags):
        """
        :param tags:  # shape: (batch_size, seq_length)
        :return:  (batch_size,entity_nums)
        """
        entities_indices = [get_entities(line_tags) for line_tags in tags]
        entities = [[entity for entity, start, end in line_entities] for line_entities in entities_indices]
        return entities


class Evaluator(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = MutilabelMetrics(list(self.data_helper.entity2id.keys()))

    def test(self):
        pred_tags = []
        true_tags = []
        for x_batch, y_batch in self.data_helper.batch_iter(data_type="val",
                                                            batch_size=Config.batch_size,
                                                            epoch_nums=1):
            x_batch = torch.tensor(x_batch, dtype=torch.long).to(Config.device)
            y_batch = torch.tensor(y_batch, dtype=torch.long).to(Config.device)
            _pred_ids = self.model(input_ids=x_batch, token_type_ids=None, attention_mask=x_batch.gt(0))
            # shape: (batch_size, seq_length)
            pred_tags.extend(_pred_ids.tolist())
            true_tags.extend(y_batch.tolist())
        assert len(pred_tags) == len(true_tags)
        acc, precision, recall, f1 = self.evaluate(true_tags, pred_tags)
        return acc, precision, recall, f1

    def evaluate(self, batch_y_ids, batch_pred_ids):
        _pred_tags = [[self.data_helper.id2tag.get(tag_id, "O") for tag_id in line_tags]
                      for line_tags in batch_pred_ids]
        _true_tags = [[self.data_helper.id2tag.get(tag_id, "O") for tag_id in line_tags]
                      for line_tags in batch_y_ids]
        acc = accuracy_score(sum(_true_tags, []), sum(_pred_tags, []))
        true_entities = self.cellect_entities(_true_tags)
        pred_entities = self.cellect_entities(_pred_tags)
        precision, recall, f1 = self.metrics.get_metrics(true_entities, pred_entities)
        return acc, precision, recall, f1
