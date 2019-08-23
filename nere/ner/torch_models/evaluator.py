import torch
from pytorch_pretrained_bert import BertTokenizer

from nere.metrics import MutilabelMetrics
from nere.ner.config import Config
from nere.ner.data_helper import DataHelper, get_entities
from nere.ner.torch_models.models import BERTCRF, BERTSoftmax
from nere.torch_utils import Saver


def cellect_entities(tags):
    """
    :param tags:  # shape: (batch_size, seq_length)
    :return:  (batch_size,entity_nums)
    """
    entities_indices = [get_entities(line_tags) for line_tags in tags]
    entities = [[entity for entity, start, end in line_entities] for line_entities in entities_indices]
    return entities


class Predictor(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.data_helper = DataHelper()
        self.model = self.load_model()
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)

    def load_model(self):
        saver = Saver(model_name=self.model_name, mode=Config.save_mode)
        if self.model_name == 'BERTSoftmax':
            model = BERTSoftmax.from_pretrained(Config.bert_pretrained_dir, num_labels=len(self.data_helper.tag2id))
        elif self.model_name == 'BERTCRF':
            model = BERTCRF.from_pretrained(Config.bert_pretrained_dir, num_labels=len(self.data_helper.tag2id))
        else:
            raise ValueError("Unknown model, must be one of 'BERTSoftmax'/'BERTCRF'")
        model = saver.load(model)
        return model


class Evaluator(Predictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = MutilabelMetrics(list(self.data_helper.entity2id.keys()))

    def test(self):
        pred_tags = []
        true_tags = []
        for x_batch, y_batch in self.data_helper.batch_iter(data_type="test",
                                                            batch_size=Config.batch_size,
                                                            epoch_nums=Config.epoch_nums):
            x_batch = torch.tensor(x_batch).to(torch.int64)
            _pred_ids = self.model(input_ids=x_batch, token_type_ids=None, attention_mask=x_batch.gt(0))
            _pred_tags = [[self.data_helper.id2tag[tag_id] for tag_id in line_tags] for line_tags in _pred_ids]
            _true_tags = [[self.data_helper.id2tag[tag_id] for tag_id in line_tags] for line_tags in y_batch]
            # shape: (batch_size, seq_length)
            pred_tags.extend(_pred_tags)
            true_tags.extend(y_batch)
        assert len(pred_tags) == len(true_tags)
        true_entities = cellect_entities(true_tags)
        pred_entities = cellect_entities(pred_tags)
        acc, precision, recall, f1 = self.metrics.get_metrics(true_entities, pred_entities)
        print("acc: {}, precision: {}, recall: {}, f1: {}".format(acc, precision, recall, f1))

    def evaluate(self, batch_y_ids, batch_pred_ids):
        _pred_tags = [[self.data_helper.id2tag[tag_id] for tag_id in line_tags] for line_tags in batch_pred_ids]
        _true_tags = [[self.data_helper.id2tag[tag_id] for tag_id in line_tags] for line_tags in batch_y_ids]
        true_entities = cellect_entities(_true_tags)
        pred_entities = cellect_entities(_pred_tags)
        acc, precision, recall, f1 = self.metrics.get_metrics(true_entities, pred_entities)
        return acc, precision, recall, f1
