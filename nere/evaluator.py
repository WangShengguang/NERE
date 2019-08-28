import os

import keras
import numpy as np
from sklearn.metrics import accuracy_score

from nere.config import Config
from nere.data_helper import DataHelper, entity_label2tag
from nere.metrics import MutilabelMetrics


class Predictor(object):
    def __init__(self, framework):
        self.framework = framework
        self.model = None
        self.data_helper = DataHelper()

    def load_model(self, task, model_name):
        if self.framework == "keras":
            model_path = os.path.join(Config.keras_ckpt_dir, task, model_name)
            assert os.path.isfile(model_path)
            model = keras.models.load_model(model_path)
        elif self.framework == "tf":
            model = None
        else:
            raise ValueError(self.framework)
        return model

    def predict_ner(self, batch_data):
        if self.framework == "torch":
            batch_pred_ids = self.model(batch_data["sents"])
        elif self.framework == "keras":
            batch_logits = self.model.predict(batch_data["sents"])
            batch_pred_ids = np.argmax(batch_logits, axis=-1)
        elif self.framework == "tf":
            batch_pred_ids = None
        else:
            raise ValueError(self.framework)
        return batch_pred_ids

    def predict_re(self, batch_data):
        if self.framework == "torch":
            batch_pred_ids = self.model(batch_data)
        elif self.framework == "keras":
            batch_pred_ids = self.model.predict()
        elif self.framework == "tf":
            batch_pred_ids = None
        else:
            raise ValueError(self.framework)
        return batch_pred_ids

    def cellect_entities(self, batch_tags):
        """
        :param tags:  # shape: (batch_size, seq_length),
                    [["O","B-NP","I-NP"],["O","B-NP","I-NP"]]
        :return:  (batch_size,entity_nums)
        """
        entities_indices = [get_entities(line_tags) for line_tags in batch_tags]
        entities = [[entity for entity, start, end in line_entities] for line_entities in entities_indices]
        return entities


class Evaluator(Predictor):
    def __init__(self, framework):
        super().__init__(framework)
        self.val_data = self.data_helper.get_joint_data("val")
        # self.ner_metrics = MutilabelMetrics(list(self.data_helper.ent_tag2id.keys()))
        self.ner_metrics = MutilabelMetrics(list(entity_label2tag.values()))
        self.re_metrics = MutilabelMetrics(list(self.data_helper.rel_label2id.keys()))

    def test(self, task, model=None):
        if isinstance(model, str):
            model = self.load_model(task, model)
        self.model = model
        if task == "ner":
            res = self.test_ner()  # acc, precision, recall, f1
        elif task == "re":
            res = self.test_re()
        elif task == "joint":
            res = self.test_joint()
        else:
            raise ValueError(task)
        return res

    def test_joint(self):
        ner_pred_tags, ner_true_tags = [], []
        re_pred_tags, re_true_tags = [], []
        re_type = "torch" if self.framework == "torch" else "numpy"
        for batch_data in self.data_helper.batch_iter(self.val_data, batch_size=Config.batch_size, re_type=re_type):
            # with torch.no_grad():  # 适用于测试阶段，不需要反向传播
            ner_logits, re_logits = self.model(batch_data, is_train=False)  # shape: (batch_size, seq_length)
            ner_pred_tags.extend(ner_logits.tolist())
            ner_true_tags.extend(batch_data["ent_tags"].tolist())
            re_pred_tags.extend(re_logits.tolist())
            re_true_tags.extend(batch_data["rel_labels"].tolist())
        assert len(ner_pred_tags) == len(ner_true_tags) == len(re_pred_tags) == len(re_true_tags)
        metrics = {}
        metrics["NER"] = self.evaluate_ner(ner_true_tags, ner_pred_tags)  # cc, precision, recall, f1
        metrics["RE"] = self.re_metrics.get_metrics_1d(re_true_tags, re_pred_tags)
        return metrics

    def test_ner(self):
        pred_tags = []
        true_tags = []
        re_type = "torch" if self.framework == "torch" else "numpy"
        for batch_data in self.data_helper.batch_iter(self.val_data, batch_size=Config.batch_size, re_type=re_type):
            batch_pred_ids = self.predict_ner(batch_data)  # shape: (batch_size, 1)
            pred_tags.extend(batch_pred_ids.tolist())
            true_tags.extend(batch_data["ent_tags"].tolist())
        assert len(pred_tags) == len(true_tags)
        acc, precision, recall, f1 = self.evaluate_ner(true_tags, pred_tags)
        return acc, precision, recall, f1

    def test_re(self):
        pred_tags = []
        true_tags = []
        re_type = "torch" if self.framework == "torch" else "numpy"
        for batch_data in self.data_helper.batch_iter(self.val_data, batch_size=Config.batch_size, re_type=re_type):
            batch_pred_ids = self.predict_re(batch_data)  # shape: (batch_size, 1)
            pred_tags.extend(batch_pred_ids.tolist())
            true_tags.extend(batch_data["rel_labels"].tolist())
        assert len(pred_tags) == len(true_tags)
        acc, precision, recall, f1 = self.re_metrics.get_metrics_1d(true_tags, pred_tags)
        return acc, precision, recall, f1

    def evaluate_ner(self, batch_y_ent_ids, batch_pred_ent_ids):
        _pred_tags = [[self.data_helper.id2ent_tag.get(tag_id, "O") for tag_id in line_tags]
                      for line_tags in batch_y_ent_ids]
        _true_tags = [[self.data_helper.id2ent_tag.get(tag_id, "O") for tag_id in line_tags]
                      for line_tags in batch_pred_ent_ids]
        acc = accuracy_score(sum(_true_tags, []), sum(_pred_tags, []))
        true_entities = self.cellect_entities(_true_tags)
        pred_entities = self.cellect_entities(_pred_tags)
        precision, recall, f1 = self.ner_metrics.get_metrics(true_entities, pred_entities)
        return acc, precision, recall, f1


def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start
