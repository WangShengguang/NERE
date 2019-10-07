import os

import keras
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from config import Config
from nere.data_helper import DataHelper


class Predictor(object):
    def __init__(self, task, model_name, framework, load_model=True):
        self.framework = framework
        self.data_helper = DataHelper()
        if load_model:
            self.model = self.load_model(task, model_name)

    def load_model(self, task, model_name):
        if self.framework == "keras":
            model_path = os.path.join(Config.keras_ckpt_dir, task, model_name)
            assert os.path.isfile(model_path)
            model = keras.models.load_model(model_path)
        elif self.framework == "tf":
            model = None
        elif self.framework == "torch":
            model = None  # torch.load
        else:
            raise ValueError(self.framework)
        if task == "ner" and model_name in ["BiLSTM_ATT"]:
            self.fixed_seq_len = Config.max_sequence_len
        model.eval()  # 切记，否则有偏差
        return model

    def set_model(self, model, fixed_seq_len=None):
        self.model = model
        self.fixed_seq_len = fixed_seq_len
        if self.framework == "torch":
            self.model.eval()

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
    def __init__(self, task, model_name, framework, load_model=True):
        """
        :param framework: torch,tf,keras
        :param task: ner,re,joint
        :param data_type: train,val,test
        """
        super().__init__(task, model_name, framework, load_model=load_model)
        self.task = task

    def test(self, data_type):
        assert self.model is not None, "please set model before test"
        if self.task == "ner":
            res = self.test_ner(data_type)  # acc, precision, recall, f1
        elif self.task == "re":
            res = self.test_re(data_type)
        elif self.task == "joint":
            res = self.test_joint(data_type)
        else:
            raise ValueError(self.task)
        return res

    def test_joint(self, data_type):
        ner_pred_tags, ner_true_tags = [], []
        re_pred_tags, re_true_tags = [], []
        re_type = "torch" if self.framework == "torch" else "numpy"
        for batch_data in self.data_helper.batch_iter(task="ner", data_type=data_type,
                                                      batch_size=Config.test_batch_size,
                                                      re_type=re_type, _shuffle=False,
                                                      fixed_seq_len=self.fixed_seq_len):
            # with torch.no_grad():  # 适用于测试阶段，不需要反向传播
            ner_logits = self.model(batch_data, is_train=False, mode="ner")  # shape: (batch_size, seq_length)
            ner_pred_tags.extend(ner_logits.tolist())
            ner_true_tags.extend(batch_data["ent_tags"].tolist())
        for batch_data in self.data_helper.batch_iter(task="re", data_type=data_type,
                                                      batch_size=Config.test_batch_size,
                                                      re_type=re_type, _shuffle=False,
                                                      fixed_seq_len=self.fixed_seq_len):
            # with torch.no_grad():  # 适用于测试阶段，不需要反向传播
            re_logits = self.model(batch_data, is_train=False, mode="re")  # shape: (batch_size, seq_length)
            re_pred_tags.extend(re_logits.tolist())
            re_true_tags.extend(batch_data["rel_labels"].tolist())
        assert len(ner_pred_tags) == len(ner_true_tags) and len(re_pred_tags) == len(re_true_tags)
        metrics = {}
        metrics["NER"] = self.evaluate_ner(ner_true_tags, ner_pred_tags)  # acc, precision, recall, f1
        metrics["RE"] = self.get_re_metrics(re_true_tags, re_pred_tags, average="macro")
        return metrics

    def test_ner(self, data_type):
        pred_tags = []
        true_tags = []
        re_type = "torch" if self.framework == "torch" else "numpy"
        for batch_data in self.data_helper.batch_iter(task=self.task, data_type=data_type,
                                                      batch_size=Config.test_batch_size,
                                                      re_type=re_type, _shuffle=False,
                                                      fixed_seq_len=self.fixed_seq_len):
            batch_pred_ids = self.predict_ner(batch_data)  # shape: (batch_size, 1)
            pred_tags.extend(batch_pred_ids.tolist())
            true_tags.extend(batch_data["ent_tags"].tolist())
        assert len(pred_tags) == len(true_tags)
        acc, precision, recall, f1 = self.evaluate_ner(true_tags, pred_tags)
        return acc, precision, recall, f1

    def test_re(self, data_type):
        pred_tags = []
        true_tags = []
        re_type = "torch" if self.framework == "torch" else "numpy"
        for batch_data in self.data_helper.batch_iter(task=self.task, data_type=data_type,
                                                      batch_size=Config.test_batch_size,
                                                      re_type=re_type, _shuffle=False,
                                                      fixed_seq_len=self.fixed_seq_len):
            batch_pred_ids = self.predict_re(batch_data)  # shape: (batch_size, 1)
            pred_tags.extend(batch_pred_ids.tolist())
            true_tags.extend(batch_data["rel_labels"].tolist())
        assert len(pred_tags) == len(true_tags)
        acc, precision, recall, f1 = self.get_re_metrics(true_tags, pred_tags, average="macro")
        return acc, precision, recall, f1

    def evaluate_ner(self, batch_y_ent_ids, batch_pred_ent_ids):
        """
        :param batch_y_ent_ids:
        :param batch_pred_ent_ids:
        :return:
        """
        _true_tags = [[self.data_helper.id2ent_tag.get(tag_id, "O") for tag_id in line_tags]
                      for line_tags in batch_y_ent_ids]
        _pred_tags = [[self.data_helper.id2ent_tag.get(tag_id, "O") for tag_id in line_tags]
                      for line_tags in batch_pred_ent_ids]
        # true_tags = sum(_true_tags, [])
        # pred_tags = sum(_pred_tags, [])
        acc = ner_accuracy_score(_true_tags, _pred_tags)
        precision, recall, f1 = ner_f1_score(_true_tags, _pred_tags)
        return acc, precision, recall, f1

    def get_re_metrics(self, y_true, y_pred, average="macro"):
        metric_labels = self.data_helper.relation_metric_labels
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, labels=metric_labels, average=average)
        precision = precision_score(y_true, y_pred, labels=metric_labels, average=average)
        f1 = f1_score(y_true, y_pred, average=average)
        return acc, precision, recall, f1


def ner_accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        # >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def ner_f1_score(y_true, y_pred, average='micro', digits=2, suffix=False):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        # >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return p, r, f1


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
