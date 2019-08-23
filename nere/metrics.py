import numpy as np

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


class MutilabelMetrics(object):
    """
    https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel
    """

    def __init__(self, all_labels=None):
        self.all_labels = all_labels
        self.labels = [idx for idx, tag in enumerate(self.all_labels)]

    def vectorization(self, sample_labels):
        """
            labels : list, optional
            The set of labels to include when ``average != 'binary'``
        :param sample_labels:
        :return:
        """
        sample_label_vectors = []
        for sample_y in sample_labels:
            _sample_y = np.zeros_like(self.all_labels)
            _sample_y[[self.all_labels.index(_label) for _label in sample_y]] = 1
            sample_label_vectors.append(_sample_y)
        return sample_label_vectors

    def get_metrics(self, y_trues, y_preds):
        """
        :param y_trues: list [["label1","label2"],["label1"]] ; compatible id or str label
        :param y_preds: [["label1","label2"],["label1"]]
        :return:
        """
        y_true = self.vectorization(y_trues)
        y_pred = self.vectorization(y_preds)
        acc = accuracy_score(sum(y_trues, []), sum(y_preds, []))
        average = "samples"
        labels = self.labels
        recall = recall_score(y_true, y_pred, labels=labels, average=average)
        precision = precision_score(y_true, y_pred, labels=labels, average=average)
        f1 = f1_score(y_true, y_pred, labels=labels, average=average)
        return acc, precision, recall, f1
