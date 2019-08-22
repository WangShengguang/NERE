import os
from pathlib import Path

from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer
from sklearn.utils import shuffle

from .config import Config


class DataHelper(object):

    def __init__(self):
        self.sequence_len = Config.sequence_len
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)
        self.load_tag()

    def load_tag(self):
        with open(os.path.join(Config.ner_data_dir, "tags.txt"), "r", encoding="utf-8") as f:
            tags = [line.strip() for line in f.readlines()]
        self.tag2id = {"O": 0}
        for tag in tags:
            if tag not in self.tag2id:
                self.tag2id[tag] = len(self.tag2id)
        self.id2tag = {id: tag for id, tag in enumerate(tags)}

    def load_data(self, data_type):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        assert data_type in ['train', 'val', 'test'], "data type not in ['train', 'val', 'test']"
        data_dir = Path(Config.ner_data_dir).joinpath(data_type)
        with open(data_dir.joinpath("sentences.txt"), "r", encoding="utf-8") as f:
            sentences = []
            for line in f:
                tokens = ['[CLS]'] + line.rstrip('\n').split(' ') + ['[SEP]']
                sentences.append(self.tokenizer.convert_tokens_to_ids(tokens))
        with open(data_dir.joinpath("tags.txt"), "r", encoding="utf-8") as f:
            pad = self.tag2id["O"]
            tags = [[pad] + [self.tag2id[tag] for tag in line.strip().split(' ')] + [pad] for line in f]
        assert len(sentences) == len(tags)
        for i in range(len(sentences)):
            assert len(tags[i]) == len(sentences[i])
        return sentences, tags

    def batch_iter(self, data_type, batch_size, epoch_nums, _shuffle=True):
        sentences, tags = self.load_data(data_type)
        x_data = pad_sequences(sentences, maxlen=Config.sequence_len, dtype='int32',
                               padding='post', truncating='post', value=0)
        y_data = pad_sequences(tags, maxlen=Config.sequence_len, dtype='int32',
                               padding='post', truncating='post', value=self.tag2id["O"])
        for epoch in range(epoch_nums):
            self.epoch_num = epoch + 1
            if _shuffle:
                x_data, y_data = shuffle(x_data, y_data, random_state=epoch)
            x_batch = []
            y_batch = []
            for x_sample, y_sample in zip(x_data, y_data):
                x_batch.append(x_sample)  # 注意，交换了位置
                y_batch.append(y_sample)
                if len(x_batch) == batch_size:
                    yield x_batch, y_batch
                    x_batch = []
                    y_batch = []
