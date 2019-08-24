import logging
import os
from pathlib import Path

from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer
from sklearn.utils import shuffle
from tqdm import trange

from .config import Config


class DataHelper(object):

    def __init__(self):
        self.sequence_len = Config.sequence_len
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)
        self.load_tag()
        self.load_entities()

    def load_tag(self):
        with open(os.path.join(Config.ner_data_dir, "tags.txt"), "r", encoding="utf-8") as f:
            tags = [line.strip() for line in f.readlines()]
        self.tag2id = {"O": 0}
        for tag in tags:
            if tag not in self.tag2id:
                self.tag2id[tag] = len(self.tag2id)
        self.id2tag = {id: tag for id, tag in enumerate(tags)}

    def load_entities(self):
        with open(os.path.join(Config.ner_data_dir, "tags.txt"), "r", encoding="utf-8") as f:
            tags = [line.strip() for line in f.readlines()]
        self.entity2id = {"O": 0}
        tags.remove("O")
        for tag in tags:
            entity = tag.split("-")[1]
            if entity not in self.entity2id:
                self.entity2id[entity] = len(self.entity2id)
        self.id2entity = {id: entity for id, entity in enumerate(tags)}
        return self.entity2id, self.id2entity

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
        for tag, sentence in zip(tags, sentences):
            assert len(tag) == len(sentence)
        return sentences, tags

    def batch_iter(self, data_type, batch_size, epoch_nums, _shuffle=True):
        sentences, tags = self.load_data(data_type)
        x_data, y_data = sentences, tags
        logging.info("* NER data_type : {} num_epochs: {}".format(data_type, epoch_nums))
        for epoch in trange(epoch_nums):
            self.epoch_num = epoch + 1
            if _shuffle:
                x_data, y_data = shuffle(x_data, y_data, random_state=epoch)
            x_batch = []
            y_batch = []
            for x_sample, y_sample in zip(x_data, y_data):
                x_batch.append(x_sample)
                y_batch.append(y_sample)
                if len(x_batch) == batch_size:
                    batch_max_len = min(max([len(s) for s in x_batch]), Config.sequence_len)
                    x_batch = pad_sequences(x_batch, maxlen=batch_max_len, dtype='int32',
                                            padding='post', truncating='post', value=0).tolist()
                    y_batch = pad_sequences(y_batch, maxlen=batch_max_len, dtype='int32',
                                            padding='post', truncating='post', value=self.tag2id["O"]).tolist()
                    yield x_batch, y_batch
                    x_batch = []
                    y_batch = []


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
