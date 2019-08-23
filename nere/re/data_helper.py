import logging
import os
import random

import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from tqdm import trange

from .config import Config

random.seed(Config.rand_seed)


class DataHelper(object):

    def __init__(self):
        self.sequence_len = Config.sequence_len
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)
        self.load_tags()

    def load_tags(self):
        with open(os.path.join(Config.re_data_dir, "rel_labels.txt"), "r", encoding="utf-8") as f:
            relation_labels = [line.strip() for line in f.readlines()]
            self.rel_label2id = {rel_label: id for id, rel_label in enumerate(relation_labels)}
        with open(os.path.join(Config.re_data_dir, "ent_labels.txt"), "r", encoding="utf-8") as f:
            entity_labels = [line.strip() for line in f.readlines()]
            self.ent_label2id = {ent_label: id for id, ent_label in enumerate(entity_labels)}
        # Other types of relationships do not participate in the assessment
        metric_labels = list(self.rel_label2id.values())
        if 'Other' in self.rel_label2id:
            self.other_label_id = self.rel_label2id['Other']
        elif '其他' in self.rel_label2id:
            self.other_label_id = self.rel_label2id['其他']
        else:
            raise ValueError("Unknown other label, must be one of 'Other'/'其他'")
        metric_labels.remove(self.other_label_id)

    def get_data(self, data_type):
        """Loads the data for each type in types from data_dir.
        :param data_type ['train', 'val', 'test']
        :return data2id
        """
        assert data_type in ['train', 'val', 'test'], "data type not in ['train', 'val', 'test']"
        ent_labels, e1_indices, e2_indices, sentences, rel_labels = [], [], [], [], []  # Convert to corresponding ids
        with open(os.path.join(Config.re_data_dir, "{}.txt".format(data_type)), "r", encoding="utf-8") as f:
            for line in f:
                splits = line.strip().split('\t')
                rel_label = splits[0]
                e1_label, e2_label = splits[1], splits[2]
                e1, e2 = splits[3], splits[4]
                sent_text = splits[5]

                ent_labels.append([self.ent_label2id[e1_label], self.ent_label2id[e2_label]])
                e1_tokens = self.tokenizer.tokenize(e1)
                e2_tokens = self.tokenizer.tokenize(e2)
                sent_tokens = ['[CLS]'] + self.tokenizer.tokenize(sent_text)
                e1_match = self.find_sub_list(sent_tokens, e1_tokens)
                e1_indices.append(e1_match)
                e2_match = self.find_sub_list(sent_tokens, e2_tokens)
                e2_indices.append(e2_match)
                if not e1_match:
                    logging.info("Exception: {}\t{}\t{}\n".format(e1, e1_tokens, sent_text))
                if not e2_match:
                    logging.info("Exception: {}\t{}\t{}\n".format(e2, e2_tokens, sent_text))

                sentences.append(self.tokenizer.convert_tokens_to_ids(sent_tokens))
                rel_labels.append(self.rel_label2id[rel_label])
            assert len(sentences) == len(rel_labels)
        # data = {}
        # data['ent_labels'] = ent_labels
        # data['e1_indices'] = e1_indices
        # data['e2_indices'] = e2_indices
        # data['sents'] = sentences
        # data['rel_labels'] = rel_labels
        return rel_labels, ent_labels, e1_indices, e2_indices, sentences

    def find_sub_list(self, all_list, sub_list):
        match_indices = []
        all_len, sub_len = len(all_list), len(sub_list)
        starts = [i for i, ele in enumerate(all_list) if ele == sub_list[0]]
        for start in starts:
            end = start + sub_len
            if end <= all_len and all_list[start: end] == sub_list:
                match_indices.extend(list(range(start, end)))
        return match_indices

    def get_fake_rel_label(self, cur_label):
        fake_rel_label = random.randint(0, len(self.rel_label2id) - 1)
        while fake_rel_label == cur_label:
            fake_rel_label = random.randint(0, len(self.rel_label2id) - 1)
        return fake_rel_label

    def batch_iter(self, data_type, batch_size, epoch_nums, _shuffle=True):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'tags' and 'size'
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_tags: (tensor) shape: (batch_size, max_len)
        """
        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        rel_labels, ent_labels, e1_indices, e2_indices, sentences = self.get_data(data_type)
        logging.info("load {} sentence data {} ...".format(data_type, len(sentences)))
        order = list(range(len(rel_labels)))
        if _shuffle:
            random.shuffle(order)
        for epoch in trange(epoch_nums):
            self.epoch_num = epoch + 1
            # one pass over data
            for batch_num in trange(len(rel_labels) // batch_size):
                # fetch sentences and tags
                batch_indexs = order[batch_num * batch_size:(batch_num + 1) * batch_size]
                batch_ent_labels = [ent_labels[idx] for idx in batch_indexs]
                batch_e1_indices = [e1_indices[idx] for idx in batch_indexs]
                batch_e2_indices = [e2_indices[idx] for idx in batch_indexs]
                batch_sents = [sentences[idx] for idx in batch_indexs]
                batch_rel_labels = [rel_labels[idx] for idx in batch_indexs]

                # batch length
                this_batch_size = len(batch_rel_labels)

                # compute length of longest sentence in batch
                batch_max_len = max([len(s) for s in batch_sents])
                max_len = min(batch_max_len, self.sequence_len)
                # prepare a numpy array with the data, initialising the data with pad_idx
                sents_padding = np.zeros((this_batch_size, max_len))
                e1_masks = np.zeros((this_batch_size, max_len))
                e2_masks = np.zeros((this_batch_size, max_len))
                # Considering that the text is too long, e2 is truncated
                # e2_dummy_masks = np.zeros((batch_len, max_len))
                fake_labels = -1 * np.ones(this_batch_size)
                # copy the data to the numpy array
                for j in range(this_batch_size):
                    logging.info("{}/{}".format(j, this_batch_size))
                    cur_len = len(batch_sents[j])
                    if cur_len <= max_len:
                        sents_padding[j][:cur_len] = batch_sents[j]
                    else:
                        sents_padding[j] = batch_sents[j][:max_len]
                    if batch_e1_indices[j]:
                        e1_masks[j][batch_e1_indices[j]] = 1
                    else:
                        print("Exception: e1_indices[{}] is empty".format(j))
                    if batch_e2_indices[j]:
                        e2_masks[j][batch_e2_indices[j]] = 1
                    else:
                        print("Exception: e2_indices[{}] is empty".format(j))
                    fake_label = self.get_fake_rel_label(rel_labels[j])
                    fake_labels[j] = fake_label

                batch_x_data = {'ent_labels': batch_ent_labels,
                                'e1_masks': e1_masks,
                                'e2_masks': e2_masks,
                                'sents': sents_padding,
                                'fake_rel_labels': fake_labels}
                yield batch_x_data, batch_rel_labels
