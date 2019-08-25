import logging
import os
import random

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer
from tqdm import trange

from .config import Config

random.seed(Config.rand_seed)

entity_tags = {'自然人主体': 'NP',
               '非自然人主体': 'NNP',
               '机动车': 'MV',
               '非机动车': 'NMV',
               '责任认定': 'DUT',
               '一般人身损害': 'GI',
               '伤残': 'DIS',
               '死亡': 'DEA',
               '人身损害赔偿项目': 'PIC',
               '财产损失赔偿项目': 'PLC',
               '保险类别': 'INS',
               '抗辩事由': 'DEF',
               '违反道路交通信号灯': 'VTL',
               '饮酒后驾驶': 'DAD',
               '醉酒驾驶': 'DD',
               '超速': 'SPE',
               '违法变更车道': 'ICL',
               '未取得驾驶资格': 'UD',
               '超载': 'OVE',
               '不避让行人': 'NAP',
               '行人未走人行横道或过街设施': 'NCF',
               '其他违法行为': 'OLA'
               }


class DataHelper(object):

    def __init__(self):
        self.sequence_len = Config.sequence_len
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)
        self.load_tags()
        self.load_ner_tags()

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

    def load_ner_tags(self):
        with open(os.path.join(Config.ner_data_dir, "tags.txt"), "r", encoding="utf-8") as f:
            tags = [line.strip() for line in f.readlines()]
        self.entity_tag2id = {"O": 0}  # B-MV I-MV
        for tag in tags:
            if tag not in self.entity_tag2id:
                self.entity_tag2id[tag] = len(self.entity_tag2id)
        self.id2entity_tag = {id: tag for id, tag in enumerate(tags)}
        self.entity_label2tag = entity_tags
        # self.entity_label2tag_id = {entity: idx for idx, (entity, tag) in enumerate(entity_tags.items())}

    def get_data(self, data_type):
        """Loads the data for each type in types from data_dir.
        :param data_type ['train', 'val', 'test']
        :return data2id
        """
        assert data_type in ['train', 'val', 'test'], "data type not in ['train', 'val', 'test']"
        ent_labels, e1_indices, e2_indices, sentences, rel_labels = [], [], [], [], []  # Convert to corresponding ids
        sentences_ent_tags = []
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
                # entity tag
                ent_tags = np.ones(len(sent_tokens)) * self.entity_tag2id["O"]
                for ent_label, ent_match in [(e1_label, e1_match), (e2_label, e2_match)]:
                    _tag = self.entity_label2tag[ent_label]
                    ent_tags[ent_match] = self.entity_tag2id["I-" + _tag]
                    ent_tags[ent_match[0]] = self.entity_tag2id["B-" + _tag]
                sentences_ent_tags.append(ent_tags)
                sentences.append(self.tokenizer.convert_tokens_to_ids(sent_tokens))
                rel_labels.append(self.rel_label2id[rel_label])
            assert len(sentences) == len(rel_labels) == len(sentences_ent_tags)
        # data = {}
        # data['ent_labels'] = ent_labels
        # data['e1_indices'] = e1_indices
        # data['e2_indices'] = e2_indices
        # data['sents'] = sentences
        # data['rel_labels'] = rel_labels
        # data["sentences_ent_tags"]=sentences_ent_tags
        return rel_labels, ent_labels, e1_indices, e2_indices, sentences, sentences_ent_tags

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
        rel_labels, ent_labels, e1_indices, e2_indices, sentences, sentences_ent_tags = self.get_data(data_type)
        order = list(range(len(rel_labels)))
        logging.info("* RE load data:{}, data_type:{}, num_epochs: {}".format(len(sentences), data_type, epoch_nums))
        for epoch in trange(epoch_nums):
            self.epoch_num = epoch + 1
            if _shuffle:
                random.shuffle(order)
            # one pass over data
            for batch_num in range(len(rel_labels) // batch_size):
                # fetch sentences and tags
                batch_indexs = order[batch_num * batch_size:(batch_num + 1) * batch_size]
                batch_ent_labels = [ent_labels[idx] for idx in batch_indexs]
                batch_e1_indices = [e1_indices[idx] for idx in batch_indexs]
                batch_e2_indices = [e2_indices[idx] for idx in batch_indexs]
                batch_sents = [sentences[idx] for idx in batch_indexs]
                batch_ent_tags = [sentences_ent_tags[idx] for idx in batch_indexs]
                batch_rel_labels = [rel_labels[idx] for idx in batch_indexs]

                # batch length
                this_batch_size = len(batch_rel_labels)

                # compute length of longest sentence in batch
                batch_max_len = max([len(s) for s in batch_sents])
                max_len = min(batch_max_len, self.sequence_len)
                # prepare a numpy array with the data, initialising the data with pad_idx
                sents_padding = np.zeros((this_batch_size, max_len))
                sents_tags_padding = np.zeros((this_batch_size, max_len))
                e1_masks = np.zeros((this_batch_size, max_len))
                e2_masks = np.zeros((this_batch_size, max_len))
                # Considering that the text is too long, e2 is truncated
                # e2_dummy_masks = np.zeros((batch_len, max_len))
                fake_labels = -1 * np.ones(this_batch_size)
                # copy the data to the numpy array
                for j in range(this_batch_size):
                    cur_len = len(batch_sents[j])
                    if cur_len <= max_len:
                        sents_padding[j][:cur_len] = batch_sents[j]
                        sents_tags_padding[j][:cur_len] = batch_ent_tags[j]
                    else:
                        sents_padding[j] = batch_sents[j][:max_len]
                        sents_tags_padding[j] = batch_ent_tags[j][:max_len]
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

                batch_data = {'ent_labels': batch_ent_labels,
                              'e1_masks': e1_masks,
                              'e2_masks': e2_masks,
                              'sents': sents_padding,
                              "sents_tags": sents_tags_padding,
                              "rel_labels": batch_rel_labels,
                              'fake_rel_labels': fake_labels}
                yield batch_data

    def get_samples(self, data_type, sample_type):
        """
        :param data_type: traib,val,test
        :param sample_type:  ner,re,joint
        :return:
        """
        rel_labels, ent_labels, e1_indices, e2_indices, sentences, sentences_ent_tags = self.get_data(data_type)
        logging.info("* RE load data:{}, data_type:{}".format(len(sentences), data_type))
        # max_sequence_len = min(Config.sequence_len, [len(s) for s in sentences])
        max_sequence_len = Config.sequence_len
        if sample_type == "ner":
            x_data = pad_sequences(sentences, maxlen=max_sequence_len, dtype=int,
                                   padding="post", truncating='post', value=0)
            y_data = pad_sequences(sentences_ent_tags, maxlen=max_sequence_len, dtype=int,
                                   padding="post", truncating='post', value=self.entity_tag2id["O"])
            return x_data, y_data
