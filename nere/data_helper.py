import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer

from config import Config

from collections import defaultdict


class DataHelper(object):
    """label : 标签体系
        tag： ：样本标注结果
        因为需要对比单模型和联合训练，所以需要保证单模型的所用数据和联合训练所用数据是相同的
    """

    def __init__(self, ner_data_dir=Config.ner_data_dir, re_data_dir=Config.re_data_dir):
        self.ner_data_dir = ner_data_dir
        self.re_data_dir = re_data_dir
        self.load_re_tags()
        self.load_ner_tags()
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)
        # iter data
        self.iter_data = None
        self.task_data_type = ""

    def load_ner_tags(self):
        with open(os.path.join(self.ner_data_dir, "tags.txt"), "r", encoding="utf-8") as f:
            ent_tags = [line.strip() for line in f.readlines()]
        self.ent_tag2id = {"O": 0}  # I-NAP、B-NMV、O  ；BIO
        for tag in ent_tags:
            if tag not in self.ent_tag2id:
                self.ent_tag2id[tag] = len(self.ent_tag2id)
        self.id2ent_tag = {id: tag for tag, id in self.ent_tag2id.items()}
        # self.entity_label2abbr_id = {entity: idx for idx, (entity, tag) in enumerate(entity_tags.items())}

    def load_re_tags(self):
        with open(os.path.join(self.re_data_dir, "rel_labels.txt"), "r", encoding="utf-8") as f:
            relation_labels = [line.strip() for line in f.readlines()]  # 搭乘、其他、投保
            self.rel_label2id = {rel_label: id for id, rel_label in enumerate(relation_labels)}
        with open(os.path.join(self.re_data_dir, "ent_labels.txt"), "r", encoding="utf-8") as f:
            entity_labels = [line.strip() for line in f.readlines()]  # 伤残、机动车、非机动车
            self.ent_label2id = {ent_label: id for id, ent_label in enumerate(entity_labels)}
        self.other_rel_label_id = self.rel_label2id['其他'] if '其他' in self.rel_label2id else self.rel_label2id['Other']
        # Other types of relationships do not participate in the assessment

    def get_ner_data(self, data_type):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        assert data_type in ['train', 'val', 'test'], "data type not in ['train', 'val', 'test']"
        data_dir = Path(self.ner_data_dir).joinpath(data_type)
        with open(data_dir.joinpath("sentences.txt"), "r", encoding="utf-8") as f:
            sentences = []
            for line in f:
                tokens = ['[CLS]'] + line.rstrip('\n').split(' ') + ['[SEP]']
                sentences.append(self.tokenizer.convert_tokens_to_ids(tokens))
        with open(data_dir.joinpath("tags.txt"), "r", encoding="utf-8") as f:
            pad = self.ent_tag2id["O"]
            tags = [[pad] + [self.ent_tag2id[tag] for tag in line.strip().split(' ')] + [pad] for line in f]
        assert len(sentences) == len(tags)
        for tag, sentence in zip(tags, sentences):
            assert len(tag) == len(sentence)
        data = {"sents": sentences, "ent_tags": tags}
        return data

    def get_re_data(self, data_type):
        """Loads the data for each type in types from data_dir.
        :param data_type ['train', 'val', 'test']
        :return data2id, 全部转换为对应的id list
        """
        assert data_type in ['train', 'val', 'test'], "data type not in ['train', 'val', 'test']"
        ent_labels, e1_indices, e2_indices, sentences, rel_labels = [], [], [], [], []  # Convert to corresponding ids
        pos1, pos2 = [], []
        entity_match_omit = 0
        with open(os.path.join(self.re_data_dir, "{}.txt".format(data_type)), "r", encoding="utf-8") as f:
            for line in f:
                _ent_match_omit = 0
                splits = line.strip().split('\t')
                rel_label = splits[0]
                e1_label, e2_label = splits[1], splits[2]
                e1, e2 = splits[3], splits[4]
                sent_text = splits[5].strip()
                e1_tokens = self.tokenizer.tokenize(e1)
                e2_tokens = self.tokenizer.tokenize(e2)
                sent_tokens = ['[CLS]'] + self.tokenizer.tokenize(sent_text) + ["[SEP]"]
                e1_match = self.find_sub_list(sent_tokens, e1_tokens)
                e2_match = self.find_sub_list(sent_tokens, e2_tokens)
                if not e1_match:
                    logging.info("Exception: {}\t{}\t{}\n".format(e1, e1_tokens, sent_text))
                    _ent_match_omit += 1
                else:
                    pos1.append([pos_encode(i - e1_match[0]) for i in range(len(sent_tokens))])
                if not e2_match:
                    logging.info("Exception: {}\t{}\t{}\n".format(e2, e2_tokens, sent_text))
                    _ent_match_omit += 1
                else:
                    pos2.append([pos_encode(i - e2_match[0]) for i in range(len(sent_tokens))])
                if _ent_match_omit:
                    continue
                e1_indices.append(e1_match)
                e2_indices.append(e2_match)
                sentences.append(self.tokenizer.convert_tokens_to_ids(sent_tokens))
                ent_labels.append([self.ent_label2id[e1_label], self.ent_label2id[e2_label]])
                rel_labels.append(self.rel_label2id[rel_label])
            assert len(sentences) == len(rel_labels)
        data = {'ent_labels': ent_labels, 'e1_indices': e1_indices, 'e2_indices': e2_indices,
                'pos1': pos1, 'pos2': pos2, 'sents': sentences, 'rel_labels': rel_labels}
        print("* get_re_data entity_match_omit/sentences: {}/{}".format(entity_match_omit, len(sentences)))
        return data

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

    def get_joint_train_data(self):
        ner_data = self.get_ner_data(data_type="train")
        ner_data_dict = {hash(tuple(sent)): tag_seq for sent, tag_seq in zip(ner_data["sents"], ner_data["ent_tags"])}
        re_data = self.get_re_data(data_type="train")
        joint_data = {'ent_labels': [], 'e1_indices': [], 'e2_indices': [], 'sents': [], 'rel_labels': [],
                      'ent_tags': [], "pos1": [], "pos2": []}
        for i, s in enumerate(re_data["sents"]):
            ent_tag = ner_data_dict[hash(tuple(s))]
            assert len(ent_tag) == len(s)
            joint_data["ent_tags"].append(ent_tag)
            joint_data["ent_labels"].append(re_data["ent_labels"][i])
            joint_data["e1_indices"].append(re_data["e1_indices"][i])
            joint_data["e2_indices"].append(re_data["e2_indices"][i])
            joint_data["pos1"].append(re_data["pos1"][i])
            joint_data["pos2"].append(re_data["pos2"][i])
            joint_data["sents"].append(re_data["sents"][i])
            joint_data["rel_labels"].append(re_data["rel_labels"][i])
        print("* get_joint_train_data, {}".format(len(joint_data["sents"])))
        return joint_data

    def get_joint_data(self, data_type):
        """
            以RE数据为准，NER数据填充之
        :param data_type:
        :return:
        """
        shash2ent_tag = {}
        for _data_type in ["train", "val", "test"]:
            _ner_data = self.get_ner_data(_data_type)
            for sentence, ent_tag in zip(_ner_data["sents"], _ner_data["ent_tags"]):
                shash2ent_tag[hash(tuple(sentence))] = ent_tag
        # joint data
        omit = 0
        unique_sents = set()
        re_data = self.get_re_data(data_type)
        re_data["ent_tags"] = []
        hash_sents = [hash(tuple(s)) for s in re_data["sents"]]
        for i, s_hash in enumerate(hash_sents):
            if s_hash in shash2ent_tag:
                re_data["ent_tags"].append(shash2ent_tag[s_hash])
                unique_sents.add(s_hash)
            else:
                for key in re_data:  # 没有ner的数据，不能联合，删除此数据
                    if key != "ent_tags":
                        re_data[key].pop(i - omit)
                omit += 1
        print("* get_joint_data, {}, omit/all:{}/{}, joint_data/unique_sents: {}/{}".format(
            data_type, omit, len(hash_sents), len(re_data["sents"]), len(unique_sents)))
        import ipdb
        assert len(re_data["sents"]) == len(re_data["ent_tags"]), ipdb.set_trace()
        return re_data

    def batch_iter(self, task, data_type, batch_size, re_type="numpy", _shuffle=True, sequence_len=None):
        """
        :param data:  dict
        :param re_type: numpy, torch
        :return:  dict padding & to type
        """
        assert task in ["ner", "re", "joint"] and data_type in ["train", "val", "test"]
        if self.iter_data is None or self.task_data_type != task + data_type:
            self.iter_data = self.get_joint_data(data_type)
            # if data_type == "train":  # joint
            #     self.iter_data = self.get_joint_train_data()
            # elif task == "ner":
            #     self.iter_data = self.get_ner_data(data_type)
            # elif task == "re":
            #     self.iter_data = self.get_re_data(data_type)
            self.task_data_type = task + data_type
        data = self.iter_data
        data_size = len(data["sents"])
        max_len = Config.max_sequence_len
        order = list(range(data_size))
        if _shuffle:
            np.random.shuffle(order)
        for batch_step in range(data_size // batch_size + 1):
            # fetch sentences and tags
            batch_idxs = order[batch_step * batch_size:(batch_step + 1) * batch_size]
            if len(batch_idxs) != batch_size:  # batch size 不可过大; 不足batch_size的数据丢弃（最后一batch）
                continue
            sents = [data['sents'][idx] for idx in batch_idxs]
            # max_len = sequence_len if sequence_len else min(max([len(s) for s in sents]), Config.max_sequence_len)
            batch_sents = np.zeros((batch_size, max_len))

            for j in range(batch_size):
                cur_len = len(sents[j])
                if cur_len <= max_len:
                    batch_sents[j][:cur_len] = sents[j]
                else:
                    batch_sents[j] = sents[j][:max_len]
            batch_data = {'sents': batch_sents}
            if task in ["ner", "joint"]:  # ner 专有属性
                ent_tags = [data['ent_tags'][idx] for idx in batch_idxs]
                batch_ent_tags = self.ent_tag2id["O"] * np.ones((batch_size, max_len))
                for j in range(batch_size):
                    cur_len = len(sents[j])
                    if cur_len <= max_len:
                        batch_ent_tags[j][:cur_len] = ent_tags[j]
                    else:
                        batch_ent_tags[j] = ent_tags[j][:max_len]
                batch_data["ent_tags"] = batch_ent_tags
            if task in ["re", "joint"]:  # re 专有属性
                ent_labels = [data['ent_labels'][idx] for idx in batch_idxs]
                e1_indices = [data['e1_indices'][idx] for idx in batch_idxs]
                e2_indices = [data['e2_indices'][idx] for idx in batch_idxs]
                pos1 = [data['pos1'][idx] for idx in batch_idxs]
                pos2 = [data['pos2'][idx] for idx in batch_idxs]
                rel_labels = [data['rel_labels'][idx] for idx in batch_idxs]
                batch_pos1 = np.zeros((batch_size, max_len)) * (Config.max_sequence_len - 100)
                batch_pos2 = np.zeros((batch_size, max_len)) * (Config.max_sequence_len - 100)
                batch_e1_masks = np.zeros((batch_size, max_len))
                batch_e2_masks = np.zeros((batch_size, max_len))
                batch_fake_labels = -1 * np.ones(batch_size)
                # copy the data to the numpy array
                for j in range(batch_size):
                    cur_len = len(sents[j])
                    if cur_len <= max_len:
                        batch_pos1[j][:cur_len] = pos1[j]
                        batch_pos2[j][:cur_len] = pos2[j]
                    else:
                        batch_pos1[j] = pos1[j][:max_len]
                        batch_pos2[j] = pos2[j][:max_len]
                    if e1_indices[j]:
                        batch_e1_masks[j][e1_indices[j]] = 1
                    else:
                        print("Exception: e1_indices[{}] is empty".format(j))
                    if e2_indices[j]:
                        batch_e2_masks[j][e2_indices[j]] = 1
                    else:
                        print("Exception: e2_indices[{}] is empty".format(j))
                    batch_fake_labels[j] = self.get_fake_rel_label(rel_labels[j])
                batch_ent_labels = ent_labels
                batch_rel_labels = rel_labels
                #
                batch_data['sents'] = batch_sents
                batch_data['ent_labels'] = batch_ent_labels
                batch_data['e1_masks'] = batch_e1_masks
                batch_data['e2_masks'] = batch_e2_masks
                batch_data['pos1'] = batch_pos1
                batch_data['pos2'] = batch_pos2
                batch_data['fake_rel_labels'] = batch_fake_labels
                batch_data["rel_labels"] = batch_rel_labels
            if re_type == "torch":
                batch_data = {k: torch.tensor(v, dtype=torch.long).to(Config.device) for k, v in batch_data.items()}
            yield batch_data

    def get_samples(self, task, data_type):
        """
        :param data_type: train,val,test
        :param task:  ner,re,joint
        :return:
        """
        sample_datas = defaultdict(list)
        for batch_data in self.batch_iter(task=task, data_type=data_type, batch_size=Config.batch_size,
                                          re_type="numpy", sequence_len=Config.max_sequence_len):
            for key, v in batch_data.items():
                sample_datas[key].extend(list(v))
        return {k: np.asarray(data) for k, data in sample_datas.items()}


def pos_encode(relative_position):
    """
    :param relative_position: 当前单词相对于实体的位置
    :return:
    """
    pos_size = Config.max_sequence_len - 100
    semi_size = pos_size // 2
    if relative_position < -semi_size:
        pos_code = 0
    elif -semi_size <= relative_position < semi_size:
        pos_code = relative_position + semi_size
    else:  # if relative_position > semi_size:
        pos_code = pos_size - 1
    return pos_code
