import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer

from config import Config


class DataHelper(object):
    """label : 标签体系
        tag： ：样本标注结果
        因为需要对比单模型和联合训练，所以需要保证单模型的所用数据和联合训练所用数据是相同的
    """

    def __init__(self):
        """
        :param task: joint,ner,re
        """
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)
        self.load_re_tags()
        self.load_ner_tags()
        # iter data
        self.iter_data = None
        self.task_data_type = ""

    def load_re_tags(self):
        with open(os.path.join(Config.re_data_dir, "rel_labels.txt"), "r", encoding="utf-8") as f:
            relation_labels = [line.strip() for line in f.readlines()]  # 搭乘、其他、投保
            self.rel_label2id = {rel_label: id for id, rel_label in enumerate(relation_labels)}
        with open(os.path.join(Config.re_data_dir, "ent_labels.txt"), "r", encoding="utf-8") as f:
            entity_labels = [line.strip() for line in f.readlines()]  # 伤残、机动车、非机动车
            self.ent_label2id = {ent_label: id for id, ent_label in enumerate(entity_labels)}
        self.other_rel_label_id = self.rel_label2id['其他'] if '其他' in self.rel_label2id else self.rel_label2id['Other']
        # Other types of relationships do not participate in the assessment

    def load_ner_tags(self):
        with open(os.path.join(Config.ner_data_dir, "tags.txt"), "r", encoding="utf-8") as f:
            ent_tags = [line.strip() for line in f.readlines()]
        self.ent_tag2id = {"O": 0}  # I-NAP、B-NMV、O  ；BIO
        for tag in ent_tags:
            if tag not in self.ent_tag2id:
                self.ent_tag2id[tag] = len(self.ent_tag2id)
        self.id2ent_tag = {id: tag for tag, id in self.ent_tag2id.items()}
        # self.entity_label2abbr_id = {entity: idx for idx, (entity, tag) in enumerate(entity_tags.items())}

    def get_re_data(self, data_type):
        """Loads the data for each type in types from data_dir.
        :param data_type ['train', 'val', 'test']
        :return data2id, 全部转换为对应的id list
        """
        assert data_type in ['train', 'val', 'test'], "data type not in ['train', 'val', 'test']"
        ent_labels, e1_indices, e2_indices, sentences, rel_labels = [], [], [], [], []  # Convert to corresponding ids
        pos1, pos2 = [], []
        with open(os.path.join(Config.re_data_dir, "{}.txt".format(data_type)), "r", encoding="utf-8") as f:
            for line in f:
                splits = line.strip().split('\t')
                rel_label = splits[0]
                e1_label, e2_label = splits[1], splits[2]
                e1, e2 = splits[3], splits[4]
                sent_text = splits[5].strip()
                ent_labels.append([self.ent_label2id[e1_label], self.ent_label2id[e2_label]])
                e1_tokens = self.tokenizer.tokenize(e1)
                e2_tokens = self.tokenizer.tokenize(e2)
                sent_tokens = ['[CLS]'] + self.tokenizer.tokenize(sent_text) + ["[SEP]"]
                e1_match = self.find_sub_list(sent_tokens, e1_tokens)
                e1_indices.append(e1_match)
                e2_match = self.find_sub_list(sent_tokens, e2_tokens)
                e2_indices.append(e2_match)
                pos1.append([pos_encode(i - e1_match[0]) for i in range(len(sent_tokens))])
                pos2.append([pos_encode(i - e2_match[0]) for i in range(len(sent_tokens))])
                if not e1_match:
                    logging.info("Exception: {}\t{}\t{}\n".format(e1, e1_tokens, sent_text))
                if not e2_match:
                    logging.info("Exception: {}\t{}\t{}\n".format(e2, e2_tokens, sent_text))
                sentences.append(self.tokenizer.convert_tokens_to_ids(sent_tokens))
                rel_labels.append(self.rel_label2id[rel_label])
            assert len(sentences) == len(rel_labels)
        data = {}
        data['ent_labels'] = ent_labels
        data['e1_indices'] = e1_indices
        data['e2_indices'] = e2_indices
        data['pos1'] = pos1
        data['pos2'] = pos2
        data['sents'] = sentences
        data['rel_labels'] = rel_labels
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

    def get_ner_data(self, data_type):
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
            pad = self.ent_tag2id["O"]
            tags = [[pad] + [self.ent_tag2id[tag] for tag in line.strip().split(' ')] + [pad] for line in f]
        assert len(sentences) == len(tags)
        for tag, sentence in zip(tags, sentences):
            assert len(tag) == len(sentence)
        data = {"sents": sentences, "ent_tags": tags}
        return data

    def get_sentences_ent_tags(self):
        sentence2ent_tags = {}
        train_hash_set = set()
        for data_type in ["train", "val", "test"]:
            ner_data = self.get_ner_data(data_type)
            for s, tag in zip(ner_data["sents"], ner_data["ent_tags"]):
                hash_s = hash(tuple(s))
                sentence2ent_tags[hash_s] = tag
                if data_type == "train":
                    train_hash_set.add(hash_s)
        return sentence2ent_tags, train_hash_set

    def get_joint_data(self, task, data_type):
        """
            验证集暂时不使用
        :param task:
        :param data_type:
        :return:
        """
        omit_count = 0
        sentences_hash2ent_tags, train_sents_hash = self.get_sentences_ent_tags()
        re_data = self.get_re_data(data_type)
        joint_data = {'ent_labels': [], 'e1_indices': [], 'e2_indices': [], 'sents': [], 'rel_labels': [],
                      'ent_tags': [], "pos1": [], "pos2": []}
        unique_sents_hashs = set()
        for i, s in enumerate(re_data["sents"]):
            sent_hash = hash(tuple(s))
            if task == "ner":
                if sent_hash in unique_sents_hashs:  # ner 保证任务数据集单一
                    continue
                elif data_type == "test" and sent_hash in train_sents_hash:  # test not in train #todo joint
                    continue
            if sent_hash in sentences_hash2ent_tags:
                ent_tag = sentences_hash2ent_tags[sent_hash]
                assert len(ent_tag) == len(s)
                unique_sents_hashs.add(sent_hash)
                joint_data["ent_tags"].append(ent_tag)
                joint_data["ent_labels"].append(re_data["ent_labels"][i])
                joint_data["e1_indices"].append(re_data["e1_indices"][i])
                joint_data["e2_indices"].append(re_data["e2_indices"][i])
                joint_data["pos1"].append(re_data["pos1"][i])
                joint_data["pos2"].append(re_data["pos2"][i])
                joint_data["sents"].append(re_data["sents"][i])
                joint_data["rel_labels"].append(re_data["rel_labels"][i])
            else:
                # print(self.tokenizer.convert_ids_to_tokens(s), "\n\n")
                omit_count += 1
        print("***task {} data_type:{} omit/sample_count: {}/{}".format(task, data_type, omit_count,
                                                                        len(joint_data["sents"])))
        return joint_data

    def batch_iter(self, task, data_type, batch_size, re_type="numpy", _shuffle=True, sequence_len=None):
        """
        :param data:  dict
        :param re_type: numpy, torch
        :return:  dict padding & to type
        """
        # one pass over data
        if self.iter_data is None or self.task_data_type != task + data_type:
            self.iter_data = self.get_joint_data(task, data_type)
            self.task_data_type = task + data_type
        data = self.get_joint_data(task, data_type)
        data_size = len(data["sents"])
        order = list(range(data_size))
        for batch_step in range(data_size // batch_size + 1):
            # fetch sentences and tags
            batch_idxs = order[batch_step * batch_size:(batch_step + 1) * batch_size]
            if len(batch_idxs) != batch_size:  # batch size 不可过大
                continue
            ent_labels = [data['ent_labels'][idx] for idx in batch_idxs]
            e1_indices = [data['e1_indices'][idx] for idx in batch_idxs]
            e2_indices = [data['e2_indices'][idx] for idx in batch_idxs]
            pos1 = [data['pos1'][idx] for idx in batch_idxs]
            pos2 = [data['pos2'][idx] for idx in batch_idxs]
            sents = [data['sents'][idx] for idx in batch_idxs]
            rel_labels = [data['rel_labels'][idx] for idx in batch_idxs]
            ent_tags = [data['ent_tags'][idx] for idx in batch_idxs]
            # batch size
            # batch_size = len(batch_idxs)
            if sequence_len is None:
                # compute length of longest sentence in batch
                batch_max_len = max([len(s) for s in sents])
                max_len = min(batch_max_len, Config.max_sequence_len)
            else:
                max_len = sequence_len

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_sents = np.zeros((batch_size, max_len))
            batch_pos1 = np.zeros((batch_size, max_len)) * (Config.max_sequence_len - 100)
            batch_pos2 = np.zeros((batch_size, max_len)) * (Config.max_sequence_len - 100)
            batch_ent_tags = self.ent_tag2id["O"] * np.ones((batch_size, max_len))
            batch_e1_masks = np.zeros((batch_size, max_len))
            batch_e2_masks = np.zeros((batch_size, max_len))
            # Considering that the text is too long, e2 is truncated
            # e2_dummy_masks = np.zeros((batch_size, max_len))
            batch_fake_labels = -1 * np.ones(batch_size)

            # copy the data to the numpy array
            for j in range(batch_size):
                cur_len = len(sents[j])
                if cur_len <= max_len:
                    batch_sents[j][:cur_len] = sents[j]
                    batch_ent_tags[j][:cur_len] = ent_tags[j]
                    batch_pos1[j][:cur_len] = pos1[j]
                    batch_pos2[j][:cur_len] = pos2[j]
                else:
                    batch_sents[j] = sents[j][:max_len]
                    batch_ent_tags[j] = ent_tags[j][:max_len]
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
            if re_type == "torch":
                # since all data are indices, we convert them to torch LongTensors
                batch_ent_labels = torch.tensor(batch_ent_labels, dtype=torch.long).to(Config.device)
                batch_e1_masks = torch.tensor(batch_e1_masks, dtype=torch.long).to(Config.device)
                batch_e2_masks = torch.tensor(batch_e2_masks, dtype=torch.long).to(Config.device)
                batch_pos1 = torch.tensor(batch_pos1, dtype=torch.long).to(Config.device)
                batch_pos2 = torch.tensor(batch_pos2, dtype=torch.long).to(Config.device)
                batch_sents = torch.tensor(batch_sents, dtype=torch.long).to(Config.device)
                batch_ent_tags = torch.tensor(batch_ent_tags, dtype=torch.long).to(Config.device)
                batch_fake_labels = torch.tensor(batch_fake_labels, dtype=torch.long).to(Config.device)
                batch_rel_labels = torch.tensor(batch_rel_labels, dtype=torch.long).to(Config.device)

            batch_data = {'ent_labels': batch_ent_labels,
                          'e1_masks': batch_e1_masks,
                          'e2_masks': batch_e2_masks,
                          'pos1': batch_pos1,
                          'pos2': batch_pos2,
                          'sents': batch_sents,
                          'ent_tags': batch_ent_tags,
                          'fake_rel_labels': batch_fake_labels,
                          "rel_labels": batch_rel_labels}
            yield batch_data

    def get_samples(self, task, data_type):
        """
        :param data_type: train,val,test
        :param task:  ner,re,joint
        :return:
        """
        sample_datas = {'ent_labels': [], 'e1_masks': [], 'e2_masks': [], "pos1": [], "pos2": [],
                        'sents': [], 'ent_tags': [], 'fake_rel_labels': [], "rel_labels": []}
        for batch_data in self.batch_iter(task=task, data_type=data_type, batch_size=Config.batch_size, re_type="numpy",
                                          sequence_len=Config.max_sequence_len):
            for key, v in batch_data.items():
                sample_datas[key].extend(list(v))
        for k, v in sample_datas.items():
            sample_datas[k] = np.asarray(v)
        return sample_datas


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
