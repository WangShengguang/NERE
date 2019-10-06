import difflib
import os
import shutil
from collections import defaultdict
from typing import List

import numpy as np
from pytorch_pretrained_bert import BertTokenizer

from config import Config


class PrepareJointData(object):
    """
    联合训练，因为需要对比，必需使得NER和RE训练集 使用的句子 相同 才能联合
    测试时，NER和RE可使用各自独立的测试数据，这样可保证不会丢弃太多数据
    """

    def get_re_data(self):
        """Loads the data for each type in types from data_dir.
        :param data_type ['train', "valid", 'test']
        :return data2id, 全部转换为对应的id list
        """
        tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)
        re_data = defaultdict(list)
        for data_type in ["train", "valid", "test"]:
            with open(os.path.join(Config.re_data_dir, "{}.txt".format(data_type)), "r", encoding="utf-8") as f:
                for line in f:
                    splits = line.strip().split('\t')
                    sent_text = "".join(tokenizer.tokenize(splits[5].strip()))  # 用作key，需要和NER进行同样的处理
                    re_data[sent_text].append(line.strip())  # 一个句子多个标注 sample
        return re_data

    def get_ner_data(self):
        all_sentences = []
        all_tags = []
        for data_type in ["train", "valid", "test"]:
            with open(os.path.join(Config.ner_data_dir, data_type, "sentences.txt"), "r", encoding="utf-8") as f:
                _sentences = [line.strip() for line in f.readlines()]
                all_sentences.extend(_sentences)
            with open(os.path.join(Config.ner_data_dir, data_type, "tags.txt"), "r", encoding="utf-8") as f:
                _tags = [line.strip() for line in f.readlines()]
                all_tags.extend(_tags)
        ner_data = {}
        for sentence_str, tag_str in zip(all_sentences, all_tags):
            sent_txt = "".join(sentence_str.rstrip('\n').split(' '))
            ner_data[sent_txt] = [sentence_str, tag_str]  # 一个句子一个标注sample
        return ner_data

    def get_joint_data(self):
        ner_data = self.get_ner_data()
        re_data = self.get_re_data()
        #
        ner_sents_set = set(ner_data.keys())
        re_sents_set = set(re_data.keys())
        joint_sents = ner_sents_set & re_sents_set
        ner_unique_sents = ner_sents_set - joint_sents
        re_unique_sents = re_sents_set - joint_sents

        print("\n* joint_sents: {}, ner_unique_sents/ner_data: {}/{}, re_unique_sents/re_data: {}/{}".format(
            len(joint_sents), len(ner_unique_sents), len(ner_data), len(re_unique_sents), len(re_data)))

        # test
        # re_unique_sents_list = list(re_unique_sents)
        # sim_sentor = SimSent(list(ner_sents_set))
        # sims = sim_sentor.find_sim_sents(re_unique_sents_list[0])
        # print(re_unique_sents_list[0], "\n", sims)

        # Joint data
        joint_train_sents_li = sorted(joint_sents)
        joint_train_cnt = int(len(joint_train_sents_li) * 0.9)
        joint_train_sents = joint_train_sents_li[:joint_train_cnt]
        joint_test_sents = joint_train_sents_li[joint_train_cnt:]
        # NER data
        os.makedirs(Config.joint_ner_data_dir, exist_ok=True)
        shutil.copyfile(src=os.path.join(Config.ner_data_dir, "tags.txt"),
                        dst=os.path.join(Config.joint_ner_data_dir, "tags.txt"))
        ner_unique_sents_li = joint_test_sents + sorted(ner_unique_sents)
        ner_test_cnt = len(ner_unique_sents_li) // 2
        ner_test_sents = ner_unique_sents_li[:ner_test_cnt]
        ner_val_sents = ner_unique_sents_li[ner_test_cnt:]
        write_ner_data_to_file(joint_train_sents, data_type="train", ner_data=ner_data)
        write_ner_data_to_file(ner_test_sents, data_type="test", ner_data=ner_data)
        write_ner_data_to_file(ner_val_sents, data_type="valid", ner_data=ner_data)
        # RE data
        os.makedirs(Config.joint_re_data_dir, exist_ok=True)
        shutil.copyfile(src=os.path.join(Config.re_data_dir, "ent_labels.txt"),
                        dst=os.path.join(Config.joint_re_data_dir, "ent_labels.txt"))
        shutil.copyfile(src=os.path.join(Config.re_data_dir, "rel_labels.txt"),
                        dst=os.path.join(Config.joint_re_data_dir, "rel_labels.txt"))
        re_unique_sents_li = joint_test_sents + sorted(re_unique_sents)
        re_test_cnt = int(len(re_unique_sents_li) * 0.7)
        re_test_sents = re_unique_sents_li[:re_test_cnt]
        re_val_sents = re_unique_sents_li[re_test_cnt:]
        write_re_data_to_file(joint_train_sents, data_type="train", re_data=re_data)
        write_re_data_to_file(re_test_sents, data_type="test", re_data=re_data)
        write_re_data_to_file(re_val_sents, data_type="valid", re_data=re_data)
        # import ipdb
        # ipdb.set_trace()


def write_re_data_to_file(sent_txts, data_type, re_data):
    """
    :param sent_txts:
    :param data_type: train, val, test
    :return:
    """
    os.makedirs(Config.joint_re_data_dir, exist_ok=True)
    file_path = os.path.join(Config.joint_re_data_dir, "{}.txt".format(data_type))
    cnt = 0
    with open(file_path, "w", encoding="utf-8") as f:
        for s in sent_txts:
            sents = re_data[s]
            for sent in sents:
                f.write(sent + "\n")
                cnt += 1
    print("re data_type: {}, {}".format(data_type, cnt))


def write_ner_data_to_file(sent_txts, data_type, ner_data):
    """
    :param sent_txts:
    :param data_type: train, val, test
    :return:
    """
    os.makedirs(os.path.join(Config.joint_ner_data_dir, data_type), exist_ok=True)
    sentences_file = os.path.join(Config.joint_ner_data_dir, data_type, 'sentences.txt')
    tags_file = os.path.join(Config.joint_ner_data_dir, data_type, 'tags.txt')
    sents, tags = [], []
    for s in sent_txts:
        sent, tag = ner_data[s]
        sents.append(sent)
        tags.append(tag)
    with open(sentences_file, 'w') as writer_sent, open(tags_file, 'w') as writer_tag:
        for sent, tag_seq in zip(sents, tags):
            writer_sent.write(sent + '\n')
            writer_tag.write(tag_seq + '\n')
    print("ner data_type: {}, {}".format(data_type, len(sents)))


class SimSent(object):
    def __init__(self, sentences: List):
        self.sentences = sentences

    def delete_distance(self, a, b):
        """ 删除距离，最大子串
        https://stackoverflow.com/questions/41275345/deletion-distance-between-words
        """
        diff_cnt = len([s for s in difflib.ndiff(a, b) if s[0] != ' '])
        return diff_cnt

    def find_sim_sents(self, s, n=3):
        distances = [self.delete_distance(s, _s) for _s in self.sentences]
        indices = np.array(distances).argsort()[:n]
        sim_sents = [self.sentences[i] for i in indices]
        return sim_sents


def create_joint_data():
    PrepareJointData().get_joint_data()
