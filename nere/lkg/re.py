"""Relation Extraction"""

import os
import re

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertConfig

from nere.config import Config
from nere.evaluator import Predictor
from nere.re.torch_models import BERTMultitask, BERTSoftmax


def get_nearest_punc(sent_tokens, punc, ent_pos, direction):
    # search from left
    pattern = re.compile('[{}]'.format(punc))
    pos = ent_pos
    if direction == 'left':
        while pos > 0:
            if pattern.search(sent_tokens[pos]):
                break
            pos -= 1
        pos = pos + 1 if pos > 0 else 0
    # search from right
    elif direction == 'right':
        sent_length = len(sent_tokens)
        while pos < sent_length - 1:
            if pattern.search(sent_tokens[pos]):
                break
            pos += 1
    else:
        raise ValueError("direction must be one of 'left' / 'right' ")
    return pos


def find_sub_list(all_list, sub_list, verbose=False):
    match_indices = []
    all_len, sub_len = len(all_list), len(sub_list)
    starts = [i for i, ele in enumerate(all_list) if ele == sub_list[0]]
    for start in starts:
        end = start + sub_len
        if end <= all_len and all_list[start: end] == sub_list:
            if verbose:
                match_indices.append(list(range(start, end)))
            else:
                match_indices.append((start, end))
    return match_indices


class RelationExtraction(Predictor):
    def __init__(self, model_name, max_len=Config.max_sequence_len):
        super().__init__(framework="torch")
        self.max_len = max_len
        self.other_rel_label_id = self.data_helper.other_rel_label_id
        # R E
        self.label2idx = self.data_helper.rel_label2id  # for encoding
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}  # for decoding
        # NER
        self.ent_label2idx = self.data_helper.ent_label2id  # for encoding
        self.ent_label_pairs = self.load_ent_label_pairs()

        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        num_rel_tags = len(self.data_helper.rel_label2id)
        config = BertConfig.from_json_file(Config.bert_config_path)
        if model_name == 'BERTSoftmax':
            model = BERTSoftmax(config, num_labels=num_rel_tags)
        elif model_name == 'BERTMultitask':
            model = BERTMultitask(config, num_labels=num_rel_tags)
        else:
            raise ValueError(model_name)
        model_dir = os.path.join(Config.torch_ckpt_dir, "re")
        model_path = os.path.join(model_dir, model_name + ".bin")
        model.load_state_dict(torch.load(model_path))  # 断点续训
        return model

    def parse(self, data):
        # for d, e in data:
        #     print('{}\n{}'.format(''.join(d), e))
        re_data = self.load_data(data)
        re_sults = self.get_re_results(re_data)
        return re_sults

    def get_re_results(self, all_data):
        results = set()
        for data in all_data:
            # print('{}\n'.format(data))
            e1_label, e2_label, e1_tokens, e2_tokens = data[0], data[1], data[2], data[3]
            e1_mention = ''.join(e1_tokens).replace('#', '')
            e2_mention = ''.join(e2_tokens).replace('#', '')
            tensor_data = self.to_tensor_data(data)
            pred_idx = self.model(tensor_data)
            pred_idx = pred_idx.numpy()[0]
            if pred_idx == self.other_rel_label_id:
                continue
            label = self.idx2label[pred_idx]
            results.add(((e1_mention, e1_label), (e2_mention, e2_label), label))
        return results

    def load_ent_label_pairs(self):
        predefined_file = os.path.join(Config.torch_ckpt_dir, "re", 'predefined.txt')
        ent_label_pairs = {}  # {e1_label: [e21_label, e21_label]}
        with open(predefined_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                splits = line.strip().split(' ')
                if len(splits) < 3:
                    continue
                e1_label = splits[1].split(':')[-1].strip(',')
                e2_label = splits[2].split(':')[-1].strip(',')
                if e1_label not in ent_label_pairs:
                    ent_label_pairs[e1_label] = [e2_label]
                elif e2_label not in ent_label_pairs[e1_label]:
                    ent_label_pairs[e1_label].append(e2_label)
        return ent_label_pairs

    def load_data(self, data):
        re_data = []
        for sent_tokens, sent_entities in data:
            sent_length = len(sent_tokens)
            for e1 in sent_entities:
                e1_tokens, e1_label = e1['ent_tokens'], e1['ent_label']
                if e1_label not in self.ent_label_pairs:
                    continue
                for e2 in sent_entities:
                    if e1 == e2:
                        continue
                    e2_tokens, e2_label = e2['ent_tokens'], e2['ent_label']
                    if e2_label not in self.ent_label_pairs[e1_label]:
                        continue
                    if sent_length <= self.max_len:
                        new_data = [e1_label, e2_label, e1_tokens, e2_tokens, sent_tokens]
                        re_data.append(new_data)
                    else:
                        span_tokens = self.get_nearest_triple_span(sent_tokens, e1_tokens, e2_tokens)
                        if len(span_tokens) > 5:
                            new_data = (e1_label, e2_label, e1_tokens, e2_tokens, span_tokens)
                            re_data.append(new_data)
        return re_data

    def get_nearest_triple_span(self, sent_tokens, e1_tokens, e2_tokens):
        e1_pos_list = find_sub_list(sent_tokens, e1_tokens)
        e2_pos_list = find_sub_list(sent_tokens, e2_tokens)
        e1_pos_res, e2_pos_res = None, None
        min_dist = self.max_len
        for e1_pos in e1_pos_list:
            for e2_pos in e2_pos_list:
                min_start = min(e1_pos[0], e2_pos[0])
                max_end = max(e1_pos[1], e2_pos[1])
                cur_dist = max_end - min_start
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    e1_pos_res, e2_pos_res = e1_pos, e2_pos
        if e1_pos_res and e2_pos_res:
            span = self.get_gold_triple_span(sent_tokens, e1_pos_res, e2_pos_res)
        else:
            span = ''
        return span

    def get_gold_triple_span(self, sent_tokens, e1_pos, e2_pos):
        """Long sentence processing due to maximum length limit (MAX_LEN)
        Args:
            text: (str)
            punc: (str)
            e1_pos: (tuple)
            e2_pos: (e2_Pos)
        """
        e1_start, e1_end = e1_pos[0], e1_pos[1]
        e2_start, e2_end = e2_pos[0], e2_pos[1]
        min_start = min(e1_start, e2_start)
        max_end = max(e1_end, e2_end) - 1
        min_punc_pos = get_nearest_punc(sent_tokens, '；;', min_start, 'left')
        max_punc_pos = get_nearest_punc(sent_tokens, '；;', max_end, 'right')
        span = ''
        # "；" <-> "；"
        if max_punc_pos < min_punc_pos:
            span = ''
        elif max_punc_pos - min_punc_pos <= self.max_len:
            span = sent_tokens[min_punc_pos: max_punc_pos + 1]
        else:
            # "；" <-> "，"
            max_punc_pos = get_nearest_punc(sent_tokens, '；;，,', max_end, 'right')
            if max_punc_pos < min_punc_pos:
                span = ''
            elif max_punc_pos - min_punc_pos <= self.max_len:
                span = sent_tokens[min_punc_pos: max_punc_pos + 1]
            else:
                # "，" <-> "，"
                min_punc_pos = get_nearest_punc(sent_tokens, '；;，,', min_start, 'left')
                if max_punc_pos < min_punc_pos:
                    span = ''
                elif max_punc_pos - min_punc_pos <= self.max_len:
                    span = sent_tokens[min_punc_pos: max_punc_pos + 1]
                else:
                    span = ''
        return span

    def to_tensor_data(self, data):
        e1_label, e2_label, e1_tokens, e2_tokens, sent_tokens = data
        ent_labels = [self.ent_label2idx[e1_label], self.ent_label2idx[e2_label]]
        ent_labels = torch.tensor(ent_labels, dtype=torch.long).unsqueeze(0)
        e1_indices = find_sub_list(sent_tokens, e1_tokens, verbose=True)
        e2_indices = find_sub_list(sent_tokens, e2_tokens, verbose=True)

        e1_masks = np.zeros(len(sent_tokens))
        for indices in e1_indices:
            e1_masks[indices] = 1
        e1_masks = torch.tensor(e1_masks, dtype=torch.uint8).unsqueeze(0)
        e2_masks = np.zeros(len(sent_tokens))
        for indices in e2_indices:
            e2_masks[indices] = 1
        e2_masks = torch.tensor(e2_masks, dtype=torch.uint8).unsqueeze(0)

        sents = self.tokenizer.convert_tokens_to_ids(sent_tokens)
        sents = torch.tensor(sents, dtype=torch.long).unsqueeze(0)
        new_data = {}
        new_data['ent_labels'] = ent_labels
        new_data['e1_masks'] = e1_masks
        new_data['e2_masks'] = e2_masks
        new_data['sents'] = sents
        return new_data
