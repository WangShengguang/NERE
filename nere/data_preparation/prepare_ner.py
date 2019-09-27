import os
import random
import re

import intervals as I
from pytorch_pretrained_bert import BertTokenizer

from config import Config

entity_label2abbr = {'自然人主体': 'NP',
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
# filters = ['抗辩事由', '违反道路交通信号灯', '饮酒后驾驶', '醉酒驾驶', '超速', '违法变更车道', '未取得驾驶资格', '超载', '不避让行人', '行人未走人行横道或过街设施', '其他违法行为']
filters = ['抗辩事由', '其他违法行为']

# entity statistics
statistics = {}


class PrepareNer(object):
    def __init__(self, max_len=Config.max_sequence_len):
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)
        self.max_len = max_len

    def get_data(self, txt_file, ann_file):
        """Get a piece of data from a annotated file"""
        # Get annotated entity dict
        entities = {}
        with open(ann_file, 'r') as reader_ann:
            for line in reader_ann:
                line = line.strip()
                if line.startswith('T'):
                    splits = line.split('\t')
                    label_pos = splits[1]
                    ent_label = label_pos.split(' ')[0]
                    ent = splits[2]
                    if ';' in label_pos:
                        continue
                    if ent_label not in statistics:
                        statistics[ent_label] = 0
                    else:
                        statistics[ent_label] += 1

                    # Cross line annotation, example:  T49   其他违法行为 1320 1372;1374 1417
                    if ent not in entities and ent_label not in filters:
                        entities[ent] = ent_label
        entity_items = []
        for ent, ent_label in entities.items():
            ent_tokens = self.tokenizer.tokenize(ent)
            entity_items.append((ent_tokens, ent_label))
        entity_items = sorted(entity_items, key=lambda x: len(x[0]), reverse=True)

        # The length of the segmentation fragment for each long sentence, no more than 2
        sents = self.get_fact(txt_file)
        sents = [self.tokenizer.tokenize(sent.strip()) for sent in sents if len(sent.strip()) > 5]
        tags = [['O'] * len(sent) for sent in sents]
        for i in range(len(sents)):
            assert len(sents[i]) == len(tags[i])

        for i, sent_tokens in enumerate(sents):
            intervals = []
            for ent_tokens, ent_label in entity_items:
                ent_tag = entity_label2abbr[ent_label]
                positions = self.find_sub_list(sent_tokens, ent_tokens)
                for pos in positions:
                    interval = I.closed(pos[0], pos[-1])
                    overlap = self.is_overlap(intervals, interval)
                    if not overlap:
                        tags[i][pos[0]] = 'B-' + ent_tag
                        if len(pos) > 1:
                            for p in pos[1:]:
                                tags[i][p] = 'I-' + ent_tag
                        intervals.append(interval)
        return sents, tags

    def get_fact(self, txt_file):
        with open(txt_file, 'r') as reader_txt:
            txt_text = reader_txt.read()
        facts = txt_text.split('\n')  # fact list
        new_facts = []  # The length of the segmentation fragment for each long sentence, no more than 2
        for i, fact in enumerate(facts):
            if len(fact) <= 5:
                continue
            clauses_period = self.split_text(fact, '。')
            clauses_period = list(filter(lambda x: len(x) > 5, clauses_period))
            for clause in clauses_period:
                if len(clause) <= self.max_len:
                    new_facts.append(clause)
                else:
                    clauses_semi = self.split_text(clause, '；;')
                    clauses_semi = list(filter(lambda x: len(x) > 5, clauses_semi))
                    len_semi = len(clauses_semi)
                    if len_semi == 1:
                        clauses_comma = self.split_text(clause, '，,')
                        clauses_comma = list(filter(lambda x: len(x) > 5, clauses_comma))
                        len_comma = len(clauses_comma)
                        if len_comma <= 2:
                            new_facts.extend(clauses_comma)
                        else:  # len_comma >= 3
                            mid = len_comma // 2
                            if len_comma % 2 == 1:  # The number of elements is odd
                                if clauses_comma[0] < clauses_comma[-1]:
                                    mid = mid + 1
                            new_facts.extend(clauses_comma[:mid])
                            new_facts.extend(clauses_comma[mid:])
                    elif len_semi == 2:
                        new_facts.extend(clauses_semi)
                    else:
                        mid = len_semi // 2
                        if len_semi % 2 == 1:
                            if clauses_semi[0] < clauses_semi[-1]:
                                mid = mid + 1
                        new_facts.extend(clauses_semi[:mid])
                        new_facts.extend(clauses_semi[mid:])
        return new_facts

    def split_text(self, text, punc):
        # (*x) - Retain the punctuation separator
        splits = re.split('([{}])'.format(punc), text)
        # Put the separator after the sentence if available
        clauses = [''.join(split).strip() for split in zip(splits[0::2], splits[1::2])]
        if not text.endswith(punc):
            clauses.append(splits[-1].strip())
        return clauses

    def find_sub_list(self, all_list, sub_list):
        match_indices = []
        all_len, sub_len = len(all_list), len(sub_list)
        starts = [i for i, ele in enumerate(all_list) if ele == sub_list[0]]
        for start in starts:
            end = start + sub_len
            if end <= all_len and all_list[start: end] == sub_list:
                match_indices.append(list(range(start, end)))
        return match_indices

    def is_overlap(self, intervals, interval):
        flag = False
        for i in intervals:
            if interval.overlaps(i):
                flag = True
                break
        return flag


def prepare_data():
    """Data processing and data partitioning"""
    prapare_ner = PrepareNer()
    tasks = os.listdir(Config.annotation_data_dir)
    task_dirs = [os.path.join(Config.annotation_data_dir, task) for task in tasks]
    task_dirs = filter(lambda x: os.path.isdir(x), task_dirs)

    dataset = []
    for task_dir in task_dirs:
        files = os.listdir(task_dir)
        file_paths = [os.path.join(task_dir, file) for file in files]
        file_paths = list(filter(lambda x: os.path.isfile(x) and x.endswith('.txt'), file_paths))
        for file_txt in file_paths:
            file_ann = file_txt.replace('txt', 'ann')
            sents, tags = prapare_ner.get_data(file_txt, file_ann)
            dataset.append((sents, tags))
    print('data size: {}'.format(len(dataset)))
    order = list(range(len(dataset)))
    random.shuffle(order)
    train_dataset = [dataset[idx] for idx in order[:430]]
    val_dataset = [dataset[idx] for idx in order[430:485]]
    test_dataset = [dataset[idx] for idx in order[485:]]

    write_to_file(os.path.join(Config.ner_data_dir, 'train'), train_dataset)
    write_to_file(os.path.join(Config.ner_data_dir, 'val'), val_dataset)
    write_to_file(os.path.join(Config.ner_data_dir, 'test'), test_dataset)


def write_to_file(data_dir, dataset):
    os.makedirs(data_dir, exist_ok=True)
    sentences_file = os.path.join(data_dir, 'sentences.txt')
    tags_file = os.path.join(data_dir, 'tags.txt')
    with open(sentences_file, 'w') as writer_sent, open(tags_file, 'w') as writer_tag:
        for sents, tags in dataset:
            for sent, tag_seq in zip(sents, tags):
                writer_sent.write(' '.join(sent) + '\n')
                writer_tag.write(' '.join(tag_seq) + '\n')


def build_tags():
    tags = set()
    data_types = ['train', 'val', 'test']
    for data_type in data_types:
        data_path = os.path.join(Config.ner_data_dir, data_type, 'tags.txt')
        with open(data_path, 'r') as reader:
            for line in reader:
                line = line.strip()
                tags.update(list(line.strip().split(' ')))
    tags_data_path = os.path.join(Config.ner_data_dir, 'tags.txt')
    with open(tags_data_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tags))
    return tags


def draw_histogram(statistics):
    """Draw according to statistical results"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

    items = sorted(statistics.items(), key=lambda x: x[1], reverse=True)
    print('sorted statistics: {}'.format(items))
    labels = [item[0] for item in items]
    num = [item[1] for item in items]
    plt.bar(labels, num)
    plt.xticks(rotation=300)
    plt.xlabel('Label')
    plt.ylabel('Number')
    axes = plt.gca()
    axes.yaxis.grid(linestyle='--')
    for l, n in zip(labels, num):
        plt.text(l, n + 0.05, '%.0f' % n, ha='center', va='bottom')
    plt.show()


def create_ner_data():
    prepare_data()
    print('statistics: {}'.format(statistics))
    # build tag set
    build_tags()
