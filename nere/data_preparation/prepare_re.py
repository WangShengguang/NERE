import os
import random
import re
from collections import defaultdict
from pathlib import Path

import intervals as I

from config import Config

# relation_labels = { '发生事故': 'Accident',
#                     '驾驶': 'Drive',
#                     '搭乘': 'Take',
#                     '所有': 'Own',
#                     '实施': 'Implement',
#                     '承担': 'Undertake',
#                     '遭受': 'Suffer',
#                     '投保': 'Insure',
#                     '未投保': 'Not-insure',
#                     '具备': 'Have',
#                     '指代': 'Refer'}

OTHER_LABER = '其他'
MAX_LEN = 400
MIN_LEN = 8  # If the length is less than MIN_LEN, let go
distance_statistics = {'[0,50]': 0,
                       '(50,100]': 0,
                       '(100-150]': 0,
                       '(150-200]': 0,
                       '(200-250]': 0,
                       '(250-300]': 0,
                       '(300-350]': 0,
                       '(350-400]': 0,
                       '(400-450]': 0,
                       '(450-500]': 0,
                       '(500-)': 0}

filters = ['具备', '指代', '未投保']
ent_filters = ['其他违法行为', '抗辩事由']

pattern_insur_1 = re.compile('(交强|强制)')
pattern_insur_2 = re.compile('(三者|第三|三责|^商业)')

pattern_motor_1 = re.compile('^小轿车|^事故车|^车辆')
pattern_motor_2 = re.compile('^摩托车')
pattern_motor_3 = re.compile('^电动车')
pattern_motor_4 = re.compile('^自行车')

pattern_para = re.compile('\n\n')  # Paragraph separator
pattern_sent = re.compile('\n')  # Sentence separator, period (。) is identifier
pattern_blank = re.compile('[ \t]')


def get_negative_triple_span(text, e1, e2):
    e1_pos_list = [(match.start(), match.end()) for match in re.finditer(re.escape(e1), text)]
    e2_pos_list = [(match.start(), match.end()) for match in re.finditer(re.escape(e2), text)]
    min_dist = MAX_LEN
    e1_pos_res, e2_pos_res = None, None
    for e1_pos in e1_pos_list:
        for e2_pos in e2_pos_list:
            min_start = min(e1_pos[0], e2_pos[0])
            max_end = max(e1_pos[1], e2_pos[1])
            cur_dist = max_end - min_start
            if cur_dist < min_dist:
                min_dist = cur_dist
                e1_pos_res, e2_pos_res = e1_pos, e2_pos
    if e1_pos_res and e2_pos_res:
        span = get_gold_triple_span(text, (e1_pos_res[0], e1_pos_res[1]), (e2_pos_res[0], e2_pos_res[1]))
    else:
        span = ''
    return span


def get_gold_triple_span(text, e1_pos, e2_pos):
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
    min_punc_pos = get_nearest_punc(text, '；;', min_start, 'left')
    max_punc_pos = get_nearest_punc(text, '；;', max_end, 'right')
    span = ''
    # "；" <-> "；"
    if max_punc_pos < min_punc_pos:
        span = ''
    elif max_punc_pos - min_punc_pos <= MAX_LEN:
        span = text[min_punc_pos: max_punc_pos + 1]
    else:
        # "；" <-> "，"
        max_punc_pos = get_nearest_punc(text, '；;，,', max_end, 'right')
        if max_punc_pos < min_punc_pos:
            span = ''
        elif max_punc_pos - min_punc_pos <= MAX_LEN:
            span = text[min_punc_pos: max_punc_pos + 1]
        else:
            # "，" <-> "，"
            min_punc_pos = get_nearest_punc(text, '；;，,', min_start, 'left')
            if max_punc_pos < min_punc_pos:
                span = ''
            elif max_punc_pos - min_punc_pos <= MAX_LEN:
                span = text[min_punc_pos: max_punc_pos + 1]
            else:
                span = ''
    return span


def get_nearest_punc(text, punc, ent_pos, direction):
    punc_positions = [match.start() for match in re.finditer('[{}]'.format(punc), text)]
    # search from left
    if direction == 'left':
        pos = 0
        for punc_pos in punc_positions:
            if punc_pos < ent_pos:
                pos = punc_pos + 1
            else:
                break
        pos = pos + 1 if pos > 0 else 0
    elif direction == 'right':
        pos = len(text)
        for punc_pos in punc_positions[::-1]:
            if punc_pos > ent_pos:
                pos = punc_pos
            else:
                break
    else:
        raise ValueError("direction must be one of 'left' / 'right' ")
    return pos


def get_data(txt_file, ann_file, entity_label_pairs, samples_statistics):
    """Get data from a judgment"""
    # Get annotated entities
    triples = {}  # ((e1_label, e1_pos1, e1_pos2), (e2_label, e2_pos1, e2_pos2)) -> relation_label
    entities = {}  # ID -> (label, pos1, pos2)
    with open(ann_file, 'r', encoding="utf-8") as reader_ann:
        for line in reader_ann:
            line = line.strip()
            if line.startswith('T'):
                splits = line.split('\t')
                ID = splits[0]  # e.g. T19
                if ';' in splits[1]:
                    temp = splits[1].replace(';', ' ').split(' ')
                else:
                    temp = splits[1].split(' ')
                pos1, pos2 = int(temp[1]), int(temp[2])
                label = temp[0]
                if label not in filters:
                    entities[ID] = (label, pos1, pos2)  # T2 -> (自然人主体, 35, 38)
    with open(ann_file, 'r', encoding="utf-8") as reader_ann:
        for line in reader_ann:
            line = line.strip()
            if line.startswith('R'):
                splits = line.split('\t')[1].split(' ')
                label = splits[0]
                samples_statistics[label] += 1
                e1_id = splits[1].split(':')[1]  # e.g. T5
                e1_label = entities[e1_id][0]
                e2_id = splits[2].split(':')[1]
                e2_label = entities[e2_id][0]
                if label not in filters and e1_label not in ent_filters and e2_label not in ent_filters:
                    triples[(entities[e1_id], entities[e2_id])] = label

    # distance statistics between two entities
    for triple in triples:
        e1_label, e1_start, e1_end = triple[0]
        e2_label, e2_start, e2_end = triple[1]

        distance = abs(e1_start - e2_start)
        if distance <= 50:
            distance_statistics['[0,50]'] += 1
        elif distance <= 100:
            distance_statistics['(50,100]'] += 1
        elif distance <= 150:
            distance_statistics['(100-150]'] += 1
        elif distance <= 200:
            distance_statistics['(150-200]'] += 1
        elif distance <= 250:
            distance_statistics['(200-250]'] += 1
        elif distance <= 300:
            distance_statistics['(250-300]'] += 1
        elif distance <= 350:
            distance_statistics['(300-350]'] += 1
        elif distance <= 400:
            distance_statistics['(350-400]'] += 1
        elif distance <= 450:
            distance_statistics['(400-450]'] += 1
        elif distance <= 500:
            distance_statistics['(450-500]'] += 1
        else:
            distance_statistics['(500-)'] += 1

    sent_pos_list = []
    case_text = ""
    with open(txt_file, 'r', encoding="utf-8") as f:
        sent_start = 0
        for line in f:
            case_text += line
            sent_end = sent_start + len(line)
            sent_pos_list.append((sent_start, sent_end - 1))  # closed loop []
            sent_start = sent_end
    data_list = []
    for sent_pos in sent_pos_list:
        sent_start, sent_end = sent_pos[0], sent_pos[1]
        sent_text = case_text[sent_start:sent_end]
        sent_length = len(sent_text)
        if sent_length <= MIN_LEN:
            continue
        sent_interval = I.closedopen(sent_start, sent_end)

        flag_insur_1 = False
        flag_insur_2 = False
        flag_motor_1_1 = False
        flag_motor_1_2 = False
        flag_motor_2_1 = False
        flag_motor_2_2 = False
        flag_motor_3_1 = False
        flag_motor_3_2 = False
        flag_motor_4_1 = False
        flag_motor_4_2 = False

        entity_dict = {}
        entity_pais = []  # The actual pair of entities in the sentence
        for triple in triples:
            e1_label, e1_start, e1_end = triple[0]
            e2_label, e2_start, e2_end = triple[1]

            if e1_label in ent_filters or e2_label in ent_filters:
                continue

            e1_interval = I.closedopen(e1_start, e1_end)
            e2_interval = I.closedopen(e2_start, e2_end)
            if e1_interval in sent_interval and e2_interval in sent_interval:
                e1, e2 = case_text[e1_start:e1_end], case_text[e2_start:e2_end]
                # if "9167轻型厢式货车" in e1 + e2 or "×号/人民保" in e1 + e2: TODO Debug(wsg,标注问题)
                #     import ipdb
                #     ipdb.set_trace()
                if e1 not in entity_dict:
                    entity_dict[e1] = e1_label
                if e2 not in entity_dict:
                    entity_dict[e2] = e2_label
                entity_pais.append((e1, e2))
                if e2_label == '保险类别':
                    if pattern_insur_1.search(e2):
                        flag_insur_1 = True
                    elif pattern_insur_2.search(e2):
                        flag_insur_2 = True
                if pattern_motor_1.search(e1):
                    flag_motor_1_1 = True
                elif pattern_motor_1.search(e2):
                    flag_motor_1_2 = True

                elif pattern_motor_2.search(e1):
                    flag_motor_2_1 = True
                elif pattern_motor_2.search(e2):
                    flag_motor_2_2 = True

                elif pattern_motor_3.search(e1):
                    flag_motor_3_1 = True
                elif pattern_motor_3.search(e2):
                    flag_motor_3_2 = True

                elif pattern_motor_4.search(e1):
                    flag_motor_4_1 = True
                elif pattern_motor_4.search(e2):
                    flag_motor_4_2 = True

                rel_label = triples[triple]
                if sent_length <= MAX_LEN:
                    data = (rel_label, e1_label, e2_label, e1, e2, sent_text)  # sentence features
                    if data not in data_list:
                        data_list.append(data)
                else:
                    # the entity position relative to the sentence
                    span = get_gold_triple_span(sent_text, (e1_start - sent_start, e1_end - sent_start),
                                                (e2_start - sent_start, e2_end - sent_start))
                    if len(span) > MIN_LEN:
                        data = (rel_label, e1_label, e2_label, e1, e2, span)
                        if data not in data_list:
                            data_list.append(data)
        # Negative sampling
        for e1 in entity_dict:
            e1_label = entity_dict[e1]
            # There is no relationship between e1 and e2 at all
            if e1_label not in entity_label_pairs:
                continue
            for e2 in entity_dict:
                # e1 and e2 are the same entity
                if e1 == e2:
                    continue
                # (e1, e2) is a pair of real entities
                if (e1, e2) in entity_pais:
                    continue

                e2_label = entity_dict[e2]
                # There is no relationship between e1 and e2 at all
                if e2_label not in entity_label_pairs[e1_label]:
                    continue

                if flag_insur_1 == True and pattern_insur_1.search(e2):  # e_label -> Insurcece
                    continue
                if flag_insur_2 == True and pattern_insur_2.search(e2):
                    continue

                if flag_motor_1_1 == True and pattern_motor_1.search(e1):
                    continue
                if flag_motor_1_2 == True and pattern_motor_1.search(e2):
                    continue

                if flag_motor_2_1 == True and pattern_motor_2.search(e1):
                    continue
                if flag_motor_2_2 == True and pattern_motor_2.search(e2):
                    continue

                if flag_motor_3_1 == True and pattern_motor_3.search(e1):
                    continue
                if flag_motor_3_2 == True and pattern_motor_3.search(e2):
                    continue

                if flag_motor_4_1 == True and pattern_motor_4.search(e1):
                    continue
                if flag_motor_4_2 == True and pattern_motor_4.search(e2):
                    continue

                rel_label = OTHER_LABER  # Other relation label
                data = None
                if sent_length <= MAX_LEN:
                    data = (rel_label, e1_label, e2_label, e1, e2, sent_text)
                else:
                    span = get_negative_triple_span(sent_text, e1, e2)

                    if len(span) > MIN_LEN:
                        data = (rel_label, e1_label, e2_label, e1, e2, span)
                if data is not None and random.random() >= 0.5:
                    if data not in data_list:
                        data_list.append(data)

    return data_list


def prepare_data():
    ent_label_pairs = {}  # {e1_label: [e2_1_label, e2_2_label]}
    with open(Config.predefined_file, 'r') as reader:
        for line in reader:
            splits = line.strip().split(' ')
            if len(splits) < 3:
                continue
            label = splits[0]  # for relation
            e1_label = splits[1].split(':')[-1].strip(',').strip()
            e2_label = splits[2].split(':')[-1].strip()
            if e1_label in ent_filters or e2_label in ent_filters:
                continue
            if label in filters:
                continue
            if e1_label not in ent_label_pairs:
                ent_label_pairs[e1_label] = [e2_label]
            elif e2_label not in ent_label_pairs[e1_label]:
                ent_label_pairs[e1_label].append(e2_label)
    # print('entity_label_pairs: {}'.format(entity_label_pairs))
    # relation samples_statistics
    samples_statistics = defaultdict(int)  # samples_statistics based on `.ann` file
    dataset = []  # for doc-level
    for ann_file in Path(Config.annotation_data_dir).rglob("*.ann"):
        txt_file = str(ann_file.with_suffix('.txt'))
        # data obtained from a case file, including negative examples
        data_li = get_data(txt_file, ann_file, ent_label_pairs, samples_statistics)
        data = ["\t".join((rel_label, e1_label, e2_label, e1, e2, sent_text))
                for (rel_label, e1_label, e2_label, e1, e2, sent_text) in data_li]
        dataset.append(data)
    all_case_num = len(dataset)
    train_count = int(all_case_num * 0.8)
    valid_count = int(all_case_num * 0.1)
    test_count = all_case_num - train_count - valid_count

    order = list(range(all_case_num))
    random.shuffle(order)
    train_dataset = [dataset[idx] for idx in order[:train_count]]
    valid_dataset = [dataset[idx] for idx in order[train_count:train_count + valid_count]]
    test_dataset = [dataset[idx] for idx in order[train_count + valid_count:]]

    train_samples_count = write_to_file(os.path.join(Config.re_data_dir, 'train.txt'), train_dataset)
    valid_samples_count = write_to_file(os.path.join(Config.re_data_dir, 'valid.txt'), valid_dataset)
    test_samples_count = write_to_file(os.path.join(Config.re_data_dir, 'test.txt'), test_dataset)

    print('\nall cases num: {}'.format(all_case_num))
    print("train cases: {}, samples: {}".format(train_count, train_samples_count))
    print("valid cases: {}, samples: {}".format(valid_count, valid_samples_count))
    print("test cases: {}, samples: {}".format(test_count, test_samples_count))
    return dict(samples_statistics)


def write_to_file(dataset_path, dataset):
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    print(dataset_path)
    sample_num = 0
    with open(dataset_path, 'w', encoding="utf-8") as writer:
        for data_doc in dataset:
            for data in data_doc:
                writer.write(data + '\n')
                sample_num += 1
    return sample_num


def build_labels():
    """build entity label and relation label"""
    rel_labels = set()
    ent_labels = set()
    data_types = ['train', "valid", 'test']
    for data_type in data_types:
        dataset_path = os.path.join(Config.re_data_dir, '{}.txt'.format(data_type))
        with open(dataset_path, 'r', encoding="utf-8") as reader:
            for line in reader:
                splits = line.strip().split('\t')
                rel_label = splits[0]
                ent1_label, ent2_label = splits[1], splits[2]
                rel_labels.add(rel_label)
                ent_labels.update([ent1_label, ent2_label])

    rel_label_file = os.path.join(Config.re_data_dir, 'rel_labels.txt')
    ent2_label_file = os.path.join(Config.re_data_dir, 'ent_labels.txt')
    with open(rel_label_file, 'w') as writer_rel, open(ent2_label_file, 'w') as writer_ent:
        writer_rel.write('\n'.join(rel_labels) + '\n')
        writer_ent.write('\n'.join(ent_labels) + '\n')
    return rel_labels, ent_labels


def get_actual_statistics():
    samples_num = 0
    samples_statistics = defaultdict(int)
    data_types = ['train', "valid", 'test']
    for data_type in data_types:
        dataset_path = os.path.join(Config.re_data_dir, '{}.txt'.format(data_type))
        with open(dataset_path, 'r') as f:
            for line in f:
                label = line.strip().split('\t')[0]
                samples_num += 1
                samples_statistics[label] += 1
    return samples_num, dict(samples_statistics)


def draw_histogram(samples_statistics):
    """Draw according to statistical results"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

    items = sorted(samples_statistics.items(), key=lambda x: x[1], reverse=True)
    print('sorted samples_statistics: {}'.format(items))
    labels = [item[0] for item in items]
    num = [item[1] for item in items]
    plt.bar(labels, num)
    plt.xticks(rotation=0)
    plt.xlabel('Label')
    plt.ylabel('Number')
    axes = plt.gca()
    axes.yaxis.grid(linestyle='--')
    for l, n in zip(labels, num):
        plt.text(l, n + 0.05, '%.0f' % n, ha='center', va='bottom')
    plt.show()


def create_re_data():
    # Prepare data and perform relation samples_statistics
    samples_statistics = prepare_data()
    print('\nsamples_statistics: {}'.format(samples_statistics))

    # Statistics based on actual sentence-level
    actual_samples_num, actual_samples_statistics = get_actual_statistics()
    print('\nactual_samples_num: {}'.format(actual_samples_num))
    print('actual_samples_statistics: {}'.format(actual_samples_statistics))

    # get labels set
    rel_labels, ent_labels = build_labels()

    # draw histogram
    # draw_histogram(distance_statistics)
    # draw_histogram(statistics)
    # draw_histogram(samples_statistics_actual)
