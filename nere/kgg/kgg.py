"""Automatic generation of law knowledge graph"""

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import Config
from nere.kgg.neo4j_module import ImportNeo4j
from nere.kgg.ner import EntityRecognition
from nere.kgg.preprocessor import Preprocessing
from nere.kgg.re import RelationExtraction
from nere.kgg.utils import create_triple


class KGG(object):
    """Automatic generation of knowledge map for a case judgement; KGG
    Args:
        case_file: case file
        output_file: output file
    """

    def __init__(self, ner_model_name, re_model_name):
        self.ner = EntityRecognition(ner_model_name)
        self.re = RelationExtraction(re_model_name)
        self.preprocessor = Preprocessing()
        self.neo4j = ImportNeo4j()

    def parse(self, case_file):
        """Give a case file and parse it"""
        case_id, basic_fact, fact_text = self.preprocessor.process(case_file)
        if basic_fact == None:
            print("文件{}不存在案情事实".format(case_file))
            return False, None, None
        ner_result = self.ner.parse(fact_text)  # shit
        re_result = self.re.parse(ner_result)
        # new_result = self.suffer(basic_fact, re_result)
        new_result = create_triple(basic_fact, re_result)
        print('case_id: {}\n'.format(case_id))
        print('basic_fact: {}\n'.format(basic_fact))
        print('fact_text: {}\n'.format(fact_text))
        print('ner_result: {}\n'.format(ner_result))
        print('re_result: {}\n'.format(re_result))
        print('new_result: {}\n'.format(new_result))
        return True, case_id, new_result
        # self.neo4j.import_data(basic_fact, new_result)
        # return True

    def suffer(self, basic_fact, re_result):
        """去除被告遭受医疗费的情况"""
        new_result = set()
        defens = basic_fact['被告']
        defens = [defen['名字'] for defen in defens]
        for res in re_result:
            e1, e2, rel = res[0], res[1], res[2]
            e1_mention, e1_label = e1[0], e1[1]
            e2_mention, e2_label = e2[0], e2[1]
            if e1_mention in defens and rel == '遭受' and (e2_label == '人身损害赔偿项目' or e2_label == '财产损失赔偿项目'):
                continue
            new_result.add(res)
        return list(new_result)


def get_triple_result(raw_data=Config.kgg_raw_data_dir):
    ner_model = "BERTCRF"
    re_model = "BERTMultitask"
    kgg = KGG(ner_model, re_model)
    start_time = time.time()
    err_file = []
    long_sentence = 0
    result = {}
    triples = set()
    files = list(Path(raw_data).glob("*.txt"))
    for case_file in tqdm(files, desc="get_triple_result"):
        case_file_name = case_file.name
        # print(case_file)
        try:
            flag, case_id, res = kgg.parse(case_file)
            if flag:
                result[case_id] = list(res)
                for tep in res:
                    triples.add(tep)
            else:
                err_file.append(case_file_name)
        except KeyboardInterrupt:
            exit(1)
        except Exception as e:
            logging.error(e)
            err_file.append(case_file_name)
            long_sentence += 1
            print("句子长度超过512：", case_file_name)
    end_time = time.time()
    print('running time: {}'.format(end_time - start_time))
    print("case num: {}".format(len(files)))
    print('Average running time: {}'.format((end_time - start_time) / len(files)))
    print("不存在案情事实的文件有{}个".format(len(err_file)))
    print(err_file)
    print("句子长度超过512：{}".format(long_sentence))
    with open(Config.result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    triples = list(triples)
    with open(Config.triples_result_file, 'w', encoding='utf-8') as f:
        for item in triples:
            if len(item) < 3:
                continue
            f.write(f"{item[0]}\t{item[1]}\t{item[2]}\n")


def gat_cate_ids():
    file_list = ['311.txt', '376.txt', '105.txt', '271.txt', '441 (80).txt', '089.txt', '360.txt', '441 (248).txt',
                 '441 (150).txt', '310.txt', '427.txt', '398.txt', '042.txt', '441 (232).txt', '184.txt', '268.txt',
                 '441 (67).txt', '397.txt', '097.txt', '437.txt', '356.txt', '139.txt', '206.txt', '368.txt', '194.txt',
                 '101.txt', '078.txt', '441 (96).txt', '353.txt', '441 (163).txt', '441 (88).txt', '264.txt', '109.txt',
                 '441 (103).txt', '006.txt', '275.txt', '098.txt', '441 (85).txt', '402.txt', '020.txt', '075.txt',
                 '279.txt', '428.txt', '115.txt', '441 (209).txt', '324.txt', '441 (66).txt', '199.txt', '234.txt',
                 '002.txt', '090.txt', '186.txt', '441 (131).txt', '441 (254).txt', '441 (64).txt', '009.txt',
                 '421.txt',
                 '240.txt', '345.txt', '312.txt', '392.txt', '157.txt', '441 (62).txt', '012.txt', '216.txt',
                 '441 (57).txt', '201.txt', '158.txt', '441 (253).txt', '143.txt', '132.txt', '441 (176).txt',
                 '441 (73).txt', '093.txt', '061.txt', '441 (72).txt', '441 (65).txt', '001.txt', '441 (8).txt',
                 '410.txt',
                 '305.txt', '441 (130).txt', '084.txt']
    cate_ids = []
    for case_file in Path(Config.kgg_raw_data_dir).glob("*.txt"):
        if case_file.name in file_list:
            continue
        case_file = str(case_file)
        with open(case_file, 'r', encoding='utf-8') as f:
            id = f.readline().split('\t')[0]
            cate_ids.append(id)
            dst_file = os.path.join(Config.kgg_out_data_dir, f"{id}.txt")
        print(dst_file)
        # shutil.copyfile(case_file, dst_file)


def create_ke_train_data(triples_result_file=Config.triples_result_file):
    """
    :param file_path:
    :return:
    """
    entities = {}
    relations = {}
    fw_tri = open(Config.out_triple_file, 'w')
    ent_idx = 0
    rel_idx = 0
    with open(triples_result_file, 'r', encoding='utf-8') as f:
        for line in f:
            rows = line.strip().split('\t')
            if len(rows) != 3:
                continue
            if rows[0] not in entities:
                entities[rows[0]] = ent_idx
                ent_idx += 1
            if rows[2] not in entities:
                entities[rows[2]] = ent_idx
                ent_idx += 1
            if rows[1] not in relations:
                relations[rows[1]] = rel_idx
                rel_idx += 1
            fw_tri.write(f"{entities[rows[0]]}\t{entities[rows[2]]}\t{relations[rows[1]]}\n")
    with open(Config.out_entity_vocab, 'w', encoding='utf-8') as fw:
        fw.write(f"{len(entities)}\n")
        for ent, ent_id in entities.items():
            fw.write(f"{ent}\t{ent_id}\n")
    with open(Config.out_relation_vocab, 'w', encoding='utf-8') as fw:
        fw.write(f"{len(relations)}\n")
        for rel, rel_id in relations.items():
            fw.write(f"{rel}\t{rel_id}\n")


def train2ke(train_file=Config.out_triple_file):
    df = pd.read_csv(train_file, header=None, sep="\t")  # train.txt
    print("df head 5: \n", df.head(5), "\n\ndf tail 5: \n", df.tail(5), "\n")
    # df = df.sample(frac=1.0)

    train_size = int(0.6 * df.shape[0])
    test_size = int(0.2 * df.shape[0])
    valid_size = df.shape[0] - train_size - test_size

    print("train_size: ", train_size, "\ntest_size: ", test_size, "\nvalid_size: ", valid_size, "\n")

    train_df = df.loc[: train_size - 1]
    test_df = df.loc[train_size: train_size + test_size - 1]
    valid_df = df.loc[train_size + test_size: train_size + test_size + valid_size - 1]

    with open(Config.train2id_file, "w") as f_train:
        f_train.write(str(train_size) + "\n")

    with open(Config.test2id_file, "w") as f_test:
        f_test.write(str(test_size) + "\n")

    with open(Config.valid2id_file, "w") as f_valid:
        f_valid.write(str(valid_size) + "\n")

    train_df.to_csv(Config.train2id_file, header=None, sep='\t', index=False, mode='a')
    test_df.to_csv(Config.test2id_file, header=None, sep='\t', index=False, mode='a')
    valid_df.to_csv(Config.valid2id_file, header=None, sep='\t', index=False, mode='a')


def create_lawdata():
    get_triple_result(raw_data=Config.kgg_raw_data_dir)
    create_ke_train_data(triples_result_file=Config.triples_result_file)
    train2ke(train_file=Config.out_triple_file)
