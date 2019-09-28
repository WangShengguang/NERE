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


class KGG2KE(object):
    def __init__(self, data_set):
        self.data_set = data_set
        self.init_path()

    def init_path(self):
        self.base_path_dir = Path(Config.data_dir).joinpath(self.data_set)
        self.cases_triples_result_json_file = Config.cases_triples_result_json_file_tmpl.format(data_set=self.data_set)
        self.entity2id_path = Config.entity2id_path_tmpl.format(data_set=self.data_set)
        self.relation2id_path = Config.relation2id_path_tmpl.format(data_set=self.data_set)
        #
        self.train_triple_file = Config.train_triple_file_tmpl.format(data_set=self.data_set)
        self.train2id_file_path = Config.train2id_file_tmpl.format(data_set=self.data_set)
        self.valid2id_file_path = Config.valid2id_file_tmpl.format(data_set=self.data_set)
        self.test2id_file_path = Config.test2id_file_tmpl.format(data_set=self.data_set)
        os.makedirs(os.path.dirname(self.test2id_file_path), exist_ok=True)

    def get_triple_result(self):
        ner_model = "BERTCRF"
        re_model = "BERTMultitask"
        kgg = KGG(ner_model, re_model)
        start_time = time.time()
        err_file = []
        long_sentence = 0
        triple_result = {}
        files = list(self.base_path_dir.glob("*.txt"))
        print("* kgg files count: {}".format(len(files)))
        for case_file in tqdm(files, desc="kgg get_triple_result"):
            case_file_name = case_file.name
            # print(case_file)
            try:
                flag, case_id, res = kgg.parse(case_file)
                if flag:
                    triple_result[case_id] = list(res)
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
        with open(self.cases_triples_result_json_file, 'w', encoding='utf-8') as f:
            json.dump(triple_result, f, ensure_ascii=False, indent=4)
        return triple_result

    def create_ke_train_data(self):
        with open(self.cases_triples_result_json_file, 'r', encoding='utf-8') as f:
            triples = json.load(f)
        entities = set()
        relations = set()
        unique_triples = set()
        for case_id, case_triples in triples.items():
            for triple in case_triples:
                if len(triple) != 3:
                    print(case_id, triple)
                    continue
                entity1, relation, entity2 = triple
                unique_triples.add((entity1, entity2, relation))
                entities.add(entity1)
                entities.add(entity2)
                relations.add(relation)
        entity2id = {ent: ent_id for ent_id, ent in enumerate(entities)}
        relation2id = {rel: rel_id for rel_id, rel in enumerate(relations)}
        with open(self.train_triple_file, "w", encoding="utf-8") as f:
            for entity1, entity2, relation in unique_triples:
                f.write(f"{entity2id[entity1]}\t{entity2id[entity2]}\t{relation2id[relation]}\n")
        with open(self.entity2id_path, 'w', encoding='utf-8') as f:
            f.write(f"{len(entity2id)}\n")
            for ent, ent_id in entity2id.items():
                f.write(f"{ent}\t{ent_id}\n")
        with open(self.relation2id_path, 'w', encoding='utf-8') as f:
            f.write(f"{len(relation2id)}\n")
            for rel, rel_id in relation2id.items():
                f.write(f"{rel}\t{rel_id}\n")

    def data_split(self):
        df = pd.read_csv(self.train_triple_file, header=None, sep="\t")  # train.txt
        print("df head 5: \n", df.head(5), "\n\ndf tail 5: \n", df.tail(5), "\n")
        # df = df.sample(frac=1.0)
        train_size = int(0.6 * df.shape[0])
        test_size = int(0.2 * df.shape[0])
        valid_size = df.shape[0] - train_size - test_size

        print("train_size: ", train_size, "\ntest_size: ", test_size, "\nvalid_size: ", valid_size, "\n")

        train_df = df.loc[: train_size - 1]
        test_df = df.loc[train_size: train_size + test_size - 1]
        valid_df = df.loc[train_size + test_size: train_size + test_size + valid_size - 1]

        with open(self.train2id_file_path, "w") as f_train:
            f_train.write(str(train_size) + "\n")

        with open(self.test2id_file_path, "w") as f_test:
            f_test.write(str(test_size) + "\n")

        with open(self.valid2id_file_path, "w") as f_valid:
            f_valid.write(str(valid_size) + "\n")

        train_df.to_csv(self.train2id_file_path, header=None, sep='\t', index=False, mode='a')
        test_df.to_csv(self.test2id_file_path, header=None, sep='\t', index=False, mode='a')
        valid_df.to_csv(self.valid2id_file_path, header=None, sep='\t', index=False, mode='a')

    def run(self):
        # self.get_triple_result()
        self.create_ke_train_data()
        self.data_split()
