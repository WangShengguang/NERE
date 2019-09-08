"""Automatic generation of law knowledge graph"""

from nere.lkg.neo4j_module import ImportNeo4j
from nere.lkg.ner import EntityRecognition
from nere.lkg.preprocessor import Preprocessing
from nere.lkg.re import RelationExtraction
from nere.lkg.utils import create_triple


class KGG(object):
    """Automatic generation of knowledge map for a case judgement
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
