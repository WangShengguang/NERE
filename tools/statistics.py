import os
from collections import defaultdict
from pathlib import Path

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)


class Config(object):
    """
        数据和模型所在文件夹
    """
    data_dir = os.path.join(root_dir, "data")
    ner_data_dir = os.path.join(data_dir, "ner")
    re_data_dir = os.path.join(data_dir, "re")


entity_label2tag = {'自然人主体': 'NP',
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
                    '其他违法行为': 'OLA'}


class DataHelper(object):
    """label : 标签体系
        tag： ：样本标注结果
    """

    def __init__(self):
        """
        :param task: joint,ner,re
        """
        self.load_re_tags()

    def load_re_tags(self):
        with open(os.path.join(Config.re_data_dir, "rel_labels.txt"), "r", encoding="utf-8") as f:
            self.relation_labels = [line.strip() for line in f.readlines()]  # 搭乘、其他、投保
        with open(os.path.join(Config.re_data_dir, "ent_labels.txt"), "r", encoding="utf-8") as f:
            self.entity_labels = [line.strip() for line in f.readlines()]  # 伤残、机动车、非机动车

    def statics_re(self):
        relations = defaultdict(int)
        for data_type in ['train', 'val', 'test']:
            with open(os.path.join(Config.re_data_dir, "{}.txt".format(data_type)), "r", encoding="utf-8") as f:
                for line in f:
                    splits = line.strip().split('\t')
                    rel_label = splits[0]
                    e1_label, e2_label = splits[1], splits[2]
                    e1, e2 = splits[3], splits[4]
                    sent_text = splits[5].strip()
                    relations[rel_label] += 1
        relations_li = sorted([(k, v) for k, v in relations.items()], key=lambda x: x[1],reverse=True)
        return relations_li

    def statics_ner(self):
        tag2label = {v: k for k, v in entity_label2tag.items()}
        entities = defaultdict(int)
        for data_type in ['train', 'val', 'test']:
            data_dir = Path(Config.ner_data_dir).joinpath(data_type)
            with open(data_dir.joinpath("tags.txt"), "r", encoding="utf-8") as f:
                for line in f:
                    for tag in line.strip().split(' '):
                        if tag.startswith("B-"):
                            entities[tag2label[tag.lstrip("B-")]] += 1
        entities_li = sorted([(k, v) for k, v in entities.items()], key=lambda x: x[1],reverse=True)
        return entities_li


if __name__ == "__main__":
    res = DataHelper().statics_ner()
    print(res)
    res = DataHelper().statics_re()
    print(res)
