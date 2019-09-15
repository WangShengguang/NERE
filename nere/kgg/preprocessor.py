import re

from nere.data_helper import entity_label2tag

entity_abbr = entity_label2tag  # entity 2 abbrave


class Preprocessing():
    def __init__(self):
        # Regular expression
        self.pattern_prefix = re.compile('^(原告|被告)$')
        self.pattern_repre = re.compile('(负责人|代理人|代表人)')
        self.pattern_colon = re.compile('[：:]')
        self.pattern_bracket = re.compile('[\(\)（）\[\]【】]')
        self.pattern_info = re.compile('[，,。.；;、]')
        # self.pattern_abbr1 = re.compile('(?<=[(（](以下|如下|下面)简(称为|写为)).*?(?=[)）)])')
        # self.pattern_abbr2 = re.compile('(?<=[(（](以下|如下|下面)简(称|写))(?!为).*?(?=[）)])')
        self.pattern_abbr = re.compile(
            '((?<=[(（](以下|如下|下面)简(称为|写为)).*?(?=[)）)])|(?<=[(（](以下|如下|下面)简(称|写))(?!为).*?(?=[）)]))')

    def process(self, case_file, file_name=None):
        """Judgement cases pre-processing and pre-annotate
        Args:
            case_file: (str) Case file in `txt` format after paragraph segmentation
            file_name: (str) Pre-annotated or to be annotated file name, without suffix
        """
        with open(case_file, 'r', encoding='utf-8') as file_in:
            lines = file_in.readlines()
            case_id = lines[0].split('\t')[0].strip()
            # Judging whether the instrument is a first instance judgment
            temp = lines[0].split('\t')
            if len(temp) < 3:
                return case_id, None, None
            title_para_type = temp[1].strip()
            if len(title_para_type) >= 3 or title_para_type == '0':
                return case_id, None, None
            party_info, fact_info, basic_fact = self.get_party_fact(lines)
            if not party_info or not fact_info:
                return None, None
            plaintiffs, defendants, entities_ann, entities_repl, entities_comp, entities_abbr = self.extract_parties(
                party_info)
            basic_fact['原告'] = plaintiffs
            basic_fact['被告'] = defendants
            fact_text = self.replace_complete(fact_info, entities_repl, entities_comp, entities_abbr)
            return case_id, basic_fact, fact_text

    def get_party_fact(self, lines):
        """get party and fact information
        paragraph type:
                               party information  Court found facts
              First instance:                  3                  7
             Second instance:                103          [105,110]
                     Retrial:                203      [205,210,225]
                Non-judgment:                303
        """
        type_label = 15  # judgement types
        party_label = 3  # party information
        fact_label = 7  # courts found facts

        basic_fact = {}  # basic facts
        case_id = lines[0].split('\t')[0].strip()
        basic_fact['编号'] = case_id
        party_info = []  # party information
        fact_info = []  # court found facts
        # get text
        for line in lines:
            splits = line.strip().split('\t')
            if len(splits) < 3:
                return party_info, fact_info, basic_fact
            para_type = int(splits[1].strip())
            content = splits[-1].strip()
            if para_type == 1:
                basic_fact['标题'] = content
            elif para_type == 14:
                basic_fact['受理法院'] = content
            elif para_type == 2:
                basic_fact['案号'] = content
            elif para_type == party_label:
                party_info.append(content)
            elif para_type == fact_label:
                fact_info.append(content)
        return party_info, fact_info, basic_fact

    def extract_parties(self, party_info):
        """extract party information, including plaintiffs and defendants
        """
        plaintiffs, defendants = [], []
        entities_ann = {}  # pre-labeled entities, entity<->label
        entities_repl = {}  # entities to be replaced, "原告"/"被告"<->entity, up to 2 elements
        entities_comp = {}  # entities to be completed, entity<->"原告"或"被告"
        entities_abbr = {}  # entities abbreviation, entity<->entity_abbr
        if party_info[0][:2] == '原告':
            plain_num = 0
            defed_num = 0
            for para in party_info:
                prefix = para[:2]
                if not self.pattern_prefix.search(prefix):
                    continue
                splits = self.pattern_info.split(para)
                # 原告委托代理人
                if len(splits[0]) <= 2 or (
                        self.pattern_repre.search(splits[0]) and not self.pattern_bracket.search(splits[0])):
                    continue
                entity = splits[0][2:]
                # Standard colon-separated party information
                if self.pattern_colon.search(para):
                    splits = self.pattern_colon.split(para)
                    entity = self.pattern_info.split(splits[1])[0]  # entity name

                # Entity can not be bracketed
                # e.g. "被告：蔡铜良(兼被告董瑞贤的委托诉讼代理人)"
                entity_abbr = ''
                if self.pattern_bracket.search(entity):
                    # bracket may be abbreviated
                    # e.g. "被告英大泰和财产保险股份有限公司某某分公司(以下简称英大泰和某某分公司)。"
                    match = self.pattern_abbr.search(entity)
                    if match:
                        entity_abbr = match.group()
                    entity = self.pattern_bracket.split(entity)[0]

                if len(entity) == 0:
                    continue
                party = {}  # a party
                if prefix == '原告':
                    plain_num += 1
                    party['编号'] = plain_num
                else:
                    defed_num += 1
                    party['编号'] = defed_num

                party['名字'] = entity

                if entity_abbr:
                    entities_abbr[entity] = entity_abbr
                    party['简称'] = entity_abbr
                    # Natural people generally do not have abbreviations
                    if '保险' in entity:
                        entities_ann[entity_abbr] = '非自然人主体'

                    entities_repl[prefix] = entity_abbr
                    entities_comp[entity_abbr] = prefix
                else:
                    if len(entity) <= 4:
                        entities_ann[entity] = '自然人主体'
                    elif '保险' in entity:  # "非自然人主体" only includes subclass of "保险"
                        entities_ann[entity] = '非自然人主体'
                    entities_repl[prefix] = entity
                    entities_comp[entity] = prefix
                if prefix == '原告':
                    plaintiffs.append(party)
                else:
                    defendants.append(party)
            if plain_num >= 2:
                del entities_repl['原告']
            if defed_num >= 2:
                del entities_repl['被告']
        return plaintiffs, defendants, entities_ann, entities_repl, entities_comp, entities_abbr

    def replace_complete(self, fact_info, entities_repl, entities_comp, entities_abbr):
        """Replacement and completion of plaintiff and defendant
        Args:
            fact_info: (list) the court found facts, paragraph list
            entities_repl: (dict) entities to be replaced, "原告"/"被告"<->entity, up to 2 elements
            entities_comp: (dict) entities to be completed
        Returns:
            fact_text: (str or list) the court found facts after processing
        """
        paras = []
        for para in fact_info:
            if entities_repl:
                for prefix in entities_repl:
                    # Replace "原告" and "被告" with the entity actually referred to, and check for completion
                    entity = entities_repl[prefix]
                    try:
                        # Check entities by reference "原告" or "被告"
                        prefix_ends = [match.end() for match in re.finditer(prefix, para)]
                        length = len(prefix_ends)
                        for i, prefix_end in enumerate(prefix_ends):
                            # Determine if temp and entity are equal
                            if entities_abbr and entity in entities_abbr:
                                entity = entity_abbr[entity]  # entity abbreviations
                            entity_len = len(entity)
                            temp = para[prefix_end: prefix_end + entity_len]
                            if not self.temp_equal_entity(temp, entity):
                                para = para[:prefix_end] + entity + para[prefix_end:]
                                # After adding entities, the corresponding ends in prefix_ends[i+1:] are moved backwards by entity_len lengths
                                if i + 1 < length:
                                    prefix_ends[i + 1:] = [end + entity_len for end in prefix_ends[i + 1:]]
                    except Exception as e:
                        print('Exception of replace occured in: ' + entity)
                        print(e)

            # Entity prefix completion, entity-by-entity traversal (equivalent to looking up the dictionary)
            for entity in entities_comp:
                # completing the prefix before the entity
                prefix = entities_comp[entity]
                try:
                    # According to the entity search ("原告" or "被告")
                    entity_starts = [match.start() for match in re.finditer(re.escape(entity), para)]
                    length = len(entity_starts)
                    for i, entity_start in enumerate(entity_starts):
                        temp = para[entity_start - 2: entity_start]
                        if temp != prefix:
                            para = para[:entity_start] + prefix + para[entity_start:]
                            if i + 1 < length:
                                entity_starts[i + 1:] = [start + 2 for start in
                                                         entity_starts[i + 1:]]  # prefix_len=len(prefix)=2
                except Exception as e:
                    print('Exception of completion occured in: ' + entity)
                    print(entities_comp)
                    print(e)
            paras.append(para)

        fact_text = []
        # The paragraph is divided according to the punctuation "。"
        for para in paras:
            # (。)-Retain the period "。" separator
            splits = re.split('(。)', para)
            # Put the separator after the sentence
            sents = [''.join(split).strip() for split in zip(splits[0::2], splits[1::2])]
            # If the end of the paragraph is not the end of the period, fill in the last sentence
            if not para.endswith('。'):
                sents.append(splits[-1].strip())
            # A paragraph as an element, a paragraph may contain multiple sentences
            if len(sents) > 0:
                fact_text.append(sents)
        return fact_text

    def temp_equal_entity(self, temp, entity):
        """Determine whether the mention and the entity are the same entity for accurate replacement of the entity
            Determine whether the same entity is based on the matching degree between the same length string after the "原告" or "被告",
            and the entity obtained in the structured text.
        """
        match_max_len = self.get_lcs_len(temp, entity)
        match_degree = match_max_len / len(entity)
        if match_degree >= 0.6:
            return True
        else:
            return False

    def get_lcs_len(self, str1, str2):
        """Get the longest common subssequence length of two strings"""
        temp = [[0 for j in range(len(str2))] for i in range(len(str1))]
        max_len = 0
        for i in range(len(str1)):
            for j in range(len(str2)):
                if str1[i] == str2[j]:
                    if i > 0 and j > 0:
                        temp[i][j] = temp[i - 1][j - 1] + 1
                        if temp[i][j] > max_len:
                            max_len = temp[i][j]
                    else:
                        temp[i][j] = 1
                else:
                    temp[i][j] = 0
        return max_len


if __name__ == '__main__':
    case_file = 'data/sample.txt'
    preprocessing = Preprocessing()
    case_id, basic_fact, fact_text = preprocessing.process(case_file)
    print(case_id)
    print(basic_fact)
    print(fact_text)
