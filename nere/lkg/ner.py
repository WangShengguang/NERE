"""Named Entity Recognition"""

import os
import re

import torch
from pytorch_pretrained_bert import BertTokenizer, BertConfig

from nere.config import Config
from nere.data_helper import entity_label2tag
from nere.evaluator import Predictor
from nere.ner.torch_models import BERTSoftmax, BERTCRF


class EntityRecognition(Predictor):
    def __init__(self, model_name, max_len=Config.max_sequence_len):
        super().__init__(framework="torch")
        self.max_len = max_len
        self.idx2tag = self.data_helper.id2ent_tag
        self.abbr2label = {label: abbr for label, abbr in entity_label2tag.items()}
        self.tokenizer = BertTokenizer.from_pretrained(Config.bert_pretrained_dir, do_lower_case=True)
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        num_ent_tags = len(self.data_helper.ent_tag2id)
        config = BertConfig.from_json_file(Config.bert_config_path)
        if model_name == 'BERTSoftmax':
            model = BERTSoftmax(config, num_labels=num_ent_tags)
        elif model_name == 'BERTCRF':
            model = BERTCRF(config, num_labels=num_ent_tags)
        else:
            raise ValueError(model_name)
        model_dir = os.path.join(Config.torch_ckpt_dir, "ner")
        model_path = os.path.join(model_dir, model_name + ".bin")
        model.load_state_dict(torch.load(model_path))  # 断点续训
        return model

    def parse(self, fact_text):
        sents, sent_spans = self.get_sent_list(fact_text)
        sent_tokens, sent_spans_data = self.load_data(sents, sent_spans)
        ner_results = self.get_ner_results(sent_tokens, sent_spans_data)
        return ner_results

    def get_sent_list(self, fact_text):
        sents = []
        sent_spans = []
        for para_fact in fact_text:
            for fact in para_fact:
                spans = []
                if len(fact) <= 5:
                    continue
                sents.append(fact)
                clauses_period = self.split_text(fact, '。')
                clauses_period = list(filter(lambda x: len(x) > 5, clauses_period))
                for clause in clauses_period:
                    if len(clause) <= self.max_len:
                        spans.append(clause)
                    else:
                        clauses_semi = self.split_text(clause, '；;')
                        clauses_semi = list(filter(lambda x: len(x) > 5, clauses_semi))
                        len_semi = len(clauses_semi)
                        if len_semi == 1:
                            clauses_comma = self.split_text(clause, '，,')
                            clauses_comma = list(filter(lambda x: len(x) > 5, clauses_comma))
                            len_comma = len(clauses_comma)
                            if len_comma <= 2:
                                spans.extend(clauses_comma)
                            else:  # len_comma >= 3
                                mid = len_comma // 2
                                if len_comma % 2 == 1:  # The number of elements is odd
                                    if clauses_comma[0] < clauses_comma[-1]:
                                        mid = mid + 1
                                spans.extend(clauses_comma[:mid])
                                spans.extend(clauses_comma[mid:])
                        elif len_semi == 2:
                            spans.extend(clauses_semi)
                        else:
                            mid = len_semi // 2
                            if len_semi % 2 == 1:
                                if clauses_semi[0] < clauses_semi[-1]:
                                    mid = mid + 1
                            spans.extend(clauses_semi[:mid])
                            spans.extend(clauses_semi[mid:])
                if spans:
                    sent_spans.append(spans)
        return sents, sent_spans

    def split_text(self, text, punc):
        # (*x) - Retain the punctuation separator
        splits = re.split('([{}])'.format(punc), text)
        # Put the separator after the sentence if available
        clauses = [''.join(split).strip() for split in zip(splits[0::2], splits[1::2])]
        if not text.endswith(punc):
            clauses.append(splits[-1].strip())
        return clauses

    def load_data(self, sents, sent_spans):
        sent_tokens = []
        sent_spans_data = []
        for sent, spans in zip(sents, sent_spans):
            sent_tokens.append(self.tokenizer.tokenize(sent))
            data = []  # for a sentence
            for span in spans:
                tokens = ['[CLS]'] + self.tokenizer.tokenize(span) + ['[SEP]']
                token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long)
                token_ids = token_ids.unsqueeze(0)  # batch_size = 1
                data.append((tokens, token_ids))
            sent_spans_data.append(data)
        return sent_tokens, sent_spans_data

    def get_ner_results(self, sent_tokens, sent_spans_data):
        """get ner result"""
        results = []  # for relation extraction
        for sent_tokens, spans_data in zip(sent_tokens, sent_spans_data):
            sent_entities = []
            for span_tokens, token_ids in spans_data:
                pred_indices = self.model(token_ids)
                pred_indices = pred_indices.squeeze(0).numpy()
                pred_tags = [self.idx2tag[idx] for idx in pred_indices]
                assert len(span_tokens) == len(pred_tags)
                i = 0
                span_len = len(span_tokens)
                while i < span_len:
                    ent_tokens = []
                    if pred_tags[i].startswith('B-'):
                        ent_tokens.append(span_tokens[i])
                        abbr = pred_tags[i].lstrip('B-')
                        label = self.abbr2label[abbr]
                        i += 1
                        mid_tag = 'I-' + abbr
                        while i < span_len and pred_tags[i] == mid_tag:
                            ent_tokens.append(span_tokens[i])
                            i += 1
                        if ent_tokens and ent_tokens not in sent_entities:
                            sent_entities.append({'ent_tokens': ent_tokens, 'ent_label': label})
                    else:
                        i += 1
            if sent_entities:
                results.append((sent_tokens, sent_entities))
        return results
