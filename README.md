# NERE
Named Entity Recognition &amp; Relation Extraction




## 联合训练对比 

### 实体识别
|模型|accuracy|precision|recall|f1|
|---|---|---|---|---|
|BERTCRF|0.9965|0.9923|0.9811|0.9850|
|BERTSoftmax|0.9918|0.9522|0.9820|0.9613|
|BiLSTM|0.9852|0.9895|0.9662|0.9741|


|BiLSTMCRF|0.9852|0.9895|0.9662|0.9741|



### 关系识别
|模型|accuracy|precision|recall|f1|
|---|---|---|---|---|
|BERTMultitask|0.9352|0.9083|0.9105|0.9086|
|BERTSoftmax|0.7874|0.7684|0.6965|0.7184|
|BiLSTM|0.4852|0.2516|0.2014|0.1896|
|ACNN|0.1300|0.0944|0.2079|0.1134|
|BiLSTM_ATT|0.1300|0.0944|0.2079|0.1134|



### 联合训练 
|模型|组合方式|任务|accuracy|precision|recall|f1|
|---|---|---|---|---|---|---|
|BERTCRF+BERTMultitask|ner_loss + 10 * re_loss|NER|0.9918|0.9764|0.9830|0.9768|
|BERTCRF+BERTMultitask|ner_loss + 10 * re_loss|RE|0.9423|0.9333|0.9165|0.9226|
|BERTCRF+BERTMultitask|ner_loss + 5 * re_loss|NER|0.9901|0.9841|0.9800|0.9797|
|BERTCRF+BERTMultitask|ner_loss + 5 * re_loss|RE|0.9371|0.9162|0.9213|0.9184|

1+1
* NER acc: 0.9932, precision: 0.9791, recall: 0.9795, f1: 0.9768
* RE acc: 0.9392, precision: 0.9230, recall: 0.9147, f1: 0.9177

5+1



- [Multi-label classification with keras](https://www.kaggle.com/roccoli/multi-label-classification-with-keras)
- [Named Entity Recognition (NER) with keras and tensorflow](https://towardsdatascience.com/named-entity-recognition-ner-meeting-industrys-requirement-by-applying-state-of-the-art-deep-698d2b3b4ede)
- [Build a POS tagger with an LSTM using Keras](https://nlpforhackers.io/lstm-pos-tagger-keras/)



https://github.com/buppt/ChineseNRE/blob/master/BiLSTM_ATT.py

- [awesome-relation-extraction](https://github.com/roomylee/awesome-relation-extraction)

- [理论](http://nlpprogress.com/english/relationship_extraction.html)


- [ABCNN](https://github.com/lsrock1/abcnn_pytorch/blob/master/abcnn.py)

- [ACNN](https://github.com/lawlietAi/pytorch-acnn-model)
