# NERE
Named Entity Recognition &amp; Relation Extraction

---
## 联合训练对比 

### 实体识别
|模型|accuracy|precision|recall|f1|
|---|---|---|---|---|
|BERTCRF|0.9965|0.9923|0.9811|0.9850|
|BERTSoftmax|0.9918|0.9522|0.9820|0.9613|
|BiLSTM|0.9852|0.9895|0.9662|0.9741|
|BiLSTMCRF|0.9841|0.8923|0.9123|0.8961|

### 关系识别
|模型|accuracy|precision|recall|f1|
|---|---|---|---|---|
|BERTMultitask|0.9352|0.9083|0.9105|0.9086|
|BERTSoftmax|0.9283|0.9032|0.8868|0.8931|
|BiLSTM|0.8166|0.8027|0.7321|0.7500|
|BiLSTM_ATT|0.8169|0.8213|0.7025|0.7344|
|ACNN|0.1300|0.0944|0.2079|0.1134|


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
* NER acc: 0.9902, precision: 0.9764, recall: 0.9798, f1: 0.9754
* RE acc: 0.9410, precision: 0.9135, recall: 0.9060, f1: 0.9091

1+15
* NER acc: 0.9918, precision: 0.9759, recall: 0.9816, f1: 0.9759
* RE acc: 0.9427, precision: 0.9222, recall: 0.9096, f1: 0.9153

1+8
* NER acc: 0.9796, precision: 0.9777, recall: 0.9638, f1: 0.9667
* RE acc: 0.9375, precision: 0.8992, recall: 0.9264, f1: 0.9116

 model:joint_0.1BERTCRF_0.89BERTMultitask_0.01TransE, NER acc: 0.9214, precision: 0.9603, recall: 0.9286, f1: 0.9345
 model:joint_0.1BERTCRF_0.89BERTMultitask_0.01TransE,  RE acc: 0.9678, precision: 0.9589, recall: 0.9379, f1: 0.9476


- [Multi-label classification with keras](https://www.kaggle.com/roccoli/multi-label-classification-with-keras)
- [Named Entity Recognition (NER) with keras and tensorflow](https://towardsdatascience.com/named-entity-recognition-ner-meeting-industrys-requirement-by-applying-state-of-the-art-deep-698d2b3b4ede)
- [Build a POS tagger with an LSTM using Keras](https://nlpforhackers.io/lstm-pos-tagger-keras/)



https://github.com/buppt/ChineseNRE/blob/master/BiLSTM_ATT.py

- [awesome-relation-extraction](https://github.com/roomylee/awesome-relation-extraction)

- [理论](http://nlpprogress.com/english/relationship_extraction.html)


- [ABCNN](https://github.com/lsrock1/abcnn_pytorch/blob/master/abcnn.py)

- [ACNN](https://github.com/lawlietAi/pytorch-acnn-model)
