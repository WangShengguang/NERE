# NERE
Named Entity Recognition &amp; Relation Extraction




## 联合训练对比 

### 实体识别
|模型|accuracy|precision|recall|f1|
|---|---|---|---|---|
|BERTCRF|0.9928|0.9938|0.9804|0.9852|
|BiLSTM|0.9852|0.9895|0.9662|0.9741|


### 关系抽取
|模型|accuracy|precision|recall|f1|
|---|---|---|---|---|
|BERTMultitask|0.9623|0.9536|0.9446|0.9488|


### 联合训练 
|模型|任务|accuracy|precision|recall|f1|
|---|---|---|---|---|---|
|BERTCRF+BERTMultitask|NER|0.9909|0.9726|0.9899|0.9782|
|BERTCRF+BERTMultitask|RE|0.9669|0.9513|0.9424|0.9460|

 








- [Multi-label classification with keras](https://www.kaggle.com/roccoli/multi-label-classification-with-keras)
- [Named Entity Recognition (NER) with keras and tensorflow](https://towardsdatascience.com/named-entity-recognition-ner-meeting-industrys-requirement-by-applying-state-of-the-art-deep-698d2b3b4ede)
- [Build a POS tagger with an LSTM using Keras](https://nlpforhackers.io/lstm-pos-tagger-keras/)



