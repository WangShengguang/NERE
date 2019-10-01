

dataset
* joint_sents: 3684, ner_unique_sents/ner_data: 3924/7608, re_unique_sents/re_data: 53/3737 

ner data_type: train, 3315
ner data_type: test, 2146
ner data_type: val, 2147
re data_type: train, 11876
re data_type: test, 944
re data_type: val, 365


-  数据集  

|NER Datasets|Number of Cases|Number of Entities|Number of Relations| 
|---|---|---|---|
|Training Set|3315|20|9|
|Development Set|365|20|9|
|Test Set|2146|944|9|

---

|RE Datasets|Number of Cases|Number of Entities|Number of Relations| 
|---|---|---|---|
|Training Set|11876|20|9|
|Development Set|2147|20|9|
|Test Set|2146|20|9|


---
201929

NER
数据集为RE的数据集 
* model: ner BERTCRF, test acc: 0.9909, precision: 0.9813, recall: 0.9434, f1: 0.9565  
* model: ner BERTSoftmax, test acc: 0.9966, precision: 0.9834, recall: 0.9740, f1: 0.9760        
* model: ner BiLSTM, test acc: 0.9882, precision: 0.9618, recall: 0.9369, f1: 0.9431            
bilstm test acc: 0.9993, precision: 0.9954, recall: 0.9983, f1: 0.9963                           
bilstm_crf test acc: 0.9147, precision: 0.8065, recall: 0.4195, f1: 0.5376                       
   

RE
 model: re BERTMultitask, test acc: 0.9343, precision: 0.9087, recall: 0.9243, f1: 0.9151  
* model: re BERTSoftmax, test acc: 0.9400, precision: 0.9263, recall: 0.8864, f1: 0.8993
* model: re BiLSTM_ATT, test acc: 0.8148, precision: 0.7406, recall: 0.6402, f1: 0.6394
* model: re ACNN, test acc: 0.3509, precision: 0.1579, recall: 0.1886, f1: 0.1520
* model: re BiLSTM, test acc: 0.8022, precision: 0.6416, recall: 0.5904, f1: 0.5977

## tips

batch_size问题

RE影响不大；NER讲20% f1
