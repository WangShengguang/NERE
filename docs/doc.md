

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
run 记录  
2019-09-27 23:34 
nohup python3 manage.py --joint --mode train &>joint_train.out&
nohup python3 manage.py --ner all --mode train &>ner_all_train.out&
nohup python3 manage.py --re all --mode train &>re_all_train.out&

