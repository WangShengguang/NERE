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

ner BERTCRF, test acc: 0.9844, precision: 0.9186, recall: 0.9848, f1: 0.9407
* model: ner BERTSoftmax, test acc: 0.9825, precision: 0.9104, recall: 0.9810, f1: 0.9317
bilstm test acc: 0.9949, precision: 0.9894, recall: 0.9780, f1: 0.9816
bilstm_crf test acc: 0.9923, precision: 0.9741, recall: 0.9650, f1: 0.9665



### 关系识别
|模型|accuracy|precision|recall|f1|
|---|---|---|---|---|
|BERTMultitask|0.9352|0.9083|0.9105|0.9086|
|BERTSoftmax|0.9283|0.9032|0.8868|0.8931|
|BiLSTM|0.8166|0.8027|0.7321|0.7500|
|BiLSTM_ATT|0.8169|0.8213|0.7025|0.7344|
|ACNN|0.1300|0.0944|0.2079|0.1134|


### 联合训练 
模型：BERTCRF+BERTMultitask+TransE

|组合方式(a\*ner+b\*re+c*transe)|任务|accuracy|precision|recall|f1|
|---|---|---|---|---|---|
|ner_loss + 10 * re_loss|NER|0.9918|0.9764|0.9830|0.9768|
|ner_loss + 10 * re_loss|RE|0.9423|0.9333|0.9165|0.9226|
|ner_loss + 5 * re_loss|NER|0.9901|0.9841|0.9800|0.9797|
|ner_loss + 5 * re_loss|RE|0.9371|0.9162|0.9213|0.9184|

|ner_loss + 5 * re_loss|RE|0.9371|0.9162|0.9213|0.9184|
|ner_loss + 5 * re_loss|RE|0.9371|0.9162|0.9213|0.9184|

---
*joint_0.2BERTCRF_0.4BERTMultitask_0.4TransE test NER acc: 0.9841, precision: 0.9803, recall: 0.9580, f1: 0.9645              
*joint_0.2BERTCRF_0.4BERTMultitask_0.4TransE test RE acc: 0.9492, precision: 0.9169, recall: 0.9198, f1: 0.9176 



 *joint_0.6BERTCRF_0.3BERTMultitask_0.1TransE test NER acc: 0.9791, precision: 0.9698, recall: 0.9494, f1: 0.9532    
 *joint_0.6BERTCRF_0.3BERTMultitask_0.1TransE test RE acc: 0.9470, precision: 0.9051, recall: 0.8934, f1: 0.8980 

 *joint_0.7BERTCRF_0.2BERTMultitask_0.1TransE test NER acc: 0.9735, precision: 0.9749, recall: 0.9506, f1: 0.9570
 *joint_0.7BERTCRF_0.2BERTMultitask_0.1TransE test RE acc: 0.9315, precision: 0.9059, recall: 0.9086, f1: 0.9045

 *joint_0.8BERTCRF_0.15BERTMultitask_0.05TransE test NER acc: 0.9843, precision: 0.9822, recall: 0.9493, f1: 0.9602  
 *joint_0.8BERTCRF_0.15BERTMultitask_0.05TransE test RE acc: 0.9399, precision: 0.9172, recall: 0.8776, f1: 0.8937 


*joint_0.9BERTCRF_0.05BERTMultitask_0.05TransE test NER acc: 0.9835, precision: 0.9769, recall: 0.9577, f1: 0.9616 
*joint_0.9BERTCRF_0.05BERTMultitask_0.05TransE test RE acc: 0.9245, precision: 0.9075, recall: 0.8759, f1: 0.8893 


*joint_0.5BERTCRF_0.49BERTMultitask_0.01TransE test NER acc: 0.9803, precision: 0.9802, recall: 0.9476, f1: 0.9580
*joint_0.5BERTCRF_0.49BERTMultitask_0.01TransE test RE acc: 0.9337, precision: 0.8888, recall: 0.9115, f1: 0.8987 

*joint_0.6BERTCRF_0.39BERTMultitask_0.01TransE test NER acc: 0.9878, precision: 0.9689, recall: 0.9833, f1: 0.9718  
*joint_0.6BERTCRF_0.39BERTMultitask_0.01TransE test RE acc: 0.9355, precision: 0.8958, recall: 0.9030, f1: 0.8985  


*joint_0.7BERTCRF_0.29BERTMultitask_0.01TransE test NER acc: 0.9866, precision: 0.9731, recall: 0.9748, f1: 0.9699                     
*joint_0.7BERTCRF_0.29BERTMultitask_0.01TransE test RE acc: 0.9439, precision: 0.9160, recall: 0.9121, f1: 0.9127  


*joint_0.1BERTCRF_0.899BERTMultitask_0.001TransE test NER acc: 0.9827, precision: 0.9691, recall: 0.9655, f1: 0.9617  
*joint_0.1BERTCRF_0.899BERTMultitask_0.001TransE test RE acc: 0.9360, precision: 0.8856, recall: 0.9114, f1: 0.8978  

*joint_0.3BERTCRF_0.699BERTMultitask_0.001TransE test NER acc: 0.9884, precision: 0.9772, recall: 0.9734, f1: 0.9716                   
*joint_0.3BERTCRF_0.699BERTMultitask_0.001TransE test RE acc: 0.9337, precision: 0.9080, recall: 0.8591, f1: 0.8792 

*joint_0.5BERTCRF_0.4999BERTMultitask_0.0001TransE test NER acc: 0.9825, precision: 0.9789, recall: 0.9339, f1: 0.9493  
*joint_0.5BERTCRF_0.4999BERTMultitask_0.0001TransE test RE acc: 0.9390, precision: 0.9051, recall: 0.9236, f1: 0.9133   

*joint_0.8BERTCRF_0.1999BERTMultitask_0.0001TransE test NER acc: 0.9829, precision: 0.9782, recall: 0.9711, f1: 0.9706 
*joint_0.8BERTCRF_0.1999BERTMultitask_0.0001TransE test RE acc: 0.9346, precision: 0.9017, recall: 0.8993, f1: 0.9001 

 *joint_0.2BERTCRF_0.4BERTMultitask_0.4TransE test NER acc: 0.9841, precision: 0.9803, recall: 0.9580, f1: 0.9645  
 *joint_0.2BERTCRF_0.4BERTMultitask_0.4TransE test RE acc: 0.9492, precision: 0.9169, recall: 0.9198, f1: 0.9176 

*joint_0.7BERTCRF_0.29BERTMultitask_0.01TransE test NER acc: 0.9866, precision: 0.9731, recall: 0.9748, f1: 0.9699  
*joint_0.7BERTCRF_0.29BERTMultitask_0.01TransE test RE acc: 0.9439, precision: 0.9160, recall: 0.9121, f1: 0.9127 

*joint_0.5BERTCRF_0.4999BERTMultitask_0.0001TransE test NER acc: 0.9825, precision: 0.9789, recall: 0.9339, f1: 0.9493 
*joint_0.5BERTCRF_0.4999BERTMultitask_0.0001TransE test RE acc: 0.9390, precision: 0.9051, recall: 0.9236, f1: 0.9133

*joint_0.2BERTCRF_0.4BERTMultitask_0.4TransE test NER acc: 0.9841, precision: 0.9803, recall: 0.9580, f1: 0.9645  
*joint_0.2BERTCRF_0.4BERTMultitask_0.4TransE test RE acc: 0.9492, precision: 0.9169, recall: 0.9198, f1: 0.9176  

*joint_0.7BERTCRF_0.29BERTMultitask_0.01TransE test NER acc: 0.9866, precision: 0.9731, recall: 0.9748, f1: 0.9699 
*joint_0.7BERTCRF_0.29BERTMultitask_0.01TransE test RE acc: 0.9439, precision: 0.9160, recall: 0.9121, f1: 0.9127   





---


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

新生成的数据集、新transx模型、大为试验



---




train 9420 


NER, train: 3684,valid: 1962,test: 1962
RE, train: 11100,valid: 1043,test: 1042



NER&RE -> 生成数据集三元组（模拟线上环境） -> KE 
用已训练的数据来测试 

2019 10万文书
500 裁判文书 -》 测一遍 kgg

SimilarRanking 推荐系统
导出编码，A_TransE_embedding.json ，作为输入，运行的result.json。
推荐10篇最相近的文章（10个算法）

500抽10,每篇10个算法，每个算法10篇==1000篇; 每个推荐10个

KE 1000篇和原始数据 

