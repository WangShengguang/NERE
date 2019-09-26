

##  试验解读

-  数据集  

|Datasets|Number of Cases|Number of Entities|Number of Relations| 
|---|---|---|---|
|Training Set|9680|20|9|
|Development Set|1094|20|9|
|Test Set|2300|20|9|


- 实体识别 

|模型|accuracy|precision|recall|f1| 
|---|---|---|---|---|
|BiLSTM|0.9274|0.8775|0.5486|0.6571|
|BiLSTM-CRF|||0.657|
|BERTSoftmax|0.9259|0.8568|0.9744|0.9019|
|BERTCRF|0.9885|0.9840|0.9402|0.9555|

 ----

 model: ner BERTCRF, test acc:         0.9887, precision: 0.9890, recall: 0.9503, f1: 0.9638  
 BERTSoftmax, test         acc: 0.9274, precision: 0.8688, recall: 0.9694, f1: 0.9059  
bilstm test acc: 0.9903, precisi        on: 0.9736, recall: 0.9919, f1: 0.9804             
  bilstm_crf test acc: 0.9858, pre        cision: 0.9403, recall: 0.9586, f1: 0.9448
  
## 关系抽取  


|模型|accuracy|precision|recall|f1| 
|---|---|---|---|---|
|BiLSTM|0.8001|0.6134|0.6191|0.6126|
|BiLSTM-ATT|0.7936|0.6449|0.6060|0.6096|
|ACNN|0.2343|0.0596|0.1199|0.0775|
|BERTSoftmax|0.9395|0.9238| 0.8615|0.8823|
|BERTMultitask|0.9342| 0.9305|0.8621|0.8842|


