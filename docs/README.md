

##  试验解读

-  数据集  

|Datasets|Number of Cases|Number of Entities|Number of Relations| 
|---|---|---|---|
|Training Set|9680|20|9|
|Development Set|1094|20|9|
|Test Set|2300|20|9|

取宏平均（Macroaveraged）的准确率、精确率、召回率及 F1 进⾏效果评估 

- 实体识别 

|模型|accuracy|precision|recall|f1|
|---|---|---|---|---|
|BERTCRF|0.9965|0.9923|0.9811|0.9850|
|BERTSoftmax|0.9918|0.9522|0.9820|0.9613|
|BiLSTM|0.9852|0.9895|0.9662|0.9741|
|BiLSTMCRF|0.9841|0.8923|0.9123|0.8961|


- 关系抽取 

|模型|accuracy|precision|recall|f1|
|---|---|---|---|---|
|BERTMultitask|0.9352|0.9083|0.9105|0.9086|
|BERTSoftmax|0.9400|0.9263|0.8864|0.8993|
|BiLSTM|0.8166|0.8027|0.7321|0.7500|
|BiLSTM_ATT|0.8169|0.8213|0.7025|0.7344|
|ACNN|0.1300|0.0944|0.2079|0.1134|


- 联合训练 

为了进一步提升模型的整体效果，我们尝试将NER和RE任务做了联合训练。
在联合模型中，将NER和RE部分的输出通过加权方式合并到一起，
得到损失函数， $ loss=\alpha * BERTCRF+ \beta * BERTMultitask $ 

|Task|accuracy|precision|recall|f1|
|---|---|---|---|---|
|NER|0.9918|0.9764|0.9830|0.9768|
|RE|0.9423|0.9333|0.9165|0.9226|

最终联合训练模型的表现相对于最好的RE单模型BERTMultitask F1有1.4的提升，NER单模型BERTCRF F1有0.9个点的下降。


- 知识表示 



