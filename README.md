# 1. NERE
Named Entity Recognition &amp; Relation Extraction

---

## 1.1 数据准备  
    python3 manage.py --data_prepare ner/re/joint   
    
    source_file:   
        Config.raw_data(data/raw_data)  
    out_data:   
        将标记结果转换为可读的文本，并划分训练集测试集 作为输入 
        Config.ner_data (data/ner/)   
        Config.ner_data (data/re/)

## 1.2 实体识别  
```bash
python3 manage.py --ner model_name --mode train
```
|模型|accuracy|precision|recall|f1|
|---|---|---|---|---|
|BERTCRF|0.9965|0.9923|0.9811|0.9850|
|BERTSoftmax|0.9918|0.9522|0.9820|0.9613|
|BiLSTM|0.9852|0.9895|0.9662|0.9741|
|BiLSTMCRF|0.9841|0.8923|0.9123|0.8961|

## 1.3 关系抽取
```bash
python3 manage.py --re model_name --mode train
```
|模型|accuracy|precision|recall|f1|
|---|---|---|---|---|
|BERTMultitask|0.9352|0.9083|0.9105|0.9086|
|BERTSoftmax|0.9283|0.9032|0.8868|0.8931|
|BiLSTM|0.8166|0.8027|0.7321|0.7500|
|BiLSTM_ATT|0.8169|0.8213|0.7025|0.7344|
|ACNN|0.1300|0.0944|0.2079|0.1134|


## 1.4 联合训练 
> 模型：$\alpha$*BERTCRF+$\beta$*BERTMultitask+$\gamma$TransE
```bash
python3 manage.py --joint --mode train
```
|参数|任务|accuracy|precision|recall|f1|
|---|---|---|---|---|---|
|ner_loss + 10 * re_loss|NER|0.9918|0.9764|0.9830|0.9768|
|ner_loss + 10 * re_loss|RE|0.9423|0.9333|0.9165|0.9226|
|ner_loss + 5 * re_loss|NER|0.9901|0.9841|0.9800|0.9797|
|ner_loss + 5 * re_loss|RE|0.9371|0.9162|0.9213|0.9184|


## 参考 
- [Multi-label classification with keras](https://www.kaggle.com/roccoli/multi-label-classification-with-keras)
- [Named Entity Recognition (NER) with keras and tensorflow](https://towardsdatascience.com/named-entity-recognition-ner-meeting-industrys-requirement-by-applying-state-of-the-art-deep-698d2b3b4ede)
- [Build a POS tagger with an LSTM using Keras](https://nlpforhackers.io/lstm-pos-tagger-keras/)
- [awesome-relation-extraction](https://github.com/roomylee/awesome-relation-extraction)
- [relationship_extraction](http://nlpprogress.com/english/relationship_extraction.html)
- [ABCNN](https://github.com/lsrock1/abcnn_pytorch/blob/master/abcnn.py)
- [ACNN](https://github.com/lawlietAi/pytorch-acnn-model)
- [BiLSTM_ATT](https://github.com/buppt/ChineseNRE/blob/master/BiLSTM_ATT.py)

------

# 2. KGG
Automatic generation of law knowledge graph

----

## 2.1 数据准备 
    
    input_data  
        Config.kgg_data_dir (data/kgg/dataset)
    output_data
        Config.kgg_out_dir (output/kgg/dataset)
        三元组，并切分训练集 


|数据集|input|output|
|---|---|---|
|lawdata|原始标注集|通过NERE后得到|
|lawdata_new|与lawdata同，避免生成数据覆盖而命名|通过NERE后得到|
|traffic_all|未标注数据|通过NERE后得到|


训练
```bash
python3 manage.py --kgg 
```


## tips
    测试时 
    test_batch_size必须一致，否则导致batch_seq_len变化，导致指标变化
    torch测试前必须做
    model.eval() 
    --- 
    源代码RE预处理忘记加+ ["[SEP]"]，效果差一点 ？还有其他因素...
    