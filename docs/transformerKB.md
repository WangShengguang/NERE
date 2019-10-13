# TransformerKB 

类似词向量中的平移不变性，C(king)-C(queen)~=C(man)-C(woman),词向量能够捕捉到单词之间某钟隐藏的语义关系。 受次启发，Bordes等人提出TransE模型，将知识库中的关系看作实体间的某种平移关系。对每个三元组(h,r,t)，d对应向量(l_h,l_r,l_t),将l_r看作l_h到l_t的平移，也可将l_r看作l_h到l_t的翻译,因此TransE又称翻译模型。对于每个三元组，TransE优化目标为l_h+l_r~=l_t。TransE模型简单有效，参数少，复杂度低，性能提升明显。TransE模型已近成为知识表示的代表模型，此后的大部分模型都是基于TransE的改进。

ConvE和ConvKB作为卷积网络在知识表示中的应用，取得了较好的效果。ConvKB score function $f(h,r,t)=concat(g([v_h,v_r,v_t])*\omega).w$ , 将g和w取适当的值，可得到TransE，因此，可将ConvKB看作TransE的扩展；将(h,r,t)当做一个句子，ConvKB也可视为对句子正确性判断的二分类问题。知识表示其实就是word embedding ,sentence embedding的进一步细化。知识表示最终的目标是随意遮掉(h,r,t)之一，可以通过另外两个得到它,这和预测句子中缺失词语的任务可以说十分相似。因此,一个优秀的文本表示模型，稍加改造,也可以作为知识表示模型。受到TransE，ConvKB启发,结合目前的工作，Self-Attention机制在翻译和文本方面取得巨大成功，因此，我们考虑尝试将ConvKB其中的卷积部分替换为Transformer encoder,并做适当调整，得到 TransKB score function $\mathit{score}~(h,r,t) = {transformer}~([h,r,t]) \cdot \mathbf{w}$. we named it as TransformerKB, 是一个新的基于Transformer特征提取器的知识表示模型。

in TransformerKB，三元组中的entity，relation被初始化为一个等长的k-dimension embedding.每个三元组对应得到k*3的input 矩阵(就是一个长度为3的句子，embedding后的结果),送入transformer encoder get a feature vector which is then computed with a weight vector via a dot product to produce a score for the triple (h, r, t). This score is used to infer whether the triple (h, r, t) is valid or not .考虑到(h,t,r) 复杂度较低,我们取num_blocks=3, num_heads=4。相对于ConvKB的优势是TransKB更加复杂,也可以直接获取到三元组中更丰富的信息,可扩展性更好。

目前的知识表示模型全部使用三元组当做训练语料开始训练，并未引入外部信息，孤立的三元组丧失了其上下文信息，因此我们正在考虑在进一步的工作中，将包含三元组(h,r,t)的原始文本先学习一个文本表示用来初始化知识表示任务的embedding矩阵，相当于文本表示问题中引入外部语料的预训练词向量。而TransKB可以做到两个任务的统一。


