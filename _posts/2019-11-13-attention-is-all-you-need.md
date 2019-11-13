---
title: Attention Is All You Need
author: sigmeta
layout: post
categories: [ML, paper]
tags: [NLP, paper, Transformer]
---

# Attention Is All You Need 论文笔记
本文主要讲述Self-Attention机制+Transformer模型。自己看过论文与其他人文章的总结，不是对论文的完整翻译。

> 论文原文翻译可看[这篇](https://www.yiyibooks.cn/yiyibooks/Attention_Is_All_You_Need/index.html)，翻译质量还可以。
> 关于Attention的讲解可以看这里：[Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
> 关于Transformer，这篇文章讲解很棒：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## 背景
Attention机制允许对依赖关系进行建模，而不考虑其在输入输出中的距离。在多数情况下，Attention与循环神经网络一起使用。

Transformer模型提出的原因：循环神经网络模型无法并行计算

## Tansformer模型简介


![image](https://github.com/sigmeta/PaperNotes/raw/master/image/attention/transformer.PNG){:width="400"}


transformer模型


Transformer模型是**纯attention**模型，完全依赖attention机制来描述输入与输出的全局依赖。

在Transformer模型中，依赖关系的计算降低到固定次数（这里就是说计算速度快啦），尽管由于对用attention权重化的位置取平均降低了效果，但是我使用Multi-Head Attention进行抵消。

Self-attention，有时称为intra-attention，是一种attention机制，它关联单个序列的不同位置以计算序列的表示。

#### Attention & Self-Attetion

- attention：每一步输出时，找到输入中最应该注意的部分。（对齐）
- self-attention：在模型处理某个词时，self-attention允许模型查看其他位置，来寻找更好编码该词的线索。

## 论文细节

### Attention

1. 首先要计算出三个向量q，k，v。
    1. q是query，是当前要处理的词对应的向量；k是key，通过计算q与k的关系可以得到当前需要对其他词的关注度；v是value，表示的是其他单词。
    2. 这三个向量是通过训练出三个矩阵与词的embedding相乘得到。
    3. q，k，v维度一般小于embedding。（非必要，但是可以减小运算量）
    4. 在论文中，embedding维度512，q，k，v维度$d_k$=64
2. q，k相乘得到一个分数。 这里是相当于计算了一个相似度，找到当前词到其他词的相关关系。
3. 该分数除以$\sqrt{d_k}$。这里是为了更稳定的梯度，否则做softmax容易导致一个1其他全是0。
4. 对3中的值做softmax。softmax分数决定了在该位置其他词表达多少。显然本词自身占比会最高，但也会关注到相关的词。
5. softmax乘以v。保留关注的词，丢弃不相关的。
6. 对5产生的vectors求和，就产生了该位置的self-attention输出。
$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$
$$Q \in \mathbb{R}^{n\times d_k}, K \in \mathbb{R}^{m\times d_k},V \in \mathbb{R}^{m\times d_v}$$

![attention示意](https://github.com/sigmeta/PaperNotes/raw/master/image/attention/self-attention-matrix-calculation-2.png){:width="400"}

在普通的attention中，K，V对应编码器输出，Q对应解码器当前的输入。self-attention中，Q，K，V都对应于当前的输入X。

![enter image description here](https://github.com/sigmeta/PaperNotes/raw/master/image/attention/self-attention-matrix-calculation.png){:width="400"}

### Multi-head attention
将self-attention做多次后，进行拼接，点乘一个训练得到的矩阵$W_0$，得到self-attention输出。

![enter image description here](https://github.com/sigmeta/PaperNotes/raw/master/image/attention/transformer_attention_heads_weight_matrix_o.png){:width="800"}

1. 它扩展了模型关注不同位置的能力。
2. 它为attention层提供了多个“表示子空间”。Transformer使用8头。这些集合中每一个都是随机初始化的，在训练之后，每组用于将输入embedding投影到不同的表示子空间中。
![single attention](https://github.com/sigmeta/PaperNotes/raw/master/image/attention/transformer_self-attention_visualization.png){:width="400"}
![Multi-head attention](https://github.com/sigmeta/PaperNotes/raw/master/image/attention/transformer_self-attention_visualization_3.png){:width="400"}

完整过程：

![enter image description here](https://github.com/sigmeta/PaperNotes/raw/master/image/attention/transformer_multi-headed_self-attention-recap.png){:width="800"}

### Position embedding
本文采用的position embedding是直接计算得到的。
$$ f(x)=\left\{
\begin{aligned}
PE_{2i}(p)=\sin(p/10000^{2i/d_{pos}}) \\
PE_{2i+1}(p)=\cos(p/10000^{2i/d_{pos}}) \\
\end{aligned}
\right.
$$
与词向量维度一样，直接相加。

### Transformer其他部分

![image](https://github.com/sigmeta/PaperNotes/raw/master/image/attention/transformer.PNG){:width="400"}

Feed-forward层：两个线性变换，之间ReLu。
$$FFN(x)=\max(0,XW_1+b_1)W_2+b_2$$

子层之间有残差连接。每个子层的输出：$LayerNorm(x+Sublayer(x))$。为方便残差连接，维度都为512。

编码器：
- Encoder中，所有词一起过self-attention子层，分别单独过Feed-forward子层。
- 最顶层Encoder输出转成K、V给Decoder。

![enter image description here](https://github.com/sigmeta/PaperNotes/raw/master/image/attention/transformer_resideual_layer_norm_2.png){:width="400"}

解码器：
- 解码器中的self-attention有所不同，只允许关注该词之前的词。这通过在self-attention的计算中，在softmax之前对后面的位置置-inf实现。
- Encoder-Decoder attention（第二个子层），和self-attention类似，但它的Q是从下面层的输出得到，而K、V来自encoder。

![enter image description here](https://github.com/sigmeta/PaperNotes/raw/master/image/attention/transformer_resideual_layer_norm_3.png){:width="800"}

输出层就是做softmax，找到概率最大的词。


## 参考文献
1. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
2. [Attention Is All You Need 翻译](https://www.yiyibooks.cn/yiyibooks/Attention_Is_All_You_Need/index.html)
3. [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
4. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)



