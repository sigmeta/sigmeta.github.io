---
layout: post
title: How to Fine-Tune BERT for Text Classification 论文笔记
author: sigmeta
date: '2019-11-13 14:35:23 +0530'
category: paper
summary: "BERT在NLP任务中效果十分优秀，这篇文章对于BERT在文本分类的应用上做了非常丰富的实验，介绍了一些调参以及改进的经验，进一步挖掘BERT的潜力。"
thumbnail: hello.jpg
---


# How to Fine-Tune BERT for Text Classification 论文笔记

论文地址：[How to Fine-Tune BERT for Text Classification？](https://arxiv.org/abs/1905.05583?context=cs.CL)

BERT在NLP任务中效果十分优秀，这篇文章对于BERT在文本分类的应用上做了非常丰富的实验，介绍了一些调参以及改进的经验，进一步挖掘BERT的潜力。

实验主要在8个被广泛研究的数据集上进行，在BERT-base模型上做了验证。

## 文章的主要结论如下：

### 1.微调（fin-tune）策略
1. 对于长文本，尝试了（1）取头部510 tokens，（2）尾部510 tokens，（3）头部128 tokens+尾部382 tokens，（4）分片并进行最大池化、平均池化、attention，发现方法（3）最好。因为文章的关键信息一般在开头和结尾。
2. 分层训练，上层对文本分类更加重要。
3. 灾难性遗忘：在下游finetune可能会遗忘预训练的知识。需要设置较小的学习率，如2e-5.
4. 分层衰减学习率（Layer-wise Decreasing Layer Rate），对下层设置更小的学习率可以得到更高的准确率，在lr=2e-5，衰减率$\xi$=0.95

### 2. 继续预训练（Further Pretraining）
任务内（within-task） 和同领域（in-domain）的继续预训练可以大大提高准确率。
In-domain比within-task要好。

### 3. 多任务微调（Multi-task Finetuning）
在单任务微调之前的多任务微调有帮助，但是提升效果小于Further pretraining。

### 4. 小数据集
BERT对小数据集提升很大，这个大家都知道的。Further pretraining对小数据集也有帮助。


