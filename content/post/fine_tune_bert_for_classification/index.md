---
title: "How to Fine-Tune BERT for Text Classification"
date: 2021-03-19
lastmod: 2021-03-19
draft: True
authors: ["Roymond Liao"]
categories:
    - NLP
    - Deep Learning
tags: ["BERT", "Classification", "Fine-tune"]
markup: goldmark
image:
  placement: 2
  caption: ""
  focal_point: "Center"
  preview_only: false
---

Forcus on two main topics reseasch:

* Fine-tun the pre-tarining model with three strategies:
  1. Further pre-train BERT on **within-task** training data or **in-domain** data.
  2. Fine-tune BERT with multi-task learning.
  3. Fine-tune BERT for the target task.

* Investigate the fine-tuning method for BERT on servel sides.
  1. Pre-process of long text
  2. Layer selection
  3. Layer wise learning rate.
  4. Catastrophi forgetting.
  5. Low-shot learning problems.



At final layer add softmax layer to the top of BERT and to predict the probability of label c:
$$
p(c|h) = softmax(Wh)
$$
where $W$ is the task-specific parameter matrix.

We could imagines $W$ is a jointly by maximizing the log-probability of the correct label.

## Methods

On below figure the author show three differnets training strategiess of fine-turn BERT.

<figure class="image"> 
<center>
  <img src="./figure_1.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1905.05583">Paper</a>
  </figcaption>
</center>
</figure>
### 1. Fine-Tuning Strategies

There have serveral factores need consideration:

1. Long text processing: The maximum size is 512 in BERT.
2. Layer selection: The official BERT-base model is builded by 12 encoder lyers. We need to chose which layer is more effective layer for text classification task.
3. Overcome overfitting problem.

Intuitively, the lower layer may contain more geneal information and the higher layer may extracte specific feature information for task. So, we can fine-tun them with differenct learning rates.

Following Howard and Ruder (2018)[^1] , the authors to split the parameters $\theta$ into $ \{\theta^1, \dots, \theta^L\}$ where the $\theta$ contatins the parameters of the $l$-th layer of BERT.
$$
\theta_t^l = \theta_{t-1}^l - \eta^L \nabla_{\theta^l}J(\theta)
$$
where $\eta^l$ represents the learning rate of the $l$-th layer.

We set the base learning rate to $\eta^l$ and use the decay factor $\xi$ to set different learning rate $\eta^{k-1} = \xi\cdot\eta^k$ in different layers. When $\xi < 1$ the lower layers will have the lower learning rate than higher layers. When $\xi = 1$ , all layers have the same learning rate.



### 2. Further Pre-training

### 3. Mulit-Task Fine-Tuning

## Reference

[^1]: Jeremy Howard and Sebastian Ruder. 2018. Universal language model ﬁne-tuning for text classiﬁcation. arXiv preprint arXiv:1801.06146.

