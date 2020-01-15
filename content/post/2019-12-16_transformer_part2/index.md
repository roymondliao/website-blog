---
title: "Attention Is All You Need"
date: 2019-11-09
lastmod: 2019-11-09
draft: true
authors: ["Roymond Liao"]
categories:
    - NLP
    - Deep Learning
tags: ["Attention", "Self-attention", "Transformer"]
markup: mmark
image:
  placement: 2
  caption: ""
  focal_point: ""
  preview_only: false
---
# Attention Mechanism

**Attention** 的概念在 2014 年被 Bahdanau et al. [3] 所提出，解決了 encoder-decoder 架構的模型在 decoder 必須依賴一個固定向量長度的 context vector 的問題。實際上 attention mechanism 也符合人類在生活上的應用，例如：當你在閱讀一篇文章時，會從上下文的關鍵字詞來推論句子所以表達的意思，又或者像是在聆聽演講時，會捕捉講者的關鍵字，來了解講者所要描述的內容，這都是人類在注意力上的行為表現。

用比較簡單的講法： attention mechanism 可以幫助模型對輸入 sequence 的每個部分賦予不同的權重， 然後抽出更加關鍵的重要訊息，使模型可以做出更加準確的判斷。

Attention model 的架構如圖四：

<figure class="image">
<center>
  <img src="./attention_bahdanau.png" style="zoom:60%" />
  <figcaption>
  圖四(Image credit:[3])
  </figcaption>
</center>
</figure>

Decoder's conditional probabilit: $P\left(y_i|y_1, y_2,\dots,y_{i-1}, x\right) = g\left(y_{i-1}, s_i, c_i\right)$

$s_i$ is hidden state: $s_i = f(s_{i-1}, y_{i-1}, c_i)$

Here the probability is conditioned on a distinct context vector $c_i$ for each target word $y_i$

The context vector $c_i$ is depends on a sequence of annotation $(h_1, h_2,\dots,h_{T_x})$  to which an encoder maps the input sentence.

$c_i$ 是針對 $h_j$ 進行 weight sum 計算 :$c_i = \displaystyle\sum_{j=1}^{T_x}\alpha_{ij}h_j$

$\alpha_{ij}$ 則是對應 $h_j$ 的權重： $\alpha_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}{T_x}exp(e_{ik})}$

$e_{ij}$ 是 alignment model which scores how well the inputs around position j and the output at position i match. The score is based on the RNN hidden state $s_{i−1}$  and the $j-th$ annotation $h_j$ of the input sentence.：$e_{ij} = a(s_{i-1}, h_j)$

attenion value and query 的理解不要被公式混淆，而是從 attention 的概念去了解，query 就是

## Refenece

Illustrate:

1. [The IIIustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
2. [w淺談神經機器翻譯 & 用 Transformer 與 Tensorflow2](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html#top)
3. [Attention is all you need 解讀](https://zhuanlan.zhihu.com/p/34781297)
4. [Transformer model for language understanding by google](https://www.tensorflow.org/tutorials/text/transformer)
5. [How Self-Attention with Relative Position Representations works](https://medium.com/@_init_/how-self-attention-with-relative-position-representations-works-28173b8c245a)

Tutorial:

1. [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt#background-on-the-attention-mechanism)
2. https://www.tensorflow.org/tutorials/text/transformer
3. [Guide annotating the paper with PyTorch implementation](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

Visualization:

1. https://github.com/jessevig/bertviz

