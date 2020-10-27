---
title: "[Paper-NLP] ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"
date: 2020-10-08
lastmod: 2020-10-27
draft: False
authors: ["Roymond Liao"]
categories:
    - NLP
    - Deep Learning
tags: ["ALBERT", "BERT Series", "NLP"]
markup: goldmark
image:
  placement: 2
  caption: 
  focal_point: "Center"
  preview_only: false
---

# ALBERT

在 2017 年 Transformer 的誕生，突破了 RNN、LSTM、GRU ... 等在計算上的限制，也帶來新的觀點，爾後再 2018 年底 **Google** 發表了 **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** 開啟了通往偉大到航道的路線，也帶起了 pre-training model 的各種應用，不用再辛苦的從頭開始訓練，為了資料問題所苦惱。在 BERT 之後，湧出各種基於 BERT 的架構下進行優化改進，例如： GPT-2、XLNet、RoBERTa、ERNIE ... 等這些耳熟能詳的模型，而這次將是為大家介紹也是基於 BERT 的架構下 Google 在 2019 年推出的輕量化版本的 BERT，**ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**。

## Introduction

如前言所述，在 BERT 後的時代透過大量的資料進行自監督(self-supervised[^1])的訓練，提高模型參數與更深層的結構，讓模型取得更好的表現，但也因為硬體上的限制，要訓練參數量大的模型就可能需要做到平行化處理以及記憶體內存的管控，但這樣的方式並沒有解決成本上的問題。基於這樣的情況，作者提出了下列問題：

<div align="center"><b> Is haveing better NLP models as easy as hvaing larger models?  </b></div>

也因為這個問題的討論，造就了 **A Lite BERT(ALBERT)** 的模型架構出來。

## Model architecture 

ALBERT 的模型架構與 BERT 相似，都是使用 transformer encoder 搭配 GELU nonlinearities 為主軸，但為了降低模型的參數並且獲得更好的表現，採用了兩種降低模型參數的方法與更換不同的預訓練任務，接下來將會一一介紹。

### Reduction techniques

ALBERT 採用了兩種減少參數的方法，解決在預訓練時模型擴展的問題，以下分別來解說採用的方法。

#### 1. Factorized embedding parameterization

在 BERT、XLNet、RoBERTa 中，都採用了 **WordPiece** 的方法，其 WordPiece embedding size $E$ 與 hidden layer size $H$ 是綁定在一起的，也就是說 size  大小一模一樣。這樣的方式對於模型來說是次優的辦法，而非最好的選擇，原因在於：

* WordPiece embedding 學習的是語句上下文相互獨立的表示方式
* Hidden-layer embedding 學習的是語句上下文有相關的表示方式

由於各別 embedding 在學習語句上的概念不同，所以作者在這邊進行了拆解，將原本 $E$ 與 $H$ 的 size 個別獨立，從 $ E \equiv H \rightarrow H \gg E $，這樣的拆解可以更有效優化模型的參數，也大幅降低了總參數量。因為如果採用原本的方式，綁定在一起，那麼當中提高 $H$ 的 size，那根據 vocabulary size $V$ 的大小(通常在一般的情況下 $V$ 是很大的，而在 BERT 中 $V$ 大約為 30,000)，WordPiece embedding  將會是 $V \times E$ 的矩陣大小，這樣是很容易得到一個有十億級別的模型參數，而且大部分的 embedding 在訓練期間都很少量的更新。

因此 ALBERT 對 WordPiece embedding 將行因式分解(factorization)，將其拆解成兩個小矩陣。先將 One-hot vector 投影到大小為 $E$ 的低維度空間中，然後再從低維度空間投影回 Hidden-layer embedding。模型在 embedding parameters 從原本的 $O\left(V \times H \right) \rightarrow O\left( V \times E + E \times H \right)$，大幅的降低模型的參數。


#### 2. Cross-layer parameter sharing

參數共享的這個想法，作者指出在 Transformer[^3] 時就有被拿來討論過，並使用在 encoder-decoder 的任務上而非是針對 pretraining/finetuning 的訓練上(這邊作者指的部分我認為應該是說 Embedding layer 與 pre-softmax linear transformation layer 的參數共享部分)。在 Dehghani et al. (2018)[^4] 所提出的 **Universal Transformer** 就展現了cross-layer parameter sharing 的方法在語言模型上得到更好的結果，在近年 Bai et al. (2019)[^5] 所提出的 **Deep Equilibrium Model (DQE)** 發現某一些層的 input embedding 與 output embedding 會達到一個平衡點(這邊平衡的解釋，需要去閱讀該篇 paper 才能比較了解，對此初步的了解認為在較深的 layer 時參數會達到一個平穩的狀態，不再震盪，達到收斂)。

參數共享的方法有很多種，比如說 only sharing  feed-forward network parameters、only sharing attention parameters 或是 sharing all parameters across layers ...等，在 ALBERT 中採用的是 sharing all parameters across layers。作者藉由計算向量相c似度的方法 L2 distances 與 cosine similarity 來衡量 input embedding 與 output embedding 在深層網路中是收斂還是震盪，如下圖：

<figure class="image"> 
<center>
  <img src="./figure_1.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>

透過圖的顯示，作者在這邊觀察到 embedding 在 BERT 中是呈現震盪變化，而採用 cross-layer parameter sharing 的 ALBERT 在深層網路中是逐漸平滑的趨勢，這樣的現象也告訴我們 weight-sharing 的方法對於穩定網路的參數有著顯著的影響。

在這部分作者也提到說，僅管的 BERT 相比，有明顯的下降震盪的狀況，但即使經過了 24 層，也不會收斂到 0 (也就是 input embedding 與 output embedding 完全相似)，對比前面提到得 DQE 所找到的解決方案有很大的不同。(這 DQE 感覺很神奇啊！需要好好來閱讀一番)

### Task change

BERT 的預訓練有兩個項目，一個是 masked language modeling (MLM)，另一項是 Next-sentence prediction (NSP)，在 ALBERT 的預訓練任務也參考了 BERT 的任務，採用了MLM 作為訓練任務之一，但另一項任務並非使用 NSP，而是採用了新的預訓練任務稱為 Sentence ordering objectives (SOP)，接下來將會解說兩者的差異，為什麼 ALBERT 要更換預訓練任務。

#### Sentence ordering objectives

* Next-sentence prediction (NSP)
  * 目的：學習語句之間的關係，預測第二個句子是否為上一個句的下一句，為一個二分類的訓練
  * Targets 產生的方式:
    * 是下一句(positive) $\rightarrow$ 同一文檔內的連續句子
    * 不是下一句(negative) $\rightarrow$ 不同文檔的句子組合
  * Positive 與 Negative 資料比例各站 50 %

* Sentence ordering objectives (SOP)
  * 目的：與 NSP 一樣，都是學習語句之間的關係，預測第二句子是否為上一句的下一句
  * Targets 產生的方式:
    * 是下一句(positive) $\rightarrow$ 同一文檔內的連續句子
    * 不是下一句(negative) $\rightarrow$ **同一文檔內的連續句子，但是順序對調**
  * Positive 與 Negative 資料比例各站 50 %

從上面的介紹可以明顯地發現兩個任務的差異在於 negative 樣本的創造方式，而為什麼要這樣去修改呢？

其實在 Liu et al., 2019[^6] 中就有提到 NSP 對於 downstream tasks 的表現會有影響，所以在 RoBERTa 的預訓練任務中將其剔除(這部分值得深讀一下 RoBERTa)。而作者在這邊認為在學習句子之間的關係是很注重 `語句的連貫性(coherence)與銜接性(cohesion)` ，假設說前一句與後一句所要描述的東西大不相同，那麼別說是機器，人類也很難理解想要表達的意思。

此外，NSP 的學習融合了兩個主軸，一個是 **topic prediction**，一個是 **coherence prediction**，由上述的 NSP 資料，其實可以了解到，在不同文檔的句子主組合會很容易學習到 topic prediction，因為所講的東西完全不同，但卻很難學習到 coherence prediction，而這樣的情況其實與 MLM 的學習有同疊到，所以作者認為語句組合的建構是語言理解中很重要的一部分，所以採用了 SOP，避免產生 topic prediction 的問題，專注於學習 inter-sentence coherence，並且也提出了基於 coherence 的 loss，**Inter-sentence coherence loss**。

## Experimental Results

講解完 ALBERT 的主要優化與改變的地方後，接下來看看作者對於 ALBERT 與其他模型的各種實驗比較。

### Model Setup

Tabel 1 列出 ALBERT 與 BERT 的對於模型框架不同的參數設定，在 paper 中的實驗結果大多以 12 層的網路架構為主，因為在相同配置下採用 24 層的網路架構所獲得的結果是相似的，但計算成本卻比較高。

<figure class="image"> 
<center>
  <img src="./table_1.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>

### Experimental Setup

實驗設定方面，為了確保比較結果是具有可參考性的，這邊作者的設定是依據 BERT 所採用的 BOOK CORPUS (Zhu et al., 2015)8 與 English Wikipedia (Devlin et al., 2019)9 來作為預訓練任務資料集，總共大約 16 GB 的資料量。Input length 最大長度限制為 512， 並以 10% 的機率隨機產生長度小於 512 的語句，也與 BERT 依樣， vocabulary size 也是 30,000 個字詞，並且使用 SentencePiece (Kudo & Richardson, 2018)[^10]作為斷詞的方法，XLNet 也有同樣有採用這個斷詞方法。

對於 MLM 任務的輸入，BERT 採用的是 random masking 的方式，對 15% 的**詞**作 mask，而 ALBET 採用 n-gram masking (Joshi et al., 2019)[^11]的方式，$n$ 最大取 3，相對的比較保有語意的訊息。n- gram  masking 的機率如下：

$$
p\left(n\right) = \frac{1/n}{\sum_{k=1}^{N} 1/k}
$$

當 n 越大，被選擇 mask 的機率就相對較低，例如：1-gram 的機率為  6/11，2-gram 的機率 3/11，3-gram 的機率  2/11 這樣。

其他參數設定：

* Batch size: 4096

* Optimizer: LAMB with learning rate 0.00176

* Train steps: 125,000 steps

* Device: Gloud TPU V3

以上的設定都是用於實驗中的所有模型。

### Comparison between BERT and ALBERT

底下展示 BERT 與 ALBERT 在不同參數下的模型表現，作者在文中將 BERT-large 當作比較的 baseline，ALBERT-xxlarge 在參數量上只有 BERT-large 的 70%，而且各項向下游任務都表現得較好，雖然訓練速度上慢了有約 3 倍，但仍顯示出 ALBERT 的設計方式對模型的表現有很大的改善。不過這邊有一點要注意的是這邊比較是訓練的時間，如果是在推論的狀況下，速度上應該是不會有太大的差異。

<figure class="image"> 
<center>
  <img src="./table_2.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>

### Factorized Embedding Parameterization

Table 3 進行了不同 vocabulary embedding size 的設定對模型的節果影響，採用的模型為 ALBERT-base，其 configurations 的設定如 Table 1 所提的一樣，實驗的設定採用 not-shared parameters(BERT-style) 與 all-shared parameters(ALBERT-style) 的比較，從結果可以看得出 $E$  的大小對於實驗結果並不一定帶來正相關的反應，因為在後續模型在使用 shared parameters 的條件下 $E$ 都是設為 128，因為在 shared parameters 的條件下 $E = 128$  表現最好 。

<figure class="image"> 
<center>
  <img src="./table_3.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>

### Cross-layer parameter sharing

在這一部分的實驗，採用幾個不同的參數分享策略：

* All-shared
* Not-shared
* Only the attention parameters arr shared
* Only the FFN parameters arr shared

<figure class="image"> 
<center>
  <img src="./table_4.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>
在 paper 中作者還有提及另外一種參數分享策略，就是將模型的堆疊層數區分為 $N$ 組，每一組的大小為 $M$ 層做參數共享，實驗結果顯示當 $M$ 越小，其表現越好，但相對的較低 $M$ 的大小，也提升了整體參數的數量。

### Sentence Order Prediction (SOP)

Table 5 展示了在 `sentence-prediction loss` 的影響下，對於固有任務(Intrinsic Tasks)與下游任務(Downstream Tasks)的模型表現：

<figure class="image"> 
<center>
  <img src="./table_5.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>
由上表可以看到在固有任務的表現，NSP 的訓練效果在 SOP 的預測上與隨機預測沒什麼差異，可以說是完全預測不出來，相反 SOP 的訓練下，在 NSP 的表現依然非常好(78.9% 的準確度)，由此可以確認 NSP 的任務學習只有對於 topic prediction 的部分有效。另外對於下游任務的表現也是基於 SOP 的訓練模型表現較好。

### Train with same amount of time 

前面 Table 2 比較了在相同的訓練下 steps 下的模型表現差異，作者還比較了在同樣的訓練時間下模型的表現情況，如 Table 6 的結果，雖然 ALBERT-xxlarge 在差不多的同樣訓練時間內走的步數不如 BERT-large，但是在表現上卻優於  BERT-large，所以並非是訓練的步數越多或是訓練的時間越久就能夠得到更好的結果。

<figure class="image"> 
<center>
  <img src="./table_6.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>

### Additional training data and dropout effects

借鏡 XLNet (Yang et al., 2019)[^7] 與 RoBERTa (Liu et al., 2019)[^6] 都有透過增加而外的資料來提升模型的表現，作者也比較了增加額外的資料的結果。如下表 Table 7 結果，看起來並非全部的下游任務都有改善，除了 SQuAD 以外的任務都有提升。

<figure class="image"> 
<center>
  <img src="./table_7.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>

下圖展示了在採用額外資料與否的條件下，對於 MLM 任務有明顯的差異。此外也比較了 dropout 的效用，在訓練超過 1M 步後，模型仍然沒有發生 overfitting 的問題，所以直接刪除掉 dropout，比較結果於 Table 8。其實在 Szegedy et al., 2017[^12] 的實驗與 Li et al., 2019[^13] 就已經證明結合 batch normalization 與 dropout 會有害 Convolutional Neural Networks 的結果表現。

<figure class="image"> 
<center>
  <img src="./figure_2.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>

<figure class="image"> 
<center>
  <img src="./table_8.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>

實驗的部分，作者還有測試了不同的 NLU tasks 的比較，這邊就留給讀者自己去看看，不再贅述。個人覺得在 Appenidx 的部分，值得好好看一下。

### Effect on network depth and width

在 Appenidx 的部分，作者對於網路的寬度與深度進行了實驗，模型採用的是 ALBERT-large。實驗採用的方法是在前 2 層做 trained，而當超過 3 層(包含第 3 層)的深度網路則採用 fine-tuning 的方式來訓練，底下 Table 11 可以得知，在相同的參數下，當層數的堆疊從 $1 \rightarrow 3$ 層時，模型的表現相對提升；但是當層數堆疊持續增加時，模型的提升效益就開始減緩，可以看當層數從 $12 \rightarrow 24 \rightarrow 48$ 的模型表現，24 與 48 的表現已經差不多了，所以再堆疊上去對模型而言並不會有太多的提升。

<figure class="image"> 
<center>
  <img src="./table_11.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>
比較玩深度的狀況，來比較寬度的狀況，模型採用的是 3 層堆疊的 ALBERT-large。當增加 hidden size 的時候，也發生與深度堆疊的狀況，當增加到一定的程度後，效果開始減緩，寬度網路甚至還下降。

<figure class="image"> 
<center>
  <img src="./table_12.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>

在 Tabel 11 比較了深度的影響，但是在 ALBERT-large 的 hidden size 為 1024，如果是更寬更深的堆疊網路層，是否能夠得到更好的效果。作者對此提出了疑問

> Do very wide ALBERT models need to be deep(er) too?

採用 hidden size 為 4096 的 ALBERT-xxlarge 來進行實驗，在 Table 13 可以發現效果其實差不多，因此認為對於 ALBERT 來說不需要太深層的網路，只需要 12層就足夠了。


<figure class="image"> 
<center>
  <img src="./table_13.png" style="zoom:100%" />
  <figcaption>
  Image credit: <a href="https://arxiv.org/abs/1909.11942">Paper</a>
  </figcaption>
</center>
</figure>


## Conclusion

這篇 paper 其實讀起來不難，只要有了解過 BERT 再來閱讀後，可以很快速地了解優化的部分，而且其優化的方法也並非創新，在其他的 paper 都有被討論過，比較新的部分，應該是預訓練任務的替換，SOP 的訓練確實帶來很大的改善，此外在 ERNIE 推出一個新的預訓練任務叫說 Sentence Reordering Task (SRT)，是對於學習句子的關聯性更加強的預訓練任務，可以好好看一下。另外，paper 中也不斷地提及 XLNet 與 RoBERTa 對於 ALBERT 的啟發與影響，所以這兩邊也值得好好閱讀一翻，有時間再來寫一下。

題外話，在大幅減少參數的情況下，其實不只是訓練速度加快，另外來為平行運算的方式帶來很大的優化就是傳輸量的減少，我們都知道在做 multi-gpu 運算所採用到策略有 Central Storage Strategy 與 Parameter Server Startegy 兩種，前者需要將 gradient 的結果回傳到 Master 做全局的 gradient update，而後者則是各個 worker 之間互相溝通傳遞參數，直到所有 worker 都 update 完畢，所以減少了參數量，相對的在傳輸部分也縮短了時間。

## Reference

[^1]: Self Supervised Representation Learning in NLP, Amit Chaudhary, https://amitness.com/2020/05/self-supervised-learning-nlp/

[^2]: Dan Hendrycks and Kevin Gimpel., Gaussian Error Linear Units (GELUs)., arXiv preprint arXiv:1606.08415, 2016. 
[^3]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. 2017
[^4]: Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. arXiv preprint arXiv:1807.03819, 2018.
[^5]: Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. Deep equilibrium models. In Neural Information Processing Systems (NeurIPS), 2019.
[^6]: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692, 2019.
[^7]: Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V Le. XLNet: Generalized autoregressive pretraining for language understanding. arXiv preprint arXiv:1906.08237, 2019.

[^8]: Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pp. 19–27, 2015.
[^9]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https: //www.aclweb.org/anthology/N19-1423.
[^10]: Taku Kudo and John Richardson. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 66–71, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-2012. URL https://www.aclweb.org/anthology/D18-2012.
[^11]: Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S Weld, Luke Zettlemoyer, and Omer Levy. SpanBERT: Improving pre-training by representing and predicting spans. arXiv preprint arXiv:1907.10529, 2019.

[^12]: Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alexander A Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning. In Thirty-First AAAI Conference on Artiﬁcial Intelligence, 2017.
[^13]: Xiang Li, Shuo Chen, Xiaolin Hu, and Jian Yang. Understanding the disharmony between dropout and batch normalization by variance shift. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2682–2690, 2019.