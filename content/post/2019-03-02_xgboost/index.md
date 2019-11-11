---
title: "XGBoost"
date: 2019-03-02
lastmod: 2019-03-02
draft: false
authors: ["Roymond Liao"]
categories:
    - Machine Learning
tags: ["Machine Learning", "XGBoost"]
summary: "Review note for XGBoost"
markup: mmark
image:
  caption: "[PPhoto by Hans-Peter Gauster on Unsplash](https://unsplash.com/photos/3y1zF4hIPCg)"
  placement: 2
  focal_point: ""
  preview_only: false
---
## Review note
Bagging

> **Concept**
>
> Bagging involves creating mulitple copies of the original training data set using the boostrap, fitting a seperate decision tree to each copy, and then combining all of the trees in order to create a single predcitive model. <font color="#F44336">Notably, each tree is built on a bootstrap data set, independent of the other trees.</font>
>
> **Algorithm**
>
> * Random Forest

Boosting

> **Concept**
>
> Boosting works in a similar way with bagging, except that the trees are grown <font color="#F44336">sequentially</font>. Each tree is grown using information from previous grown trees.
>
> **Algorithm**
>
> * Adaboost - [Yoav Freund and Robert Schapire](https://en.wikipedia.org/wiki/AdaBoost)
>
>   * 根據樣本的誤差來調整樣本的權重，誤差較大的樣本給予較高的權重，反之亦然。藉此著重訓練分類錯誤的資料，進而來增進模型的準確度。
>
> * Gradient boosting - [Friedman, J.H.](https://statweb.stanford.edu/~jhf/)
>
>   * 根據當前模型的殘差來調整權重的大小，其目的是為了降低殘差。通過迭代的方式，使損失函數(loss function)達到最小值(局部最小)。
>
>   * Method
>       * GBDT(Grandien Boosting Decision Tree)
>       * XGBoost(eXtreme Gradient Boosting)](https://github.com/dmlc/xgboost) - [Tianqi Chen](http://homes.cs.washington.edu/~tqchen/)
>       * LightGBM(Light Gradient Boosting Machine)](https://github.com/Microsoft/LightGBM) - Microsoft Research Asia
>





## Advantages of XGBoost

* 傳統 GBDT 是以 CART 作為分類器的基礎，但是XGBoost還可以支援線性分類器，另外在 objective function 可以加入 L1 regularization 和 L2 regularization 的方式來優化，降低了 model 的 variance，避免 overfitting 的狀況。
* GBDT 在優化部分只使用到泰勒展開式的一階導數，但 XGBoost 則使用到二階導數，所以在預測準確度上提供更多的訊息。
* XGBoost 支援平行運算與分布式運算，所以相較傳統的GBDT在計算速度上有大幅的提升。XGBoost 的平行並非是在 tree 的維度做平行化處理，而是在 features 的維度上做平行化處理，因為 tree 的生長是需要前一次迭代的結果的來進行 tree 的生長。
* 對 features 進行預排序的處理，然後保存排序的結構，以利後續再 tree 的分裂上能夠快速的計算每個 features 的 gain 的結果，最終選擇 gain 最大的 feature 進行分裂，這樣的方式就可以平行化處理。
* 加入 shrinkage 和 column subsampling 的優化技術。
* 有效地處理 missing value 的問題。
* 先從頭到尾建立所有可能的 sub trees，再從底到頭的方式進行剪枝(pruning)。

## Disadvantages of XGBoost
* 在每次的迭代過程中，都需要掃過整個訓練集合多次。如果把整個訓練集合存到 memory 會限制數據的大小;如果不存到 memory 中，反覆的讀寫訓練集合也會消耗非常多的時間。
* 預排序方法(pre-sorted): 由於需要先針對 feature 內的 value 進行排序並且保存排序的結果，以利於後續的 gain 的計算，但在這個計算上就需要消耗兩倍的 memory 空間，來執行。

## Reference
* http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
* http://mlnote.com/2016/10/05/a-guide-to-xgboost-A-Scalable-Tree-Boosting-System/
* https://www.zybuluo.com/yxd/note/611571#机器学习的关键元素
* https://en.wikipedia.org/wiki/Gradient_boosting#Shrinkage
* http://zhanpengfang.github.io/418home.html

## Paper 
* Fridman J.H. (1999). [Greedy Function Approximation: A Gradient Boosting Machine](http://statweb.stanford.edu/~jhf/ftp/trebst.pdf)
* Tianqi Chen, Carlos Gusetrin (2016). [XGBoost: A Scalable Tree Boosting System](http://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)

## Doing 
* https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
* https://adeshpande3.github.io/adeshpande3.github.io/Applying-Machine-Learning-to-March-Madness