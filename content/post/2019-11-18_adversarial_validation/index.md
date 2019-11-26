---
title: "Adversarial validation"
date: 2019-11-18
lastmod: 2019-11-18
draft: true
authors: ["Roymond Liao"]
categories:
    - Kaggle
tags: ["Adversarial validation"]]
markup: mmark

# Featured image
# To use, add an image named `featured.jpg/png` to your project's folder. 
image:
  caption: Photo by rawpixel on Unsplash
  focal_point: Smart
  preview_only: false
---
在重新回顧 Kaggle 近期的 IEEE-CIS Fraud Detection 的比賽中，發現有人提到一個 Features selection 的方法 `Adversarial validation`。

# Problem

在 model building 時常常都會遇到 training set 與 testing set 的分佈存在明顯的差異的，而在分佈不相同的狀況下，即使我們使用 Kfold 的方法來驗證 model，也不會得到較好的結果，因為在驗證所取得的 validation set 也會與 testing set 有著分佈上的差異。

在現實的處理方法，可以透過重新收集數據或是一些處理手段，來取得 training 與 testing set 分佈相同的，但在資料的比賽中， training set 與 testing set 都是給定好的數據，並無法做其他跟改，而面對這樣的狀況， Adversarial validation 就是一個很好來處理這樣的問題。

# Mothed

其實 Adversarial validation 的概念非常簡單，幾需要幾的步驟即可完成：
1. 將 training set 與 testing set 合併，並標注新的 target column `is_train`，$training = 1, testing = 0$
2. 建立一個 classifier
3. 將 training set 的預測機率按照 descending 的方式排序
4. 取 Top $n%$ 的數據當作 validation set

藉由這樣的方式所取得的 validation set 在分佈上就與 testing set 相似，如果 model 在 validation 上取得好的預測結果，那相對也能反映在 testing。

# Understanding

* Model 的 AUC 約會等於 0.5，表示 training 與 testing set 來自相同的分佈
* 



# Refenece
1. http://fastml.com/adversarial-validation-part-one/
2. http://fastml.com/adversarial-validation-part-two/
3. https://blog.csdn.net/weixin_43896398/article/details/84762922


