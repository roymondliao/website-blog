---
title: "Adversarial validation"
date: 2019-11-18
lastmod: 2019-11-18
draft: false
authors: ["Roymond Liao"]
categories:
    - Kaggle
tags: ["Adversarial Validation", "Time Consistency", "Features Selection"]
markup: mmark

# Featured image
# To use, add an image named `featured.jpg/png` to your project's folder. 
image:
  caption: Photo by rawpixel on Unsplash
  focal_point: Smart
  preview_only: false
---
在重新回顧 Kaggle 近期的 IEEE-CIS Fraud Detection 的比賽中，發現有人提到一個 Features selection 的方法 **Adversarial validation**。

# Problem

在建立模型時常常都會遇到 training set 與 testing set 的分佈存在明顯的差異的，而在分佈不相同的狀況下，即使我們使用 Kfold 的方法來驗證 model，也不會得到較好的結果，因為在驗證所取得的 validation set 也會與 testing set 有著分佈上的差異。

在現實的處理方法，可以透過重新收集數據或是一些處理手段，來取得 training set 與 testing set 分佈相同的，但在資料的比賽中， training set 與 testing set 都是給定好的數據，並無法做其他跟改，而面對這樣的狀況， Adversarial validation 就是一個很好來處理這樣的問題。

# Mothed

其實 Adversarial validation 的概念非常簡單，只需要幾個步驟：
1. 將 training set 與 testing set 合併，並標注新的 target column `is_train` ($training = 1, testing = 0$)
2. 建立一個 classifier
3. 將 training set 的預測機率按照 descending 的方式排序
4. 取 Top $n\%$ 的數據當作 validation set

藉由這樣的方式所取得的 validation set 在分佈上就與 testing set 相似，如果 model 在 validation 上取得好的預測結果，那相對地也能反映在 testing set。

# Understanding

* Model 的 AUC 大約等於 0.5，表示 training set 與 testing set 來自相同的分佈
* Model 的 AUC 非常高時，表示 training set 與 testing set 來自不相同的分佈，可以明顯地分開

# Other

這邊提一下另一個 trick 的 features selection 方法，稱為 **time consistency**。在 IEEE-CIS Fraud Detection 比賽第一名的隊伍中，[Chris Deotte](https://www.kaggle.com/cdeotte) 提出用了這個方法來去除掉對模型沒有影響力的 features。

### Problem
不管在現實的資料或是比賽的資料，部分資料都有可能因為時間的改變而分佈有所改變，這是我們在建立模型上不太希望發生的事情。因為如果 features 會因為時間的因素而分佈有明顯變化的話，在建模的過程中，受時間影響的 features 可能就會傷害模型本身，可能在時間相近的資料驗證有好的表現，但當預測時間間隔較長的資料時就會發生 overfitting。在處理上述的情況，我們期望 features 的分佈是穩定的，不希望因為時間的影響而有所改變，所以可以使用 time consistency 的方法來剔除這些受時間影響的 features。

### Mothed
Time consistency 的步驟，這邊以 IEEE-CIS Fraud Detection 的比賽資料為例：
1. 將 training set 依據`月`為單位切分資料
2. training data 與 validation data 策略，這邊的策略可以自由調整改變，以下只舉幾個例子
    * 選擇前 n 個月的資料為 training data，最後一個月的資料為 validation data
    * 選擇前 n 個月的資料為 training data，中間跳過 m 個月份，最後一個月的資料為 validation data

3. 選擇一個 feature，進行模型建立，分別查看模型的 AUC 在 training 與 validation 是否有差異

### Understanding
* 如果 training 的 AUC 與 validation 的 AUC 差不多，表示這 feature 不受時間的變化影響
* 如果 training 的 AUC 與 validation 的 AUC 有明顯差異，表示這 feature 時間的變化影響，會影響模型本身，可以考慮移除

### Code
以下是 Chris Deotte 所提供的簡單的程式碼：
```python
# ADD MONTH FEATURE
import datetime
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
train['DT_M'] = train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
train['DT_M'] = (train['DT_M'].dt.year-2017)*12 + train['DT_M'].dt.month 

# SPLIT DATA INTO FIRST MONTH AND LAST MONTH
train = train[train.DT_M==12].copy()
validate = train[train.DT_M==17].copy()

# TRAIN AND VALIDATE
lgbm = lgb.LGBMClassifier(n_estimators=500, objective='binary', num_leaves=8, learning_rate=0.02)
h = lgbm.fit(train[[col]], 
             train.isFraud, 
             eval_metric='auc', 
             eval_names=['train', 'valid'],
             eval_set=[(train[[col]],train.isFraud),(validate[[col]],validate.isFraud)],
             verbose=10)
auc_train = np.round(h._best_score['train']['auc'], 4)
auc_val = np.round(h._best_score['valid']['auc'], 4)
print('Best score in trian:{}, valid:{}'.format(auc_train, auc_val))
```
Btw，最近有看到一個驗證的方法叫做 `Double Cross-Validation`，這邊紀錄一下，有機會再來講講這方法的概念與應用。

# Refenece
1. http://fastml.com/adversarial-validation-part-one/
2. http://fastml.com/adversarial-validation-part-two/
3. https://blog.csdn.net/weixin_43896398/article/details/84762922
4. https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308

