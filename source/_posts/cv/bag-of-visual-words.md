---
title: 使用 Bag of Visual Words 做圖片分類
date: 2020-06-20 14:00:00
tags: [電腦視覺, computer vision, bag of words, bag of visual words, python]
des: "本篇介紹如何用 Bag of Visual Words (BoVW) 模型來做圖片辨識，先將取得圖片集的特徵篩選出有哪些群集，然後對每張照片根據其特徵得到他們的 Histogram，最後用分類器來辨識測試圖片屬於哪一種類的圖片。"
lang: zh
translation_key: bag-of-visual-words
---

## 1. 簡介

Bag of Visual Words (BoVW) 模型源自於 Bag of Words (BoW, 詞袋模型)，是一種圖片分類方法。

在文件檢索中，BoW 模型將文件所包含的字彙 (Words) 分裝在不同的袋子中，每個文件所含的字彙種類不同則會對應到不同的袋子。舉例來說，假設有三個袋子分別包含「汽車、機車、馬路」、「醫生、感冒、生病」、「貓咪、金魚、狗狗」，那今天如果有一個文件裡面提到「汽車」，那我們就比較相信這文件屬於第一個袋子。BoW 簡單來說就是這樣做分類。

同理，在 BoVW 中，因為我們做的是圖片分類，所以這邊的 Word 就會是圖片的特徵，例如一個袋子包含一個人的「眼睛、鼻子、嘴巴」，那麼當我們在辨識一張圖片的時候，看到類似的圖片特徵，例如裡面包含「鼻子、嘴巴」，就可以將他歸類在是這個袋子，反之如果圖片特徵是「桌腳、扶手」那麼就不會是這個袋子，以此方法我們就可以做圖片分類。

## 2. BoVW 原理與實作


### 2.1 找出 Kmeans Clusters

我們想要辨識一群照片，這些照片可能有好幾種種類，例如種類分為建築物、森林、動物等。

首先將所有照片的所有特徵都蒐集起來，裡面可能包含家具、屋頂、樹葉、貓尾巴、樹枝等等特徵。

然後用 Kmeans 去歸納這些特徵，找出不同的群集 (Clusters)，這個群集就是 BoVW 中的袋子。如果特徵相近，他們就有可能都被歸類在同一個袋子，像是貓尾巴、狗耳朵因為都是動物的器官，他們可能就都屬於同一個袋子。

![Kmeans 示意圖](https://user-images.githubusercontent.com/18013815/85189084-21eebf80-b2de-11ea-93e3-4116ac2b4fcf.png)

Kmeans 的群集是我們可以決定的，一般來說會取辨識種類的 10 倍，亦即如果一群照片要被分成 5 種的話，Kmeans 會區分成 50 個群集，一個群集可以視為是一個特徵的集合。如上圖，所有跟山岳類似的特徵都會被分到「山」的特徵集合。

以下示範如何取得特徵與 Kmeans 集合：

```py
import numpy as np
import cv2
from cyvlfeat.kmeans import kmeans
from cyvlfeat.sift.dsift import dsift

def get_clusters(images, cluster_size, method="dsift"):

    bag_of_features = []

    for img in images: # 遍歷所有圖片

      if(method == "dsift"):
        _keypoints, descriptors = dsift(img, step=[5,5], fast=True) 
      elif(method == "orb"):
        orb = cv2.ORB_create()
        _keypoints, descriptors = orb.detectAndCompute(img, None)
      elif(method == "sift"):
        sift = cv2.xfeatures2d.SIFT_create()
        _keypoints, descriptors = sift.detectAndCompute(img, None)

      if descriptors is not None:
          for des in descriptors:
              bag_of_features.append(des)

    clusters = kmeans(np.array(bag_of_features).astype('float32'),
                      cluster_size, initialization="PLUSPLUS")

    return clusters

feature_clusters = get_clusters(images, 150, method="dsift")
```

在上面實作中，`descriptor` 就是一張圖片的其中一個特徵描述量，也可以說是一個 word，通常是一個描述局部區域的矩陣。取得特徵的方法有很多，像是程式碼中就提供了 DSIFT (Dense SIFT)、SIFT、ORB 三種擷取方法。根據實測 DSIFT 在分類問題中表現最好，在文後會仔細討論。

Kmeans 有很多 Python 函式庫都有提供，不過經過實測 cyvlfeat 表現比 sklean 和 scipy 都快，但如果要更快的話，還有其他用不同演算法時實現的函式庫，根據數據多寡和要分群的大小不同算法效能上也會有差。

### 2.2 圖片的 Histogram

一張照片會包含許多不同特徵，這些特徵分別屬於剛剛得到 Kmeans Clusters 中的不同群集。

如果這張照片是山嶽的話，那麼它的特徵應該大部分都會跟山、山峰、山嶽等等屬於相同的群集。於是我們可以去統計這張照片的特徵，分別屬於不同群集的數量，就可以判斷他應該是哪一種類的照片。這個方法方法就是 BoVW。

實務上，我們可以用 Histogram (統計圖) 來表示 BoVW，記錄這張圖片對應群集的分布。

![histogram 示意圖](https://user-images.githubusercontent.com/18013815/85189274-18665700-b2e0-11ea-90a1-5429862d3b81.png)

上方示意圖可以看到，每張照片對應的 Histogram 都不太一樣，以此我們可以推論他們分別屬於哪一種類的照片，如果是家具建築物可能紅色就多一點，如果是自然可能藍色特徵就多一點。

```py
def image_histogram(image, feature_clusters, method="dsift"):
            
    # 這邊我只寫出 dsift 以節省空間
    if(method == "dsift"):
      _keypoints, descriptors = dsift(image), step=[5,5], fast=True)
    
    # 比較這張照片的特徵和群集的特徵
    dist = distance.cdist(feature_clusters, descriptors, metric='euclidean')
    
    # 替這張照片每個特徵選最接近的群集特徵
    idx = np.argmin(dist, axis=0)
    
    # 建立 Histogram
    hist, bin_edges = np.histogram(idx, bins=len(feature_clusters))
    hist_norm = [float(i)/sum(hist) for i in hist]

    return hist_norm
```

這邊要注意 `get_clusters` 和 `image_histogram` 找特徵的方法要統一，例如統一用 `dsift`。

### 2.3 KNN 分類

要能辨識照片，我們會有一組訓練集和一組測試集，訓練集用來學習某個類別的照片的 Histogram 應該長怎樣，測試集則是判斷我們分類的正確性。

我們先替訓練集找出 Histogram 後，再去算要測試的照片的 Histogram，看這張照片的 Histogram 最接近哪一個種類的 Histogram，我們就可以知道這張照片屬於哪一種類。

你可以考慮用 K Nearest Neighbors (KNN) 分類器，演算法如下：

1. 找出這張測試照片的 Histogram
2. 比較這個 Histogram 與訓練集所有種類圖片的 Histogram 的差異 (向量距離差)
3. 選出與此照片 Histogram 差異最小的 K 個訓練集圖片
4. 統計這 K 個訓練圖片分別屬於哪個種類的分布
5. 選出票數最高的種類，此為測試照片的種類

### 2.4 SVM 分類

KNN 正確率大概介於 40~50%，我們可以考慮用 [SVM](https://zh.wikipedia.org/wiki/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA) 來做分類，準確率可以達 60~70%。當你不知道要用甚麼分類器的時候通常可以試試看 SVM，處理速度算滿快的。

SVM 分類的簡單原理是將不同群以一條線切開，同時這條線應該和兩邊的群都保有一定距離，如下圖所示：

![SVM image](https://user-images.githubusercontent.com/18013815/85191168-7438dc00-b2f0-11ea-9a69-7d28fe3465eb.png)
(Source: [Dhairya Kumar](https://towardsdatascience.com/demystifying-support-vector-machines-8453b39f7368))

SVM 的原理和實作可以參考這篇「[Support Vector Machine — Introduction to Machine Learning Algorithms](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)」。

實作上可以考慮用 [Scikit-Learn SVM](https://scikit-learn.org/stable/modules/svm.html_) 或是 [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/#python) 函式庫，我們將訓練集找出的 Histogram 丟進 SVM 學習，再讓 SVM 去猜測試的照片是屬於哪個種類。

**完整的程式碼範例可以在這邊找到：[連結](https://gist.github.com/tigercosmos/a5af5359b81b99669ef59e82839aed60)**

## 3. 問題討論

已知一批照片，並且用 BoVW 來做分類。照片分為 15 種類，每一個種類都有 100 張 256*256 像素的灰階照片，用來當作訓練集，此外每一類都有 10 張同規格照片作為測試集。

種類如下：
![圖片種類](https://user-images.githubusercontent.com/18013815/85193818-00053500-b2fe-11ea-95f7-cc8343688817.png)

測試環境是 Macbook Pro 2017 with 2.3 GHz Intel Core i5 搭配 Python 3.6。

實驗結果如下：

| Descriptor  | Kmeans Cluster |  KNN(%)  | Linear SVM(%)   |  Time(min) |
|---|---|---|---|---|
| DSIFT |  60 |  49.3 | 58  | 12:10  |
| DSIFT  | 150  | 40.7  | 57.3  |  20:52 |
| DSIFT  |  300 |  39.3 | **60.7**  |  51:19 |
| SIFT |  60 |  30 |  42.7 |  4:33 |
| SIFT  | 150  | 24.7  | 38.7  |  6:33 |
| SIFT  |  300 |  22.7 | 34.7  |  9:57 |
| ORB |  60 |  12.7 |  14 |  0:30 |
| ORB  | 150  | 12.7 | 17.3  |  1:15 |
| ORB  |  300 |  10.0 | 13.3  |  2:23 |

### 3.1 特徵演算法差異

可以看到 DSIFT 雖然花費時間最長，但表現也最好。令人意外的是 ORB 的雖然非常快但結果十分差。

這其實可以理解，DSIFT 是一種 Dense Feature Extraction，白話來說就是一次取用好幾個特徵來用，那麼對於物體形狀的辨識也會比較優秀。反之 ORB 這種主要拿來找 Key Points 的演算法，對於局部的理解的比較低，就比較難辨識出形狀。

### 3.2 Kmeans 群集數量差異

Kmeans 需要在獨特性和穩健度上取得平衡，分群數量較少，群集差異越大，反之分群數量較多，差異度可能不是很大，但結果可能會比較穩健，這也會影響演算法的效能。

實驗中分別設定 Kmeans 的群集數量為 60、120、300，分別是圖片種類的 2 倍、4 倍、20 倍，發現 DSIFT 在 300 個群集時表現最優，但也只好一點點，代價卻是幾乎兩倍以上的時間，因為 Kmeans 的複雜度是 $O(t \times k \times n \times d)$，其中 $t$ 是迭代次數，$k$ 是群集數量，$n$ 是多少個點，$d$ 是每個點的維度。

此外可以觀察到 SIFT 卻是 60 時最高，而 ORB 則是 150 最高。

因此在此實驗情況下分群不需要很大，也許是因為照片種類差異比較大，少量群集就可以分辨出不同種類。

實務上如果分類效果不佳，可以試試看 Hierarchically clustering，亦即將原本的群集再切割成更多小的群集。

### 3.3 不同分類器

根據實驗可以觀察到 SVM 表現是比 KNN 優秀的。

## 4. 結論

本篇介紹如何用 Bag of Visual Words (BoVW) 模型來做圖片辨識，先將取得圖片集的特徵篩選出有哪些群集，然後對每張照片根據其特徵得到他們的 Histogram，最後用分類器來辨識測試圖片屬於哪一種類的圖片。
