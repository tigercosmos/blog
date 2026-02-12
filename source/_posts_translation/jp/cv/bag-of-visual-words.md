---
title: "Bag of Visual Words（BoVW）による画像分類"
date: 2020-06-20 14:00:00
tags: [computer vision, bag of words, bag of visual words, python, image classification, kmeans, knn, svm]
des: "本記事では Bag of Visual Words（BoVW）モデルを用いた画像認識を解説します。特徴量からクラスタを構築し、各画像をクラスタ分布のヒストグラムとして表現し、分類器でテスト画像のクラスを推定します。"
lang: jp
translation_key: bag-of-visual-words
---

## 1. はじめに

Bag of Visual Words（BoVW）モデルは、Bag of Words（BoW、いわゆる「単語袋」モデル）に由来する、画像分類の手法です。

文書検索では、BoW は文書に含まれる単語（words）を「袋」に詰めて考えます。文書ごとに含まれる単語の種類が異なるため、どの袋に近いかで分類できます。たとえば、次の3つの袋があるとします：

- 「車、バイク、道路」
- 「医者、風邪、病気」
- 「猫、金魚、犬」

ある文書に「車」という単語が出てきたら、その文書は1つ目の袋に属する可能性が高い、と判断します。BoW の直感はだいたいこのようなものです。

同様に BoVW では、対象が文書ではなく画像なので、「単語」に相当するものが画像の特徴になります。たとえば、ある袋が「目、鼻、口」という特徴を含むとします。画像認識時に似た特徴（例えば「鼻、口」）が見えたら、その袋に分類します。逆に、画像の特徴が「机の脚、肘掛け」などであれば、その袋ではないと判断できます。このようにして画像分類ができます。

## 2. BoVW の原理と実装

### 2.1 K-means クラスタの作成

まず、複数カテゴリの写真集合を分類したいとします。カテゴリは建物、森、動物など、複数ありえます。

最初に、すべての写真から特徴量を集めます。そこには家具、屋根、葉、猫のしっぽ、枝など、さまざまな特徴が含まれます。

次に K-means でこれらの特徴を要約し、異なるクラスタ（clusters）を見つけます。このクラスタが BoVW における「袋」に相当します。特徴が似ていれば同じ袋に分類されやすく、例えば猫のしっぽと犬の耳はどちらも動物の器官なので、同じ袋に入る可能性があります。

![Kmeans 示意圖](https://user-images.githubusercontent.com/18013815/85189084-21eebf80-b2de-11ea-93e3-4116ac2b4fcf.png)

K-means のクラスタ数は任意に決められます。経験則としては「クラス数の10倍」程度がよく使われます。例えば5クラスに分類したい場合、50クラスタに分ける、といった具合です。1つのクラスタは、似た特徴の集合とみなせます。図では、山に似た特徴が「山」クラスタに集約されています。

以下は、特徴抽出と K-means クラスタの例です：

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

上の実装では、`descriptor` が画像中の特徴記述子（feature descriptor）であり、BoVW でいう「単語（word）」に相当します。通常は、局所領域を表す行列（ベクトル）です。特徴抽出法には様々なものがあり、コードでは DSIFT（Dense SIFT）、SIFT、ORB の3種類を例示しています。実験上は、分類問題では DSIFT が最も良い結果でした（後で議論します）。

K-means は多くの Python ライブラリで提供されていますが、実測では `cyvlfeat` が `sklean` や `scipy` より高速でした。さらに高速化したい場合は、別アルゴリズムで実装されたライブラリを検討できます。データ量やクラスタ数によっても性能は変わります。

### 2.2 画像のヒストグラム

1枚の画像には多数の特徴があり、それぞれが先ほど得た K-means クラスタのいずれかに属します。

例えば山の写真なら、山・峰などに関連するクラスタに属する特徴が多くなるはずです。そこで、画像の特徴が各クラスタにどれだけ割り当てられたか（数）を集計することで、その画像がどのカテゴリに属するか推定できます。これが BoVW の基本です。

実務では、BoVW をヒストグラム（統計）で表現します。つまり「クラスタ分布」をベクトルとして持ちます。

![histogram 示意圖](https://user-images.githubusercontent.com/18013815/85189274-18665700-b2e0-11ea-90a1-5429862d3b81.png)

図のように、画像ごとにヒストグラムが異なるため、これを使ってカテゴリを推定できます。家具・建築物なら赤が多い、自然なら青が多い、といった傾向が出るイメージです。

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

ここで重要なのは、`get_clusters` と `image_histogram` の特徴抽出法を揃えることです（例：どちらも `dsift`）。

### 2.3 KNN による分類

画像を認識するには、通常、訓練データ（training set）とテストデータ（test set）を用意します。訓練データで各クラスのヒストグラムの傾向を学習し、テストデータで分類の正しさを評価します。

訓練集合のヒストグラムを用意した後、テスト画像のヒストグラムを計算し、それがどのクラスのヒストグラムに最も近いかでクラスを推定できます。

分類器としては K Nearest Neighbors（KNN）を使うことができます。アルゴリズムは次のとおりです：

1. テスト画像のヒストグラムを計算する
2. そのヒストグラムと、訓練集合のすべてのヒストグラムを比較する（ベクトル距離）
3. 距離が最も小さい K 個の訓練画像を選ぶ
4. その K 個が属するクラス分布を数える
5. 最頻クラスをテスト画像のクラスとして採用する

### 2.4 SVM による分類

KNN の精度はおおむね 40〜50% 程度でした。そこで [SVM](https://zh.wikipedia.org/wiki/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA) を使うと、60〜70% 程度まで上がる場合があります。どの分類器を使うべきか分からないとき、SVM はとりあえず試す価値のあるベースラインで、処理速度も比較的速いです。

SVM の直感的な説明としては、異なるグループを1本の線（高次元なら超平面）で分離し、さらにその線が両側のグループから一定距離（マージン）を保つようにします：

![SVM image](https://user-images.githubusercontent.com/18013815/85191168-7438dc00-b2f0-11ea-9a69-7d28fe3465eb.png)
(Source: [Dhairya Kumar](https://towardsdatascience.com/demystifying-support-vector-machines-8453b39f7368))

SVM の原理と実装は、次の記事が参考になります：  
「[Support Vector Machine — Introduction to Machine Learning Algorithms](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)」。

実装としては、[Scikit-Learn SVM](https://scikit-learn.org/stable/modules/svm.html_) や [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/#python) が利用できます。訓練集合のヒストグラムを SVM に学習させ、テスト画像のクラスを推定します。

**完全なコード例はここにあります： [link](https://gist.github.com/tigercosmos/a5af5359b81b99669ef59e82839aed60)**

## 3. 議論

BoVW で分類する写真集合があり、写真は 15 クラスに分かれています。各クラスには 256×256 ピクセルのグレースケール画像が 100 枚あり、これを訓練データとします。さらに各クラスに同規格のテスト画像が 10 枚あります。

クラスは次のとおりです：
![圖片種類](https://user-images.githubusercontent.com/18013815/85193818-00053500-b2fe-11ea-95f7-cc8343688817.png)

実験環境は MacBook Pro 2017（2.3 GHz Intel Core i5）で、Python 3.6 を使用しました。

結果は次のとおりです：

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

### 3.1 特徴抽出アルゴリズムの違い

DSIFT は最も時間がかかる一方で、性能も最良でした。意外なのは ORB が非常に高速なのに結果がとても悪い点です。

これはある程度理解できます。DSIFT は Dense Feature Extraction で、平たく言えば多数の特徴を密にサンプリングします。そのため物体形状の認識に強くなりやすいです。一方、ORB は主にキーポイント検出向けで局所的な理解に寄りがちなので、この設定では形状を識別しづらくなります。

### 3.2 K-means クラスタ数の違い

K-means は「特徴の独自性」と「頑健性」のバランスが必要です。クラスタ数が少ないとクラスタ間の差が大きくなり、クラスタ数が多いと差は小さくなりがちですが、結果が安定する場合があります。これが性能に影響します。

実験ではクラスタ数を 60、120、300（クラス数の 2 倍、4 倍、20 倍）に設定しました。DSIFT は 300 で最も良かったものの、改善はわずかで、計算時間のコストはほぼ倍以上になりました。K-means の計算量は $O(t \times k \times n \times d)$（$t$：反復回数、$k$：クラスタ数、$n$：点数、$d$：次元）だからです。

また、SIFT は 60 が最大で、ORB は 150 が最大でした。

この実験条件では、クラスタ数を大きくしすぎる必要はないかもしれません。写真クラス間の差が大きく、少数クラスタでも判別できた可能性があります。

実務で分類性能が悪い場合は、階層クラスタリング（hierarchical clustering）を試すのも手です。つまり、既存クラスタをさらに細かいサブクラスタに分割します。

### 3.3 分類器の違い

実験結果から、SVM は KNN より良い性能でした。

## 4. まとめ

本記事では Bag of Visual Words（BoVW）モデルによる画像認識を紹介しました。画像集合から特徴を抽出してクラスタを構築し、各画像をクラスタ分布のヒストグラムとして表現し、最後に分類器でテスト画像がどのクラスに属するか推定します。

