---
title: "Image Classification with Bag of Visual Words (BoVW)"
date: 2020-06-20 14:00:00
tags: [computer vision, bag of words, bag of visual words, python, image classification, kmeans, knn, svm]
des: "This post introduces Bag of Visual Words (BoVW) for image recognition: build clusters from local features, represent each image as a histogram over clusters, and train a classifier to recognize test images."
lang: en
translation_key: bag-of-visual-words
---

## 1. Introduction

The Bag of Visual Words (BoVW) model is derived from the Bag of Words (BoW) model and is a method for image classification.

In information retrieval, BoW “packs” the words contained in a document into different bags. Different documents contain different sets of words, and thus correspond to different bags. For example, suppose we have three bags containing:

- “car, motorcycle, road”
- “doctor, cold, illness”
- “cat, goldfish, dog”

If a document mentions “car”, we are more likely to believe it belongs to the first bag. In short, BoW classifies documents in this way.

Similarly, in BoVW we classify images, so the “words” are image features. For example, if a bag contains a person’s “eyes, nose, mouth”, then when we recognize an image and see similar features (e.g., it contains “nose, mouth”), we can classify it into that bag. Conversely, if an image contains features like “table legs, armrests”, then it likely does not belong to that bag. With this idea, we can do image classification.

## 2. BoVW Principles and Implementation

### 2.1 Finding K-means Clusters

Suppose we want to recognize a set of photos, and the photos may belong to multiple categories (e.g., buildings, forests, animals, etc.).

First, we collect all features from all images. They may include features corresponding to furniture, rooftops, leaves, cat tails, branches, and so on.

Then we use K-means to summarize these features and find different clusters. These clusters correspond to the “bags” in BoVW. If features are similar, they are likely assigned to the same bag. For instance, cat tails and dog ears are both animal body parts, so they might end up in the same bag.

![Kmeans 示意圖](https://user-images.githubusercontent.com/18013815/85189084-21eebf80-b2de-11ea-93e3-4116ac2b4fcf.png)

You can choose the number of K-means clusters. A common heuristic is to set it to about 10× the number of classes. For example, if you want to classify photos into 5 categories, you may set K-means to create 50 clusters. Each cluster can be viewed as a set of similar features. As shown above, features similar to “mountains” are grouped into the “mountain” feature set.

Below is an example of how to extract features and build K-means clusters:

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

In the implementation above, a `descriptor` is a feature descriptor from an image—i.e., a “word”. It is typically a matrix describing a local region. There are many ways to extract features; the code shows three options: DSIFT (Dense SIFT), SIFT, and ORB. Based on experiments, DSIFT performs best for this classification task, which I will discuss later.

Many Python libraries provide K-means. In my experiments, `cyvlfeat` is faster than `sklean` and `scipy`. If you need even more speed, there are other libraries that implement K-means with different algorithms—the performance can vary depending on dataset size and the number of clusters.

### 2.2 Image Histograms

An image contains many features, and each feature belongs to one of the K-means clusters obtained above.

If an image is of mountains, most of its features should belong to clusters related to mountains, peaks, etc. So we can count how many of the image’s features fall into each cluster, and then infer which category the image belongs to. This is the core idea of BoVW.

In practice, we represent BoVW using a histogram that records the distribution of clusters for an image.

![histogram 示意圖](https://user-images.githubusercontent.com/18013815/85189274-18665700-b2e0-11ea-90a1-5429862d3b81.png)

From the diagram above, you can see that each image has a different histogram. Based on this, we can infer which class it belongs to. For example, furniture/buildings may have more “red” features, while natural scenes may have more “blue” features.

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

One important note: `get_clusters` and `image_histogram` must use the same feature extraction method (e.g., both use `dsift`).

### 2.3 KNN Classification

To recognize images, we typically have a training set and a test set. The training set is used to learn what histograms look like for each class, and the test set is used to evaluate classification accuracy.

After we compute histograms for the training images, we compute the histogram of a test image. By finding which class histogram it is closest to, we can predict the class of the test image.

You can use a K Nearest Neighbors (KNN) classifier. The algorithm is:

1. Compute the histogram of the test image
2. Compare it with histograms of all training images (vector distance)
3. Select the K training images with the smallest distance
4. Count the class distribution among these K neighbors
5. Pick the class with the most votes as the predicted class

### 2.4 SVM Classification

KNN accuracy is roughly around 40–50%. You can consider using [SVM](https://zh.wikipedia.org/wiki/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA), which can reach about 60–70%. When you are not sure which classifier to use, SVM is often a good baseline. It is also relatively fast.

Intuitively, SVM tries to separate different groups with a line (or hyperplane), while keeping a certain margin from both sides, as shown below:

![SVM image](https://user-images.githubusercontent.com/18013815/85191168-7438dc00-b2f0-11ea-9a69-7d28fe3465eb.png)
(Source: [Dhairya Kumar](https://towardsdatascience.com/demystifying-support-vector-machines-8453b39f7368))

For SVM principles and implementation, you can refer to: “[Support Vector Machine — Introduction to Machine Learning Algorithms](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)”.

In practice, you can use [Scikit-Learn SVM](https://scikit-learn.org/stable/modules/svm.html_) or [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/#python). Feed the training histograms into SVM to learn, and then let SVM predict the class of test images.

**A complete code example can be found here: [link](https://gist.github.com/tigercosmos/a5af5359b81b99669ef59e82839aed60)**

## 3. Discussion

Given a batch of photos, we use BoVW to classify them. The photos are divided into 15 classes. Each class has 100 grayscale images of size 256×256 as the training set. In addition, each class has 10 images of the same specification as the test set.

The classes are:
![圖片種類](https://user-images.githubusercontent.com/18013815/85193818-00053500-b2fe-11ea-95f7-cc8343688817.png)

The test environment is a MacBook Pro 2017 with a 2.3 GHz Intel Core i5, using Python 3.6.

The results are:

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

### 3.1 Differences Between Feature Algorithms

You can see that DSIFT takes the most time but also performs best. Surprisingly, ORB is very fast but performs poorly.

This is understandable: DSIFT is a dense feature extraction method—plainly speaking, it samples many local features. This can be advantageous for recognizing object shapes. In contrast, ORB is mainly used for detecting key points; it has weaker understanding of the overall structure, so it is harder to recognize shapes in this setup.

### 3.2 Differences in the Number of K-means Clusters

K-means needs to balance distinctiveness and robustness. With fewer clusters, cluster differences are larger. With more clusters, differences may become smaller, but results may be more robust. This affects performance.

In this experiment, I set the number of K-means clusters to 60, 120, and 300, corresponding to 2×, 4×, and 20× the number of image classes. DSIFT performs best with 300 clusters, but only slightly better, while the cost is almost more than double the time. This is because the complexity of K-means is $O(t \times k \times n \times d)$, where $t$ is the number of iterations, $k$ is the number of clusters, $n$ is the number of points, and $d$ is the dimensionality of each point.

You can also observe that SIFT achieves its best result at 60 clusters, while ORB peaks at 150 clusters.

So under this experimental setup, we may not need a large number of clusters—perhaps because the image classes are quite distinct, and a small number of clusters is enough to separate them.

In practice, if classification performance is poor, you can try hierarchical clustering—i.e., further splitting the original clusters into smaller sub-clusters.

### 3.3 Different Classifiers

From the experiments, SVM performs better than KNN.

## 4. Conclusion

This post introduced how to use Bag of Visual Words (BoVW) for image recognition: build clusters from features extracted from an image set, compute a histogram for each image based on its features, and then use a classifier to recognize which class a test image belongs to.

