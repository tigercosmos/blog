---
title: 使用 Tensorflow 開發花朵種類辨識 App
date: 2017-07-28 00:42:09
tags: [ai, tensorflow]
lang: zh
translation_key: tensorflow-flower
---

# 前言
最近機器學習、人工智慧、深度學習等等名詞非常夯，本篇用深度學習簡單實作一個可以辨識不同種類花朵的 Android App。第一部分是訓練圖片的模型，這時已經可以在電腦上作辨識了。第二部分是將第一部份訓練的模型放進 Android 裡面，實作手機辨識的部分。

<!-- more --> 

# 第一部分
## 環境設定

### Anaconda
Anaconda 可以建立虛擬環境來跑 Python，好處是可以建立很多種不同的環境來測試，並且使用 Anaconda 時 `pip` 不用 root 權限。
```
$ wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
$ bash Anaconda3-4.4.0-Linux-x86_64.sh
```

### Python & Tensorflow
建立一個虛擬環境，因為我用 python2.7 搭配 tensorflow1.1 所以環境名稱取為`py2t1.1`
```
$ conda create -n py2t1.1 python=2.7
```
安裝好環境後，要進入虛擬環境，啟用是用 `activate`
```
 $ source activate py2t1.1
 (py2t1.1)$  # 會顯示成這樣
```
而取消啟用是用 `deactivate`
```
 $ source deactivate
```

接著要安裝 `tensorflow`，我這邊版本是用 tensorflow1.1 python2.7 CPU 版本，有其他需求可以在[這裡](https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package)找其他版本。 
```
 (py2t1.1)$ pip install --ignore-installed --upgrade \
 https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl
```

這樣就把環境設置好了。

## 訓練模型

### 前置作業
這時候可以在虛擬環境裡，也可以不在。只有當要使用 python 跑 tensorflow 時才一定要進入設置好的虛擬環境。

建個資料夾
```
$ mkdir flower_classification
$ cd flower_classification
```

取得花朵照片
```
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
```

取得 `retrain.py`，這是使用 `Inception v3` 訓練好的圖片辨識模型，並在最後加上新的一層網絡，然後重新訓練一次模型來辨識照片。
```
$ curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py
```

### 開始訓練
先進入環境
```
$ source activate py2t1.1
```

```
$ python retrain.py \
  --bottleneck_dir=bottlenecks \
  --model_dir=inception \
  --learning_rate=0.5 \
  --summaries_dir=training_summaries/long \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=flower_photos
```

訓練完之後會產生 `retrained_graph.pb` 和 `retrained_labels.txt`，這兩個檔案可以拿來分析新的照片。

### 辨識照片
取得 `label_image.py` 這是辨識的程式碼
```
$ curl -L https://goo.gl/3lTKZs > label_image.py
```

執行辨識
```
$ python label_image.py 要辨識的花朵照片.jpeg
```

結果大概會像這樣
```
daisy (score = 0.99071)
sunflowers (score = 0.00595)
dandelion (score = 0.00252)
roses (score = 0.00049)
tulips (score = 0.00032)
```

# 第二部分
## 前置作業
`tensorflow-for-poets-2` 是 googlecodelab 的範例 Andriod App 專案 
```
$ cd ~/
$ git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
$ cd tensorflow-for-poets-2
```
將剛剛訓練好的花朵辨識整個資料夾 `flower_classification` 複製進去
```
$ cp -r ~/flower_classification .
```

## 處理模型
優化
```
$ python -m tensorflow.python.tools.optimize_for_inference \
  --input=flower_classification/retrained_graph.pb \
  --output=flower_classification/optimized_graph.pb \
  --input_names="Cast" \
  --output_names="final_result"
```
壓縮
```
$ python -m scripts.quantize_graph \
  --input=flower_classification/optimized_graph.pb \
  --output=flower_classification/rounded_graph.pb \
  --output_node_names=final_result \
  --mode=weights_rounded
```

將 `rounded_graph.pb` 和 `retrained_labels.txt` 放進 `android/assets` 資料夾
```
$ cp flower_classification/rounded_graph.pb flower_classification/retrained_labels.txt android/assets/ 
```

## 安裝 AndroidStudio
https://developer.android.com/studio/index.html

## 開啟 Android 專案
打開 AndroidStudio，選擇 "Open an existing Android Studio project"，選擇資料夾 `tensorflow-for-poets-2/android`。 然後它會問是否用 gradle，選擇"ok"。 等 gradle 更新好之後，就可以 build 了。最後再把 APK 安裝到手機上。

## 參考資料
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#0
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/#0
