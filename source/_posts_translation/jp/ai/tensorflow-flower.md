---
title: "TensorFlow で花の種類判別アプリを作る"
date: 2017-07-28 00:42:09
tags: [ai, tensorflow]
lang: jp
translation_key: tensorflow-flower
---

# 前書き

最近は機械学習・人工知能・深層学習などのキーワードがとても盛り上がっています。本記事では深層学習を使って、花の種類を判別できるシンプルな Android アプリを実装します。第 1 部では画像分類モデルを学習し、まずは PC 上で判別できる状態にします。第 2 部では、第 1 部で学習したモデルを Android に組み込み、スマホで判別できるようにします。

<!-- more --> 

# 第 1 部
## 環境設定

### Anaconda
Anaconda は Python を動かすための仮想環境を作れます。複数の環境を作って試せるのが便利です。また、Anaconda を使う場合は `pip` に root 権限が不要です。
```
$ wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
$ bash Anaconda3-4.4.0-Linux-x86_64.sh
```

### Python & TensorFlow
仮想環境を作成します。私は Python 2.7 と TensorFlow 1.1 を使ったので、環境名を `py2t1.1` にしました。
```
$ conda create -n py2t1.1 python=2.7
```
インストール後、仮想環境に入ります。有効化は `activate` を使います：
```
 $ source activate py2t1.1
 (py2t1.1)$  # こう表示されます
```
無効化は `deactivate` です：
```
 $ source deactivate
```

次に `tensorflow` をインストールします。ここでは TensorFlow 1.1（Python 2.7 / CPU 版）を使います。他のバージョンが必要なら [こちら](https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package) から探せます。
```
 (py2t1.1)$ pip install --ignore-installed --upgrade \
 https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl
```

これで環境設定は完了です。

## モデルの学習

### 前準備
この時点では仮想環境の中でも外でも構いません。Python で TensorFlow を実行するときだけ、作成した仮想環境に入る必要があります。

フォルダを作成します：
```
$ mkdir flower_classification
$ cd flower_classification
```

花の写真を取得します：
```
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
```

`retrain.py` を取得します。これは `Inception v3` で学習済みの画像認識モデルを利用し、最後に新しい層を追加して再学習することで、花の種類を判別できるようにするスクリプトです。
```
$ curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py
```

### 学習開始
まず仮想環境に入ります：
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

学習が終わると `retrained_graph.pb` と `retrained_labels.txt` が生成されます。この 2 つのファイルを使って新しい写真を判別できます。

### 写真の判別
判別用のスクリプト `label_image.py` を取得します：
```
$ curl -L https://goo.gl/3lTKZs > label_image.py
```

判別を実行します：
```
$ python label_image.py 要辨識的花朵照片.jpeg
```

結果は次のようになります：
```
daisy (score = 0.99071)
sunflowers (score = 0.00595)
dandelion (score = 0.00252)
roses (score = 0.00049)
tulips (score = 0.00032)
```

# 第 2 部
## 前準備
`tensorflow-for-poets-2` は Google Codelab のサンプル Android アプリプロジェクトです。
```
$ cd ~/
$ git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
$ cd tensorflow-for-poets-2
```
先ほど学習した花の分類フォルダ `flower_classification` をプロジェクトに丸ごとコピーします：
```
$ cp -r ~/flower_classification .
```

## モデルの処理
最適化：
```
$ python -m tensorflow.python.tools.optimize_for_inference \
  --input=flower_classification/retrained_graph.pb \
  --output=flower_classification/optimized_graph.pb \
  --input_names="Cast" \
  --output_names="final_result"
```
量子化（圧縮）：
```
$ python -m scripts.quantize_graph \
  --input=flower_classification/optimized_graph.pb \
  --output=flower_classification/rounded_graph.pb \
  --output_node_names=final_result \
  --mode=weights_rounded
```

`rounded_graph.pb` と `retrained_labels.txt` を `android/assets` フォルダへ配置します：
```
$ cp flower_classification/rounded_graph.pb flower_classification/retrained_labels.txt android/assets/ 
```

## Android Studio のインストール
https://developer.android.com/studio/index.html

## Android プロジェクトを開く
Android Studio を起動し、「Open an existing Android Studio project」を選択して `tensorflow-for-poets-2/android` を指定します。その後、Gradle を使うか聞かれるので「ok」を選択します。Gradle の同期が終わったら build できるようになります。最後に APK をスマホへインストールします。

## 参考資料
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#0
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/#0

