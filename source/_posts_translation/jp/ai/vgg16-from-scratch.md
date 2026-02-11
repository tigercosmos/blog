---
title: "VGG16 をスクラッチ実装する 2 つの方法：CPU 上の C++ と GPU 上の CUDA"
date: 2020-12-02 05:30:00
tags: [ai, deep learning, vgg16, c++, cuda, cudnn, cublas]
des: "本記事では VGG16 を 2 つのバージョンで紹介します。1 つ目は CPU 上で動く純粋な C++ 実装、2 つ目は CUDA / cuDNN / cuBLAS を用いて GPU 上で動かす実装です。"
lang: jp
translation_key: vgg16-from-scratch
---

## Introduction

VGG16 は、オックスフォード大学の K. Simonyan と A. Zisserman により論文 “Very Deep Convolutional Networks for Large-Scale Image Recognition” で提案された畳み込みニューラルネットワーク（CNN）モデルです。ILSVRC-2014 の画像認識で有名なモデルで、AlexNet、GoogleNet、ResNet などと並んでよく知られています。VGG16 の概要は、たとえば "[VGG16 – Convolutional Network for Classification and Detection](https://neurohive.io/en/popular-networks/vgg16/)" が参考になります。

本記事では VGG16 をスクラッチで 2 つのバージョンとして実装しました。1 つ目は CPU 上で動く純粋な C++ 実装で、各層の演算をすべて C++ のみで実装しています。2 つ目は CUDA / cuDNN / cuBLAS を使って GPU の利点を活かし、ニューラルネットワークの演算を高速化する実装です。

![Vgg16 architecture](https://user-images.githubusercontent.com/18013815/100784827-bd1dc080-344a-11eb-8a65-76be05809123.png)

## VGG16 in C++

このプログラムはヘッダーオンリーのフレームワークとして書きました。拡張性・スケーラビリティのある構造になっていると思います。GitHub のリポジトリは [simple-vgg16](https://github.com/tigercosmos/simple-vgg16) です。

フレームワークの使い方は次のとおりです。

```c++
    // prepare the input data
    auto input = sv::Tensor<double>(224, 224, 3); // 224 x 224 x 3

    // prepare the network
    auto network = sv::Network<double>();

    // add layers into the network
    auto *layer1_1 = new sv::ConvLayer<double>(3, 3, 64); //224 x 224 x 64
    network.addLayer(layer1_1);
    auto *layer1_2 = new sv::ConvLayer<double>(64, 3, 64); //224 x 224 x 64
    network.addLayer(layer1_2);
    auto *layer1_3 = new sv::MaxPoolLayer<double>(2, 2); // 112 x 112 x 64
    network.addLayer(layer1_3);

    // SKIP

    // predict the result by forwarding
    sv::Tensor<double> output;
    network.predict(input, output);
```

上のスニペットのように、まずネットワークを宣言し、層を追加していきます。ネットワーク構造を定義したら、`predict()` を呼び出して推論を実行します。TensorFlow や PyTorch でモデルを書くのと同じくらい簡単です。

フレームワークはシンプルで、ヘッダーは少数です：
- Activation：活性化関数の定義
- Layer：`ConvLayer`、`MaxPoolLayer` など層の定義
- Network：層を保持するネットワーク（モデル）
- Operand：畳み込み・プーリングなど、層内部で使う演算子の定義
- Tensor：テンソル型の定義

このシンプルなフレームワークはかなり分かりやすい構成だと思います。一方で最も難しいのは Operand（演算）の実装です。畳み込みなどの演算はバグを作りやすく、私は実装のデバッグに多くの時間を使いました。また、データの扱い方や使用するアルゴリズムなど、最適化の余地は多くあります。純粋な C++ は非常に遅かったので、OpenMP を導入して少し速くしました。

このフレームワークには満足しています。現時点では推論のみをサポートしていますが、学習機能を足したり、演算・層・活性化関数を拡張したりするのもそれほど難しくないと思います。

また簡単なベンチマークも用意しています。出力は次のようになります：

```
The VGG16 Net
-----------------------------
NAME:   MEM     PARAM   MAC
-----------------------------
Conv:   25690112,       1792,   86704128
Conv:   25690112,       36928,  1849688064
MaxPool:        6422528,        0,      0
Conv:   12845056,       73856,  924844032
Conv:   12845056,       147584, 1849688064
MaxPool:        3211264,        0,      0
Conv:   6422528,        295168, 924844032
Conv:   6422528,        590080, 1849688064
MaxPool:        1605632,        0,      0
Conv:   3211264,        1180160,        924844032
Conv:   3211264,        2359808,        1849688064
MaxPool:        802816, 0,      0
Conv:   802816, 2359808,        462422016
Conv:   802816, 2359808,        462422016
MaxPool:        200704, 0,      0
FC:     32768,  102764544,      102760448
FC:     32768,  16781312,       16777216
FC:     8000,   4097000,        4096000
Total:  110Mb,  133M,   11308M
-----------------------------
@@ result @@
0.0413532 0.0416434 0.0412649 0.0419855 0.0412341 ... so many ... 0.0415309 0.0406799 0.0418257 0.0416706 0.0413013
```

詳細は GitHub の [simple-vgg16](https://github.com/tigercosmos/simple-vgg16) を参照してください。ぜひ fork して自由に改造してみてください。

## VGG16 in CUDA

行列演算のように、深層学習で頻出する処理では CPU より GPU の方が速いことが多いです。深層ニューラルネットワークの実行時間を短縮するには、CUDA や cuDNN / cuBLAS といったライブラリを活用して GPU 計算を使うのが効果的です。

2 つ目の実装では、CUDA を使って VGG16 を実装しました。GitHub: [simple-vgg16-cu](https://github.com/tigercosmos/simple-vgg16-cu) にあります。先ほどの C++ 版と同様に、Proof-of-Concept 的な実装です。畳み込み層は cuDNN、全結合層は cuBLAS で実装しています。

`cudnnConvolutionBiasActivationForward`、`cudnnPoolingForward`、`cublasSgemm` のような API を使うと、CUDA 上でニューラルネットワーク（NN）の各種演算を比較的簡単に実装できます。CUDA 系ライブラリは、推論／学習の速度向上だけでなく、NN モデル開発の効率も大きく高めてくれるため、可能な限りこうしたライブラリの利用が推奨されます。

GPU は数秒で終わる一方で、CPU は数分かかります。以下は各層の実行時間です：

```
CONV 224x224x64 84 ms
CONV 224x224x64 115 ms
POOLMAX 112x112x64 108 ms
CONV 112x112x128 114 ms
CONV 112x112x128 111 ms
POOLMAX 56x56x128 70 ms
CONV 56x56x256 122 ms
CONV 56x56x256 115 ms
POOLMAX 28x28x256 92 ms
CONV 28x28x512 122 ms
CONV 28x28x512 118 ms
POOLMAX 14x14x512 73 ms
CONV 14x14x1024 63 ms
CONV 14x14x1024 114 ms
POOLMAX 7x7x1024 102 ms
FC 4096 3280 ms
FC 4096 311 ms
FC 1000 104 ms
```

なお、私のコードは最適化していません。CPU より速くはなっているものの、PyTorch の VGG16 は 100ms 台で動くこともあり、私の実装には大きな改善余地があります。

## Conclusion

PyTorch のようなフレームワークを使えば VGG16 のようなネットワークは簡単に構築できますが、スクラッチ実装することで概念理解を深められます。だからこそ私は、CPU 上の純粋な C++ と、GPU 上の CUDA という 2 つの方法で VGG16 を書きました。

CPU 計算と GPU 計算は対照的で、実行時間に非常に大きな差があることが分かります。ただし「GPU が速いから常に GPU を選ぶ」とは限りません。組み込みなど CPU しか搭載していないハードウェアもあり、その場合は CPU が唯一の選択肢です。また、CPU や GPU だけでなく、ALU や FPGA の方がさらに速いケースもあります。

何かをスクラッチで書いてみるのは常に良い練習になります。私はこうした小さな実装から多くを学びました。本記事が少しでも役に立てば嬉しいです。あなたの実装を見るのも楽しみにしています。

