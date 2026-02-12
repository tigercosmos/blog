---
title: "ハイブリッド画像（Hybrid Image）の原理と実装"
date: 2020-04-26 20:00:00
tags: [computer vision, hybrid image, image processing, python, fourier transform]
des: "ハイブリッド画像（Hybrid Image）は、2枚の画像からそれぞれ高周波成分と低周波成分を取り出して合成する手法です。本記事ではその原理と実装を解説します。"
lang: jp
translation_key: hybrid-image
---

## 1. はじめに

ハイブリッド画像（Hybrid Image）とは、2枚の画像からそれぞれ高周波成分と低周波成分を取り出し、合成して新しい画像を作る手法です。次の画像は教科書でよく出てくる有名な例です：

![hybrid image of einstein and marilyn](https://user-images.githubusercontent.com/18013815/80302650-d2a87900-87dd-11ea-90f8-1d24e3db275d.png)

この画像は何に見えますか？

近くで見るとアインシュタインのように見えますが、目を細めたり、画像を小さくして遠目に見ると、マリリン・モンローのように見えます。これは、アインシュタイン側は高周波を通すフィルタ（ハイパス）で輪郭などが強調され、マリリン側は低周波を通すフィルタ（ローパス）でぼかしが入っているためです。

「横看成嶺側成峰、遠近高低各不同（見る距離や角度で同じ山でも違って見える）」という言葉のとおり、ハイブリッド画像は1枚に複数の情報を埋め込めます。さらに、同じ画像により多くの画像情報を隠すことも可能です。

## 2. 原理

### 2.1 概念

ハイブリッド画像の背後にあるのは信号処理であり、信号処理といえば [フーリエ変換](https://zh.wikipedia.org/wiki/%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2) です。

フーリエ変換により、複雑な信号を基本信号の組み合わせに分解できます。例えば：

![gif of fourier transform form wiki](https://upload.wikimedia.org/wikipedia/commons/7/72/Fourier_transform_time_and_frequency_domains_%28small%29.gif)
(source from Wikipedia)

この図では、赤の（矩形波に近い）信号が、周波数の異なる青い正弦波の和として分解されています。

同様に、画像も複雑な信号なので、さまざまな周波数成分の重ね合わせとして扱えます。

ハイブリッド画像では、1枚目の画像をフーリエ変換して高周波成分（High-pass）を取り出し、2枚目の画像をフーリエ変換して低周波成分（Low-pass）を取り出します。両者に逆フーリエ変換（Inverse Fourier Transform）を適用し、最後に2枚を足し合わせると、ハイブリッド画像が得られます。

### 2.2 数学

画像は $\sum_{i, j}R_{ij}$ と表せます。ここで $R_{ij}$ は座標 $(i, j)$ の画素値です。

ハイパスやローパスを行うには、元信号をフィルタで通す必要があります。

フィルタ後の画像 $R$ は、フィルタカーネル（filter kernel）$H$ と元画像 $F$ の畳み込み（Convolution）で得られます：

$$
\begin{align}
R_{ij}  &= \sum_{u,v} H_{i-u, j-v} F_{u,v} \\\\
\mathbf R &= \mathbf H ** \mathbf F
\end{align}
$$

ここで $F$ は、対象画像にフーリエ変換（FFT）を適用し、さらに周波数0成分を中央に移動（FFT Shift）した 2D スペクトル（Spectrum）です。2D スペクトルでは、中心付近が低周波、境界付近が高周波に対応します。高周波成分は、急激な変化やエッジ・コーナーに対応します。

フィルタカーネル $H$ は様々に設計できますが、一般的にはガウス関数 $g(i,j)$ を用います。定義は次のとおりです：

$$g(i,j) = EXP({ -{ {(x-i)^2+(y-j)^2} \over{ 2 \sigma ^2}} })$$

ここで $\sigma$ はスケール（範囲）を表し、$(i, j)$ は画素位置、$(x, y)$ は中心点です。

### 2.3 フィルタ

なぜガウスを使うのかは [この議論](https://dsp.stackexchange.com/questions/3002/why-are-gaussian-filters-used-as-low-pass-filters-in-image-processing) が参考になります。簡単に言うと、負の値が出にくいこと、そしてフィルタ後の見え方が自然界の物理現象に近いことが利点として挙げられます。

理想フィルタ（Ideal Filter）とガウスフィルタを比べると、理想フィルタは境界がくっきりしたフィルタで、ガウスフィルタはガウス関数に従って滑らかに減衰します。下図はローパスの場合の模式図で、フーリエ変換後に中心（白）だけを残すと低周波のみが残ることを意味します。

![image](https://user-images.githubusercontent.com/18013815/80313634-a9f4a380-881e-11ea-9de2-1526ac3b1692.png)

この2種類のローパスを適用すると、ぼかしが生まれます。結果は次のとおりで、理想フィルタのほうが不自然な見え方になりやすいことが分かります。

![gaussian filter and ideal filter](https://user-images.githubusercontent.com/18013815/80313337-ee7f3f80-881c-11ea-8ba3-c69df565e8e0.png)

## 3. 実装

### 3.1 詳細

まずガウスフィルタを実装します：

```py
def makeGaussianFilter(n_row, n_col, sigma, highPass=True):

    center_x = int(n_row/2) + 1 if n_row % 2 == 1 else int(n_row/2)
    center_y = int(n_col/2) + 1 if n_col % 2 == 1 else int(n_col/2)

    def gaussian(i, j):
        coefficient = math.exp(-1.0 * ((i - center_x) **
                                       2 + (j - center_y)**2) / (2 * sigma**2))
        return 1 - coefficient if highPass else coefficient

    return numpy.array(
        [[gaussian(i, j) for j in range(n_col)] for i in range(n_row)])
```

次に画像をフィルタリングします：

```py
def filter(img, sigma, highPass):
    # 計算圖片的離散傅立葉
    shiftedDFT = fftshift(fft2(img))
    # 將 F 乘上濾鏡 H(u, v)
    filteredDFT = shiftedDFT * \
        makeGaussianFilter(
            image.shape[0], image.shape[1], sigma, highPass=isHigh)
    # 反傅立葉轉換
    res = ifft2(ifftshift(filteredDFT))
    
    return numpy.real(res)
```

`shiftedDFT` は、画像に離散フーリエ変換（`fft`）をかけた後、周波数0を中心に移動（`fftshift`）したものです。

また、最後の返り値 `numpy.real(res)` に注意してください。逆フーリエ変換は実部と虚部を生成します（phase spectrum と magnitude spectrum と呼ばれることもあります）。ここでは実部だけが必要です。実部が画像の「情報量」の大部分を担い、虚部が持つ情報は小さいからです。さらに、異なる画像でも虚部が似通っていることが多いでしょう。理由は……自然界とはそういうものです（a fact of nature）。

最後に、ハイパスとローパスを合成します：

```py
def hybrid_img(high_img, low_img, sigma_h, sigma_l):
    res = filter(high_img, sigma_h, isHigh=True) + \
        filter(low_img, sigma_l, isHigh=False)
    return res
```

### 3.2 完全なコード

```py
import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math
import imageio
import matplotlib.pyplot as plt
from skimage.transform import resize


def makeGaussianFilter(n_row, n_col, sigma, highPass=True):

    center_x = int(n_row/2) + 1 if n_row % 2 == 1 else int(n_row/2)
    center_y = int(n_col/2) + 1 if n_col % 2 == 1 else int(n_col/2)

    def gaussian(i, j):
        coefficient = math.exp(-1.0 * ((i - center_x) **
                                       2 + (j - center_y)**2) / (2 * sigma**2))
        return 1 - coefficient if highPass else coefficient

    return numpy.array([[gaussian(i, j) for j in range(n_col)] for i in range(n_row)])


def filter(image, sigma, isHigh):
    shiftedDFT = fftshift(fft2(image))
    filteredDFT = shiftedDFT * \
        makeGaussianFilter(
            image.shape[0], image.shape[1], sigma, highPass=isHigh)
    res = ifft2(ifftshift(filteredDFT))

    return numpy.real(res)


def hybrid_img(high_img, low_img, sigma_h, sigma_l):
    res = filter(high_img, sigma_h, isHigh=True) + \
        filter(low_img, sigma_l, isHigh=False)
    return res


img1 = imageio.imread("IMG_PATH", as_gray=True)
img2 = imageio.imread("IMG_PATH", as_gray=True)
resize(img2, (img1.shape[0], img1.shape[1])) # 兩張圖要一樣大

plt.show(plt.imshow(hybrid_img(img1, img2, 10, 10), cmap='gray'))
```

ここでは画像をグレースケール（1チャネル）にしていますが、同じ考え方はカラー画像（R、G、B チャネル）にも拡張できるはずです。ただ、私が各チャネルを別々に処理して合成しようとしたところ、悲劇が起きました。なのでカラー版は読者の課題として残しておきます。もし成功したらぜひ教えてください。

## 4. 結果

### 4.1 ガウスフィルタによるローパス／ハイパス

同一画像に対して、異なる $\sigma$ でハイパス／ローパスを行った結果は次のとおりです：

![image](https://user-images.githubusercontent.com/18013815/80318872-acb3c080-883f-11ea-8552-d89336d1d233.png)

ハイパスでは、$\sigma$ が大きいほどスペクトル中心に掘る「穴」が大きくなり、より周辺の高周波成分だけが残ります。そのため、画像の輪郭やエッジが主に見えるようになります。

ローパスでは、$\sigma$ が小さいほどスペクトル中心の円が小さくなり、保持される細部が失われるため、よりぼやけます。

### 4.2 ハイブリッド画像

2枚の画像に対して、異なる $\sigma$ でハイパス／ローパスを適用して合成した結果は次のとおりです：

![image](https://user-images.githubusercontent.com/18013815/80318678-4e3a1280-883e-11ea-895d-50d767d56442.png)

パラメータ設定を変えることで、さまざまな視覚効果が得られることが分かります。

## まとめ

ハイブリッド画像は、2枚の画像に対してそれぞれハイパスとローパスを適用し、合成することで作れます。この技術により、狙った特徴の出し方で2枚の画像を重ね合わせられます。ガウスフィルタの $\sigma$ を調整することで、異なる効果のハイブリッド画像が作れます。

用途の1つとしては、ステガノグラフィ（隠し透かし）があります。元画像に非常に低周波な透かし模様を足しておけば、普通に見る限りほとんど気づかれませんが、ローパス解析をすれば「実はあなたが元だ」と分かるようにできます。

アハ！無断転載がバレたね！

