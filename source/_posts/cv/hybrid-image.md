---
title: 混疊圖像 (Hybrid Image) 原理與實作
date: 2020-04-26 20:00:00
tags: [電腦視覺, hybrid image, computer vision, python, image processing]
des: "混疊圖像 (Hybrid Image) 是指將兩張圖片，分別取其中的高頻與低頻部分，合成成新的圖片。本文介紹混疊圖像的原理與實作。"
lang: zh
translation_key: hybrid-image
---

## 1. 簡介

混疊圖像 (Hybrid Image) 是指將兩張圖片，分別取其中的高頻與低頻部分，合成成新的圖片，以下圖片為經典的教科書範例：

![hybrid image of einstein and marilyn](https://user-images.githubusercontent.com/18013815/80302650-d2a87900-87dd-11ea-90f8-1d24e3db275d.png)

這張圖片是甚麼呢？

近看似乎就是愛因斯坦，但如果瞇著眼看，或是把圖片弄得很小再看，似乎就是瑪麗蓮夢露。這是因為愛因斯坦採用高通過濾造成特徵銳化，而瑪麗蓮夢露採用低通過濾而有模糊的效果。

正所謂「橫看成嶺側成峰，遠近高低各不同」，混疊圖像在一張圖片藏了不只一種訊息，甚至還可以將更多圖片藏在同一張圖片之中。

## 2. 原理

### 2.1 概念

混疊圖片的原理背後就是訊號處理，說到訊號處理就要談談[傅立葉變換](https://zh.wikipedia.org/wiki/%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2)。

傅立葉變換讓我們有辦法把複雜的訊號，一一去拆解成各種基本訊號，舉例來說：

![gif of fourier transform form wiki](https://upload.wikimedia.org/wikipedia/commons/7/72/Fourier_transform_time_and_frequency_domains_%28small%29.gif)
(source from Wikipedia)

這張圖中，把原本紅色的類似方波訊號，一一拆解成藍色的各種不同頻率 Sin 波。

同理，一張圖片其實就是一個複雜的訊號，我們也能將圖片拆解成各種訊號的疊加。

混疊圖片就是將第一張圖片取傅立葉轉換後高頻的部分 (稱為高通 High-pass)，將第二張圖片取其傅立葉轉換後低頻部分 (稱為低通 Low-Pass)，分別將訊號做反傅立葉轉換 (The Inverse Fourier Transform)，並將兩張過濾過的圖片做疊加，就會看到混疊後的圖片。

### 2.2 數學

一張圖片可以表示成 $\sum_{i, j}R_{ij}$ ，其中 $R_{ij}$ 代表作座標為 $(i, j)$ 的圖片像素。

要做到高通或低通，我們需要一個濾鏡過濾原始訊號。

一張過濾過的圖片 $R$，是由一個濾鏡核心 (filter kernel) $H$ 對原始圖片 $F$ 做卷積 (Convolution)。如下式：

$$
\begin{align}
R_{ij}  &= \sum_{u,v} H_{i-u, j-v} F_{u,v} \\\\
\mathbf R &= \mathbf H ** \mathbf F
\end{align}
$$

而這邊的 $F$ 是將目標圖片做過傅立葉轉換 (FFT)，並且將頻率零平移置中 (FFT Shift) 而產生的 2D 頻譜 (Spectrum)。在這 2D 頻譜中，靠近中央代表低頻訊號，靠近邊界則代表高頻訊號。其中高頻訊號代表劇烈改變或是邊角。

其中濾鏡核心 $H$ 可以使用各種函數，但一般使用高斯函數 $g(i,j)$ ，定義如下：

$$g(i,j) = EXP({ -{ {(x-i)^2+(y-j)^2} \over{ 2 \sigma ^2}} })$$

其中 $\sigma$ 為範圍大小，$(i, j)$ 是圖片的像素點位置，而 $(x, y)$ 是中心點。

### 2.3 濾鏡

濾鏡選擇高斯函數理由可以參考[這篇](https://dsp.stackexchange.com/questions/3002/why-are-gaussian-filters-used-as-low-pass-filters-in-image-processing)，簡單來說有幾點優點，一來是他不會產生負的值，再來是他過濾後的圖片跟自然界的物理現象比較接近。

以理想濾鏡  (Ideal Filter) 和高斯濾鏡來比較的話，理想濾鏡是一個邊界分明的濾鏡，高斯濾鏡則是套用高斯函數的濾鏡。下圖為低通情況這兩種濾鏡示意圖，圖片在經過傅立葉變換之後，若只保留中心部分 (白色)，則代表此圖片經過濾鏡後僅留下低頻訊號。

![image](https://user-images.githubusercontent.com/18013815/80313634-a9f4a380-881e-11ea-9de2-1526ac3b1692.png)

套用這兩款不同的低通濾鏡，此時低通會造成模糊，呈現的結果如下，可以看到理想濾鏡比較沒那麼自然。

![gaussian filter and ideal filter](https://user-images.githubusercontent.com/18013815/80313337-ee7f3f80-881c-11ea-8ba3-c69df565e8e0.png)

## 3. 實現

### 3.1 細部介紹

首先我們先來實現高斯濾鏡：

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

然後過濾圖片：

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

`shiftedDFT` 就是將圖片先做離算傅立葉轉換 (`fft`)，然後將做平移將頻率 0 置中 (`fftshift`)。

此外注意到最後回傳的值 `numpy.real(res)`。反傅立葉轉換其實會產生實部和虛部，稱為 phase spectrum 和 magnitude spectrum，而我們僅需要實部，這是因為實部才帶有圖片多數的「資訊量」；虛部帶有圖片的「資訊量」很小，而且會發現不同照片的虛部長得都很接近，至於原因大概就是自然界就是如此 (a fact of nature)。

最後我們要將高通和低通合併：

```py
def hybrid_img(high_img, low_img, sigma_h, sigma_l):
    res = filter(high_img, sigma_h, isHigh=True) + \
        filter(low_img, sigma_l, isHigh=False)
    return res
```

### 3.2 完整程式碼

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

這邊將照片做灰階處理 (一層)，但同樣概念應該可以擴展到彩色 (也就是會有 R、G、B 色層)，不過我自己嘗試將不同色層分別處理後合成，卻會發生悲劇，所以就彩色留給讀者去嘗試實作了，若你成功歡迎分享給我。

## 4. 結果呈現

### 4.1 高斯過濾的低通和高通

以下為同張圖片分別用不同 $\sigma$ 來做高通和低通的結果：

![image](https://user-images.githubusercontent.com/18013815/80318872-acb3c080-883f-11ea-8552-d89336d1d233.png)

高通 $\sigma$ 越大代表頻譜中間挖的洞越大，因此僅留下比較邊邊高頻的部分，因此只會看到圖片中的邊邊角角和輪廓。

低通 $\sigma$ 越小代表頻譜中央的圓越小，保留得細節資訊就會遺失，所以就會越模糊。

### 4.2 混疊圖片

以下為 2 張圖片分別用不同 $\sigma$ 做高通和低通後疊加的結果：

![image](https://user-images.githubusercontent.com/18013815/80318678-4e3a1280-883e-11ea-895d-50d767d56442.png)

可以看到不同的參數設法，可以有各種不同的視覺效果。

## 結論

混疊圖片是將兩張圖片分別以高通和低通過濾然後合成，藉由這技術我們可以將兩張不同照片以我們想要的特徵方式去疊加，根據對高斯濾鏡調整 $\sigma$ 大小，我們可以做出不同效果的混疊圖片。

一種用途可以是，藉由混疊技巧你可以將你的圖片悄悄加上浮水印，亦即原本照片加上非常低頻的浮水圖樣，別人基本上會看不出來，但一旦經過低通分析，就可以發現原來來源是你的。

啊哈！盜圖被發現了齁！
