---
title: 電腦動畫中的正向動力法 (Forward Kinematics) 和時間扭曲法 (Time Warping)
date: 2020-05-23 00:00:00
tags: [computer animation, 電腦動畫, 正向動力法, 時間扭曲法, forward kinematics, time warping]
des: "本文介紹電腦動畫中正向動力學 (forward kinematics) 和時間扭曲法 (time warping) 的原理與實作。正向動力學被廣泛應用於機器人學、電腦遊戲、電腦動畫等領域，原理是在一骨架中給予每個關節 (joint) 相對位移和旋轉，用反覆遞迴遍歷所有關節的方式計算整體運動量。時間扭曲法被應用於電腦動畫中，為關鍵影格 (keyframe) 的原理技術，當我們想要某個影格必須在特地時間才被畫出，那就必須根據關鍵影格去調整前後的撥放速度，這時就必須做加速或減速。"
lang: zh
translation_key: forward-kinematics-time-warping
---

## 簡介

本文介紹電腦動畫中正向動力學 (forward kinematics) 和時間扭曲法 (time warping) 的原理與實作。

![](https://i.imgur.com/ZUqBiQU.gif)


正向動力學被廣泛應用於機器人學、電腦遊戲、電腦動畫等領域，原理是在一骨架中給予每個關節 (joint) 相對位移和旋轉，用反覆遞迴遍歷所有關節的方式計算整體運動量。

時間扭曲法被應用於電腦動畫中，為關鍵影格 (keyframe) 的原理技術，當我們想要某個影格必須在特地時間才被畫出，那就必須根據關鍵影格去調整前後的撥放速度，這時就必須做加速或減速。

## 原理與實作

### 骨架

為了描述一個人的動作，在電腦動畫中，使用骨架 (skeleton) 來帶動人的行為，本次使用的是 acclaim 的 ASF 骨架格式，裡面會宣告每個骨頭的索引、長度、方向、neutral pose 角度和自由度。

以下是 ASF 範例：

```s
:root
   order TX TY TZ RX RY RZ
   axis XYZ
   position 0 0 0  
   orientation 0 0 0 
:bonedata
  begin
     id 1 
     name lhipjoint
     direction 0.635348 -0.713158 0.296208 
     length 2.47672 
     axis 0 0 0  XYZ
  end
  // 以下省略
```

讀完骨架會長這樣：

![](https://i.imgur.com/tAcVx5n.png)


要讓 neutral pose 可以讓骨頭去表現行為，則需要使用 acclaim 的 AMC 資訊檔，裡面描述該骨頭運動資訊。除了根節點會是對世界座標的變化量，其餘每個骨頭都只是相對於根結點的相對變化量。一個 AMC 檔會包含一個完整運動所包含所有影格中每個骨頭的變化量。

以下是 AMC 範例，最前面會是每一個影格的編號，然後包含每個骨頭在那個影格的資訊：

```s
2
root -0.303728 17.5624 -27.7253 2.02549 1.77071 -4.33872
lowerback 16.0608 -0.380636 1.35189
upperback 1.68665 -0.267024 -0.0539964
thorax -7.21419 -0.169571 -0.765959
lowerneck -2.88855 -0.493739 -1.55908
upperneck -9.88628 -0.567977 1.15901
head -2.623 -0.258251 0.642519
rclavicle -7.65321e-015 -2.38542e-015
rhumerus -42.619 18.2084 -90.2387
// ... 以下省略
```

讀完資訊檔會長這樣：

![](https://i.imgur.com/Vt5Jugt.png)


更多資訊可以參考 [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/info.php) 網站。

### 正向動力學

正向動力學可以想成是一個動力傳遞鏈，先算出第一個關節的變化，再根據第一個關節變化去計算第二個關節的變化，以此類推。

#### 概念

概念如下列幾張圖 (source: alanzucconi.com)：

**第一步**：
![](https://i.imgur.com/kDm7vRg.png)

P0 是整個骨架的 root，P1 和 P2 分別是整個骨架上的關節，並且每個關節都有三個自由度，都有自己的變化量。

**第二步**：
![](https://i.imgur.com/LrlC24S.png)

計算 P0 變化造成 P1 和 P2 變化。


**第三步**：
![](https://i.imgur.com/nxtjNKT.png)


計算 P1 變化造成 P2 變化。此時 P2 的變化就是 P0 和 P1 的疊加。


根據上述的步驟，我們就可以用正向動力學推導出一個骨架上所有關節最後的變化量。

以下為使用人體骨架 ASF 搭配跑步的 ACM 描述檔，使用正向動力法算出來的結果：

![](https://i.imgur.com/a8zYv1l.gif)


#### 數學

整個骨架的變化量為 1.骨架對於世界座標的變化量加上 2.骨架上每個關節的相對變化量。首先算出每個關節相對變化最後的結果，因為對於世界座標的變化量每個節點都一樣，所以每個關節最後再加上世界座標的變化量即可。

現在討論相對變化量的處理。

假設現在一個骨架是 $\mathbf P_0 \to \mathbf P_1 \to \mathbf P_2$，其中 $\mathbf P_i$ 代表關節向量， $\mathbf D_1 = P_0 \to P_1$ 代表第一個連桿向量。

那麼 $\mathbf P_0$ 和 $\mathbf P_1$ 的關係為：

$$ \mathbf P_1 = \mathbf P_0 + \mathbf D_1 $$

其中 $\mathbf D_1$ 是第一個連桿的位移向量。

但我們還需要加上 $P_0$ 旋轉量，以 $\mathbf P_0$ 為基準，長度 $\mathbf D_1$ 旋轉 $\alpha_0$ 度：

$$ \mathbf P_1 = \mathbf P_0 + rotate(\mathbf D_1, \mathbf P_0, \alpha_0) $$

以此類推我們可以得到 $\mathbf P_i$ 通式：

$$ \mathbf P_i = \mathbf P_{i-1} + rotate(\mathbf D_i, \mathbf P_{i-1}, \sum_{k=0}^{k-1}{\alpha_k}) $$

### 時間扭曲法

時間扭曲法讓我們可以根去某段動畫的關鍵影格去調整電腦動畫，例如一個揮拳的動作，想要讓原本在影格150 的動作延遲到影格 160，或是原本在影格 150 的動作提早到影格 140。

![](https://i.imgur.com/ZYSQXwo.png)

要做到這個效果，我們必須調整每個影格的動作變化。假設新動畫是要把一段動畫中，裡面影格 150 延遲到影格 160 才出現。對於調整一段骨架的動畫的具體步驟如下：

1. 計算新的影格在舊的影格的位置。以這個例子來說，因為新的動畫 0 到 160 相當於舊的 0 到 150，所以新的動畫在 0 到 160 之間，每個影格相當於舊動畫的每 150/160 影格。

    對照如下
    ```
    新  0  1    2     ....  160
    舊  0  0.94 1.92  ....  150
    ```
    
2. 根據舊動畫影格去內插出新動畫的影格，以新動畫第 2 個影格為例，相當於舊的第 1.92 個影格，於是我們必須拿舊動畫第 1 個和第 2 個影格來內插出第 1.92 個影格。

3. 平移 (translation) 可以使用線性內插法，旋轉 (rotation) 則需要將相量轉換成 Quaternion，再使用 Slerp 內插法才會有比較準確的內插效果。 (當然旋轉也可以用線性內插，只是會非常不準)

以正向動力法實作，並加上時間扭曲法的呈現結果如下：

![](https://i.imgur.com/Gv0nk1Z.gif)

黃色是原本影格，藍色則是使用時間扭曲後調整個影格。

值得一提的是，math 函式庫轉換很複雜，實作上會因為單位轉換容易發生錯誤，像是旋轉的單位要在 degree 和 radian 之間轉換。另外 Quaternion 在使用 [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) (C++) 函式庫下的轉換也研究很久，最後才得到 Slerp  這樣寫：

```c++
auto rot1 = ComputeRotMatXyz(ToRadian(angular_vector1);
auto rot2 = ComputeRotMatXyz(ToRadian(angular_vector2));
auto q1 = Quaternion_t(rot1);
auto q2 = Quaternion_t(rot2);
auto new_q = Slerp(q1, q2, ratio);
auto new_angular_vec = ToDegree(ComputeEulerAngleXyz(ComputeRotMat(new_q)));
```

## 結論

透過正向動立法我們可以使用每個關節的相對變化，算出一個骨架的運動變化。使用時間扭曲法我們可以實現關鍵影格的效果，變化動畫的呈現結果。兩個技術對於電腦動畫都非常基本且重要。
