---
title: 二維模式相機校正 (2D Pattern Camera Calibration)
date: 2020-04-06 00:00:00
tags: [電腦視覺, camera calibration, computer vision, 2D pattern, python]
des: "相機校正 (Camera Calibration) 是指取得相機的各種參數，包含 intrinsic parameters (相機本身參數) 和 extrinsic parameters (定向參數)。本文著重於 Zhengyou Zhang 在〈A Flexible New Technique for Camera Calibration〉所提出用 2D 模式來做相機校正的方法，針對其數學做更明白的解釋和程式碼示例。"
---

## 1. 簡介

[相機校正 (Camera Calibration)](https://en.wikipedia.org/wiki/Camera_resectioning)是指取得相機的各種參數。意思是我們可以得到這個相機拍下這張照片時，這照片上的某個 2D 點對應到他現實世界的 3D 點的所有參數，包含 intrinsic parameters (相機本身參數) 和 extrinsic parameters ([定向](https://zh.wikipedia.org/wiki/%E5%AE%9A%E5%90%91_(%E5%90%91%E9%87%8F%E7%A9%BA%E9%96%93))參數)。

照片 2D 點與現實世界 3D 點的關係如下：

$$\mathbf{\tilde m} = \mathbf A [\mathbf R \quad \mathbf t] \mathbf {\tilde M}$$

其中 $\mathbf{\tilde m}$ 是照片座標向量 $[u, v, 1]^T$，$\mathbf {\tilde M}$ 是現實座標向量 $[X, Y, Z, 1]^T$，而 $\mathbf A$ 是 intrinsic matrix、$[\mathbf R \quad \mathbf t]$ 是 extrinsic matrix。

關於相機校正可參考 Richard Szeliski 所著《Computer Vision: Algorithms and Applications》第六章 Feature-based alignment。

要取得照片對應現實世界的 3D 點座標不容易，但如果利用有模式 (pattern) 的平面物品，例如棋盤，因為每個格子相對座標都是已知的，且可以假設其 $Z$ 座標都是相同的，就可以直接藉由多張棋盤的照片，直接還原其 intrisic matrix 和 extrinsic matrix。本文著重於 Zhengyou Zhang 在[〈A Flexible New Technique for Camera Calibration〉](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)所提出用 2D 模式 (2D pattern) 來做相機校正的方法，針對其數學做更明白的解釋和程式碼示例。

## 2. 求 Homography

以下將平面模式物品皆用棋盤來代稱。當我們把現實座標的棋盤設為 $Z=0$ ，我們會得到：

$$
\begin{align}
    { 
        \begin{bmatrix}
        u \\\\
        v \\\\
        1
        \end{bmatrix}
        =
        \mathbf A [\mathbf r1 \quad \mathbf r2 \quad \mathbf r3 \quad \mathbf t] 
        \begin{bmatrix}
        X \\\\
        Y \\\\
        0 \\\\
        1
        \end{bmatrix}
    } \\\\
    { 
        =
        \mathbf A [\mathbf r1 \quad \mathbf r2 \quad \mathbf t] 
        \begin{bmatrix}
        X \\\\
        Y \\\\
        1
        \end{bmatrix}
    }
\end{align}
$$

其中 $\mathbf A [\mathbf r1 \quad \mathbf r2 \quad \mathbf t] $ 為 homography $ \mathbf H$。

在實際情況我們已經有很多棋盤的照片

![many chessboard pictures](https://user-images.githubusercontent.com/18013815/78544179-53ea9c80-782c-11ea-8c40-c0a6da560027.png)

我們經由圖片分析可以取得每張照片裡面格點的 2D 座標。

同時我們知道這些照片裡面的棋盤座標都是相同的，所以可以假設棋盤的 3D 座標是 $(0, 0, 0)$、$(0, 1, 0)$、$(1, 1, 0)$、$(1, 2, 0)$、$(2, 1, 0)$ ...。

於是我們就可以求 $\mathbf H$：

$$\mathbf{\tilde m} = \mathbf H \mathbf {\tilde M}$$

解 $\mathbf H$ 的一種方法參照 Zhang 的論文附錄 A，求以下解：

$$
\begin{bmatrix}
    {\mathbf {\tilde M}}^T & \mathbf 0^T & -u {\mathbf {\tilde M}}^T\\\\
    \mathbf 0^T   & {\mathbf{\tilde M}}^T &-v{\mathbf {\tilde M}}^T
\end{bmatrix} \mathbf x = \mathbf L \mathbf x = 0
$$

定義 $\mathbf x$ 左邊的矩陣為 $\mathbf L$，爆開來的話會長得像這樣：

$$
\begin{bmatrix}
   &X_1 &Y_1 &1 &0 &0 &0 &-uX_1 &-uY_1 &-u \\\\
   &0 &0 &0 &X_1 &Y_1 &1 &-vX_1 &-vY_1 &-v
\end{bmatrix}
$$

一張照片可以提供兩行，有 n 張照片的話，$\mathbf L$ 就會是一個 $2n \times 9$ 的矩陣。

$\mathbf x$ 的解就會是 the eigenvector of $\mathbf L^T \mathbf L$ associated with the smallest eigenvalue。用程式碼可能會比較直覺：

```py
L # 2n x 9 的 numpy 矩陣
w, v, vh = np.linalg.svd(L)
x = vh[-1]
```

透過 $\mathbf x$ 算出的 $\mathbf H$ 算出來需要補一個係數 $\rho$ 回去：

$$
\mathbf H = \rho x = \rho
\begin{bmatrix}
    x_1 &x_2 &x_3 \\\\
    x_4 &x_4 &x_5 \\\\
    x_6 &x_7 &x_8
\end{bmatrix}
$$

根據

$$\mathbf{\tilde m} = \rho \mathbf x \mathbf {\tilde M}$$

我的算法是用每個對應的座標帶入式子得到 $\rho$ 取平均。

## 3. Intrisic Parameters 的限制

定義 $\mathbf H = [\mathbf h_1 \quad \mathbf h_2 \quad \mathbf h3]$：

$$[\mathbf h_1 \quad \mathbf h_2 \quad \mathbf  h3] = \lambda \mathbf A [\mathbf r_1 \quad \mathbf r_2 \quad \mathbf t]$$

$\lambda$ 是一個任意係數，根據 $r_1$ 和 $r_2$ 是正交 (orthonormal)，可以有以下式子：

$$
\begin{align}
    {\mathbf  h_1}^T {\mathbf  A}^{-T} \mathbf  A^{-1} \mathbf h_2 &= 0 \\\\
    {\mathbf  h_1}^T {\mathbf  A}^{-T} \mathbf  A^{-1} \mathbf h_1 &= \mathbf  {h_2}^T \mathbf  A^{-T} \mathbf  A^{-1} \mathbf h_2
\end{align}
$$

## 4. 求 Intrisic Parameters

因為 intrinsic matrix 是相機本身參數，所以每張照片都會一樣。接下來都跳過論文證明過程，直接看如何求 intrinsic matrix。

根據第 2 節，我們可以取得每一張照片的 $\mathbf H$。然後令：

$$
\mathbf B = {\mathbf A}^{-T} {\mathbf A}^{-1} = 
\begin{bmatrix}
    B_{11} &B_{21} &B_{13} \\\\
    B_{12} &B_{22} &B_{23} \\\\
    B_{13} &B_{23} &B_{33}
\end{bmatrix}
$$

$\mathbf B$ 是一對稱矩陣，然後令 $\mathbf b = [B_{11}, B_{12} ,B_{22}, B_{23}, B_{33}, B_{11}]^T$

 
$\mathbf H$ 已經知道，令 $\mathbf h_i$ 指 $\mathbf H$ 的第 i 個直欄，所以 $\mathbf h_i = [h_{i1}, h_{i2} ,h_{i3}]^T$

然後定義

$$
\mathbf v_{ij} = [h_{i1} h_{j1}, h_{i1} h_{j2} + h_{i2} h_{j1}, h_{i2} h_{j2},
h_{i3} h_{j1} + h_{i1} h_{j3}, h_{i3} h_{j2} + h_{i2} h_{j3}, h_{i3} h_{j3}]^T
$$

接著只要解

$$
\begin{bmatrix}
    \mathbf v_{12}^T \\\\
    {\( \mathbf v_{11} - \mathbf v_{22}\)}^T
\end{bmatrix} \mathbf b = \mathbf V \mathbf b = 0
$$

其中 $\mathbf V$ 是一個 $2n \times 6$ 的矩陣。一張照片會有兩個式子，當我們有至少 3 個平面 ( $n >= 3$) 時，我們就能求得 $\mathbf b$ 的解。

然後解法一樣，會是 the eigenvector of $\mathbf L^T \mathbf L$ associated with the smallest eigenvalue：

```py
w, v, vh = np.linalg.svd(V)
b = vh[-1]
```

接著從 $\mathbf B$ 找出各項參數：

$$
\begin{align}
v_0 &= (B_{12} B_{13} − B_{11} B_{23})/(B_{11} B_{22} − B_{12}^2)\\\\
λ &= B_{33} − [B_{13}^2 + v_0(B_{12}B_{13} − B_{11}B_{23})]/B_{11}\\\\
α &=\sqrt{λ/B_{11}}\\\\
β &=\sqrt{λB_{11}/(B_{11}B_{22} − B^2_{12})}\\\\
γ &= −B_{12} α^2 β/λ\\\\
u_0 &= γv_0/β − B_{13}α^2/λ\\\\
\end{align}
$$

就可以得到 intrinsic matrix $\mathbf A$，根據定義代回去就好：

$$
\mathbf A = 
\begin{bmatrix}
α &γ &u_0 \\\\
0 &β &v_0 \\\\
0 &0 &1
\end{bmatrix}
$$

## 5. 求 extrinsic parameters

因為 $\mathbf A$ 是固定的，每張照片有可以得到一個 $\mathbf H$，因此我們就可以解析出 extrinsic parameters。

$$
    \mathbf H = [\mathbf h_1 \quad \mathbf h_2 \quad \mathbf h_3]
    = \mathbf A [\mathbf r_1 \quad \mathbf r_2 \quad \mathbf r_3 \quad \mathbf t] 
$$

其關係如下：

$$
\begin{align}
    \mathbf r_1 &= \lambda \mathbf A^{-1} \mathbf h_1  \\\\
    \mathbf r_2 &= \lambda \mathbf A^{-1} \mathbf h_2  \\\\
    \mathbf r_3 &= \mathbf r_1 \times \mathbf r_2  \\\\
    \mathbf t &= \lambda \mathbf A^{-1} \mathbf h_3  \\\\
\end{align} 
$$

其中 $\lambda$ 是比例係數 (與之前的 $\lambda$ 不一樣)，其值為 $ λ = 1/ \Vert \mathbf A^{−1} \mathbf h_1 \Vert = 1/ \Vert \mathbf A^{−1} \mathbf h_2 \Vert$。

到這邊我們就已經把 intrinsic parameters 和 extrinsic parameters 都求出來了。

## OpenCV 的 calibrateCamera 函數

經由上面算法，我們可以實現跟 `cv2.calibrateCamera` 一樣功能的函數，這邊對照 OpenCV 實現的 `cv2.calibrateCamera` 函數定義：

```py
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
```

發現 `cv2.calibrateCamera` 使用 rotation vector 和 translation vector 來表示 extrinsic parameters，而上面我們算出的是矩陣，所以需要做轉換。

其中 `rvecs` 為所有照片的 rotation vector, `tvecs` 為所有照片的 translation vector，而每一個照片所對應的 rotation 和 translation 算法如下：

```py
from scipy.spatial.transform import Rotation as R

# A, h1, h2, h3 為已求得 numpy 矩陣

lambda_ = 1/np.linalg.norm(np.dot(np.linalg.inv(A), h1))

r1 = lambda_*np.dot(np.linalg.inv(A), h1)
r2 = lambda_*np.dot(np.linalg.inv(A), h2)
r3 = np.cross(r1, r2
r_matrix = np.array([r1, r2, r3)]).T

t = lambda_*np.dot(np.linalg.inv(A), h3)
r = R.from_matrix(r_matrix).as_rotvec()

rvecs.append(r)
tvecs.append(t)
```

## Extrinsic 視覺化

將算出來的 extrinsics matrix 算出來之後，進行視覺化，就可以還原每張照片原本拍攝的位置與角度。小四角錐是相機，紅色方框則是棋盤。

![extrinsic visualization](https://user-images.githubusercontent.com/18013815/78570951-3a5f4a00-7858-11ea-889c-8db0a6ae7b74.jpg)


## 一些小技巧

如果照片是棋盤，可以直接呼叫 `cv2.findChessboardCorners`。

棋盤圖片可以先縮小，因為找 pattern 的點要花時間，同時因為照片座標都變小，也可以降低矩陣爆開的大小。

求 SVD 的時候，要注意 rank of matrix 夠不夠解開矩陣。
