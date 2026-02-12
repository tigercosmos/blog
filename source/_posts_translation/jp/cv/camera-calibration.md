---
title: "2Dパターンによるカメラキャリブレーション"
date: 2020-04-06 00:00:00
tags: [computer vision, camera calibration, 2D pattern, python]
des: "カメラキャリブレーション（Camera Calibration）は、intrinsic parameters（内部パラメータ）と extrinsic parameters（外部パラメータ）など、カメラの各種パラメータを推定する手法です。本記事では、Zhengyou Zhang の “A Flexible New Technique for Camera Calibration” で提案された 2D パターンによるキャリブレーションを中心に、数式をより分かりやすく整理し、コード例も示します。"
lang: jp
translation_key: camera-calibration
---

## 1. はじめに

[Camera Calibration](https://en.wikipedia.org/wiki/Camera_resectioning)（カメラキャリブレーション）とは、カメラの各種パラメータを推定することです。言い換えると、カメラが写真を撮影したとき、画像上のある 2D 点が現実世界の 3D 点に対応するために必要なすべてのパラメータを得る、ということになります。ここには intrinsic parameters（カメラ内部パラメータ）と extrinsic parameters（[定向](https://zh.wikipedia.org/wiki/%E5%AE%9A%E5%90%91_(%E5%90%91%E9%87%8F%E7%A9%BA%E9%96%93))／姿勢・外部パラメータ）が含まれます。

画像上の 2D 点と現実世界の 3D 点の関係は次のように書けます：

$$\mathbf{\tilde m} = \mathbf A [\mathbf R \quad \mathbf t] \mathbf {\tilde M}$$

ここで $\mathbf{\tilde m}$ は画像座標ベクトル $[u, v, 1]^T$、$\mathbf {\tilde M}$ は世界座標ベクトル $[X, Y, Z, 1]^T$、$\mathbf A$ は intrinsic matrix、$[\mathbf R \quad \mathbf t]$ は extrinsic matrix です。

カメラキャリブレーションについては、Richard Szeliski 著 *Computer Vision: Algorithms and Applications* の第6章 Feature-based alignment も参考になります。

画像から現実世界の 3D 点を復元するのは簡単ではありません。しかし、平面上に既知のパターン（pattern）を持つ物体、例えばチェスボードを用いると、各格子の相対座標が既知であり、さらに $Z$ 座標が一定と仮定できるため、複数枚のチェスボード画像から intrinsic matrix と extrinsic matrix を復元できます。本記事では、Zhengyou Zhang の論文 [“A Flexible New Technique for Camera Calibration”](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) で提案された 2D パターン（2D pattern）によるキャリブレーションに焦点を当て、数式の意味をより明確にしながら、コード例も示します。

## 2. Homography の計算

以降、平面パターンはチェスボードを代表として扱います。現実座標系においてチェスボード平面を $Z=0$ とすると：

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

ここで $\mathbf A [\mathbf r1 \quad \mathbf r2 \quad \mathbf t] $ が homography $\mathbf H$ です。

実際の状況では、チェスボード画像が複数枚あります：

![many chessboard pictures](https://user-images.githubusercontent.com/18013815/78544179-53ea9c80-782c-11ea-8c40-c0a6da560027.png)

画像解析により、各画像におけるコーナー点の 2D 座標を取得できます。

また、すべての画像でチェスボード上の座標は同一なので、チェスボードの 3D 座標を $(0, 0, 0)$、$(0, 1, 0)$、$(1, 1, 0)$、$(1, 2, 0)$、$(2, 1, 0)$ ... のように仮定できます。

したがって、$\mathbf H$ は次式から求められます：

$$\mathbf{\tilde m} = \mathbf H \mathbf {\tilde M}$$

$\mathbf H$ を解く方法の1つは Zhang 論文の付録Aに記載されている手法で、次を解きます：

$$
\begin{bmatrix}
    {\mathbf {\tilde M}}^T & \mathbf 0^T & -u {\mathbf {\tilde M}}^T\\\\
    \mathbf 0^T   & {\mathbf{\tilde M}}^T &-v{\mathbf {\tilde M}}^T
\end{bmatrix} \mathbf x = \mathbf L \mathbf x = 0
$$

左辺の行列を $\mathbf L$ と定義します。展開すると次の形になります：

$$
\begin{bmatrix}
   &X_1 &Y_1 &1 &0 &0 &0 &-uX_1 &-uY_1 &-u \\\\
   &0 &0 &0 &X_1 &Y_1 &1 &-vX_1 &-vY_1 &-v
\end{bmatrix}
$$

1枚の画像から2行が得られるので、画像が n 枚あると $\mathbf L$ は $2n \times 9$ 行列です。

$\mathbf x$ の解は「$\mathbf L^T \mathbf L$ の最小固有値に対応する固有ベクトル」です。コードのほうが直感的かもしれません：

```py
L # 2n x 9 的 numpy 矩陣
w, v, vh = np.linalg.svd(L)
x = vh[-1]
```

$\mathbf x$ から得られる $\mathbf H$ は、係数 $\rho$ を掛けてスケールを補う必要があります：

$$
\mathbf H = \rho x = \rho
\begin{bmatrix}
    x_1 &x_2 &x_3 \\\\
    x_4 &x_4 &x_5 \\\\
    x_6 &x_7 &x_8
\end{bmatrix}
$$

次式：

$$\mathbf{\tilde m} = \rho \mathbf x \mathbf {\tilde M}$$

に各対応点を代入して $\rho$ を求め、その平均を取るのが私のやり方です。

## 3. Intrisic Parameters の制約

$\mathbf H = [\mathbf h_1 \quad \mathbf h_2 \quad \mathbf h3]$ と定義します：

$$[\mathbf h_1 \quad \mathbf h_2 \quad \mathbf  h3] = \lambda \mathbf A [\mathbf r_1 \quad \mathbf r_2 \quad \mathbf t]$$

$\lambda$ は任意の係数です。$r_1$ と $r_2$ が直交正規（orthonormal）であることから、次が得られます：

$$
\begin{align}
    {\mathbf  h_1}^T {\mathbf  A}^{-T} \mathbf  A^{-1} \mathbf h_2 &= 0 \\\\
    {\mathbf  h_1}^T {\mathbf  A}^{-T} \mathbf  A^{-1} \mathbf h_1 &= \mathbf  {h_2}^T \mathbf  A^{-T} \mathbf  A^{-1} \mathbf h_2
\end{align}
$$

## 4. Intrisic Parameters の計算

intrinsic matrix はカメラ固有のパラメータであり、各画像で同一です。以降は論文の証明は省略し、intrinsic matrix の求め方に焦点を当てます。

第2節の手順で各画像の $\mathbf H$ が求まります。次を定義します：

$$
\mathbf B = {\mathbf A}^{-T} {\mathbf A}^{-1} = 
\begin{bmatrix}
    B_{11} &B_{12} &B_{13} \\\\
    B_{12} &B_{22} &B_{23} \\\\
    B_{13} &B_{23} &B_{33}
\end{bmatrix}
$$

$\mathbf B$ は対称行列です。さらに $\mathbf b = [B_{11}, B_{12} ,B_{22}, B_{23}, B_{33}, B_{11}]^T$ とします。

 
$\mathbf H$ は既知で、$\mathbf h_i$ を $\mathbf H$ の i 番目の列ベクトルとすると、$\mathbf h_i = [h_{i1}, h_{i2} ,h_{i3}]^T$ です。

ここで：

$$
\mathbf v_{ij} = [h_{i1} h_{j1}, h_{i1} h_{j2} + h_{i2} h_{j1}, h_{i2} h_{j2},
h_{i3} h_{j1} + h_{i1} h_{j3}, h_{i3} h_{j2} + h_{i2} h_{j3}, h_{i3} h_{j3}]^T
$$

を定義し、次を解きます：

$$
\begin{bmatrix}
    \mathbf v_{12}^T \\\\
    {\( \mathbf v_{11} - \mathbf v_{22}\)}^T
\end{bmatrix} \mathbf b = \mathbf V \mathbf b = 0
$$

$\mathbf V$ は $2n \times 6$ 行列です。1枚の画像から2つの式が得られ、少なくとも 3 つの平面（$n >= 3$）があれば $\mathbf b$ が解けます。

解法は同様に「$\mathbf L^T \mathbf L$ の最小固有値に対応する固有ベクトル」です：

```py
w, v, vh = np.linalg.svd(V)
b = vh[-1]
```

その後、$\mathbf B$ から各パラメータを復元します：

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

これで intrinsic matrix $\mathbf A$ が得られます。定義に従って代入します：

$$
\mathbf A = 
\begin{bmatrix}
α &γ &u_0 \\\\
0 &β &v_0 \\\\
0 &0 &1
\end{bmatrix}
$$

## 5. extrinsic parameters の計算

$\mathbf A$ は固定で、各画像から $\mathbf H$ が得られるため、extrinsic parameters を解析的に求められます。

$$
    \mathbf H = [\mathbf h_1 \quad \mathbf h_2 \quad \mathbf h_3]
    = \mathbf A [\mathbf r_1 \quad \mathbf r_2 \quad \mathbf r_3 \quad \mathbf t] 
$$

関係式は次のとおりです：

$$
\begin{align}
    \mathbf r_1 &= \lambda \mathbf A^{-1} \mathbf h_1  \\\\
    \mathbf r_2 &= \lambda \mathbf A^{-1} \mathbf h_2  \\\\
    \mathbf r_3 &= \mathbf r_1 \times \mathbf r_2  \\\\
    \mathbf t &= \lambda \mathbf A^{-1} \mathbf h_3  \\\\
\end{align} 
$$

ここで $\lambda$ はスケール係数（前の $\lambda$ とは別物）で、$ λ = 1/ \Vert \mathbf A^{−1} \mathbf h_1 \Vert = 1/ \Vert \mathbf A^{−1} \mathbf h_2 \Vert$ です。

これで intrinsic parameters と extrinsic parameters の両方が求まりました。

## OpenCV の `calibrateCamera`

上の手順により `cv2.calibrateCamera` と同等の機能を実装できます。OpenCV の `cv2.calibrateCamera` の定義は次です：

```py
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
```

`cv2.calibrateCamera` は extrinsic parameters を rotation vector と translation vector で表しますが、上で求めたのは行列なので変換が必要です。

`rvecs` は各画像の rotation vector、`tvecs` は各画像の translation vector です。各画像に対応する rotation と translation は次のように計算します：

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

## Extrinsic の可視化

算出した extrinsics matrix を可視化すると、各画像が撮影された位置と角度を復元できます。小さな四角錐がカメラ、赤枠がチェスボードです。

![extrinsic visualization](https://user-images.githubusercontent.com/18013815/78570951-3a5f4a00-7858-11ea-889c-8db0a6ae7b74.jpg)


## ちょっとしたコツ

チェスボード画像であれば `cv2.findChessboardCorners` を直接呼び出せます。

チェスボード画像は先に縮小しても良いです。pattern 点の検出は時間がかかりますし、画像座標が小さくなることで、展開される行列のサイズも小さくなります。

SVD を解く際は、行列の rank が十分かどうかに注意してください。

