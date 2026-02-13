---
title: "コンピュータアニメーションにおける逆運動学（Inverse Kinematics）"
date: 2020-06-09 00:00:00
tags: [電腦動畫, 反向動力法, computer animation, inverse kinematics, jacobian, pseudo inverse]
des: "コンピュータアニメーションやロボティクスでは、取っ手に手を伸ばす／ボールをキャッチするといった、特定の姿勢や動作を人物・ロボットにさせたいことがあります。正運動学（Forward Kinematics）でその姿勢に至る動きを前向きに推論するのは、次元が高くなるとほぼ不可能です。逆運動学（Inverse Kinematics）は、目標位置を与え、そこへ到達するためにどのように動くべきかを逆向きに解くことで、この問題に取り組みます。"
lang: jp
translation_key: inverse-kinematics
---

## 1. はじめに

コンピュータアニメーションやロボティクスでは、人物やロボットに特定の動作をさせたいことがあります。たとえば、手を伸ばして取っ手に触れる、ボールをキャッチする、といったケースです。

対象は多くの関節を持ち得ます。仮に各関節が 5 自由度（DoF）で、腕が 3 つの関節から構成されるとすると、すでに 15 次元になります。このような高次元空間で、正運動学（Forward Kinematics, FK）だけを用いて「どう動けば狙った姿勢になるか」を前向きに推論するのは、ほぼ不可能です。

そこで逆運動学（Inverse Kinematics, IK）を用います。対象に目標位置を与え、そこへ到達するためにどのように動くべきかを逆向きに計算します。

![ik example](https://user-images.githubusercontent.com/18013815/84060735-4cdc3800-a9ef-11ea-878b-dafc5233ff5e.gif)
（IK によって、腕を目標位置へ動かす方法を求める例）

逆運動学は一般に高次元の問題を解く必要があり、計算コストが高く、解が複数存在することも多いです。代表的な手法として、Cyclic Coordinate Descent、Jacobian Pseudoinverse、Jacobian Transpose、Levenberg–Marquardt Damped Least Squares、Quasi-Newton / Conjugate Gradient 法、ニューラルネットワークによる手法などがあります。

どの IK 手法を選ぶかは、計算時間、精度、逆算して得たい動作姿勢の性質などの要件によって変わります。たとえば Cyclic Coordinate Descent は指先を先に動かしてから腕を調整することが多く、Jacobian Pseudoinverse は全関節をまとめて更新します。ニューラルネットワークは、より現実的な結果を高速に予測できる場合があります。本記事では、IK の Jacobian Pseudoinverse 法を紹介します。

## 2. Jacobian Pseudoinverse の原理と実装

### 2.1 パラメータ定義

機械アームを考え、指先が特定の位置に到達してほしいとします。指先から腕の途中にある各関節は回転できると仮定し、第 $i$ 関節の初期位置を $\mathbf s_i$ とします。これらの中間関節の回転によって、指先が目標位置 $\mathbf t$ に到達するようにします。

<img src="https://i.imgur.com/coqDQA1.jpg" width="80%">


世界座標を $n$ 次元とします。通常は $n = 3$（$x, y, z$ の 3 次元）ですが、別の次元空間へ写像することもできます。

第 $i$ 関節の初期位置ベクトルは次のように表します：

$$ 
\mathbf s_i = 
    \begin{bmatrix}
s_x & s_y &s_z & \dots & s_n
    \end{bmatrix}
$$

目標位置ベクトルは次の通りです：

$$ 
\mathbf t = 
    \begin{bmatrix}
t_x & t_y &t_z & \dots & t_n
    \end{bmatrix}
$$

第 $i$ 関節の End Effector ベクトル $\mathbf e$ を次のように定義します：

$$\mathbf e_i = \mathbf t - \mathbf s_i$$

したがって、各関節の End Effector ベクトルは次のように表せます：


$$ 
\mathbf e = 
    \begin{bmatrix}
e_1 & e_2 &e_3 & \dots & e_N
    \end{bmatrix}
$$



全関節に含まれる可動次元（DoF）の総数を $M$ とし、それらの回転角ベクトルを次のように表します：

$$ 
\mathbf \theta = 
    \begin{bmatrix}
\theta_1 & \theta_2 & \theta_3 & \dots & \theta_M
    \end{bmatrix}
$$

### 2.2 運動学

正運動学（FK）では、回転角が与えられたときに、関数 $f$ を用いて次の End Effector $\mathbf e$ を計算できます：

$$\mathbf e = f(\mathbf \theta)$$


逆運動学（IK）ではこの関係を反転させ、End Effector が与えられたときに回転角を求めます：

$$\mathbf \theta = f^{-1}(\mathbf e)$$

### 2.3 ヤコビアン（Jacobian）

ベクトル微積分における [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) は、関数 $f$ の線形近似を表します。ヤコビアン $J$ を使うことで、End Effector $\mathbf e$ から $d\mathbf \theta$ を逆算できます。

<img src="https://i.imgur.com/z22VM1L.jpg" width="80%">


<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">J</mi></mrow><mo>=</mo><mrow class="MJX-TeXAtom-ORD"><mfrac><mrow class="MJX-TeXAtom-ORD"><mi>d</mi><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">e</mi></mrow></mrow><mrow class="MJX-TeXAtom-ORD"><mi>d</mi><mrow class="MJX-TeXAtom-ORD"><mi>&#x03B8;<!-- θ --></mi></mrow></mrow></mfrac></mrow></math>

$$d\mathbf e  = \mathbf J d \mathbf \theta$$

$\mathbf J$ は次のように表せます：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"> <mrow class="MJX-TeXAtom-ORD"> <mi mathvariant="bold">J</mi> </mrow> <mo>=</mo> <mrow> <mo>(</mo> <mtable rowspacing="4pt" columnspacing="1em"> <mtr> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mn>1</mn> </msub> </mrow> <mrow> <mi>d</mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mn>1</mn> </msub> </mrow> </mfrac> </mtd> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mn>1</mn> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mn>2</mn> </msub> </mrow> </mfrac> </mtd> <mtd> <mo>&#x22EF;<!-- ⋯ --></mo> </mtd> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mn>1</mn> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mi>n</mi> </msub> </mrow> </mfrac> </mtd> </mtr> <mtr> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mn>2</mn> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mn>1</mn> </msub> </mrow> </mfrac> </mtd> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mn>2</mn> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mn>2</mn> </msub> </mrow> </mfrac> </mtd> <mtd> <mo>&#x22EF;<!-- ⋯ --></mo> </mtd> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mn>2</mn> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mi>n</mi> </msub> </mrow> </mfrac> </mtd> </mtr> <mtr> <mtd> <mo>&#x22EE;<!-- ⋮ --></mo> </mtd> <mtd> <mo>&#x22EE;<!-- ⋮ --></mo> </mtd> <mtd> <mo>&#x22EF;<!-- ⋯ --></mo> </mtd> <mtd> <mo>&#x22EE;<!-- ⋮ --></mo> </mtd> </mtr> <mtr> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mi>k</mi> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mn>1</mn> </msub> </mrow> </mfrac> </mtd> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mi>k</mi> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mn>2</mn> </msub> </mrow> </mfrac> </mtd> <mtd> <mo>&#x22EF;<!-- ⋯ --></mo> </mtd> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mi>k</mi> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mi>n</mi> </msub> </mrow> </mfrac> </mtd> </mtr> </mtable> <mo>)</mo> </mrow></math>

$d\mathbf e  = \mathbf J d \mathbf \theta$ が得られているので、移項すると：

$$ d\mathbf \theta  = \mathbf J^{-1} d \mathbf e$$

この回転角の変化量 $d\mathbf \theta$ が求めたい値です。$d\mathbf \theta$ を使って、姿勢を目標位置へ向けて一歩ずつ更新できるためです。

<img src="https://i.imgur.com/ZP2mRyp.jpg" width="80%">


### 2.4 ヤコビアンの近似解を求める


世界座標を 3 次元（3 DoF）とします。$d\mathbf \theta  = \mathbf J^{-1} d \mathbf e$ を解くためには、まずヤコビアンが必要です。以下の式はすべて世界座標上で定義されています。

まずヤコビアンの 1 列を見てみます：

$$
\mathbf J_i =
\frac{\partial \mathbf e}{d\theta_1}=
\begin{bmatrix}
\frac{\partial e_{x}}{\partial\theta_i} \\\\
\frac{\partial e_{y}}{\partial\theta_i} \\\\
\frac{\partial e_{z}}{\partial\theta_i}
\end{bmatrix}
$$

$\mathbf \theta$ と $\mathbf e$ の線形な微小変化量を用いて、偏微分を近似します：

$$
\mathbf J_i =
\frac{\partial \mathbf e}{d\theta_i}=
\frac{\Delta \mathbf e}{\Delta\theta_i}=
\begin{bmatrix}
\frac{\Delta e_{x}}{\Delta\theta_i} \\\\
\frac{\Delta e_{y}}{\Delta\theta_i} \\\\
\frac{\Delta e_{z}}{\Delta\theta_i}
\end{bmatrix}
$$

関節 $A$ の **1 つの軸**（1 DoF）まわりの回転を考えると、その軸における End Effector $\mathbf e$ の線形変化 $d\mathbf e$ は、関節 $A$ がその軸で回転することによって生じる変化（$\mathbf \omega \mathbf rdt$）に等しくなります。

$$
\frac{d\mathbf e}{dt} = 
\vert \mathbf \omega \vert {\frac{\mathbf \omega}{\vert \mathbf \omega \vert }} \times \mathbf r =
\frac{d \mathbf \theta}{dt} \mathbf a \times \mathbf r
$$

ここで $\mathbf a= \frac{\mathbf \omega}{\vert \omega \vert}$ は単位回転ベクトルを表します。

移項すると：

$${d\mathbf e \over d\mathbf \theta} = \mathbf a \times \mathbf r$$


したがって次を得ます：

$$
\mathbf J_1 =
\frac{\Delta \mathbf e}{\Delta\theta_1}=
\begin{bmatrix}
\frac{\Delta e_{x}}{\Delta\theta_1} \\\\
\frac{\Delta e_{y}}{\Delta\theta_1} \\\\
\frac{\Delta e_{z}}{\Delta\theta_1}
\end{bmatrix}=
\mathbf a_i \times (\mathbf e - \mathbf r_i)
$$

ここで $\mathbf r_i$ は第 $i$ 関節の位置ベクトルです。

ここでの $\mathbf J_i$ は関節の **1** つの DoF に対応していることに注意してください。したがって、全関節の DoF を合計して $M$ 個あるなら、$\mathbf J$ は $3 \times M$ の行列になります。

実装では、各 $J_i$ は 1 つの DoF だけを含むので、たとえば関節 $A$ が $(x, z)$ の 2 軸で回転できるとすると、角速度は $\mathbf \omega_A = (\mathbf\omega_{Ax}, 0, \mathbf\omega_{Az})$ になります。$\mathbf \theta_i$ が $A$ の $z$ 軸を表すとすると、$\mathbf \omega_i$ は $\mathbf \omega_{A} \times (0, 0 ,1) = \mathbf\omega_{Az}$ です。

### 2.5 ヤコビアンの逆（擬似逆行列）

$d\mathbf \theta  = \mathbf J^{-1} d \mathbf e$ を解くために、$\mathbf J$ を得た後は $\mathbf J^{-1}$ を計算したくなります。しかし、逆行列は正方行列でなければ定義できないのに対し、ここでの $\mathbf J$ は $3 \times M$ の行列です。したがって直接 Inverse を取ることはできず、代わりに擬似逆行列 $\mathbf J^+$ を用います。

以下の手順を考えます：

$$ d\mathbf e = \mathbf J d\mathbf  \theta$$
$$ \mathbf J^T d\mathbf e = \mathbf J^T J d \mathbf \theta$$
$$ (\mathbf J^T \mathbf J)^{-1} \mathbf J^T d\mathbf e =(\mathbf J^T \mathbf J)^{-1}(\mathbf J^T \mathbf J)d \mathbf \theta$$
$$ (\mathbf J^T \mathbf J)^{-1} \mathbf J^T d\mathbf e =d \mathbf  \theta$$
$$ \mathbf J^+ d\mathbf e =d \mathbf \theta$$
$$ \mathbf J^+ = (\mathbf J^T \mathbf J)^{-1} \mathbf J^T $$


SVD の性質を用いると：

$$SVD(J)=UΣV^∗$$


$\mathbf J^+$ を代入して展開すると：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"> <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0em 2em 0em 2em 0em 2em 0em 2em 0em 2em 0em" displaystyle="true"> <mtr> <mtd> <msup> <mrow class="MJX-TeXAtom-ORD"> <mi mathvariant="bold">J</mi> </mrow> <mo>+</mo> </msup> </mtd> <mtd> <mi></mi> <mo>=</mo> <mo stretchy="false">(</mo> <msup> <mrow class="MJX-TeXAtom-ORD"> <mi mathvariant="bold">J</mi> </mrow> <mo>&#x2217;<!-- ∗ --></mo> </msup> <mrow class="MJX-TeXAtom-ORD"> <mi mathvariant="bold">J</mi> </mrow> <msup> <mo stretchy="false">)</mo> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> </mrow> </msup> <msup> <mrow class="MJX-TeXAtom-ORD"> <mi mathvariant="bold">J</mi> </mrow> <mo>&#x2217;<!-- ∗ --></mo> </msup> </mtd> </mtr> <mtr> <mtd /> <mtd> <mi></mi> <mo>=</mo> <mo stretchy="false">(</mo> <mi>V</mi> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <msup> <mi>U</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> <mi>U</mi> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <msup> <mi>V</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> <msup> <mo stretchy="false">)</mo> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> </mrow> </msup> <mi>V</mi> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <msup> <mi>U</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> </mtd> </mtr> <mtr> <mtd /> <mtd> <mi></mi> <mo>=</mo> <mo stretchy="false">(</mo> <mi>V</mi> <msup> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <mn>2</mn> </msup> <msup> <mi>V</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> <msup> <mo stretchy="false">)</mo> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> </mrow> </msup> <mi>V</mi> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <msup> <mi>U</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> </mtd> </mtr> <mtr> <mtd /> <mtd> <mi></mi> <mo>=</mo> <mo stretchy="false">(</mo> <msup> <mi>V</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> <msup> <mo stretchy="false">)</mo> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> </mrow> </msup> <msup> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>2</mn> </mrow> </msup> <msup> <mi>V</mi> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> </mrow> </msup> <mi>V</mi> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <msup> <mi>U</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> </mtd> </mtr> <mtr> <mtd /> <mtd> <mi></mi> <mo>=</mo> <mi>V</mi> <msup> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>2</mn> </mrow> </msup> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <msup> <mi>U</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> </mtd> </mtr> <mtr> <mtd /> <mtd> <mi></mi> <mo>=</mo> <mi>V</mi> <msup> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> </mrow> </msup> <msup> <mi>U</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> </mtd> </mtr> </mtable></math>

### 2.6 IK アルゴリズム

$\mathbf J^+$ が得られたら、1 ステップ前（更新後）の状態を推定できます：

$$\mathbf \theta_{k} = \mathbf \theta_{k+1} - dt * \mathbf J^+(\mathbf \theta_{k+1}) * d\mathbf e$$

擬似コード：

```python
while(ABS(current_position - target_position) > epsilon and
      iterator < max_iterator):
    
    1. compute Jacobian J
    2. compute Pseoduinverse Jacobain J^+
    3. compute change in joint DoFs: dθ = (J^+)de
    4. apply the changes to joints. move a small step α: θ += αdθ
    5. update current_position to close target_position
```

## 3. 考察

### 3.1 パラメータの影響

次の式：
$$\mathbf \theta_{k} = \mathbf \theta_{k+1} - dt * \mathbf J^+(\mathbf \theta_{k+1}) * d\mathbf e$$
より、ステップ幅 $dt$ は収束速度に影響します。各ステップが小さすぎると反復回数が増えますが、大きすぎると目標を行き過ぎて収束しにくくなる可能性があります。

収束しないことで反復が増えすぎるのを避けるため、誤差の閾値 $\epsilon$ を設定し、誤差が許容範囲に入ったら停止します。また、反復回数の上限も設定します。

実際、IK は常に解があるとは限りません。目標位置 $t$ が関節の最大到達距離を超えていたり、利用できる DoF の制約によって到達不可能だったりすることがあります。たとえば、どれだけ腕を伸ばしても天井には届かないですし、腕を外側へ無限に回転させ続けることもできません。これらはすべて無解のケースです。実装では、無解による破綻を避けるために $t$ を調整する、といった対策も考慮すべきです。

### 3.2 可動関節数が IK に与える影響

次の図のような初期姿勢の人物を与え、目標位置を紫色の点とします。

![](https://i.imgur.com/Q5zjvcD.png)

Case 1: 腕だけを動かせるようにして、IK で姿勢を解く：

![](https://i.imgur.com/HTNfgHB.png)
![](https://i.imgur.com/j21wFj9.png)

Case 2: 腕と上半身を動かせるようにして、IK で姿勢を解く：

![](https://i.imgur.com/Lo8DWA3.png)

Case 3: 腰より上を動かせるようにして、IK で姿勢を解く：

![](https://i.imgur.com/KLKFwme.png)

## 4. まとめ

逆運動学（IK）により、人物の現在の状態から「どのように動けば狙った姿勢に到達できるか」を逆算できます。コンピュータアニメーションでは、キーポーズ（キーフレーム）を与えて人物に狙った動作をさせることができます。ロボティクスでは、ロボットアームが目標作業位置へ到達するために IK が用いられます。

IK には多くの手法があります。本記事では Jacobian Pseudoinverse 法のみを紹介しました。この手法は全関節を一度に更新する性質があり、見た目としてはよりリアルに感じられることがあります。ただし、実際の人体動作にはより自然な「らしさ」があり、それを反映したアルゴリズムを選ぶべき場合もあります。用途に応じて適切な手法を選んでください（この部分は読者の調査に委ねます）。

## 5. 参考文献
- Samuel R. Buss. 2009. Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods. [http://graphics.cs.cmu.edu/nsp/course/15-464/Spring11/handouts/iksurvey.pdf](http://graphics.cs.cmu.edu/nsp/course/15-464/Spring11/handouts/iksurvey.pdf)

- [@shi-yan](https://epiphany.pub/@shi-yan). 2019. Inverse Kinematics Explained Interactively. [https://epiphany.pub/@shi-yan/inverse-kinematics-explained-interactively](https://epiphany.pub/@shi-yan/inverse-kinematics-explained-interactively)


- Rickard Nilsson. Inverse Kinematics Explained Interactively. 2009. [https://www.diva-portal.org/smash/get/diva2:1018821/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:1018821/FULLTEXT01.pdf)

- [Nancy Pollard](http://www.cs.cmu.edu/~nsp). 2003. CMU: Technical Animation. [http://www.cs.cmu.edu/~15464-s13/assignments/](http://www.cs.cmu.edu/~15464-s13/)
