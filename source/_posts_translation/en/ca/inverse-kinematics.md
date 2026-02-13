---
title: "Inverse Kinematics in Computer Animation"
date: 2020-06-09 00:00:00
tags: [電腦動畫, 反向動力法, computer animation, inverse kinematics, jacobian, pseudo inverse]
des: "In computer animation or robotics, we often want a character or robot to achieve a specific pose—for example, reaching a handle or catching a ball. Deriving such a pose with forward kinematics is almost impossible. Inverse kinematics addresses this by specifying a target position and then solving, in reverse, how the object should move to reach it."
lang: en
translation_key: inverse-kinematics
---

## 1. Introduction

In computer animation or robotics, we may want a character or robot to perform a specific action—for example, reaching a handle or catching a ball.

An object can have many joints. If each joint has 5 degrees of freedom (DoF) and an arm has three joints, that is 15 dimensions already. With forward kinematics (FK), it is almost impossible to reason forward and find how to reach a desired pose in such a high-dimensional space.

This is where inverse kinematics (IK) comes in. We give the object a target position, and then solve backward to compute how the object should move to reach that position.

![ik example](https://user-images.githubusercontent.com/18013815/84060735-4cdc3800-a9ef-11ea-878b-dafc5233ff5e.gif)
(Using IK to compute how to move the arm to the target position)

Inverse kinematics typically involves solving a high-dimensional problem, which makes it computationally expensive and often yields many possible solutions. Common approaches include Cyclic Coordinate Descent, Jacobian Pseudoinverse, Jacobian Transpose, Levenberg–Marquardt Damped Least Squares, Quasi-Newton and Conjugate Gradient methods, and neural network approaches.

Which IK method you choose depends on requirements such as computation time, accuracy, and the desired pose characteristics. For example, Cyclic Coordinate Descent typically moves the fingertip first and then adjusts the arm; Jacobian Pseudoinverse adjusts all joints at once; and neural networks can predict more realistic results quickly. In this post, I will introduce the Jacobian Pseudoinverse approach to IK.

## 2. Jacobian Pseudoinverse: Principles and Implementation

### 2.1 Parameter Definitions

Suppose we have a robotic arm and we want its fingertip to reach a specific position. We assume that every joint between the fingertip and the arm can rotate, and denote the initial position of joint $i$ as $\mathbf s_i$. Through rotations of these intermediate joints, the fingertip should reach the target position $\mathbf t$.

<img src="https://i.imgur.com/coqDQA1.jpg" width="80%">


We define the world coordinate system to be $n$-dimensional. Typically $n = 3$ (the $x$, $y$, and $z$ axes), but it can also be mapped into other dimensional spaces.

The initial position vector of joint $i$ is:

$$ 
\mathbf s_i = 
    \begin{bmatrix}
s_x & s_y &s_z & \dots & s_n
    \end{bmatrix}
$$

The target position vector is:

$$ 
\mathbf t = 
    \begin{bmatrix}
t_x & t_y &t_z & \dots & t_n
    \end{bmatrix}
$$

Define the end-effector vector of joint $i$ as $\mathbf e$:

$$\mathbf e_i = \mathbf t - \mathbf s_i$$

Therefore, the end-effector vector for each joint can be written as:


$$ 
\mathbf e = 
    \begin{bmatrix}
e_1 & e_2 &e_3 & \dots & e_N
    \end{bmatrix}
$$



Let the rotation-angle vector over all movable dimensions (DoF) across all joints be $M$-dimensional:

$$ 
\mathbf \theta = 
    \begin{bmatrix}
\theta_1 & \theta_2 & \theta_3 & \dots & \theta_M
    \end{bmatrix}
$$

### 2.2 Kinematics

In forward kinematics (FK), given rotation angles, we can compute the next end-effector $\mathbf e$ via a function $f$:

$$\mathbf e = f(\mathbf \theta)$$


In inverse kinematics (IK), we invert the relationship: given the end-effector, we want the rotation angles:

$$\mathbf \theta = f^{-1}(\mathbf e)$$

### 2.3 Jacobian

In vector calculus, the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) represents a linear approximation of a function $f$. We can use the Jacobian $J$ to back-solve $d\mathbf \theta$ from the end-effector $\mathbf e$.

<img src="https://i.imgur.com/z22VM1L.jpg" width="80%">


<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">J</mi></mrow><mo>=</mo><mrow class="MJX-TeXAtom-ORD"><mfrac><mrow class="MJX-TeXAtom-ORD"><mi>d</mi><mrow class="MJX-TeXAtom-ORD"><mi mathvariant="bold">e</mi></mrow></mrow><mrow class="MJX-TeXAtom-ORD"><mi>d</mi><mrow class="MJX-TeXAtom-ORD"><mi>&#x03B8;<!-- θ --></mi></mrow></mrow></mfrac></mrow></math>

$$d\mathbf e  = \mathbf J d \mathbf \theta$$

$\mathbf J$ can be written as:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"> <mrow class="MJX-TeXAtom-ORD"> <mi mathvariant="bold">J</mi> </mrow> <mo>=</mo> <mrow> <mo>(</mo> <mtable rowspacing="4pt" columnspacing="1em"> <mtr> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mn>1</mn> </msub> </mrow> <mrow> <mi>d</mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mn>1</mn> </msub> </mrow> </mfrac> </mtd> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mn>1</mn> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mn>2</mn> </msub> </mrow> </mfrac> </mtd> <mtd> <mo>&#x22EF;<!-- ⋯ --></mo> </mtd> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mn>1</mn> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mi>n</mi> </msub> </mrow> </mfrac> </mtd> </mtr> <mtr> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mn>2</mn> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mn>1</mn> </msub> </mrow> </mfrac> </mtd> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mn>2</mn> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mn>2</mn> </msub> </mrow> </mfrac> </mtd> <mtd> <mo>&#x22EF;<!-- ⋯ --></mo> </mtd> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mn>2</mn> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mi>n</mi> </msub> </mrow> </mfrac> </mtd> </mtr> <mtr> <mtd> <mo>&#x22EE;<!-- ⋮ --></mo> </mtd> <mtd> <mo>&#x22EE;<!-- ⋮ --></mo> </mtd> <mtd> <mo>&#x22EF;<!-- ⋯ --></mo> </mtd> <mtd> <mo>&#x22EE;<!-- ⋮ --></mo> </mtd> </mtr> <mtr> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mi>k</mi> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mn>1</mn> </msub> </mrow> </mfrac> </mtd> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mi>k</mi> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mn>2</mn> </msub> </mrow> </mfrac> </mtd> <mtd> <mo>&#x22EF;<!-- ⋯ --></mo> </mtd> <mtd> <mfrac> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>f</mi> <mi>k</mi> </msub> </mrow> <mrow> <mi mathvariant="normal">&#x2202;<!-- ∂ --></mi> <msub> <mi>&#x03B8;<!-- θ --></mi> <mi>n</mi> </msub> </mrow> </mfrac> </mtd> </mtr> </mtable> <mo>)</mo> </mrow></math>

Recall that we obtained $d\mathbf e  = \mathbf J d \mathbf \theta$. Rearranging gives:

$$ d\mathbf \theta  = \mathbf J^{-1} d \mathbf e$$

This change in rotation angles $d\mathbf \theta$ is exactly what we want, because we can use $d\mathbf \theta$ to step the pose toward the desired position.

<img src="https://i.imgur.com/ZP2mRyp.jpg" width="80%">


### 2.4 Approximating the Jacobian


Assume the world coordinates are three-dimensional (3 DoF). To solve $d\mathbf \theta  = \mathbf J^{-1} d \mathbf e$, we first need the Jacobian. The equations below are all defined in world coordinates.

Let’s look at one column of the Jacobian:

$$
\mathbf J_i =
\frac{\partial \mathbf e}{d\theta_1}=
\begin{bmatrix}
\frac{\partial e_{x}}{\partial\theta_i} \\\\
\frac{\partial e_{y}}{\partial\theta_i} \\\\
\frac{\partial e_{z}}{\partial\theta_i}
\end{bmatrix}
$$

We approximate the partial derivatives by taking small linear perturbations of $\mathbf \theta$ and $\mathbf e$:

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

For a rotation of joint $A$ around **one axis** (one DoF), the linear change of the end-effector $\mathbf e$ along that axis, $d\mathbf e$, equals the change induced by joint $A$’s rotation on that axis ($\mathbf \omega \mathbf rdt$).

$$
\frac{d\mathbf e}{dt} = 
\vert \mathbf \omega \vert {\frac{\mathbf \omega}{\vert \mathbf \omega \vert }} \times \mathbf r =
\frac{d \mathbf \theta}{dt} \mathbf a \times \mathbf r
$$

Here $\mathbf a= \frac{\mathbf \omega}{\vert \omega \vert}$ denotes the unit rotation vector.

Rearranging gives:

$${d\mathbf e \over d\mathbf \theta} = \mathbf a \times \mathbf r$$


Therefore, we obtain:

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

where $\mathbf r_i$ is the position vector of joint $i$.

Note that each $\mathbf J_i$ here corresponds to **one** DoF of a joint. Therefore, if all joints together have $M$ DoFs, then $\mathbf J$ is a $3 \times M$ matrix.

In implementation, because each $J_i$ contains only one DoF, suppose joint $A$ can rotate about $(x, z)$. Then its angular velocity is $\mathbf \omega_A = (\mathbf\omega_{Ax}, 0, \mathbf\omega_{Az})$. If $\mathbf \theta_i$ corresponds to the $z$ axis of $A$, then $\mathbf \omega_i$ is $\mathbf \omega_{A} \times (0, 0 ,1) = \mathbf\omega_{Az}$.

### 2.5 Inverse of Jacobian

To solve $d\mathbf \theta  = \mathbf J^{-1} d \mathbf e$, after obtaining $\mathbf J$ we would like to compute $\mathbf J^{-1}$. However, an inverse is only defined for square matrices, while our $\mathbf J$ is a $3 \times M$ matrix. So we cannot directly invert it; instead, we use the pseudo-inverse $\mathbf J^+$.

Consider the following steps:

$$ d\mathbf e = \mathbf J d\mathbf  \theta$$
$$ \mathbf J^T d\mathbf e = \mathbf J^T J d \mathbf \theta$$
$$ (\mathbf J^T \mathbf J)^{-1} \mathbf J^T d\mathbf e =(\mathbf J^T \mathbf J)^{-1}(\mathbf J^T \mathbf J)d \mathbf \theta$$
$$ (\mathbf J^T \mathbf J)^{-1} \mathbf J^T d\mathbf e =d \mathbf  \theta$$
$$ \mathbf J^+ d\mathbf e =d \mathbf \theta$$
$$ \mathbf J^+ = (\mathbf J^T \mathbf J)^{-1} \mathbf J^T $$


Using a property of SVD:

$$SVD(J)=UΣV^∗$$


Substituting and expanding $\mathbf J^+$:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"> <mtable columnalign="right left right left right left right left right left right left" rowspacing="3pt" columnspacing="0em 2em 0em 2em 0em 2em 0em 2em 0em 2em 0em" displaystyle="true"> <mtr> <mtd> <msup> <mrow class="MJX-TeXAtom-ORD"> <mi mathvariant="bold">J</mi> </mrow> <mo>+</mo> </msup> </mtd> <mtd> <mi></mi> <mo>=</mo> <mo stretchy="false">(</mo> <msup> <mrow class="MJX-TeXAtom-ORD"> <mi mathvariant="bold">J</mi> </mrow> <mo>&#x2217;<!-- ∗ --></mo> </msup> <mrow class="MJX-TeXAtom-ORD"> <mi mathvariant="bold">J</mi> </mrow> <msup> <mo stretchy="false">)</mo> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> </mrow> </msup> <msup> <mrow class="MJX-TeXAtom-ORD"> <mi mathvariant="bold">J</mi> </mrow> <mo>&#x2217;<!-- ∗ --></mo> </msup> </mtd> </mtr> <mtr> <mtd /> <mtd> <mi></mi> <mo>=</mo> <mo stretchy="false">(</mo> <mi>V</mi> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <msup> <mi>U</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> <mi>U</mi> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <msup> <mi>V</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> <msup> <mo stretchy="false">)</mo> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> </mrow> </msup> <mi>V</mi> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <msup> <mi>U</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> </mtd> </mtr> <mtr> <mtd /> <mtd> <mi></mi> <mo>=</mo> <mo stretchy="false">(</mo> <mi>V</mi> <msup> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <mn>2</mn> </msup> <msup> <mi>V</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> <msup> <mo stretchy="false">)</mo> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> </mrow> </msup> <mi>V</mi> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <msup> <mi>U</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> </mtd> </mtr> <mtr> <mtd /> <mtd> <mi></mi> <mo>=</mo> <mo stretchy="false">(</mo> <msup> <mi>V</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> <msup> <mo stretchy="false">)</mo> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> </mrow> </msup> <msup> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>2</mn> </mrow> </msup> <msup> <mi>V</mi> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> </mrow> </msup> <mi>V</mi> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <msup> <mi>U</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> </mtd> </mtr> <mtr> <mtd /> <mtd> <mi></mi> <mo>=</mo> <mi>V</mi> <msup> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>2</mn> </mrow> </msup> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <msup> <mi>U</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> </mtd> </mtr> <mtr> <mtd /> <mtd> <mi></mi> <mo>=</mo> <mi>V</mi> <msup> <mi mathvariant="normal">&#x03A3;<!-- Σ --></mi> <mrow class="MJX-TeXAtom-ORD"> <mo>&#x2212;<!-- − --></mo> <mn>1</mn> </mrow> </msup> <msup> <mi>U</mi> <mo>&#x2217;<!-- ∗ --></mo> </msup> </mtd> </mtr> </mtable></math>

### 2.6 IK Algorithm

After obtaining $\mathbf J^+$, we can estimate the previous step:

$$\mathbf \theta_{k} = \mathbf \theta_{k+1} - dt * \mathbf J^+(\mathbf \theta_{k+1}) * d\mathbf e$$

Pseudocode:

```python
while(ABS(current_position - target_position) > epsilon and
      iterator < max_iterator):
    
    1. compute Jacobian J
    2. compute Pseoduinverse Jacobain J^+
    3. compute change in joint DoFs: dθ = (J^+)de
    4. apply the changes to joints. move a small step α: θ += αdθ
    5. update current_position to close target_position
```

## 3. Discussion

### 3.1 Effect of Parameters

From the equation:
$$\mathbf \theta_{k} = \mathbf \theta_{k+1} - dt * \mathbf J^+(\mathbf \theta_{k+1}) * d\mathbf e$$
the step size $dt$ affects convergence speed. If each step is too small, you need many iterations; but if each step is too large, you might overshoot the target and the system may fail to converge.

To prevent endless iterations due to non-convergence, we set an error threshold $\epsilon$ and stop when the error is within tolerance. We also set a maximum number of iterations.

In fact, IK does not always have a solution. If the target position $t$ exceeds the maximum reachable length of the joints, or is constrained by the available DoFs, it may be impossible to reach $t$. For example, no matter how you stretch your arm, you cannot touch the ceiling; and your arm also cannot rotate outward indefinitely. These are all unsolvable cases. In practice, you should also consider adjusting $t$ to avoid crashing when the problem has no solution.

### 3.2 Effect of the Number of Movable Joints

Given a character in the initial pose shown below, the target position is the purple dot.

![](https://i.imgur.com/Q5zjvcD.png)

Case 1: Allow the arm to move; use IK to solve the pose:

![](https://i.imgur.com/HTNfgHB.png)
![](https://i.imgur.com/j21wFj9.png)

Case 2: Allow the arm and upper body to move; use IK to solve the pose:

![](https://i.imgur.com/Lo8DWA3.png)

Case 3: Allow everything above the hips to move; use IK to solve the pose:

![](https://i.imgur.com/KLKFwme.png)

## 4. Conclusion

With inverse kinematics (IK), we can start from a character’s current pose and back-solve how to move to reach the pose we want. In computer animation, IK lets us specify key poses (keyframes) and have the character reach them. In robotics, IK enables robotic arms to reach target working positions reliably.

There are many IK methods. This post only introduced the Jacobian Pseudoinverse method, whose characteristic is that it updates all joints at once, which can look more realistic. However, real human motion often has more natural preferences, and then you would want algorithms that incorporate such priors. Depending on your needs, you can choose different algorithms—this part is left to the reader to explore.

## 5. References
- Samuel R. Buss. 2009. Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods. [http://graphics.cs.cmu.edu/nsp/course/15-464/Spring11/handouts/iksurvey.pdf](http://graphics.cs.cmu.edu/nsp/course/15-464/Spring11/handouts/iksurvey.pdf)

- [@shi-yan](https://epiphany.pub/@shi-yan). 2019. Inverse Kinematics Explained Interactively. [https://epiphany.pub/@shi-yan/inverse-kinematics-explained-interactively](https://epiphany.pub/@shi-yan/inverse-kinematics-explained-interactively)


- Rickard Nilsson. Inverse Kinematics Explained Interactively. 2009. [https://www.diva-portal.org/smash/get/diva2:1018821/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:1018821/FULLTEXT01.pdf)

- [Nancy Pollard](http://www.cs.cmu.edu/~nsp). 2003. CMU: Technical Animation. [http://www.cs.cmu.edu/~15464-s13/assignments/](http://www.cs.cmu.edu/~15464-s13/)
