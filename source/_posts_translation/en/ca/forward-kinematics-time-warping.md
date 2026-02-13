---
title: "Forward Kinematics and Time Warping in Computer Animation"
date: 2020-05-23 00:00:00
tags: [computer animation, 電腦動畫, 正向動力法, 時間扭曲法, forward kinematics, time warping]
des: "This post explains the principles and implementation of forward kinematics and time warping in computer animation. Forward kinematics is widely used in robotics, games, and animation: it computes global motion by recursively traversing all joints in a skeleton given per-joint relative translations and rotations. Time warping is a keyframe-based technique used in animation: when you want a specific frame to be rendered at a specific time, you adjust the playback speed before and after that keyframe, effectively speeding up or slowing down parts of the motion."
lang: en
translation_key: forward-kinematics-time-warping
---

## Introduction

This post introduces the principles and implementation of forward kinematics and time warping in computer animation.

![](https://i.imgur.com/ZUqBiQU.gif)

Forward kinematics is widely used in robotics, computer games, and computer animation. Conceptually, you specify the relative translation and rotation for each joint in a skeleton, and then compute the overall motion by recursively traversing all joints.

Time warping is a keyframe-based technique in computer animation. When you want a particular frame to be drawn at a specific time, you adjust the playback speed before and after the keyframe, which means you must apply acceleration or deceleration to the motion.

## Principles and Implementation

### Skeleton

To describe human motion in computer animation, we use a skeleton (skeleton) to drive the character. In this post, I use the Acclaim ASF skeleton format, which declares each bone’s index, length, direction, neutral-pose angles, and degrees of freedom.

Here is an ASF example:

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

After reading the skeleton, it looks like this:

![](https://i.imgur.com/tAcVx5n.png)

To make the neutral pose actually express motion, we need an Acclaim AMC motion file, which describes the motion information for each bone. Aside from the root node (which contains the transform relative to world coordinates), every other bone is only a relative transform. An AMC file contains the per-frame transforms for every bone over a complete motion sequence.

Here is an AMC example. At the beginning, you have the frame index, followed by per-bone information for that frame:

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

After reading the motion file, it looks like this:

![](https://i.imgur.com/Vt5Jugt.png)

For more information, you can refer to the [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/info.php).

### Forward Kinematics

You can think of forward kinematics as a chain of transformations: compute the change of the first joint, then use that change to compute the change of the second joint, and so on.

#### Concept

The idea is illustrated by the following figures (source: alanzucconi.com):

**Step 1**:
![](https://i.imgur.com/kDm7vRg.png)

P0 is the root of the skeleton. P1 and P2 are joints in the skeleton. Each joint has three degrees of freedom and its own local transform.

**Step 2**:
![](https://i.imgur.com/LrlC24S.png)

Compute how the transform of P0 affects P1 and P2.

**Step 3**:
![](https://i.imgur.com/nxtjNKT.png)

Compute how the transform of P1 affects P2. At this point, the transform of P2 is the composition of the transforms from P0 and P1.

Following these steps, we can derive the final transform of every joint in the skeleton using forward kinematics.

Below is the result of running forward kinematics on a human skeleton (ASF) with a running motion described by an AMC file:

![](https://i.imgur.com/a8zYv1l.gif)

#### Math

The overall transform of the skeleton consists of (1) the skeleton’s transform relative to world coordinates and (2) the per-joint relative transforms within the skeleton. We first compute the final results of the relative transforms. Since the world transform is the same for every node, we can add the world transform to each joint at the end.

Now let’s discuss how to handle relative transforms.

Assume a skeleton is $\mathbf P_0 \to \mathbf P_1 \to \mathbf P_2$, where $\mathbf P_i$ denotes a joint position vector, and $\mathbf D_1 = P_0 \to P_1$ denotes the first link vector.

Then the relationship between $\mathbf P_0$ and $\mathbf P_1$ is:

$$ \mathbf P_1 = \mathbf P_0 + \mathbf D_1 $$

where $\mathbf D_1$ is the displacement vector of the first link.

However, we also need to apply the rotation at $P_0$. Taking $\mathbf P_0$ as the pivot, we rotate $\mathbf D_1$ by $\alpha_0$ degrees:

$$ \mathbf P_1 = \mathbf P_0 + rotate(\mathbf D_1, \mathbf P_0, \alpha_0) $$

By analogy, we can obtain the general form for $\mathbf P_i$:

$$ \mathbf P_i = \mathbf P_{i-1} + rotate(\mathbf D_i, \mathbf P_{i-1}, \sum_{k=0}^{k-1}{\alpha_k}) $$

### Time Warping

Time warping allows us to adjust an animation based on keyframes. For example, for a punching motion, you might want the action that originally occurred at frame 150 to be delayed until frame 160, or moved earlier to frame 140.

![](https://i.imgur.com/ZYSQXwo.png)

To achieve this effect, we must adjust the motion at every frame. Suppose the new animation delays the action at original frame 150 so that it appears at new frame 160. The concrete steps to warp the timing of a skeleton animation are as follows:

1. Compute where each new frame falls in the timeline of the old animation. In this example, because new frames 0 to 160 correspond to old frames 0 to 150, each frame in the new animation between 0 and 160 corresponds to every $150/160$ frame in the old animation.

    The correspondence looks like:
    ```
    New  0  1    2     ....  160
    Old  0  0.94 1.92  ....  150
    ```
    
2. Interpolate the new animation frames from the old animation frames. For example, the 2nd frame in the new animation corresponds to the 1.92nd frame in the old animation, so we interpolate between the 1st and 2nd frames of the old animation to obtain frame 1.92.

3. For translation, you can use linear interpolation. For rotation, you should convert to quaternions and use SLERP for more accurate interpolation. (Of course, you can also linearly interpolate rotations, but it will be very inaccurate.)

Below is the result of implementing forward kinematics with time warping:

![](https://i.imgur.com/Gv0nk1Z.gif)

Yellow indicates the original frames, and blue indicates the frames after time warping.

It is worth mentioning that conversion in math libraries can be complex, and in practice it is easy to make mistakes due to unit conversion (e.g., degrees vs radians). I also spent a long time figuring out quaternion conversions in [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) (C++). In the end, SLERP looked like this:

```c++
auto rot1 = ComputeRotMatXyz(ToRadian(angular_vector1);
auto rot2 = ComputeRotMatXyz(ToRadian(angular_vector2));
auto q1 = Quaternion_t(rot1);
auto q2 = Quaternion_t(rot2);
auto new_q = Slerp(q1, q2, ratio);
auto new_angular_vec = ToDegree(ComputeEulerAngleXyz(ComputeRotMat(new_q)));
```

## Conclusion

With forward kinematics, we can compute the motion of a skeleton using the relative transforms of each joint. With time warping, we can realize keyframe timing effects and change how the animation is played back. Both techniques are fundamental and important for computer animation.

