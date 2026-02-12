---
title: "Differences Between CPU and GPU Floating-Point Computation"
date: 2020-12-05 00:06:40
tags: [cpu, gpu, floating-point number, ‎IEEE 754]
des: "CPU and GPU can produce different results when computing floating-point numbers. This post gives a concrete example and explains why."
lang: en
translation_key: floating-number-cpu-gpu
---

## Introduction

CPU and GPU can produce different results when computing floating-point numbers.

I had heard about this before, but I never truly felt it. Recently, while designing homework for the [NYCU Parallel Programming](https://nycu-sslab.github.io/PP-f20/) course, I got a painful lesson—turns out you only really understand after stepping on the landmine yourself.

Here is the situation. I wanted students to compute the [Mandelbrot set](https://zh.wikipedia.org/wiki/%E6%9B%BC%E5%BE%B7%E5%8D%9A%E9%9B%86%E5%90%88) using CUDA. It is a set of points that forms a fractal on the complex plane, and you can compute the value at a coordinate by iterating a recurrence.

![Mandelbrot set](https://user-images.githubusercontent.com/18013815/101222730-6112a080-36c5-11eb-8c17-631e7c03d62e.png)

Previously, the homework already had a CPU version where students used `std::thread` to speed it up. At that time, the way we verified correctness was to provide a single-thread reference implementation and compare the multi-threaded result against it. So I planned to reuse the same framework: compare the GPU version against the CPU version.

To compute a value at a position in the Mandelbrot set, you can use the following function:

```c++
int diverge_cpu(float c_re, float c_im, int max)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < max; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}
```

Here `c_re` is the x coordinate on the complex plane, `c_im` is the y coordinate, and `max` is the number of iterations. The return value `i` is the iteration count (the result).

After I finished writing the GPU version, I could not get it to match the CPU version. I spent quite some time validating it. I was sure that the `c_re` and `c_im` used by both CPU and GPU were identical, and the algorithm was literally the same. But in some cases, the results still differed. I felt desperate—was I really that unlucky to hit a case where CPU and GPU compute different results? And as it turned out, this situation is not nearly as rare as I initially thought.

## A Concrete Example: CPU vs GPU Floating-Point Differences

Below is a CUDA sample program that demonstrates the difference. `diverge_cpu` and `diverge_gpu` are identical implementations of the Mandelbrot algorithm. In this example, both CPU and GPU use the same `INPUT_X` and `INPUT_Y`.

`test.cu`:

```c
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_X -0.0612500f
#define INPUT_Y -0.9916667f

int diverge_cpu(float c_re, float c_im, int max)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < max; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__device__ int diverge_gpu(float c_re, float c_im, int max)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < max; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void kernel(int *c, int n)
{
  // 取得 global ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // 通通設一樣的值
  c[id] = diverge_gpu(INPUT_X, INPUT_Y, 256);
}

int main(int argc, char *argv[])
{
  int n = 100;
  int *h_c;
  int *d_c;
  h_c = (int *)malloc(n * sizeof(int));
  cudaMalloc(&d_c, n * sizeof(int));

  int blockSize = 1024;
  int gridSize = 1;

  // 這邊是算 GPU 的部分
  kernel<<<gridSize, blockSize>>>(d_c, n);
  cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

  // 這邊是算 CPU 的部分
  int cpu_result = diverge_cpu(INPUT_X, INPUT_Y, 256);

  printf("GPU vs CPU: %d, %d\n", h_c[0], cpu_result);

  cudaFree(d_c);
  free(h_c);

  return 0;
}
```

If you do not fully understand the code, that is fine. The key point is: every part of this program is the same—the input is the same, and the CPU and GPU formulas are the same.

We can run it:

```shell
$ nvcc test.cu; ./a.out
GPU vs CPU: 39, 40
```

Then we discover that when `INPUT_X` and `INPUT_Y` are `-0.0612500` and `-0.9916667`, the CPU and GPU produce different answers. You can tweak the value a bit—for example, change `INPUT_Y` to `-0.9916669`—and then the results become 35 and 34.

In short, this small program proves that floating-point computation can indeed differ.

## Why CPU and GPU Floating-Point Results Differ

The widely adopted standard for floating-point numbers today is [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754). Both GPU and CPU support it, so in theory the results “should” match—but they do not.

So where is the problem? If everything follows the standard, who is correct—CPU, GPU, or both?

Even if the CPU, GPU, and even the compiler strictly follow IEEE 754, differences can still exist. The paper [Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://developer.nvidia.com/sites/default/files/akamai/cuda/files/NVIDIA-CUDA-Floating-Point.pdf) points out:

> Even in the strict world of IEEE 754 operations, minor details such as organization of parentheses or thread counts can affect the final result. Take this into account when doing comparisons between implementations.

In other words: even under IEEE 754, small details—such as how parentheses are arranged or how many threads are used—can affect the final result, so you need to be extra careful when comparing different implementations.

The simplest example is round-off error:

```
x = (x * y) * z; // 不等於  x *= y * z;
z = (x - y) + y ; // 不等於 z = x;
z = x + x * y; // 不等於 z = x * (1.0 + y);
y = x / 5.0; // 不等於 y = x * 0.2;
```

My guess is that the GPU hardware or compiler handles rounding differently (while still conforming to the specification), which leads to differences in the small program above. In fact, the [NVIDIA official documentation](https://docs.nvidia.com/cuda/floating-point/index.html#cuda-and-floating-point) also mentions:

> The consequence is that different math libraries cannot be expected to compute exactly the same result for a given input. This applies to GPU programming as well. Functions compiled for the GPU will use the NVIDIA CUDA math library implementation while functions compiled for the CPU will use the host compiler math library implementation (e.g., glibc on Linux). Because these implementations are independent and neither is guaranteed to be correctly rounded, the results will often differ slightly.

So you can say both the GPU and CPU results are “correct”—they are just slightly different. After all, we can only represent numbers with a finite number of bits. Also, this [discussion thread](https://github.com/HPCE/hpce-2016-cw5/issues/10) is very helpful and worth reading.

## Conclusion

CPU and GPU can produce different floating-point results. This means we must be especially careful when exchanging data between CPU and GPU. For example, for heterogeneous computing where kernels can run on both CPU and GPU, it is possible to get different outputs from the two sides.

So how did I handle the homework in the end? I later confirmed that my GPU version was not wrong—it was the CPU comparison strategy that was wrong. So I changed the correctness check to compare against a GPU reference output instead. Then the problem was solved.
