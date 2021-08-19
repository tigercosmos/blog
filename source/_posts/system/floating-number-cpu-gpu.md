---
title: CPU 與 GPU 計算浮點數的差異
date: 2020-12-05 00:06:40
tags: [cpu, gpu, floating-point number, ‎IEEE 754]
des: "CPU 與 GPU 在計算浮點數 (floating-point number) 時會有差異，本文實際舉例並解釋原因。"
---

## 簡介

CPU 與 GPU 在計算浮點數 (floating-point number) 時會有差異。

雖然之前就略有耳聞，但一直沒體會過，近日我在設計[交大平行程式](https://nycu-sslab.github.io/PP-f20/)課程的作業的時候，就被深深教訓了一番，果然還是要踩過雷才會真的懂。

問題是這樣的，我打算讓學生用 CUDA 去算 [Mandelbrot set](https://zh.wikipedia.org/wiki/%E6%9B%BC%E5%BE%B7%E5%8D%9A%E9%9B%86%E5%90%88) (曼德博集合)。這是一種在複數平面上組成碎形的點的集，可以透過不斷迭代來取得平面座標的數值。

![Mandelbrot set](https://user-images.githubusercontent.com/18013815/101222730-6112a080-36c5-11eb-8c17-631e7c03d62e.png)

之前作業已經有出過 CPU 版本要學生用 std::thread 去加速，那時候比對正確性方式就是提供一個單執行緒版本去和多執行緒版本做比對，因此打算我沿用之前的框架，讓這次 GPU 跑的版本跟用 CPU 跑版本做比對。

要算出 Mandelbrot Set 上的某個位置可以用下面的算式：

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

其中 `c_re` 就是複數平面的 x，`c_im` 就是複數平面的 y，`max` 就是要迭代幾次。`i` 就是迭代完後的結果。

我自己寫完 GPU 版本答案後跟 CPU 版本一直對不起來，花了不少時間做驗證，我已經確定 CPU 和 GPU 所使用的 `c_re` 和 `c_im` 都一樣，兩者使用的算法也長一模一樣，但某些時候出來的結果就會不一樣。此時我心中感到很絕望，難道我就這麼衰，遇上 CPU 和 GPU 會算出不一樣的情況？而事實證明，這種情況不像我一開始以為的極少發生。

## GPU 與 CPU 浮點數運算差異示例

以下為 CUDA 範例程式碼，實際展示會有差異。其中 `diverge_cpu` 和 `diverge_gpu` 長得一模一樣，是 Mandelbrot 的演算法，在這邊我讓 CPU 和 GPU 在執行他們的時候都會使用一樣的 `INPUT_X` 和 `INPUT_Y`。

`test.cu`：

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

程式碼如果沒看很懂沒關係，只要知道說這個程式所有環節都一樣，不管是 input 或是 CPU 與 GPU 的算式都長一樣。

我們可以執行程式：

```shell
$ nvcc test.cu; ./a.out
GPU vs CPU: 39, 40
```

然後我們就會發現當 `INPUT_X` 和 `INPUT_Y` 是 `-0.0612500` 和 `-0.9916667` 情況下，CPU 和 GPU 會得到不一樣的答案。這個數值你也可以稍微改一下，像是把 `INPUT_Y` 改成 `-0.9916669` 後答案會變成 35 跟 34。

總之這個小程式證明了浮點數運算確實會有差異。

## GPU 與 CPU 浮點數運算的差別

現在浮點數的通行標準是 [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754)，這個不管是 GPU 或是 CPU 都支援，理論上應該要一樣了？但結果卻不然。

所以問題出在哪，如果都照著標準，到底誰是對的，CPU？GPU？或者都對。

儘管 CPU、GPU 甚至編譯器都嚴格遵照 IEEE 754，仍然會存在差異。在 [Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://developer.nvidia.com/sites/default/files/akamai/cuda/files/NVIDIA-CUDA-Floating-Point.pdf) 論文中便有提出：

> Even in the strict world of IEEE 754 operations, minor details such as organization of parentheses or thread counts can affect the final result. Take this into account when doing comparisons between implementations.
>
> 即使嚴格遵照 IEEE 754 的運算，小細節像是括號或執行緒計數都有機會影響到最終結果。當要對不同實作做比對時，要特別小心。

最簡單的例子就是 Round-off Error (四捨五入誤差)：

```
x = (x * y) * z; // 不等於  x *= y * z;
z = (x - y) + y ; // 不等於 z = x;
z = x + x * y; // 不等於 z = x * (1.0 + y);
y = x / 5.0; // 不等於 y = x * 0.2;
```

我的猜測是 GPU 硬體或編譯器在 Round-off 的處理上有不一樣，並且是在符合規範的情況下，進而導致上面的小程式會有差異。事實上 [Nvidai 官方文件](https://docs.nvidia.com/cuda/floating-point/index.html#cuda-and-floating-point)也有提到：

> The consequence is that different math libraries cannot be expected to compute exactly the same result for a given input. This applies to GPU programming as well. Functions compiled for the GPU will use the NVIDIA CUDA math library implementation while functions compiled for the CPU will use the host compiler math library implementation (e.g., glibc on Linux). Because these implementations are independent and neither is guaranteed to be correctly rounded, the results will often differ slightly.

所以 GPU 和 CPU 的答案都可以說是對的，只是結果仍然有差，沒辦法誰叫我們的電腦只能用有限位元來表示數字呢。此外這邊有個很有幫助的[討論串](https://github.com/HPCE/hpce-2016-cw5/issues/10)非常值得參考。

## 結論

CPU 與 GPU 在計算浮點數時會有差異。這代表我們在處理 CPU 與 GPU 相互資料的時候要特別小心，像是異質運算 kernel 同時會在 CPU 跑和在 GPU 跑時，數據跑出來的結果就有機會不一樣。

那最後作業怎麼處理呢，後來我發現不是我 GPU 版本有寫錯，而是不能與 CPU 去比較，就把比對正確的部分改成與 GPU 的參考答案相比，問題就解決了。
