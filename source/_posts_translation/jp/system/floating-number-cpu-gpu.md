---
title: "CPU と GPU における浮動小数点計算の違い"
date: 2020-12-05 00:06:40
tags: [cpu, gpu, floating-point number, ‎IEEE 754]
des: "CPU と GPU は浮動小数点数（floating-point number）の計算結果が異なる場合があります。本記事では実例を示し、その理由を説明します。"
lang: jp
translation_key: floating-number-cpu-gpu
---

## イントロダクション

CPU と GPU は、浮動小数点数（floating-point number）を計算する際に結果が異なる場合があります。

以前から話としては聞いていたのですが、正直なところ実感したことはありませんでした。ところが最近、[陽明交大の平行プログラミング](https://nycu-sslab.github.io/PP-f20/) の課題を設計していたときに、痛い目を見て深く学ぶことになりました。やはり「踏んでみて」初めて本当に理解できますね。

状況はこうです。学生に CUDA を使って [Mandelbrot set](https://zh.wikipedia.org/wiki/%E6%9B%BC%E5%BE%B7%E5%8D%9A%E9%9B%86%E5%90%88)（マンデルブロ集合）を計算してもらおうとしました。これは複素平面上にフラクタルを構成する点集合で、反復計算によって座標の値を求められます。

![Mandelbrot set](https://user-images.githubusercontent.com/18013815/101222730-6112a080-36c5-11eb-8c17-631e7c03d62e.png)

以前の課題では、CPU 版として `std::thread` を使って高速化してもらうものがあり、正しさの検証は「単一スレッドの参照実装」と「マルチスレッド版」を比較する方法を取りました。そこで今回も同じ枠組みを使い、GPU 版の結果を CPU 版と比較しようと考えました。

Mandelbrot set のある点の値は、次の式で計算できます。

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

ここで `c_re` は複素平面の x、`c_im` は複素平面の y、`max` は反復回数です。戻り値の `i` が反復の結果になります。

ところが GPU 版を書き終えて CPU 版と突き合わせると、どうしても一致しませんでした。かなり時間をかけて検証した結果、CPU と GPU で使っている `c_re` と `c_im` は同一であること、アルゴリズムもまったく同じであることは確認できました。それでも一部のケースで結果が違ってしまいます。正直かなり絶望しました。まさか自分が「CPU と GPU が異なる結果を出すケース」に当たるなんて……と思ったのですが、実際にはこの状況は私が最初に想像していたほどレアではありませんでした。

## GPU と CPU の浮動小数点演算差分の例

以下は、実際に差分が出ることを示す CUDA のサンプルプログラムです。`diverge_cpu` と `diverge_gpu` は同一の Mandelbrot アルゴリズムで、CPU と GPU の両方が同じ `INPUT_X` と `INPUT_Y` を使って実行します。

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

コードが完全に分からなくても大丈夫です。重要なのは、このプログラムでは入力も、CPU と GPU の式も、すべて同一であるという点です。

実行してみます：

```shell
$ nvcc test.cu; ./a.out
GPU vs CPU: 39, 40
```

すると、`INPUT_X` と `INPUT_Y` が `-0.0612500` と `-0.9916667` の場合、CPU と GPU で異なる答えが出ることが分かります。値を少し変えてみると、たとえば `INPUT_Y` を `-0.9916669` にすると、結果は 35 と 34 になります。

つまり、この小さなプログラムは「浮動小数点演算には差が出ることがある」ことを実際に示しています。

## GPU と CPU の浮動小数点演算が違う理由

現在、浮動小数点数の標準は [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) です。GPU も CPU もこれをサポートしているので、理屈の上では同じ結果になりそうですが、現実にはそうはなりません。

では問題はどこにあるのでしょうか。標準に従っているなら、CPU が正しいのか、GPU が正しいのか、それとも両方正しいのでしょうか。

CPU・GPU そしてコンパイラが IEEE 754 に厳密に従っていたとしても、差分が生じることがあります。[Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://developer.nvidia.com/sites/default/files/akamai/cuda/files/NVIDIA-CUDA-Floating-Point.pdf) では次のように述べられています：

> Even in the strict world of IEEE 754 operations, minor details such as organization of parentheses or thread counts can affect the final result. Take this into account when doing comparisons between implementations.

つまり、IEEE 754 の世界でも、括弧の付け方やスレッド数といった細部によって最終結果が変わり得るため、実装間の比較を行う際には特に注意が必要だ、ということです。

もっとも単純な例は Round-off Error（四捨五入誤差）です：

```
x = (x * y) * z; // 不等於  x *= y * z;
z = (x - y) + y ; // 不等於 z = x;
z = x + x * y; // 不等於 z = x * (1.0 + y);
y = x / 5.0; // 不等於 y = x * 0.2;
```

私の推測では、GPU のハードウェアやコンパイラが Round-off の扱いを異なる形で行い（しかも仕様に準拠した範囲で）、その結果として上の小さなプログラムに差が出たのだと思います。実際、[NVIDIA の公式ドキュメント](https://docs.nvidia.com/cuda/floating-point/index.html#cuda-and-floating-point) にも次の記述があります：

> The consequence is that different math libraries cannot be expected to compute exactly the same result for a given input. This applies to GPU programming as well. Functions compiled for the GPU will use the NVIDIA CUDA math library implementation while functions compiled for the CPU will use the host compiler math library implementation (e.g., glibc on Linux). Because these implementations are independent and neither is guaranteed to be correctly rounded, the results will often differ slightly.

したがって、GPU と CPU のどちらの答えも「正しい」と言えます。ただし結果には差が出る可能性があります。有限ビットで数を表している以上、仕方がありません。また、この [discussion thread](https://github.com/HPCE/hpce-2016-cw5/issues/10) も非常に参考になります。

## 結論

CPU と GPU は浮動小数点計算で結果が異なる場合があります。これは、CPU と GPU の間でデータをやり取りする際に特に注意が必要であることを意味します。たとえば異種計算（heterogeneous computing）で同じカーネルが CPU と GPU の両方で走る場合、両者で出力が一致しない可能性があります。

では課題は最終的にどうしたかというと、後から分かったのは「GPU 版が間違っていた」のではなく、「CPU と比較するのが良くなかった」という点でした。そこで正しさの比較対象を GPU の参照出力に変更し、問題は解決しました。
