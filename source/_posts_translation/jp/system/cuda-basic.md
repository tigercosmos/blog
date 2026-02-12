---
title: "CUDA 開発環境の設定と簡単なプログラム例"
date: 2020-12-10 07:00:00
tags: [cuda, gpu, parallel programming]
des: "本記事では CUDA の開発環境の設定を簡単に紹介し、シンプルな CUDA プログラム例を解説します。"
lang: jp
translation_key: cuda-basic
---

## 1. はじめに

GPU はもともとグラフィックス処理に使われていましたが、その後、汎用 GPU 計算（General Purpose GPU, GPGPU）技術の登場により、一般計算にも広く使われるようになりました。GPU アーキテクチャは非常に多くのスレッドを持つため、プログラム実行時に大量の数値演算を GPU に任せ、ロジック部分を CPU に任せる、といった使い方が一般的です。NVIDIA は GPGPU の概念を最初に提唱した企業であり、[CUDA](https://zh.wikipedia.org/wiki/CUDA) 技術を提案しました。これにより開発者は、C に近い文法で GPU を駆動して計算を行えます。

## 2. CUDA のインストール

CUDA には多くのバージョンがあります。単に `apt install cuda` を実行しても、必ずしも必要なバージョンが入るとは限りません。

CUDA のプログラム開発だけでなく、PyTorch や TensorFlow のために CUDA を入れたい場合もあります。その場合は、NVIDIA の公式サイトで必要なバージョンを選ぶのがよいです。ただし [CUDA Toolkit Archive](https://developer.nvidia.com/CUDA-TOOLKIT-ARCHIVE) のページからバージョンを選ぶことを忘れないでください。そうしないと、NVIDIA のサイトは最新バージョンへ誘導してきます。

以下の手順に従ってインストールできます。

### 2.1 CUDA のインストール

ここでは CUDA 11.0 を例にします。まず Archive ページに行き、11.0 を選択し、自分の環境に合わせてオプションを選びます。

![CUDA 11.0 インストール例](https://user-images.githubusercontent.com/18013815/101561057-94c03400-39ff-11eb-9351-303e3cedc170.png)

たとえば上の例では x86 Ubuntu 20.04 を選択しており、runfile、deb local、deb network を選べます。

3 つの違いは、それぞれ「大きなインストーラ一式」「deb にまとめた大きなインストーラ」「ネットワークからダウンロードしてインストールする deb」です。deb かどうかの違いは、後からパッケージマネージャで管理できるかどうかに影響します。

私は通常 deb local を選びますが、好みに合わせて選んでください。

選択が終わると、親切にも長いコマンド列が提示されます。基本的にはその通りに実行すればインストールが完了します。

CUDA 11.0 Ubuntu 20.04 x86 のインストール手順：

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

`dpkg -i` で問題がある場合は、代わりに `apt ./xxx.deb` を使っても構いません。

### 2.2 cuDNN のインストール

ついでに cuDNN もインストールできます。cuDNN は CUDA の深層学習向けに最適化された関数ライブラリで、PyTorch を使う場合は必要になります。OS に合わせてリンクを調整してください。たとえば `ubuntu18.04` は Ubuntu 18 を意味します。

```shell
$ sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
$ sudo apt install libcudnn7
```

### 2.3 OpenCL のインストール

CUDA を入れたなら、基本的に OpenCL も一緒に入れておくとよいです。NVIDIA GPU の OpenCL は CUDA を利用するため、これで後から OpenCL を実行する際にもそのまま使えます。

```shell
$ sudo apt install -y nvidia-opencl-dev
$ sudo apt install opencl-headers
```

### 2.4 システム設定

まず **再起動** してください。これはとても重要です！！！！

インストールが完了したらパス設定も忘れずに行いましょう。`~/.bashrc` に次を追加します。

```shell
export PATH=$PATH:/usr/local/cuda/bin
export CUDADIR=/usr/local/cuda
```

### 2.5 インストール確認

インストールが正しくできたか確認します。

CUDA が正常に入っていれば、次のコマンドはいずれも反応するはずです。

```shell
$ nvidia-smi  # Driver 
$ nvcc --version # CUDA
$ /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn # CuDNN
```

### 2.6 その他

上の手順でうまくいかなかった場合は、次の記事も参考になります：

- [Easy installation of Cuda Toolkit on Ubuntu 18.04](https://medium.com/@ayush.goc/easy-installation-of-cuda-toolkit-on-ubuntu-18-04-7931394c1233)
- [Tutorial: CUDA v10.2 + CUDNN v7.6.5 Installation @ Ubuntu 18.04](https://sh-tsang.medium.com/tutorial-cuda-v10-2-cudnn-v7-6-5-installation-ubuntu-18-04-3d24c157473f)
- [Installing CUDA 10.1 on Ubuntu 20.04](https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0)


## 3. CUDA プログラム例

CUDA を学ぶ最良の方法は、公式ドキュメントの [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) を読むことです。NVIDIA 自身の製品なので、やはり一次情報が一番です。また、Gerassimos による [Multicore and GPU Programming: An Integrated Approach](https://www.tenlong.com.tw/products/9787111557685?list_name=srh) という書籍も良く、入門に向いています。

ここでは簡単な例を示します：

`matadd.cu`：

```c
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 512
#define BLOCK_SIZE 16

// GPU 的 Kernel
__global__ void MatAdd(float *A, float *B, float *C)
{
    // 根據 CUDA 模型，算出當下 thread 對應的 x 與 y
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // 換算成線性的 index
    int idx = j * N + i;

    if (i < N && j < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    int i;

    // 宣告 Host 記憶體 (線性)
    h_A = (float *)malloc(N * N * sizeof(float));
    h_B = (float *)malloc(N * N * sizeof(float));
    h_C = (float *)malloc(N * N * sizeof(float));

    // 初始化 Host 的數值
    for (i = 0; i < (N * N); i++)
    {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
        h_C[i] = 0.0;
    }

    // 宣告 Device (GPU) 記憶體
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    // 將資料傳給 Device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(N / BLOCK_SIZE, N / BLOCK_SIZE);

    // 執行 MatAdd kernel
    MatAdd<<<numBlock, blockSize>>>(d_A, d_B, d_C);

    // 等待 GPU 所有 thread 完成
    cudaDeviceSynchronize();

    // 將 Device 的資料傳回給 Host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 驗證正確性
    for (i = 0; i < (N * N); i++)
    {
        if (h_C[i] != 3.0)
        {
            printf("Error:%f, idx:%d\n", h_C[i], i);
            break;
        }
    }

    printf("PASS\n");

    // free memory

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

GPU 計算では、最小の実行単位を Kernel と呼び、GPU 上の各スレッドが実行する関数に相当します。そのため `__global__ void MatAdd()` という関数を宣言しています。`__global__` は GPU 関数であることをコンパイラに伝える指定で、`MatAdd()` は A と B の行列を同じ index で足し合わせて C に書き込むだけです。

データは Host 側と Device 側に分かれます。Host は CPU、Device は GPU を指します。Host のメモリ確保は通常の `malloc` ですが、GPU のメモリ確保は `cudaMalloc` を使います。

計算を GPU に移したいので、Host 側のデータは `cudaMemcpy` で先に GPU に転送する必要があります。その後 `MatAdd<<<numBlock, blockSize>>>` を実行し、計算が終わったら再び `cudaMemcpy` で GPU のデータを CPU に戻して、Host 側で利用します。

Kernel 起動時の `MatAdd<<<numBlock, blockSize>>>` という記法は、GPU のスレッド階層に関係しています。

![GPU thread grid](https://user-images.githubusercontent.com/18013815/101697693-381e5100-3ab3-11eb-8ea7-d6698b9f8a31.png)

上図の通り、GPU には多くの Block があり、各 Block に多くの Thread があります。そのため Kernel を実行する際は、Block 数と Block 内の Thread 数を指定します。

大まかに理解できたら、コンパイルして実行します。

```shell
$ nvcc matadd.cu; ./a.out
PASS
```

実行中に別ウィンドウで `nvidia-smi` を叩けば、`matadd` の GPU 使用状況を確認できます。

また `nvprof` で CUDA の性能を見てみることもできます：

```shell
$ nvprof ./a.out
==27161== NVPROF is profiling process 27161, command: ./a.out
PASS
==27161== Profiling application: ./a.out
==27161== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   70.06%  263.96us         3  87.986us  87.581us  88.285us  [CUDA memcpy HtoD]
                   21.55%  81.181us         1  81.181us  81.181us  81.181us  [CUDA memcpy DtoH]
                    8.40%  31.647us         1  31.647us  31.647us  31.647us  MatAdd(float*, float*, float*)
      API calls:   98.65%  133.57ms         3  44.524ms  3.6540us  133.50ms  cudaMalloc
                    0.78%  1.0564ms         4  264.10us  169.89us  383.89us  cudaMemcpy
                    0.19%  256.03us         3  85.342us  20.249us  148.41us  cudaFree
                    0.14%  187.38us         1  187.38us  187.38us  187.38us  cuDeviceTotalMem
                    0.13%  169.85us        97  1.7510us     193ns  69.776us  cuDeviceGetAttribute
                    0.07%  98.576us         1  98.576us  98.576us  98.576us  cudaDeviceSynchronize
                    0.02%  29.226us         1  29.226us  29.226us  29.226us  cudaLaunchKernel
                    0.02%  26.508us         1  26.508us  26.508us  26.508us  cuDeviceGetName
                    0.00%  4.0950us         1  4.0950us  4.0950us  4.0950us  cuDeviceGetPCIBusId
                    0.00%  1.3720us         3     457ns     266ns     811ns  cuDeviceGetCount
                    0.00%  1.0800us         2     540ns     192ns     888ns  cuDeviceGet
                    0.00%     365ns         1     365ns     365ns     365ns  cuDeviceGetUuid
```

上の結果を見ると、大半の時間がデータ転送に費やされていることが分かります。これは妥当で、計算自体がとても単純なため、オーバーヘッドはメモリ転送に支配されます。

## 4. 結論

本記事では CUDA の開発環境の設定を簡単に紹介し、シンプルな CUDA プログラム例を解説しました。GPGPU は大量計算において計算を効率的に高速化でき、数値計算ライブラリや機械学習・深層学習フレームワークの多くも、GPGPU を用いて計算を加速しています。
