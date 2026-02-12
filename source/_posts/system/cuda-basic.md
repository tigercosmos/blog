---
title: "CUDA 開發環境設定與簡易程式範例"
date: 2020-12-10 07:00:00
tags: [cuda, gpu, parallel programming]
des: "本文簡單介紹 CUDA 開發環境設置，並解說簡單的 CUDA 程式範例。"
lang: zh
translation_key: cuda-basic
---

## 1. 簡介

GPU 以前是用來算圖形介面，後來有了通用 GPU 運算 (General Purpose GPU, GPGPU) 技術的誕生，因為 GPU 架構擁有非常多執行緒，在跑程式的時候可以將大量的數值運算交給 GPU，而邏輯的部分交給 CPU，這也是常見的 GPGPU 使用方式。其中 Nvidia 便是第一個提出 GPGPU 概念的公司，其公司提出 [CUDA](https://zh.wikipedia.org/wiki/CUDA) 技術，讓開發者可以用一般寫 C 的語法去驅動 GPU 來做運算。

## 2. CUDA 安裝

CUDA 有很多個版本，如果直接用 `apt install cuda` 不一定安裝到自己需要的版本。

除了一般的 CUDA 程式，我們可能還會想要給 Pytorch 或 Tensorflow 使用。比較好的做法是去官方網站找自己需要的版本，不過要記住要去 [CUDA Toolkit Archive](https://developer.nvidia.com/CUDA-TOOLKIT-ARCHIVE) 的頁面選版本，不然 Nvidia 官網會直接導引到最新的版本。

大家可以照著我下面指示安裝。

### 2.1 CUDA 安裝

以下以 CUDA 11.0 版本為範例。首先去 Archive 頁面，選 11.0 點進去，然後照個你的需求把選項勾一勾。

![cuda 11.0 安裝示範](https://user-images.githubusercontent.com/18013815/101561057-94c03400-39ff-11eb-9351-303e3cedc170.png)

例如上面就是選 x86 Ubuntu 20.04 版本，然後你可以選 runfile、deb local 或 deb network。

三種差異分別是一大包安裝檔、deb 包裝好的一大包安裝檔、完全靠網路下載的 deb 安裝檔。有沒有 deb 差異在於之後可不可以用套件管理員去管理。

通常我會選 deb local，你也可以用其他的，端看個人習慣。

選完之後他很貼心，就會給你一大串指令，基本上照著打就安裝完了。

Cuda 11.0 Ubuntu 20.04 x86 安裝流程：

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

其中 `dpkg -i` 如果有問題的話，也可以直接用 `apt ./xxx.deb` 來代替。

### 2.2 CuDNN 安裝

我們可以順便裝一下 CuDNN，這是 CUDA 針對 Deep Learning 優化過的函數庫，假設有使用 Pytorch 就會用到。記得根據你的作業系統改一下連結，像是 `ubuntu18.04` 是指 Ubuntu 18。

```shell
$ sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
$ sudo apt install libcudnn7
```

### 2.3 OpenCL 安裝

如果裝了 CUDA 幾本上 OpenCL 也可以順便裝，因為 Nvidia GPU 使用 OpenCL 也是吃 CUDA，這樣之後要跑 OpenCL 就可以直接用。

```shell
$ sudo apt install -y nvidia-opencl-dev
$ sudo apt install opencl-headers
```

### 2.4 系統設定

然後首先要**重新開機**，這非常重要！！！！

順利裝完之後記得要把路徑設定好，在 `~/.bashrc` 裡面加入

```shell
export PATH=$PATH:/usr/local/cuda/bin
export CUDADIR=/usr/local/cuda
```

### 2.5 檢查安裝

然後我們就可以檢查一下有沒有裝好。

CUDA 有裝好的話，以下指令應該都有反應。

```shell
$ nvidia-smi  # Driver 
$ nvcc --version # CUDA
$ /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn # CuDNN
```

### 2.6 其他

如果上面安裝方法沒用的話，也可以參考這幾篇有用的文章：

- [Easy installation of Cuda Toolkit on Ubuntu 18.04](https://medium.com/@ayush.goc/easy-installation-of-cuda-toolkit-on-ubuntu-18-04-7931394c1233)
- [Tutorial: CUDA v10.2 + CUDNN v7.6.5 Installation @ Ubuntu 18.04](https://sh-tsang.medium.com/tutorial-cuda-v10-2-cudnn-v7-6-5-installation-ubuntu-18-04-3d24c157473f)
- [Installing CUDA 10.1 on Ubuntu 20.04](https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0)


## 3. CUDA 程式範例

想學好 CUDA 最好的辦法是看官方教學文件 [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)，畢竟是 Nvidia 自己家的產品。Gerassimos 寫的 [Multicore and GPU Programming: An Integrated Approach](https://www.tenlong.com.tw/products/9787111557685?list_name=srh) 這本書也挺好的，很適合入門。

這邊提供一個簡單的範例：

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

在 GPU 運算中，最小執行單位稱作 Kernel，等同 GPU 上的每個 Thread 要執行的函數。所以可以看到我宣告 `__global__ void MatAdd()` 函數，`__global__` 告訴編譯器這是 GPU 的函數，`MatAdd()` 裡面顧名思義是將 A、B 兩個矩陣的同個 index 相加到 C。

資料有分 Host 端跟 Device 端，Host 就是我們的 CPU，而 Device 則是指 GPU。Host 宣告記憶體就是一般的 `malloc`，但 GPU 要宣告記憶體則要用 `cudaMalloc`。

因為我們是要把運算搬到 GPU 做，所以可以看到我們在 Host 的資料會需要先用 `cudaMemcpy` 搬到 GPU，才能去執行 `MatAdd<<<numBlock, blockSize>>>`，算完之後還要再用 `cudaMemcpy` 把 GPU 的資料搬回給 CPU，Host 才能拿資料做事。

那我們看到使用 Kernel 時語法是 `MatAdd<<<numBlock, blockSize>>>`，這跟 GPU 的 Thread 架構有關。

![GPU thread grid](https://user-images.githubusercontent.com/18013815/101697693-381e5100-3ab3-11eb-8ea7-d6698b9f8a31.png)

可以參考上圖，GPU 包含許多 Block，每個 Block 又有許多 Thread。所以我們在跑 Kernel 時要指定要用多少 Block，以及裡面使用多少 Thread。

大概知道在幹嘛之後，我們就可以編譯執行囉！

```shell
$ nvcc matadd.cu; ./a.out
PASS
```

跑的時候可以在另一個視窗呼叫 `nvidia-smi`，就可以看到 `matadd` 使用 GPU 狀況。

我們也可以用 `nvprof` 來看一下 CUDA 的效能：

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

上面顯示我們大部時間都花在搬資料，的確合理，因為我們運算的東西很簡單，導致 Overhead 發生在記憶體搬移上。

## 4. 結論

本文簡單介紹 CUDA 開發環境設置，並解說簡單的 CUDA 程式範例。GPGPU 幫助我們在大量運算時可以有效加速計算，許多數值計算函式庫抑或機器學習、深度學習框架也都會使用 GPGPU 來加速計算。
