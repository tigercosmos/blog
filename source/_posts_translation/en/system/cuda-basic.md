---
title: "Setting Up a CUDA Development Environment and a Simple Program Example"
date: 2020-12-10 07:00:00
tags: [cuda, gpu, parallel programming]
des: "This post briefly introduces how to set up a CUDA development environment and explains a simple CUDA program example."
lang: en
translation_key: cuda-basic
---

## 1. Introduction

GPUs were originally used for graphics. Later, with the emergence of General Purpose GPU computing (General Purpose GPU, GPGPU), GPUs became widely used for general computation. Because GPU architectures provide a massive number of threads, we can offload large-scale numerical computation to the GPU while keeping the control and logic on the CPU—this is a common GPGPU usage pattern. NVIDIA was the first company to propose the concept of GPGPU, and it introduced the [CUDA](https://zh.wikipedia.org/wiki/CUDA) technology, which allows developers to drive GPUs for computation using C-like syntax.

## 2. Installing CUDA

CUDA has many versions. If you simply run `apt install cuda`, you may not get the version you actually need.

Besides writing CUDA programs directly, you may also want CUDA for PyTorch or TensorFlow. A better approach is to find the version you need on NVIDIA’s official website. Make sure you select the version on the [CUDA Toolkit Archive](https://developer.nvidia.com/CUDA-TOOLKIT-ARCHIVE) page; otherwise, NVIDIA’s website will redirect you to the latest version.

You can follow the steps below to install it.

### 2.1 CUDA Installation

Here I use CUDA 11.0 as an example. First, go to the Archive page, select 11.0, and then choose the options that match your environment.

![CUDA 11.0 installation example](https://user-images.githubusercontent.com/18013815/101561057-94c03400-39ff-11eb-9351-303e3cedc170.png)

For example, the screenshot above selects x86 Ubuntu 20.04, and you can then choose runfile, deb local, or deb network.

The differences are: a large standalone installer, a large installer packaged as a deb, and a deb installer that downloads everything from the network. Whether it’s a deb package affects whether you can manage it later via your package manager.

I usually choose deb local, but you can pick whatever you prefer.

After you select the options, the page conveniently provides a long list of commands. In practice, you can just run them as-is to finish the installation.

CUDA 11.0 on Ubuntu 20.04 x86 installation steps:

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

If `dpkg -i` has issues, you can also use `apt ./xxx.deb` as a replacement.

### 2.2 cuDNN Installation

You can also install cuDNN. It is a library optimized for deep learning in CUDA, and you will need it if you use PyTorch. Remember to adjust the link for your OS; for example, `ubuntu18.04` means Ubuntu 18.

```shell
$ sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
$ sudo apt install libcudnn7
```

### 2.3 OpenCL Installation

If you installed CUDA, you can usually install OpenCL along the way as well, because on NVIDIA GPUs, OpenCL is backed by CUDA. Then you can run OpenCL programs directly later.

```shell
$ sudo apt install -y nvidia-opencl-dev
$ sudo apt install opencl-headers
```

### 2.4 System Configuration

First, you **must reboot**. This is very important!!!!

After installation, remember to set up your paths. Add the following lines to `~/.bashrc`:

```shell
export PATH=$PATH:/usr/local/cuda/bin
export CUDADIR=/usr/local/cuda
```

### 2.5 Verify Installation

Now you can check whether everything is installed correctly.

If CUDA is installed properly, the following commands should all work.

```shell
$ nvidia-smi  # Driver 
$ nvcc --version # CUDA
$ /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn # CuDNN
```

### 2.6 Others

If the installation method above doesn’t work for you, these articles may be helpful:

- [Easy installation of Cuda Toolkit on Ubuntu 18.04](https://medium.com/@ayush.goc/easy-installation-of-cuda-toolkit-on-ubuntu-18-04-7931394c1233)
- [Tutorial: CUDA v10.2 + CUDNN v7.6.5 Installation @ Ubuntu 18.04](https://sh-tsang.medium.com/tutorial-cuda-v10-2-cudnn-v7-6-5-installation-ubuntu-18-04-3d24c157473f)
- [Installing CUDA 10.1 on Ubuntu 20.04](https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0)


## 3. A CUDA Program Example

The best way to learn CUDA is to read the official tutorial documentation, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html), since it is NVIDIA’s own product. The book [Multicore and GPU Programming: An Integrated Approach](https://www.tenlong.com.tw/products/9787111557685?list_name=srh) by Gerassimos is also quite good and is suitable for beginners.

Here is a simple example:

`matadd.cu`:

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

In GPU computing, the smallest execution unit is called a kernel, which corresponds to the function executed by each thread on the GPU. That’s why we declare `__global__ void MatAdd()`: `__global__` tells the compiler this is a GPU function, and `MatAdd()` simply adds matrices A and B at the same index into C.

Data is divided into the host side and the device side. Host refers to the CPU, and device refers to the GPU. Allocating memory on the host uses the normal `malloc`, while allocating GPU memory uses `cudaMalloc`.

Since we want to move computation to the GPU, the host data must be copied to the GPU first via `cudaMemcpy` before we can run `MatAdd<<<numBlock, blockSize>>>`. After the computation, we need to copy the data back from the GPU to the CPU via `cudaMemcpy` so the host can use the results.

When launching a kernel, the syntax `MatAdd<<<numBlock, blockSize>>>` is related to the GPU thread hierarchy.

![GPU thread grid](https://user-images.githubusercontent.com/18013815/101697693-381e5100-3ab3-11eb-8ea7-d6698b9f8a31.png)

As shown above, a GPU contains many blocks, and each block contains many threads. Therefore, when launching a kernel, you must specify how many blocks to use and how many threads to use within each block.

Once you roughly understand what is happening, you can compile and run:

```shell
$ nvcc matadd.cu; ./a.out
PASS
```

While it is running, you can call `nvidia-smi` in another terminal window to see GPU usage by `matadd`.

You can also use `nvprof` to take a look at CUDA performance:

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

The output above shows that most of the time is spent on data transfers, which makes sense. The computation itself is very simple, so the overhead is dominated by memory copies.

## 4. Conclusion

This post briefly introduced how to set up a CUDA development environment and explained a simple CUDA program example. GPGPU can effectively accelerate computation for large-scale workloads, and many numerical libraries as well as machine learning and deep learning frameworks also use GPGPU to speed up their computations.
