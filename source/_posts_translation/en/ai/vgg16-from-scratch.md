---
title: "VGG16 net from scratch in two ways: C++ on CPU and CUDA on GPU"
date: 2020-12-02 05:30:00
tags: [ai, deep learning, vgg16, c++, cuda, cudnn, cublas]
des: "This article introduces VGG16 model in two versions: the first one is written in pure C++ running on CPU; the second one is using CUDA, cudnn and cublas running on GPU."
lang: en
translation_key: vgg16-from-scratch
---

## Introduction

VGG16 is a convolutional neural network model introduced by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. It is a famous model for image recognition in ILSVRC-2014, and it has some well-known counterparts like AlexNet, GoogleNet, and ResNet. The more detail about VGG16 can refer to this article, "[VGG16 – Convolutional Network for Classification and Detection](https://neurohive.io/en/popular-networks/vgg16/)". 

In this post, I built VGG16 model from scratch in two versions. The first one is written in pure C++, where all operations of layers are implemented only by C++ running on CPU. The second one is using CUDA, cudnn and cublas, which leverages the advantages of GPU to speed up the operations of the neural network.

![Vgg16 architecture](https://user-images.githubusercontent.com/18013815/100784827-bd1dc080-344a-11eb-8a65-76be05809123.png)

## VGG16 in C++

I wrote the program as header-only framework. I believe it has good architecture to be scalable and extendable. You can check the repo on Github:[simple-vgg16](https://github.com/tigercosmos/simple-vgg16). 

Take a look at the way to use the framework.

```c++
    // prepare the input data
    auto input = sv::Tensor<double>(224, 224, 3); // 224 x 224 x 3

    // prepare the network
    auto network = sv::Network<double>();

    // add layers into the network
    auto *layer1_1 = new sv::ConvLayer<double>(3, 3, 64); //224 x 224 x 64
    network.addLayer(layer1_1);
    auto *layer1_2 = new sv::ConvLayer<double>(64, 3, 64); //224 x 224 x 64
    network.addLayer(layer1_2);
    auto *layer1_3 = new sv::MaxPoolLayer<double>(2, 2); // 112 x 112 x 64
    network.addLayer(layer1_3);

    // SKIP

    // predict the result by forwarding
    sv::Tensor<double> output;
    network.predict(input, output);
```

As you can see in the snippet above, you can first declare a network, and add layers into the network. Once the network model is defined, just run `predict()` to drive the model. It is just as easy as how you write models in Tensorflow or Pytorch.

The framework is simple. It has only few headers:
- Activation: define activation functions
- Layer: define layers, such as `ConvLayer`, `MaxPoolLayer` 
- Network: the network with layers; a model
- Operand: define the operators in layers, like convolution, pooling
- Tensor: define the tensor datatype

Basically I think this simple framework is really straightforward. The most complicated part is implementing operands. It is easy to make bugs while writing the operations like convolution. I spend a lot of time to debug for the operations implementation. There are also many opportunities to optimize the framework, such as how to deal with the data or what algorithm should be used. The pure C++ runs really slow, so I added OpenMP for it, which makes it a little faster.

I am satisfied with this simple framework. Currently it only support inference, but I think it is not quite hard to add features for it, no matter adding training part, or extending with more operations, layers or activation functions.

The framework also does some benchmarks. The following is how it looks like:

```
The VGG16 Net
-----------------------------
NAME:   MEM     PARAM   MAC
-----------------------------
Conv:   25690112,       1792,   86704128
Conv:   25690112,       36928,  1849688064
MaxPool:        6422528,        0,      0
Conv:   12845056,       73856,  924844032
Conv:   12845056,       147584, 1849688064
MaxPool:        3211264,        0,      0
Conv:   6422528,        295168, 924844032
Conv:   6422528,        590080, 1849688064
MaxPool:        1605632,        0,      0
Conv:   3211264,        1180160,        924844032
Conv:   3211264,        2359808,        1849688064
MaxPool:        802816, 0,      0
Conv:   802816, 2359808,        462422016
Conv:   802816, 2359808,        462422016
MaxPool:        200704, 0,      0
FC:     32768,  102764544,      102760448
FC:     32768,  16781312,       16777216
FC:     8000,   4097000,        4096000
Total:  110Mb,  133M,   11308M
-----------------------------
@@ result @@
0.0413532 0.0416434 0.0412649 0.0419855 0.0412341 ... so many ... 0.0415309 0.0406799 0.0418257 0.0416706 0.0413013
```

You can gain insight in [simple-vgg16](https://github.com/tigercosmos/simple-vgg16) on Github. Welcome to fork it and modify it.

## VGG16 in CUDA

CPU run slower than GPU in certain cases, such as matrix operations, which are used a lot in deep learning models. In order to improve the speed of the running time of deep neural network, adapting CUDA or its libraries like cudnn or cublas does help by exploiting GPU computing.

I use CUDA to implement the VGG16 net as the second version. You can check it on Github: [simple-vgg16-cu](https://github.com/tigercosmos/simple-vgg16-cu). Just like the previous C++ one, it is a work for proof-of-concept. The convolution layers are implemented by cudnn, and the fully-connected layer is implemented in cublas.

By using API like `cudnnConvolutionBiasActivationForward`, `cudnnPoolingForward`, `cublasSgemm`, it is easy to use these neural network (NN) APIs to implement models using CUDA to run on GPU. The CUDA series libraries really improve the efficiency of developing NN models with a considerable inference/training speedup, so it is always encouraged to use these kind of libraries.

GPU run only seconds while the CPU runs about minutes. The following is the running time for each layers. 

```
CONV 224x224x64 84 ms
CONV 224x224x64 115 ms
POOLMAX 112x112x64 108 ms
CONV 112x112x128 114 ms
CONV 112x112x128 111 ms
POOLMAX 56x56x128 70 ms
CONV 56x56x256 122 ms
CONV 56x56x256 115 ms
POOLMAX 28x28x256 92 ms
CONV 28x28x512 122 ms
CONV 28x28x512 118 ms
POOLMAX 14x14x512 73 ms
CONV 14x14x1024 63 ms
CONV 14x14x1024 114 ms
POOLMAX 7x7x1024 102 ms
FC 4096 3280 ms
FC 4096 311 ms
FC 1000 104 ms
```

Note that my code is not optimized. Despite the fact that it run faster than CPU, VGG16 using Pytorch can run with in a hundred of milliseconds, which means my implementation has a large room for improvement.

## Conclusion

Although we can build a neural network like VGG16 by frameworks like Pytorch, it is able to gain a deeper understanding of the concepts by implement the models from scratch. That's why I wrote the VGG16 in two way: pure C++ running on CPU and CUDA running on GPU.

CPU computing stands in stark contrast to GPU computing. We can see the running time is enormously different. It is not to said we always choose GPU since it is faster. Some embedded hardwares only have CPU, so in that situation, CPU is the only choice. Sometimes we even don't use CPU or GPU: ALU or FPGA may run even faster.

It is always a good practice to write something from scratch. I learned a lot after writing these simple stuffs. I hope this article do help you and I am looking forward to see your implementation.

