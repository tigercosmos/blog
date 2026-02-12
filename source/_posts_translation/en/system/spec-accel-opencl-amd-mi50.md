---
title: "Run SPEC ACCEL OpenCL Benchmarks on AMD MI50"
date: 2020-08-14 00:08:00
tags: [spec accel, note, amd mi50, opencl, benchmark]
des: "This post explains how to run the OpenCL benchmarks in SPEC ACCEL on an AMD MI50, provides a brief analysis of the results, and compares them with GeForce GTX 1050 results published on the SPEC website."
lang: en
translation_key: spec-accel-opencl-amd-mi50
---

## Introduction

[SPEC ACCEL](https://www.spec.org/accel/) is an accelerator benchmark suite created by The Standard Performance Evaluation Corporation (SPEC). It supports OpenACC, OpenMP, and OpenCL—three frameworks that allow code to offload work to GPUs. Because of its strong credibility, many research projects use it as an evaluation benchmark.

SPEC ACCEL is freely available to research institutions, unlike SPEC CPU which costs a large amount of money (I “borrowed” it from the lab next door 😂). In short, free is great. I happened to have a machine equipped with an AMD MI50 that needed performance evaluation, so I used SPEC ACCEL (the OpenCL part) to see how this GPU performs.

![MI50](https://user-images.githubusercontent.com/18013815/90196045-88372080-ddfd-11ea-99d1-aa6b24a70ca6.png)

## Installation and Running

The installation process is clearly described in the official “[Install Guide Unix](https://www.spec.org/accel/docs/install-guide-unix.html)”. You likely won’t encounter any issues, so I won’t repeat the steps here—please refer to the documentation.

Since this is an AMD GPU, you need to install the driver [ROCM](https://github.com/RadeonOpenCompute/ROCm). After installation, you can use the `radeontop` tool to check whether you can connect to the GPU.

After everything is set up, before running, you need to modify the OpenCL library path in the `.cfg` file. It should be inside the ROCm installation.

Then you can run:

```shell
runspec --config=my.cfg --platform AMD --device GPU opencl
```

The results can be converted into a web format using tools provided by SPEC, and you can also choose to upload them to the official website for others to reference.

## Results

### SPEC ACCEL OpenCL Benchmarks

The OpenCL benchmarks in SPEC ACCEL are as follows:

![image](https://user-images.githubusercontent.com/18013815/90194424-de09c980-ddf9-11ea-8550-79ac0ef4c458.png)

(source: [SPEC ACCEL: A Standard Application Suite for Measuring Hardware Accelerator Performance](https://link.springer.com/chapter/10.1007/978-3-319-17248-4_3))

### AMD MI50 Results

The results look like this. Some benchmarks fail during execution; I didn’t investigate the cause, so they are left blank.

![image](https://user-images.githubusercontent.com/18013815/90194468-0265a600-ddfa-11ea-91a0-504888e3e807.png)

Visualized as a chart:

![image](https://user-images.githubusercontent.com/18013815/90194588-39d45280-ddfa-11ea-9bbf-2a8cbab75a57.png)

### GPU Utilization Ratio

Below is the GPU utilization ratio reported in another paper using NVIDIA Tesla K20:

![image](https://user-images.githubusercontent.com/18013815/90194641-4d7fb900-ddfa-11ea-9734-f70b0da84ef7.png)

(source: [SPEC ACCEL: A Standard Application Suite for Measuring Hardware Accelerator Performance](https://link.springer.com/chapter/10.1007/978-3-319-17248-4_3))

I didn’t know how to measure GPU utilization ratio myself. However, since the OpenCL code is the same, the utilization ratio should not differ too much across devices, so I used it as a reference.

### Analysis

I marked the AMD MI50 results in the figure above: green indicates cases where AMD MI50 performs better, and red indicates the opposite.

From the figure, we can see that the two factors “GPU time ratio” and “transfer time” do not have a direct correlation with performance, which is somewhat surprising. Intuitively, higher GPU utilization might suggest faster performance due to a stronger GPU. Similarly, longer transfer time might suggest worse performance. But the results do not show a clear correlation.

One interesting case is `kmeans`, where the Base Ratio is below 1. This means it is slower than the official reference machine. However, the reference machine should generally be weaker, and most benchmarks are far above 1.

For `cutcp`, the benchmark description page mentions it is compute-bound and strongly affected by accelerator performance, so it is not surprising that MI50 performs well.

For other benchmarks, whether they are particularly good or particularly bad, it is often hard to understand why just from the benchmark description.

To truly understand why some benchmarks are better or worse, you likely need to inspect the OpenCL code and the MI50 architecture design. That is too complex, and I did not dig further.

### MI50 vs GeForce GTX 1050

Next, let’s compare with GeForce GTX 1050 as a baseline.

The figure below shows a general scoring comparison for the two GPUs. It suggests that 1050 is only about 87% of MI50.

![image](https://user-images.githubusercontent.com/18013815/90195134-4efdb100-ddfb-11ea-828e-0fe5746154e8.png)

Then I referenced the GeForce GTX 1050 results contributed by others on the SPEC ACCEL website and compared them:

![image](https://user-images.githubusercontent.com/18013815/90195423-e95df480-ddfb-11ea-8885-27b2d738d95b.png)

In theory, MI50 should be roughly 13% faster. But architectural differences can make this vary, which is reasonable.

In particular, `nw` and `ge` are cases where MI50 is outperformed by GTX 1050 by a noticeable margin, which is interesting. For the other benchmarks, MI50 more or less leads, which aligns with what the scoring website suggests.

The performance differences between MI50 and GTX 1050 can be used to analyze how their architectures behave in different scenarios.

## Conclusion

This post explained how to run the OpenCL benchmarks in SPEC ACCEL on an AMD MI50, provided a brief analysis of the results, and compared them with GeForce GTX 1050 results published on the SPEC website.
