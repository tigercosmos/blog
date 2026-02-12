---
title: "How to Run OpenCL on an Intel GPU"
date: 2023-09-13 00:18:40
tags: [gpu, opencl, intel]
des: "This post explains how to run OpenCL on an Intel GPU: compile the kernel ahead of time (AOT) and execute OpenCL using the precompiled kernel binary."
lang: en
translation_key: intel-gpu-opencl
---

![many workers on GPU](https://github.com/tigercosmos/blog/assets/18013815/5d281088-938f-402c-bc63-fd6e3035870e)

How do you run OpenCL on an Intel GPU?

First, you need to install Intel’s [Compute Runtime](https://github.com/intel/compute-runtime/releases/tag/23.22.26516.18) as the driver. If you install via `apt install`, the version may be too old.

After installation, you can also install `sudo apt install clinfo`, and then run `clinfo` to check whether OpenCL can actually retrieve GPU information.

Although OpenCL can compile kernels at runtime by reading `.cl` files directly, we want to compile the kernel in advance—i.e., use AOT (Ahead-Of-Time) compilation—so we don’t have to recompile at runtime. Therefore, we need to use the `ocloc` compiler to precompile the kernel. Depending on the platform, the `-device` parameter differs. You can find the corresponding code name in Intel’s documentation, “[Use AOT for Integrated Graphics](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/ahead-of-time-compilation.html)”.

For example, the code name for the Intel HD Graphics P530 I used is `glk`.

Below is a CMake example. In `CMakeList.txt`, we use the following snippet:

```sh
# 找到 OpenCL
find_package(OpenCL REQUIRED)

# 找到 ocloc
set(OCLOC_EXECUTABLE "/usr/bin/ocloc")

# 使用 ocloc 編譯 kernel 檔案
execute_process(
    COMMAND ${OCLOC_EXECUTABLE} -device glk -file ${KERNEL_SOURCE} -output kernel.bin
    OUTPUT_VARIABLE OCLOC_OUTPUT
    ERROR_VARIABLE OCLOC_ERROR
    RESULT_VARIABLE OCLOC_RESULT
    WORKING_DIRECTORY /tmp
)

# 安裝檔案到指定位置
install(FILES "/tmp/kernel.bin_glk.bin" DESTINATION /somewhere/path/kernel.bin_glk.bin)
```

> Side note: even if you set ` -output kernel.bin`, after compilation `ocloc` still automatically appends the suffix `bin_glk`, which is honestly a bit annoying.

With this, when writing your OpenCL program later, you can directly read the precompiled kernel binary at `/somewhere/path/kernel.bin_glk.bin`.

Reading the binary looks like this:

```c++
#define CL_HPP_TARGET_OPENCL_VERSION 300 // use OpenCL 3.0
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

cl::Device device;          
cl::Program program;         
cl::Context context;               
cl::CommandQueue queue;          
std::vector<int> binaryStatus;
cl_int clError = 0;

std::vector<cl::Platform> platforms;
cl::Platform::get(&platforms);

cl::Platform& platform = platforms.front();
std::vector<cl::Device> devices;
platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

if (devices.empty()) {
   std::cerr << "no device found" << std::endl;
}

device = devices.front();
context = cl::Context(device);
queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

// 讀取 OpenCL Binary
std::ifstream binaryFile(binaryFilePath, std::ios::binary);
if (!binaryFile.is_open()) {
    std::cerr << "cannot open file" << std::endl;
}
const std::vector<uint8_t> binaryBuffer(std::istreambuf_iterator<char>(binaryFile), {});
binaryFile.close();

cl::Program::Binaries binaries;
binaries.push_back(binaryBuffer);

// 使用 Binary 建立 cl::Program
program = cl::Program(context, {device}, binaries, &binaryStatus, &clError);

program.build();
if (clError != CL_SUCCESS) {
    std::cerr << program.getBuildInfo<CLprogram_BUILD_STATUS>(device) << std::endl;
    std::cerr << program.getBuildInfo<CLprogram_BUILD_LOG>(device) << std::endl;
}

// 一般的 OpenCL 寫法 ...
```

The remaining parts are standard OpenCL programming, so I won’t go into detail here.
