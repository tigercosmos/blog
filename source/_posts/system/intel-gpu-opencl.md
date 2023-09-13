---
title: 如何在 Intel GPU 上跑 OpenCL
date: 2023-09-13 00:18:40
tags: [gpu, opencl, intel]
des: "本文介紹如何在 Intel GPU 上跑 OpenCL，使用 AOT 事先編譯 Kernel 檔案，並使用編譯好的 Kernel Binary 來執行 OpenCL"
---

![many workers on GPU](https://github.com/tigercosmos/blog/assets/18013815/5d281088-938f-402c-bc63-fd6e3035870e)

如何在 Intel GPU 上跑 OpenCL?

首先，我們得先去裝 Intel 的 [Compute Runtime](https://github.com/intel/compute-runtime/releases/tag/23.22.26516.18) 來裝驅動程式，直接用 `apt install` 裝的可能會太舊。

裝完之後可以順邊裝一下 `sudo apt install clinfo`，然後使用 `clinfo` 指令看一下 OpenCL 是不是可以真的抓到 GPU 資訊。

雖然 OpenCL 可以執行時直接讀取 `.cl` 檔案來進行編譯 Kernel，但我們希望事先編譯好 Kernel，也就是使用 AOT 技術，這樣執行時才不用重新編譯，因此我們會需要使用 ocloc 編譯器事先編譯。根據平台我們要下的 `-device` 參數不一樣，可以在 Intel 文件中「[Use AOT for Integrated Graphics](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/ahead-of-time-compilation.html)」找到對應的代碼。

例如我使用的 Intel HD Graphics P530 其代碼就是 `glk`。

以下以 CMake 作為範例，`CMakeList.txt` 中我們使用以下程式碼：

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

> 題外話，即使設置 ` -output kernel.bin`，ocloc 編譯完還是會自動加後綴 `bin_glk` 其實有點惱人

這樣之後寫 OpenCL 程式的時候，就可以直接去 `/somewhere/path/kernel.bin_glk.bin` 讀取編譯好的 Kernel Binary 檔案。

讀取 Binary 的部份是這樣寫：

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

剩下的部分就是一般的 OpenCL 程式設計，本文就不多介紹了。
