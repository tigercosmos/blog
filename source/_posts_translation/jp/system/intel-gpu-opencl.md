---
title: "Intel GPU 上で OpenCL を動かす方法"
date: 2023-09-13 00:18:40
tags: [gpu, opencl, intel]
des: "本記事では Intel GPU 上で OpenCL を動かす方法を紹介します。AOT で Kernel を事前コンパイルし、生成された Kernel バイナリを用いて OpenCL を実行します。"
lang: jp
translation_key: intel-gpu-opencl
---

![many workers on GPU](https://github.com/tigercosmos/blog/assets/18013815/5d281088-938f-402c-bc63-fd6e3035870e)

Intel GPU 上で OpenCL を動かすにはどうすればよいでしょうか？

まず、ドライバとして Intel の [Compute Runtime](https://github.com/intel/compute-runtime/releases/tag/23.22.26516.18) をインストールする必要があります。`apt install` で入るものは古すぎる可能性があります。

インストール後は `sudo apt install clinfo` も入れておくと便利です。`clinfo` コマンドで OpenCL が GPU 情報を正しく取得できているか確認できます。

OpenCL は実行時に `.cl` ファイルを直接読み込んで Kernel をコンパイルできますが、ここでは Kernel を事前にコンパイルしておきたいです。つまり AOT（Ahead-Of-Time）を使い、実行時に再コンパイルしないようにします。そのため、`ocloc` コンパイラを使って Kernel を事前コンパイルします。プラットフォームごとに `-device` 引数が異なるので、Intel のドキュメント「[Use AOT for Integrated Graphics](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/ahead-of-time-compilation.html)」で対応するコードを確認できます。

たとえば、私が使っている Intel HD Graphics P530 のコードは `glk` です。

以下では CMake の例を示します。`CMakeList.txt` で次のコードを使います：

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

> 余談：` -output kernel.bin` を設定しても、`ocloc` はコンパイル後に自動で `bin_glk` というサフィックスを付けてしまうので、正直少し面倒です。

これで OpenCL のプログラムを書くときに、`/somewhere/path/kernel.bin_glk.bin` から事前コンパイル済みの Kernel バイナリを直接読み込めます。

バイナリの読み込みは次のように書けます：

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

残りは通常の OpenCL プログラミングなので、本記事ではこれ以上は触れません。
