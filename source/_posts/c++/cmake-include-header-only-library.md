---
title: 如何在 Cmake 中引入 Header-Only Library
date: 2022-12-19 15:00:00
tags: [c++, cmake, header-only, ]
des: "本文介紹如何在 Cmake 中引入 Header-Only Library"
---

在 C++ 開發環境中，Header-Only Library 是一種很長見的形式，好處是想要使用該函示庫（Library）的時候，只需要 `include` 標頭檔（`.h`, `.hpp`）即可，不需要額外去編譯函示庫或是去連結函示目的物件（Object, `.o`, `.so`），幫我們簡化了建構（Build）的複雜度，同時因為只給你標頭檔，每次都會是自己編譯，所以也比較容跨平台。不過反之就會有編譯出來的程式碼比較肥大，編譯時間比較久等等的壞處（兩者都是因為你直接把函示庫的程式碼直接嵌入到你的程式碼中）。

一些比較知名的專案像是 [RapidJson](https://github.com/Tencent/rapidjson) 或 [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) 都是 Header-Only Library，那我們如何使用 Cmake 去把 Header-Only Library 加入到專案中呢？以下是簡單範例。

![Cover Image](https://user-images.githubusercontent.com/18013815/208432889-84323b86-e97e-4a74-9cc7-87c299045d5a.png)

## 引入 RapidJson 的專案

假設你要用 RapidJson，以下是典型的 Cmake 專案的設置

```log
My_Project
├── CMakeLists.txt
├── include
│   └── ...
├── src
│   ├── my_project.cc
│   ├── ...
│   └── CMakeLists.txt
├── cmake
│   └── rapidjson.cmake
└── ...
```

## 透過 Cmake 來下載 RapidJson

雖然 RapidJson 是 Header-Only Library，所以我們可以自己下載原始碼（Source Code）然後放進專案。但是我們並不想自己下載原始碼然後放進專案中，我們希望的是透過 Cmake 自動去抓最新（或特定版本）的原始碼。

這時候我們將 Cmake 檔案都歸類在 `cmake` 資料夾中，在裡面建立 `rapidjson.cmake` 來處理 RapidJson 的 Cmake 腳本。

`rapidjson.cmake` 長的如下：

```bash
# 下載 RapidJson
ExternalProject_Add(
    rapidjson
    PREFIX "rapidjson"
    GIT_REPOSITORY "https://github.com/Tencent/rapidjson.git"
    GIT_TAG 80b6d1c83402a5785c486603c5611923159d0894 # 放入你想要的版本
    TIMEOUT 10
    CMAKE_ARGS
        -DRAPIDJSON_BUILD_TESTS=OFF
        -DRAPIDJSON_BUILD_DOC=OFF
        -DRAPIDJSON_BUILD_EXAMPLES=OFF
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

# RapidJSON 是 Header-Only Library，將其放入 include
ExternalProject_Get_Property(rapidjson source_dir)
# 看你想要把原始碼放哪，也可以是 ${source_dir}/vendor/include，但之後使用的時候要注意路徑
set(RAPIDJSON_INCLUDE_DIR ${source_dir}/include)
```

## 引入 RapidJson 到專案

在專案的 Root 路徑的 `CmakeLists.txt` 中，先是宣告有 `ExternalProject`，並且要 `include` 剛剛的 `rapidjson.cmake`  以及 `include_directories` 其原始碼的資料夾。

```bash
cmake_minimum_required(VERSION 3.16)
project(My_Project)

include(ExternalProject)
include("${CMAKE_SOURCE_DIR}/cmake/rapidjson.cmake")

include_directories(${RAPIDJSON_INCLUDE_DIR})
message(STATUS "RAPIDJSON_INCLUDE_DIR: ${RAPIDJSON_INCLUDE_DIR}")

# ...略
```

## 連結 RapidJson 到函示庫

你的專案中的 `scr/CmakeLists.txt` 原本可能長的像以下這樣：

```bash
# 編譯 my_project 為函示庫
add_library(my_project SHARED ${SRC_CC})
# 連結其他函示庫到 my_project
target_link_libraries(my_project Threads::Threads ${Boost_LIBRARIES})
```

接著我們把 RapidJson 也加入，在原本的 `add_library()` 或 `add_executable()` 的下面加入下面這行：

```bash
add_dependencies(my_project rapidjson)
```

## 開始使用

這樣我們就大功告成啦！

在 `my_project.cc` 文件中，我們就可以開始使用 RapidJson 了：

```cpp
#include "rapidjson/document.h"
```

接下來你應該可以引用和編譯 RapidJson 相關的程式碼了！

## 參考資料

- [Add RapidJSON with CMake](https://www.jibbow.com/posts/rapidjson-cmake/)
- [Stake Overflow: Benefits of header-only libraries](https://stackoverflow.com/questions/12671383/benefits-of-header-only-libraries)
