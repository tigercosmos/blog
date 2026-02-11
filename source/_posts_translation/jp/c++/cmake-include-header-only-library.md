---
title: "CMake で Header-Only ライブラリを導入する方法"
date: 2022-12-19 15:00:00
tags: [c++, cmake, header-only, ]
des: "本記事では、CMake で Header-Only ライブラリを導入する方法を紹介します。"
lang: jp
translation_key: cmake-include-header-only-library
---

C++ の開発では、Header-Only ライブラリは非常によく使われる形態です。利点としては、ライブラリを利用したいときにヘッダファイル（`.h`, `.hpp`）を `#include` するだけでよく、別途ライブラリをビルドしたり、ライブラリの成果物（例：`.o`, `.so`）をリンクしたりする必要がありません。そのため、ビルド（Build）の複雑さを減らせます。また、ヘッダだけが提供され、毎回プロジェクト側でコンパイルされるため、プラットフォーム間で扱いやすいことも多いです。一方で、生成されるバイナリが大きくなったり、コンパイル時間が長くなったりする欠点もあります（いずれも、ライブラリの実装コードがコンパイル時にプロジェクト側へ埋め込まれることに起因します）。

[RapidJson](https://github.com/Tencent/rapidjson) や [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) のような有名プロジェクトも Header-Only です。では、CMake を使って Header-Only ライブラリをプロジェクトへ追加するにはどうすればよいでしょうか？以下に簡単な例を示します。

![Cover Image](https://user-images.githubusercontent.com/18013815/208432889-84323b86-e97e-4a74-9cc7-87c299045d5a.png)

## RapidJSON をプロジェクトに導入する

RapidJSON を利用するとします。典型的な CMake プロジェクトの構成例は次のとおりです。

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

## CMake で RapidJSON をダウンロードする

RapidJSON は Header-Only なので、ソースコードを自分でダウンロードしてプロジェクトに置くこともできます。しかし、毎回手動で管理するのではなく、CMake で最新（または特定バージョン）のソースコードを自動取得したい場合が多いでしょう。

ここでは、CMake 関連ファイルを `cmake` ディレクトリにまとめ、その中に `rapidjson.cmake` を作成して RapidJSON 用の CMake スクリプトを記述します。

`rapidjson.cmake` は次のようになります：

```bash
# Download RapidJson
ExternalProject_Add(
    rapidjson
    PREFIX "rapidjson"
    GIT_REPOSITORY "https://github.com/Tencent/rapidjson.git"
    GIT_TAG 80b6d1c83402a5785c486603c5611923159d0894 # 使いたいバージョンを指定
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

# RapidJSON は Header-Only ライブラリなので include/ を公開する
ExternalProject_Get_Property(rapidjson source_dir)
# ソースコードの配置場所は好みで決めてよい（例：${source_dir}/vendor/include など）。
# ただし、その場合は include パスに注意すること。
set(RAPIDJSON_INCLUDE_DIR ${source_dir}/include)
```

## RapidJSON をプロジェクトに組み込む

プロジェクトのルートにある `CmakeLists.txt` で、まず `include(ExternalProject)` を宣言し、次に `rapidjson.cmake` を `include` します。最後に、RapidJSON の include ディレクトリを `include_directories` に追加します。

```bash
cmake_minimum_required(VERSION 3.16)
project(My_Project)

include(ExternalProject)
include("${CMAKE_SOURCE_DIR}/cmake/rapidjson.cmake")

include_directories(${RAPIDJSON_INCLUDE_DIR})
message(STATUS "RAPIDJSON_INCLUDE_DIR: ${RAPIDJSON_INCLUDE_DIR}")

# ...略
```

## RapidJSON をライブラリへ依存関係として追加する

プロジェクト内の `scr/CmakeLists.txt` が元々次のようになっているとします：

```bash
# Build my_project as a library
add_library(my_project SHARED ${SRC_CC})
# Link other libraries to my_project
target_link_libraries(my_project Threads::Threads ${Boost_LIBRARIES})
```

ここに RapidJSON も追加します。`add_library()` や `add_executable()` の直下に、次の 1 行を追加してください：

```bash
add_dependencies(my_project rapidjson)
```

`ExternalProject_Add` はデフォルトで非同期実行されるため、RapidJSON の clone が完了する前にコンパイルが開始されてしまう可能性があります。そのため、依存するターゲットには `add_dependencies` を追加し、CMake に依存関係を明示する必要があります。これにより、`my_project` をビルドする際には、先に RapidJSON のダウンロードが完了するようになります。

## 使い始める

これで導入は完了です！

`my_project.cc` では、次のように RapidJSON を include して使い始められます：

```cpp
#include "rapidjson/document.h"
```

これで RapidJSON に依存するコードを include してコンパイルできるはずです。

## 参考資料

- [Add RapidJSON with CMake](https://www.jibbow.com/posts/rapidjson-cmake/)
- [Stake Overflow: Benefits of header-only libraries](https://stackoverflow.com/questions/12671383/benefits-of-header-only-libraries)
