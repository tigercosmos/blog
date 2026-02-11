---
title: "Including a Header-Only Library in CMake"
date: 2022-12-19 15:00:00
tags: [c++, cmake, header-only, ]
des: "This post explains how to include a header-only library in CMake."
lang: en
translation_key: cmake-include-header-only-library
---

In C++ development, header-only libraries are very common. The main advantage is that when you want to use the library, you only need to `#include` the header files (`.h`, `.hpp`)—you do not need to compile the library separately or link against library artifacts (e.g., `.o`, `.so`). This simplifies the build process. Also, since the library is provided as headers and is compiled together with your project each time, it is often easier to use across platforms. On the other hand, there are downsides such as larger generated binaries and longer compile times (both are consequences of embedding the library’s implementation directly into your codebase during compilation).

Some well-known projects like [RapidJson](https://github.com/Tencent/rapidjson) and [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) are header-only libraries. So, how do we use CMake to add a header-only library to a project? Below is a simple example.

![Cover Image](https://user-images.githubusercontent.com/18013815/208432889-84323b86-e97e-4a74-9cc7-87c299045d5a.png)

## Add RapidJSON to a Project

Assume you want to use RapidJSON. A typical CMake project layout might look like this:

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

## Download RapidJSON via CMake

Although RapidJSON is a header-only library and you could download the source code manually and place it into your project, that is usually not what we want. Instead, we want CMake to automatically fetch the latest (or a specific) version of the source code.

In this example, we group CMake helper scripts under the `cmake` directory, and create `rapidjson.cmake` to handle the RapidJSON CMake logic.

`rapidjson.cmake` looks like this:

```bash
# Download RapidJson
ExternalProject_Add(
    rapidjson
    PREFIX "rapidjson"
    GIT_REPOSITORY "https://github.com/Tencent/rapidjson.git"
    GIT_TAG 80b6d1c83402a5785c486603c5611923159d0894 # Put the version you want here
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

# RapidJSON is a header-only library; expose it via include/
ExternalProject_Get_Property(rapidjson source_dir)
# Choose where you want to place the source; you could also use ${source_dir}/vendor/include,
# but then be careful about include paths when you use it.
set(RAPIDJSON_INCLUDE_DIR ${source_dir}/include)
```

## Add RapidJSON to Your Project

In the project root `CmakeLists.txt`, first declare `include(ExternalProject)`, then `include` the `rapidjson.cmake` script, and finally `include_directories` for the RapidJSON include directory.

```bash
cmake_minimum_required(VERSION 3.16)
project(My_Project)

include(ExternalProject)
include("${CMAKE_SOURCE_DIR}/cmake/rapidjson.cmake")

include_directories(${RAPIDJSON_INCLUDE_DIR})
message(STATUS "RAPIDJSON_INCLUDE_DIR: ${RAPIDJSON_INCLUDE_DIR}")

# ... omitted
```

## Link RapidJSON to Your Library

Your project’s `scr/CmakeLists.txt` might originally look like this:

```bash
# Build my_project as a library
add_library(my_project SHARED ${SRC_CC})
# Link other libraries to my_project
target_link_libraries(my_project Threads::Threads ${Boost_LIBRARIES})
```

Then, add RapidJSON as well. Right below your `add_library()` or `add_executable()`, add this line:

```bash
add_dependencies(my_project rapidjson)
```

`ExternalProject_Add` is asynchronous by default, so compilation might start while RapidJSON is still being cloned. Therefore, for any targets that depend on it, you need `add_dependencies` to tell CMake about the dependency relationship. This way, when you build `my_project`, CMake will ensure RapidJSON has been downloaded first.

## Start Using It

That’s it—done!

In `my_project.cc`, you can now include and use RapidJSON:

```cpp
#include "rapidjson/document.h"
```

At this point, you should be able to include and compile code that depends on RapidJSON.

## References

- [Add RapidJSON with CMake](https://www.jibbow.com/posts/rapidjson-cmake/)
- [Stake Overflow: Benefits of header-only libraries](https://stackoverflow.com/questions/12671383/benefits-of-header-only-libraries)
