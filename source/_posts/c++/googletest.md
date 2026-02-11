---
title: 如何使用 GoogleTest 寫 C++ 單元測試
date: 2023-05-10 15:00:00
tags: [c++, unit test, googletest]
des: "本文簡單介紹使用 GoogleTest 幫 C++ 程式寫單元測試"
lang: zh
translation_key: googletest
---

![Cover](https://github.com/solvcon/modmesh/assets/18013815/d4b634f5-d4f4-4cb4-8f73-40aca4fe9349)

## 簡介

C++ 是一種強大的程式語言，常用於開發高效能的系統軟體和應用程式。在開發複雜的 C++ 程式時，[單元測試](https://zh.wikipedia.org/wiki/%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95)是一個非常重要的步驟，因為它可以讓程式開發人員在開發過程中確保程式碼的品質和穩定性。

單元測試可以幫助開發人員驗證 C++ 程式碼的正確性，並且可以提前發現和解決潛在的錯誤。這種測試方式是對程式碼中的單個組件進行測試，例如函數、類別和方法。透過進行單元測試，開發人員可以迅速定位問題，從而節省大量的時間和成本。

此外，單元測試還可以幫助開發人員進行程式碼重構和最佳化。藉由單元測試的幫助，我們可以確保重構後的程式碼仍然能夠正確運行，同時也可以確保最佳化後的程式碼沒有導致任何錯誤。


例如我們可以幫 `sum` 寫一個簡單的測試：

```cpp
int sum(int a, int b) {
    return a + b;
}

bool test_equal(int testing, int answer) {
    if testing != answer {
        return false;
    }
    return true;
}

test_equal(sum(3, 4), 7) // true
```

如此一來如果 `sum` 裡面寫錯，比方說不小心寫成 `return a + b + 1;`，`test_equal(sum(3, 4), 7)` 就會報 `false`，因為 `sum(3, 4)` 就會是 8，這時我們就知道自己的實作有問題了。


我們當然可以土砲一個簡易的測試框架來做單元測試，不過在 C++ 中，一個常見的框架是 [GoogleTest](http://google.github.io/googletest/)，許多大型 C++ 專案都採用 GoogleTest 來作為單元測試的方案，基本上可以應付各種使用情景。

以下就簡單介紹在 C++ 專案中，如何導入 GoogleTest。本文以 CMake 專案作為範例。

## 在 CMake 專案中導入 GoogleTest

完整的範例可以再 [Github Repo](https://github.com/tigercosmos/googletest-tutorial) 中找到，可以先下載試試看：

```bash
git clone https://github.com/tigercosmos/googletest-tutorial
cd googletest-tutorial
mkdir build; cd build
cmake ..; make
ctest # 執行 GoogleTest
```

請務必試試看在接著讀下去。

### 原始專案

在範例專案中，我們原本有以下結構的專案：

```
myproject
  - CMakeLists.txt
  - main.cpp
  - foo.hpp
  - foo.cpp
```

簡單來說有個 `class Foo` 並且被 `main.cpp` 所使用的小專案。原本的 `CMakeLists.txt` 長的如下:

```makefile
cmake_minimum_required(VERSION 3.5)
project(myproject)

# 建立 foo 函示庫
add_library(foo STATIC foo.cpp)

# 加入 include 資料夾給 foo.hpp
target_include_directories(foo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 建立可執行檔 main
add_executable(main main.cpp)

# 將可執行檔和函示庫做連結
target_link_libraries(main PUBLIC foo)
```

### 加入 GoogleTest 的專案

接下來我們想加入 GoogleTest 來測試 `Foo` 的正確性，並且等等我們會建立一個 `test_foo.cpp` 的測試檔案。

新的架構如下：
```
myproject
  - CMakeLists.txt
  - main.cpp
  - foo.hpp
  - foo.cpp
  - test_foo.cpp
```

首先得先把 GoogleTest 加進 CMake 裡面，我們在 `CMakeLists.txt` 裡面加入：

```makefile
# 使用 FetchContent 模組
include(FetchContent)
# 下載和引入 Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
)
FetchContent_MakeAvailable(googletest)

# 建立一個測試用的可執行檔 test_foo
add_executable(test_foo test_foo.cpp)
# 連結必要函示庫給 test_foo
target_link_libraries(test_foo PRIVATE foo gtest gtest_main)

# 開啟 CMake 測試
enable_testing()

# 把 test_foo 加進命名 my_project_test 的測試
add_test(
    NAME my_project_test
    COMMAND test_foo
)
```

接著我們來看看 [`test_foo.cpp`](https://github.com/tigercosmos/googletest-tutorial/blob/master/test_foo.cpp) 裡面在做啥？

### 使用 TEST 和 TEST_F 來測試

#### TEST

我們可以使用 `TEST(測試集, 測試名)` Macro 來直接做測試，裡面就是用我們一般呼叫函數的方法和執行步驟，然後可以使用 `EXPECT_EQ` 來比較結果。

像是 `TEST(Foo, PublicSum)` 中，要測試的是 `PublicSum` 公開的函數，所以可以直接呼叫來使用。

```cpp
// 測試 Foo Public 可以直接使用 Foo
TEST(Foo, PublicSum)
{
    Foo foo;
    EXPECT_EQ(foo.PublicSum(1, 3), 4);
}
```

那如果想測試私有函數怎麼辦，像是 `Foo._PrivateSum` 一般是不能從外部呼叫。這時候我們可以用兩種技巧，其一是寫一個 `Bar` 繼承 `Foo`，並讓 `Foo` 原本的私有物件變成 `protected`，這樣繼承者就可以去呼叫使用。

```cpp
class Foo {
protected:
    int ProtectedGetValue();
}

class Bar : Foo {
public:
    int GetValue() {
        return ProtectedGetValue();
    }
}
```

如此一來就可以像一般測試公開函數一樣，我們呼叫 `Bar.GetValue` 就可以直接測到 `Bar.ProtectedGetValue` 也就是 `Foo.ProtectedGetValue`。

#### TEST_F

我們也可以使用 `TEST_F(Fixture名字, 測試名)` Macro，在 Google Test 中，F 代表測試夾具（Test Fixture）是設置一個共同環境來進行一組相關測試的方式。測試夾具被定義為一個 C++ 類別，每個類別實例為測試提供了一個特定的上下文。

測試夾具類別通常有一個建構函數（`SetUp`）和一個解構函數（`TearDown`），它們分別在每個測試用例之前和之後被呼叫。建構函數可以用於設置測試所需的任何共享資源或狀態，而解構函數可以用於在測試完成後清理資源。

以下例子展示了使用測試夾具類別來做測試：

```cpp
class FooTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        foo = new Foo();
    }

    void TearDown() override
    {
        delete foo;
    }

    int CallPrivateSum(int a, int b)
    {
        return foo->_PrivateSum(a, b);
    }

    Foo *foo;
};


// 測試 Foo Private 我們要使用 FooTest 的測試環境
TEST_F(FooTest, PrivateSum)
{
    EXPECT_EQ(CallPrivateSum(3, 4), 7);
}
```

由於我們不能直接呼叫 `Foo._PrivateSum`，所以我們建立了一個 `FooTest` 測試夾具，裡面包含了 `Foo *foo`。但我們還是不能呼叫 `Foo` 的私有物件。

這時候我們去原本 `class Foo` 的 `private` 裡面加上 `friend FooTest`，使 `FooTest` 成為 friend，讓 `FooTest` 可以去使用 `Foo` 的私有物件。於是我們就可以讓 `CallPrivateSum` 去呼叫 `foo->_PrivateSum`，巧妙避開了不能直接呼叫 Foo 私有物件的限制。

這個例子主要試想要展示 `TEST_F` 的用法，包含 `SetUp` 和 `TearDown`，當然一樣的邏輯也可以用 `TEST` 去辦到，只是測試夾具直接把測試的環境放進一個類別，不使用測試夾具的話，就必須在 `TEST` 裡面去陳述怎樣去建立環境。

### 執行測試

在我們的 `CMakeLists.txt` 中，我們定義了 `test_foo` 執行檔，所以我們可以直接執行 `./test_foo` 來跑測。

不過因為我們也在 `CMakeLists.txt` 中定義了 `add_test`，所以也可以用 CMake 的指令 `ctest` 來執行測試：

```bash
googletest-tutorial/build$ ctest
Test project /mnt/c/Users/tiger/googletest-tutorial/build
    Start 1: my_project_test
1/1 Test #1: my_project_test ..................   Passed    0.02 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.05 sec
```

## 結論

整體來說幫 C++ 程式碼做單元測試是一個挺好的投資也是良好的習慣，可以提高程式碼的品質和可靠性，減少錯誤和維護成本，同時也可以幫助開發人員更好地理解程式碼的運作方式。而本文簡單介紹如何在 CMake 專案中導入 GoogleTest，並以此來寫單元測試。
