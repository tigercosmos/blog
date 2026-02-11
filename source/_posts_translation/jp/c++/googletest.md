---
title: "GoogleTest で書く C++ のユニットテスト"
date: 2023-05-10 15:00:00
tags: [c++, unit test, googletest]
des: "本記事では、GoogleTest を使って C++ のユニットテストを書く方法を簡単に紹介します。"
lang: jp
translation_key: googletest
---

![Cover](https://github.com/solvcon/modmesh/assets/18013815/d4b634f5-d4f4-4cb4-8f73-40aca4fe9349)

## 概要

C++ は強力なプログラミング言語であり、高性能なシステムソフトウェアやアプリケーションの開発に広く利用されています。複雑な C++ プログラムを開発する際には、[ユニットテスト（単体テスト）](https://zh.wikipedia.org/wiki/%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95) は非常に重要なステップです。開発プロセスの中でコードの品質と安定性を継続的に担保できるからです。

ユニットテストは C++ コードの正しさを検証し、潜在的な不具合を早い段階で発見・解決するのに役立ちます。テストの対象は、関数、クラス、メソッドなど、コードの個々のコンポーネントです。ユニットテストを行うことで問題を迅速に特定でき、結果として時間とコストを大きく節約できます。

さらに、ユニットテストはリファクタリングや最適化にも有効です。テストがあれば、リファクタリング後のコードが正しく動作することや、最適化によって新たな不具合が入っていないことを確認できます。

たとえば、`sum` に対して簡単なテストを用意できます：

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

このようにしておけば、`sum` の実装を間違えて、たとえば `return a + b + 1;` と書いてしまった場合、`test_equal(sum(3, 4), 7)` は `false` になります。`sum(3, 4)` が 8 になってしまうからです。これで実装に問題があることが分かります。

もちろん、自作の簡易テストフレームワークを作ることもできますが、C++ では [GoogleTest](http://google.github.io/googletest/) がよく使われます。多くの大規模 C++ プロジェクトが GoogleTest を採用しており、基本的にさまざまな利用シーンに対応できます。

以下では、C++ プロジェクトに GoogleTest を導入する方法を簡単に紹介します。この記事では CMake プロジェクトを例として扱います。

## CMake プロジェクトに GoogleTest を導入する

完全なサンプルはこの [Github Repo](https://github.com/tigercosmos/googletest-tutorial) にあります。まずはダウンロードして試してみてください：

```bash
git clone https://github.com/tigercosmos/googletest-tutorial
cd googletest-tutorial
mkdir build; cd build
cmake ..; make
ctest # GoogleTest を実行
```

ぜひ、読み進める前に試してください。

### 元のプロジェクト

サンプルプロジェクトは、最初は次のような構成です：

```
myproject
  - CMakeLists.txt
  - main.cpp
  - foo.hpp
  - foo.cpp
```

`class Foo` があり、それを `main.cpp` が利用している小さなプロジェクトです。元の `CMakeLists.txt` は次のようになります：

```makefile
cmake_minimum_required(VERSION 3.5)
project(myproject)

# foo ライブラリを作成
add_library(foo STATIC foo.cpp)

# foo.hpp のために include ディレクトリを追加
target_include_directories(foo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 実行ファイル main を作成
add_executable(main main.cpp)

# 実行ファイルとライブラリをリンク
target_link_libraries(main PUBLIC foo)
```

### GoogleTest を追加したプロジェクト

次に、`Foo` の正しさをテストするために GoogleTest を追加し、テストファイル `test_foo.cpp` を作成します。

新しい構成は次のとおりです：

```
myproject
  - CMakeLists.txt
  - main.cpp
  - foo.hpp
  - foo.cpp
  - test_foo.cpp
```

まずは GoogleTest を CMake に取り込みます。`CMakeLists.txt` に次を追加します：

```makefile
# FetchContent モジュールを使用
include(FetchContent)
# Google Test をダウンロードして利用可能にする
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
)
FetchContent_MakeAvailable(googletest)

# テスト用実行ファイル test_foo を作成
add_executable(test_foo test_foo.cpp)
# test_foo に必要なライブラリをリンク
target_link_libraries(test_foo PRIVATE foo gtest gtest_main)

# CMake のテストを有効化
enable_testing()

# test_foo を my_project_test という名前のテストとして登録
add_test(
    NAME my_project_test
    COMMAND test_foo
)
```

続いて、[`test_foo.cpp`](https://github.com/tigercosmos/googletest-tutorial/blob/master/test_foo.cpp) の中身を見てみましょう。

### TEST と TEST_F によるテスト

#### TEST

`TEST(テストスイート名, テスト名)` マクロを使うと、直接テストを書けます。中身は普段どおりの関数呼び出しと実行手順で書き、結果の比較には `EXPECT_EQ` を使えます。

たとえば `TEST(Foo, PublicSum)` では公開関数 `PublicSum` をテストするため、普通に呼び出して使えます：

```cpp
// Foo の public メソッドはそのままテストできる
TEST(Foo, PublicSum)
{
    Foo foo;
    EXPECT_EQ(foo.PublicSum(1, 3), 4);
}
```

では `Foo._PrivateSum` のような private 関数をテストしたい場合はどうでしょう？通常、外部から呼び出せません。この場合のテクニックは 2 つあります。1 つは `Foo` を継承した `Bar` を作り、`Foo` の private メンバを `protected` にして継承側から呼べるようにする方法です。

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

これで、通常の公開関数をテストするのと同様に `Bar.GetValue` を呼べば、`Bar.ProtectedGetValue`（つまり `Foo.ProtectedGetValue`）をテストできます。

#### TEST_F

`TEST_F(Fixture名, テスト名)` マクロも利用できます。Google Test では `F` は Test Fixture（テストフィクスチャ）を意味し、関連するテスト群のために共通の環境を用意して実行する仕組みです。フィクスチャは C++ クラスとして定義され、そのインスタンスがテストのコンテキストになります。

テストフィクスチャクラスには通常、各テストケースの前後で呼ばれる `SetUp` と `TearDown`（それぞれ初期化と後片付け）が用意されます。`SetUp` は共有リソースや状態の準備に、`TearDown` はテスト終了後のクリーンアップに使います。

次の例はテストフィクスチャを使ったテストです：

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


// Foo の private メソッドは FooTest の環境を使ってテストする
TEST_F(FooTest, PrivateSum)
{
    EXPECT_EQ(CallPrivateSum(3, 4), 7);
}
```

`Foo._PrivateSum` は直接呼べないので、`Foo *foo` を保持する `FooTest` フィクスチャを作ります。ただし、これだけではまだ `Foo` の private メンバにはアクセスできません。

そこで、元の `class Foo` の `private` 節に `friend FooTest` を追加し、`FooTest` を friend にします。すると `FooTest` から `Foo` の private メンバを利用できるようになります。結果として `CallPrivateSum` で `foo->_PrivateSum` を呼び出せるようになり、private の制限をうまく回避できます。

この例は主に `TEST_F` の使い方（`SetUp` と `TearDown` を含む）を示すものです。同じロジックは `TEST` でも可能ですが、フィクスチャを使わない場合は `TEST` 内で毎回「環境をどう構築するか」を書く必要があります。

### テストを実行する

`CMakeLists.txt` で `test_foo` 実行ファイルを定義しているため、`./test_foo` を直接実行してテストできます。

ただし、`CMakeLists.txt` で `add_test` も定義しているので、CMake の `ctest` コマンドでもテストを実行できます：

```bash
googletest-tutorial/build$ ctest
Test project /mnt/c/Users/tiger/googletest-tutorial/build
    Start 1: my_project_test
1/1 Test #1: my_project_test ..................   Passed    0.02 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.05 sec
```

## 結論

C++ コードにユニットテストを書くことは、良い投資であり良い習慣です。コードの品質と信頼性を向上させ、バグや保守コストを減らし、さらにコードの動作をより深く理解する助けにもなります。本記事では、CMake プロジェクトに GoogleTest を導入し、ユニットテストを書く方法を簡単に紹介しました。
