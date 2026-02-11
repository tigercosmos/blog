---
title: "Writing C++ Unit Tests with GoogleTest"
date: 2023-05-10 15:00:00
tags: [c++, unit test, googletest]
des: "This post briefly introduces how to write unit tests for C++ with GoogleTest."
lang: en
translation_key: googletest
---

![Cover](https://github.com/solvcon/modmesh/assets/18013815/d4b634f5-d4f4-4cb4-8f73-40aca4fe9349)

## Introduction

C++ is a powerful programming language commonly used to build high-performance system software and applications. When developing non-trivial C++ programs, [unit testing](https://zh.wikipedia.org/wiki/%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95) is a critical step because it helps developers ensure code quality and stability throughout the development process.

Unit tests help verify the correctness of C++ code, and they allow you to discover and fix potential bugs earlier. This type of testing focuses on individual components of the code, such as functions, classes, and methods. With unit tests, developers can locate problems quickly, saving a lot of time and cost.

In addition, unit tests are very helpful for refactoring and optimization. With unit tests in place, you can ensure refactored code still works correctly, and that optimizations do not introduce new issues.

For example, you can write a very simple test for `sum`:

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

This way, if you wrote `sum` incorrectly—for example, accidentally writing `return a + b + 1;`—then `test_equal(sum(3, 4), 7)` would return `false`, because `sum(3, 4)` would be 8. At that point you know your implementation is wrong.

Of course, you can hack together a tiny test framework yourself, but in C++ a common choice is [GoogleTest](http://google.github.io/googletest/). Many large C++ projects use GoogleTest as their unit testing solution, and it generally covers a wide range of use cases.

Below is a brief introduction to how to integrate GoogleTest into a C++ project. This post uses a CMake project as an example.

## Integrate GoogleTest into a CMake Project

You can find the complete example in this [Github Repo](https://github.com/tigercosmos/googletest-tutorial). You can download and try it first:

```bash
git clone https://github.com/tigercosmos/googletest-tutorial
cd googletest-tutorial
mkdir build; cd build
cmake ..; make
ctest # run GoogleTest
```

Please try it before reading on.

### Original Project

In the example project, we start with the following structure:

```
myproject
  - CMakeLists.txt
  - main.cpp
  - foo.hpp
  - foo.cpp
```

In short, it is a small project with a `class Foo` used by `main.cpp`. The original `CMakeLists.txt` looks like this:

```makefile
cmake_minimum_required(VERSION 3.5)
project(myproject)

# build foo as a library
add_library(foo STATIC foo.cpp)

# add include directory for foo.hpp
target_include_directories(foo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# build executable main
add_executable(main main.cpp)

# link the executable with the library
target_link_libraries(main PUBLIC foo)
```

### Project with GoogleTest

Next, we want to add GoogleTest to test the correctness of `Foo`, and we will create a test file named `test_foo.cpp`.

The new structure becomes:

```
myproject
  - CMakeLists.txt
  - main.cpp
  - foo.hpp
  - foo.cpp
  - test_foo.cpp
```

First, we need to add GoogleTest to CMake. In `CMakeLists.txt`, add:

```makefile
# use FetchContent module
include(FetchContent)
# download and import Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
)
FetchContent_MakeAvailable(googletest)

# build a test executable test_foo
add_executable(test_foo test_foo.cpp)
# link required libraries to test_foo
target_link_libraries(test_foo PRIVATE foo gtest gtest_main)

# enable CMake testing
enable_testing()

# register test_foo as a test named my_project_test
add_test(
    NAME my_project_test
    COMMAND test_foo
)
```

Now, let’s see what [`test_foo.cpp`](https://github.com/tigercosmos/googletest-tutorial/blob/master/test_foo.cpp) does.

### Testing with TEST and TEST_F

#### TEST

You can use the `TEST(test_suite_name, test_name)` macro to write a test. Inside, you write code much like normal function calls and execution steps, and you can use `EXPECT_EQ` to compare results.

For example, in `TEST(Foo, PublicSum)`, we test the public function `PublicSum`, so we can call it directly:

```cpp
// test Foo public method directly
TEST(Foo, PublicSum)
{
    Foo foo;
    EXPECT_EQ(foo.PublicSum(1, 3), 4);
}
```

But what if you want to test a private function, such as `Foo._PrivateSum`, which usually cannot be called from outside? Here are two techniques. One is to write a `Bar` that inherits from `Foo`, and make the original private members of `Foo` become `protected`, so the subclass can call them.

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

Then, like testing a normal public function, you can call `Bar.GetValue` to test `Bar.ProtectedGetValue`, which is actually `Foo.ProtectedGetValue`.

#### TEST_F

You can also use the `TEST_F(FixtureName, TestName)` macro. In Google Test, `F` stands for a test fixture. A test fixture is a way to set up a shared environment for a group of related tests. A fixture is defined as a C++ class, and each instance provides a specific context for tests.

The fixture class usually has a constructor-like hook (`SetUp`) and a destructor-like hook (`TearDown`). They are called before and after each test case, respectively. `SetUp` can initialize shared resources and state, and `TearDown` cleans them up after the test finishes.

The following example demonstrates how to test using a fixture:

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


// test Foo private method using FooTest fixture environment
TEST_F(FooTest, PrivateSum)
{
    EXPECT_EQ(CallPrivateSum(3, 4), 7);
}
```

Since we cannot call `Foo._PrivateSum` directly, we create a `FooTest` fixture that contains `Foo *foo`. But we still cannot access `Foo`’s private members directly.

At this point, we add `friend FooTest` inside the `private` section of `class Foo`, making `FooTest` a friend. Then `FooTest` can access `Foo`’s private members. That way, `CallPrivateSum` can call `foo->_PrivateSum`, cleverly bypassing the private access restriction.

This example primarily shows how `TEST_F` works, including `SetUp` and `TearDown`. The same logic can be done with `TEST` as well, but without a fixture you would need to describe how to set up the environment inside each `TEST` manually.

### Run the Tests

In `CMakeLists.txt`, we define the `test_foo` executable, so we can run `./test_foo` directly.

But since we also defined `add_test` in `CMakeLists.txt`, we can run tests using the CMake command `ctest`:

```bash
googletest-tutorial/build$ ctest
Test project /mnt/c/Users/tiger/googletest-tutorial/build
    Start 1: my_project_test
1/1 Test #1: my_project_test ..................   Passed    0.02 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.05 sec
```

## Conclusion

Overall, writing unit tests for C++ code is a good investment and a healthy habit. It improves code quality and reliability, reduces bugs and maintenance costs, and also helps developers better understand how the code behaves. This post briefly showed how to integrate GoogleTest into a CMake project and write unit tests with it.
