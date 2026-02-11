---
title: "Converting Between Container Types with C++ Template Functions (std::list<int> to std::vector<double>)"
date: 2023-07-02 00:01:00
tags: [c++, template, vector]
des: "This post explains how to use template functions to convert C++ containers into other container types (e.g., converting std::list<int> to std::vector<double>), and walks through multiple template techniques and ways of thinking."
lang: en
translation_key: template-for-std-containter-conversion
---

![Cover](https://github.com/tigercosmos/blog/assets/18013815/f695f82e-f849-400d-abf0-9c7ab23ebdd2)
(Mt. Fuji, Shimoyoshida Honcho-dori)

## Preface

This post is mainly a translated and organized version of Raymond Chen’s article: “[Reordering C++ template type parameters for usability purposes, and type deduction from the future](https://devblogs.microsoft.com/oldnewthing/20230609-00/?p=108318)”. I fixed some incorrect code and also added a few things I discovered while digging into the topic.

Imagine you have a requirement to convert between container types. For example, you have a `std::list<int>` and you want to convert it into a `std::vector<double>`. How can we do it?

First, we obviously need to write a function. And to make that function reusable, we want to make it a template. So let’s see how we can use templates to achieve what we want.

All examples below are named `to_vectorN`, where `N` is a number indicating the version.

## Approach 0: The Super Barebones Version

A very direct approach:

```cpp
template<typename Container>
auto to_vector0(Container&& c)
{
    using ElementType = std::decay_t<decltype(*c.begin())>;
    std::vector<ElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
```

`decltype` can obtain the type of the container element (`*c.begin()`), but that type could be a reference or `const`, e.g. `const T&`. With `decay_t`, we can get the plain type `T` without those extra qualifiers.

Then we create a new `std::vector`, and copy elements one by one from `Container &&c` into the vector.

With the code above, we can do:

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v = to_vector0(l);
```

But now `v` is a `std::vector<int>`. What if we want the output element type to be another numeric type?

A more practical function should let us specify the desired output type. Ideally, we want a `to_vector<T>` syntax to directly specify the output element type:

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v = to_vector<double>(l); // desired usage
```

However, our previous example `to_vector0` cannot do this, because the vector element type is always the same as the input container’s element type.

## Approach 1: Barebones Version (Modified)

OK, then let’s add another template parameter to specify the element type.

The code becomes:

```cpp
template <typename ElementType, typename Container>
auto to_vector1(Container &&c) {
    std::vector<ElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
```

`to_vector1` achieves what we want: it can successfully convert `std::list<int>` into `std::vector<double>`:

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v1 = to_vector1<double>(l);
```

But now even if the element type is the same—like converting `std::list<int>` to `std::vector<int>`—you still have to spell out the type. That’s annoying for users of your function. Why do they have to type the same thing again?

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v1 = to_vector1<int>(l); // why do we have to explicitly write int?
```

## Approach 2: Barebones Version (Modified Again)

We cannot write this, because `Container` is declared later than `ElementType`:

```cpp
// compilation error
template <
    typename ElementType
        = std::decay_t<decltype(*std::declval<Container>().begin())>, // Container not defined yet
    typename Container> 
auto to_vector2_wrong(Container &&c) {
    std::vector<ElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
```

But we can swap the order:

```cpp
template <
    typename Container,
    typename ElementType
        = std::decay_t<decltype(*std::declval<Container>().begin())>>
auto to_vector2(Container &&c) {
    std::vector<ElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
```

Unfortunately, this is even worse, because now you always need to explicitly specify the `Container` type:

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v2 = to_vector2<std::list<int>&, double>(l);
```

Even more annoying!

## Approach 3: Leverage Template Parameters Properly

Luckily, this problem is not hard to solve. Let’s go back and modify `to_vector1`:

```cpp
template <typename ElementType = void, typename Container> 
auto to_vector3(Container &&c) {
    using ActualElementType =
        std::conditional_t<std::is_same_v<ElementType, void>, std::decay_t<decltype(*c.begin())>, ElementType>;
    std::vector<ActualElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
```

For the earlier case where `ElementType` would require `Container` but `Container` is not yet defined, we set `ElementType` to `void` first, and then replace it via `ActualElementType`. This cleverly solves the problem.

Let’s translate what `ActualElementType` means: if `ElementType` is still the default `void`, it means the user did not provide `ElementType` when using the template, so we use the input container’s element type as the default. Otherwise, we use the user-provided `ElementType`.

Now it works!

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v3_1 = to_vector3(l); // std::vector<int>
auto v3_2 = to_vector3<double>(l); // std::vector<double>
```

## Approach 4: Custom Allocator Version

Most container libraries allow a custom allocator.

What if we also want to support a custom allocator?

Assume we already have a `MyAllocator`, like this (generated by ChatGPT just for the example):

```cpp
template<typename T>
class MyAllocator {
public:
    using value_type = T;
    MyAllocator() noexcept {}
    ~MyAllocator() noexcept {}
    T* allocate(std::size_t n) { return static_cast<T*>(::operator new(n * sizeof(T)));}
    void deallocate(T* p, std::size_t n) { ::operator delete(p);}
    template<typename U, typename... Args> void construct(U* p, Args&&... args) { ::new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);}
    template<typename U> void destroy(U* p) { p->~U();}
};
```

Now it becomes more complex, because you need to pass the custom `MyAllocator` into `to_vector` as well. The desired usage would be `to_vector4<ElementType, AllocatorType>(container, allocator)`.

Using the `std::conditional_t<std::is_same_v<...>, ...>` trick we just learned, we can write:

```cpp
template <
    typename ElementType = void,
    typename Allocator = void,
    typename Container>
auto to_vector4(
    Container &&c,
    // decide the parameter type of Allocator
    std::conditional_t<std::is_same_v<Allocator, void>, // whether Allocator is default
        std::allocator<std::conditional_t<std::is_same_v<ElementType, void>, // whether ElementType is default
            std::decay_t<decltype(*std::declval<Container>().begin())>, ElementType>>, // std::allocator of Container element type or ElementType
    Allocator>  // use custom Allocator
    al = {}
) {
    using ActualElementType =
        std::conditional_t<std::is_same_v<ElementType, void>,
                           std::decay_t<decltype(*std::declval<Container>().begin())>, ElementType>;
    using ActualAllocator =
        std::conditional_t<std::is_same_v<Allocator, void>, std::allocator<ActualElementType>, Allocator>;
    std::vector<ActualElementType, ActualAllocator> v(al);
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
```

The parameter definition is a bit complex, so let’s explain it in plain language.

The second parameter of `to_vector4` takes the custom allocator type, and we need to deduce it based on the template parameters. So the logic becomes:

```
if Allocator uses the default
    if ElementType uses the default
        use std::allocator of the container's element type
    else
        use std::allocator of ElementType
else
    use the user-defined Allocator
```

Much clearer, right?

## Approach 5: Cleaner Version with Custom Allocator Support

Since `to_vector4` is pretty messy, we can simplify it into the following version:

```cpp
template <typename ElementType = void,
          typename Allocator = void,
          typename Container,
          typename ActualElementType =
              std::conditional_t<std::is_same_v<ElementType, void>,
                                 std::decay_t<decltype(*std::declval<Container>().begin())>, ElementType>,
          typename ActualAllocator =
              std::conditional_t<std::is_same_v<Allocator, void>, std::allocator<ActualElementType>, Allocator>>
auto to_vector5(Container &&c, ActualAllocator al = ActualAllocator()) {
    std::vector<ActualElementType, ActualAllocator> v(al);
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
```

Usage:

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
MyAllocator<double> al;

auto v5_1 = to_vector5(l);
auto v5_2 = to_vector5<double>(l);
auto v5_3 = to_vector5<double, MyAllocator<double>>(l, al);
```

## Approach 6: Overloading Function Templates

All approaches above rely on a single function while manipulating template parameters. In practice, we can also solve it by overloading function templates:

```cpp
// version 1
template <typename ElementType, typename Allocator = void, typename Container>
constexpr auto to_vector6(Container &&c, Allocator al = {}) {
    return std::vector<ElementType, Allocator>(std::begin(c), std::end(c), al);
}
// version 2
template <typename ElementType, typename Container> constexpr auto to_vector6(Container &&c) {
    return to_vector6<ElementType>(std::forward<Container>(c), std::allocator<ElementType>{});
}
// version 3
template <typename Container> constexpr auto to_vector6(Container &&c) {
    return to_vector6<
        std::decay_t<typename std::iterator_traits<decltype(std::begin(std::declval<Container>()))>::value_type>>(
        std::forward<Container>(c));
}
```

Doesn’t it look much cleaner?

The usage stays the same:

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
MyAllocator<double> al;

auto v6_1 = to_vector6(l); // uses version 3
auto v6_2 = to_vector6<double>(l); // uses version 2
auto v6_3 = to_vector6<double, MyAllocator<double>>(l, al); // uses version 1
```

## Summary

I know everyone is lazy, so I organized everything for you. You can copy it directly 😂

Just expand it to see it~

<details>

```cpp
// g++ test.cpp -std=c++17

#include <iostream>
#include <list>
#include <vector>

template <typename T> class MyAllocator {
  public:
    using value_type = T;
    MyAllocator() noexcept {}
    ~MyAllocator() noexcept {}
    T *allocate(std::size_t n) { return static_cast<T *>(::operator new(n * sizeof(T))); }
    void deallocate(T *p, std::size_t n) { ::operator delete(p); }
    template <typename U, typename... Args> void construct(U *p, Args &&...args) {
        ::new (reinterpret_cast<void *>(p)) U(std::forward<Args>(args)...);
    }
    template <typename U> void destroy(U *p) { p->~U(); }
};
/* ---------------------------------------------------------------------------*/
template <typename Container> auto to_vector0(Container &&c) {
    using ElementType = std::decay_t<decltype(*c.begin())>;
    std::vector<ElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
/* ---------------------------------------------------------------------------*/
template <typename ElementType, typename Container> auto to_vector1(Container &&c) {
    std::vector<ElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
/* ---------------------------------------------------------------------------*/
template <typename Container, typename ElementType = std::decay_t<decltype(*std::declval<Container>().begin())>>
auto to_vector2(Container &&c) {
    std::vector<ElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
/* ---------------------------------------------------------------------------*/
template <typename ElementType = void, typename Container> auto to_vector3(Container &&c) {
    using ActualElementType =
        std::conditional_t<std::is_same_v<ElementType, void>, std::decay_t<decltype(*c.begin())>, ElementType>;
    std::vector<ActualElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
/* ---------------------------------------------------------------------------*/
template <typename ElementType = void, typename Allocator = void, typename Container>
auto to_vector4(
    Container &&c,
    std::conditional_t<
        std::is_same_v<Allocator, void>,
        std::allocator<std::conditional_t<std::is_same_v<ElementType, void>,
                                          std::decay_t<decltype(*std::declval<Container>().begin())>, ElementType>>,
        Allocator>
        al = {}) {
    using ActualElementType =
        std::conditional_t<std::is_same_v<ElementType, void>,
                           std::decay_t<decltype(*std::declval<Container>().begin())>, ElementType>;
    using ActualAllocator =
        std::conditional_t<std::is_same_v<Allocator, void>, std::allocator<ActualElementType>, Allocator>;
    std::vector<ActualElementType, ActualAllocator> v(al);
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
/* ---------------------------------------------------------------------------*/
template <typename ElementType = void, typename Allocator = void, typename Container,
          typename ActualElementType =
              std::conditional_t<std::is_same_v<ElementType, void>,
                                 std::decay_t<decltype(*std::declval<Container>().begin())>, ElementType>,
          typename ActualAllocator =
              std::conditional_t<std::is_same_v<Allocator, void>, std::allocator<ActualElementType>, Allocator>>
auto to_vector5(Container &&c, ActualAllocator al = ActualAllocator()) {
    std::vector<ActualElementType, ActualAllocator> v(al);
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
/* ---------------------------------------------------------------------------*/
template <typename ElementType, typename Allocator = void, typename Container>
constexpr auto to_vector6(Container &&c, Allocator al = {}) {
    return std::vector<ElementType, Allocator>(std::begin(c), std::end(c), al);
}
template <typename ElementType, typename Container> constexpr auto to_vector6(Container &&c) {
    return to_vector6<ElementType>(std::forward<Container>(c), std::allocator<ElementType>{});
}
template <typename Container> constexpr auto to_vector6(Container &&c) {
    return to_vector6<
        std::decay_t<typename std::iterator_traits<decltype(std::begin(std::declval<Container>()))>::value_type>>(
        std::forward<Container>(c));
}
/* ---------------------------------------------------------------------------*/
int main() {
    MyAllocator<double> al;
    std::list<int> l = {1, 2, 3, 4, 5};

    auto v1 = to_vector1<double>(l);

    auto v2 = to_vector2<std::list<int> &, double>(l);

    auto v3_1 = to_vector3(l);
    auto v3_2 = to_vector3<double>(l);

    auto v4_1 = to_vector5(l);
    auto v4_2 = to_vector5<double>(l);
    auto v4_3 = to_vector5<double, MyAllocator<double>>(l, al);

    auto v5_1 = to_vector5(l);
    auto v5_2 = to_vector5<double>(l);
    auto v5_3 = to_vector5<double, MyAllocator<double>>(l, al);

    auto v6_1 = to_vector6(l);
    auto v6_2 = to_vector6<double>(l);
    auto v6_3 = to_vector6<double, MyAllocator<double>>(l, al);
}
```

</details>

## Conclusion

By discussing several techniques for working with C++ templates, we walked step by step through how to convert one container type into another. You can see there are many possible approaches, and the deduction process itself is pretty fun.

The examples in this post only require C++17. Even more interestingly, after C++20 introduced Concepts, there are even more ways to implement this:

```cpp
template <typename ElementType, typename Container>
    requires std::constructible_from<ElementType, std::ranges::range_value_t<Container>>
auto to_vector(Container&& c) {
    return std::vector<ElementType>(std::begin(c), std::end(c));
}
```

This uses C++20’s `std::constructible_from` and `std::ranges::range_value_t`. I won’t go into details here—you can ask ChatGPT 😜

Cool, right?
