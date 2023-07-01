---
title: 如何用 C++ 模版函數將容器輸出成不同容器型別（std::list<int> 轉成 std::vector<double>）
date: 2023-07-02 00:01:00
tags: [c++, template, vector]
des: "本文介紹如何使用模版函數將 C++ 的容器輸出成其他的容器型別，例如將 std::list<int> 輸出成 std::vector<double>，說明多種模版的操作方式以及思路。"
---

![Cover](https://github.com/tigercosmos/blog/assets/18013815/f695f82e-f849-400d-abf0-9c7ab23ebdd2)
（富士山，下吉田本町通）

## 前言

這篇文章主要是翻譯整理 Raymond Chen 所寫的文章「[Reordering C++ template type parameters for usability purposes, and type deduction from the future](https://devblogs.microsoft.com/oldnewthing/20230609-00/?p=108318)」，修正了一些錯誤的程式碼，然後補上我自己鑽研時候的一些發現。

想像一下，今天你有個需求，要做容器（Container）型別之間的轉換，例如有個 `std::list<int>` 你想要把他換成 `std::vector<double>`，這時候我們可以怎麼做？

首先我們一定要寫一個函式對吧，然後為了讓我們函式更廣泛的使用，我們還要加上模版（Template），接著就讓我們來看可以怎用用模版還實現我們想要的功能。

以下的所有範例程式都會以 `to_vectorN` 來標明，其中 `N` 為數字代表編號。

## 作法 0 超級陽春版

所以一個直觀的作法：
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

`decltype` 可以得到容器的元素（`*c.begin()`）的型別，但這型別可能是一個參考或是 const，例如 `const T&`，而使用 `decay_t` 可以得到什麼其他屬性都沒有的型別 `T`。

然後我們就建立一個新的 `std::vector`，把資料一個一個從 `Container &&c` 拷貝到 Vector 裡面。

透過上面程式碼，我們可以做到：

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v = to_vector0(l);
```

但這時候 `v` 是 `std::vector<int>`，如果我們想要裡面是其他數值型別呢？

一個比較實用函式應該可以讓我們指定想要的型別，所以我們希望可以使用 `to_vector<T>` 語法來直接指定輸出的型別。

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v = to_vector<double>(l); // 期待的用法
```

但是我們前面的範例程式 `to_vector0` 卻辦不到，因為 Vector 的元素型別會跟給定的 Container 元素型別一樣。

## 作法 1 陽春修改版

OK，那我們就多給 Template 一個參數吧，順便指定元素型別。

於是乎，新的程式碼會長的像以下：

```cpp
template <typename ElementType, typename Container>
auto to_vector1(Container &&c) {
    std::vector<ElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
```

`to_vector1` 可以達到我們的目的了，可以順利把 `std::list<int>` 轉成 `std::vector<double>`

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v1 = to_vector1<double>(l);
```

可是這下連元素型別一樣，例如  `std::list<int>` 轉成 `std::vector<int>` 都得申明型別，這下用你程式的人就不樂意了，為啥要多打一次一樣的東西？

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v1 = to_vector1<int>(l); // 為啥 int 還要特地聲明？
```

## 作法 2 陽春修改版（改）

可是我們無法寫成以下，因為 `Container` 比 `ElementType` 還晚定義。

```cpp
// 編譯錯誤
template <
    typename ElementType
        = std::decay_t<decltype(*std::declval<Container>().begin())>, // Container 未定義
    typename Container> 
auto to_vector2_wrong(Container &&c) {
    std::vector<ElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
```

但我們可以把順序交換，變成以下：

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

不過這樣更慘，因為這下以後都得特地申明 Container 的型別了。

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v2 = to_vector2<std::list<int>&, double>(l);
```

變的更麻煩！

## 作法 3 善用模版參數版

好險其實要解決這問題不難，我們回去修改一下 `to_vector1`：

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

對於剛剛 `ElementType` 會有 `Container` 還沒定義的情況，我們先把 `ElementType` 設定成 `void`，之後再用 `ActualElementType` 去取代，巧妙解決了問題！

翻譯一下 `ActualElementType` 的語意，如果 `ElementType` 還是預設的 `void`，代表使用者在使用 Template 時沒有填入 `ElementType`，那就是採用預設的 `Container` 的元素型別，否則就用使用者提供的 `ElementType` 型別。

這次沒問題了！

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v3_1 = to_vector3(l); // std::vector<int>
auto v3_2 = to_vector3<double>(l); // std::vector<double>
```

## 作法 4 自定義配置器版

一般的容器函式庫都會允許使用客製化的配置器（Allocator）。

如果還想要加上自定義的 Allocator 怎麼辦？

假設我們已經有個 `MyAllocator`，長的像以下（隨便讓 ChatGPT 給我生出來的）：

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

情況變的更複雜了，因為你會需要將自定義的 `MyAllocator` 也傳入 `to_vector`，用法應該要是 `to_vector4<ElementType, AllocatorType>(container, allocator)`。

套用我們剛剛學過的 `std::conditional_t<std::is_same_v<...>, ...>` 技巧，我們可以得到以下的程式碼：

```cpp
template <
    typename ElementType = void,
    typename Allocator = void,
    typename Container>
auto to_vector4(
    Container &&c,
    // 決定 Allocator 的參數型別
    std::conditional_t<std::is_same_v<Allocator, void>, // Allocator 是否為預設
        std::allocator<std::conditional_t<std::is_same_v<ElementType, void>, // ElementType 是否為預設
            std::decay_t<decltype(*std::declval<Container>().begin())>, ElementType>>, // Container 元素型別或 ElementType 的 std::allocator
    Allocator>  //採用自定義的 Allocator
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

上面參數定義的地方有點複雜，讓我們用白話文解釋。

`to_vector4` 的第二個參數要收自定義的 Allocator 的型別，我們得依據 Template 的型別去做推定，於是乎會有一下的邏輯：

```
if Allocator 採用預設
    if ElementType 採用預設
        使用 Container 的元素型別的 std::allocator
    else
        使用 ElementType 的 std::allocator
else
    使用使用者定義的 Allocator
```

是不是清楚多了呢！

## 作法 5 支援自定義配置器簡潔版

因為 `to_vector4` 這樣寫真的很亂，上面程式可以簡化成下面版本：

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

使用方法為：

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
MyAllocator<double> al;

auto v5_1 = to_vector5(l);
auto v5_2 = to_vector5<double>(l);
auto v5_3 = to_vector5<double, MyAllocator<double>>(l, al);
```

## 作法 6 函式模版多載版

上面都是單一函數透過操控模版的參數的方法，事實上我們也可以透過函式模版多載（Overloading Function Templates）就好：

```cpp
// 編號一
template <typename ElementType, typename Allocator = void, typename Container>
constexpr auto to_vector6(Container &&c, Allocator al = {}) {
    return std::vector<ElementType, Allocator>(std::begin(c), std::end(c), al);
}
// 編號二
template <typename ElementType, typename Container> constexpr auto to_vector6(Container &&c) {
    return to_vector6<ElementType>(std::forward<Container>(c), std::allocator<ElementType>{});
}
// 編號三
template <typename Container> constexpr auto to_vector6(Container &&c) {
    return to_vector6<
        std::decay_t<typename std::iterator_traits<decltype(std::begin(std::declval<Container>()))>::value_type>>(
        std::forward<Container>(c));
}
```

是不是感覺乾淨很多？

用法依舊一樣：

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
MyAllocator<double> al;

auto v6_1 = to_vector6(l); // 使用編號三
auto v6_2 = to_vector6<double>(l); // 使用編號二
auto v6_3 = to_vector6<double, MyAllocator<double>>(l, al); // 使用編號一
```

## 總整理

我知道大家很懶，幫你都整理在一起了，可以直接拿去抄 😂

點開就可以看了～

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


## 結論

藉由討論如何操作 C++ 模版的幾種技巧，我們一步一步了解如何將一個容器型別轉換成另一個容器型別，可以發現作法可以有很多種，推導的過程也挺好玩的。

不過本文的範例只需用到 C++17，更有趣的是，在 C++ 20 之後出了 Concept 的概念，又有更多的實作可能性：

```cpp
template <typename ElementType, typename Container>
    requires std::constructible_from<ElementType, std::ranges::range_value_t<Container>>
constexpr auto to_vector(Container&& c) {
    return std::vector<ElementType>(std::begin(c), std::end(c));
}
```

這邊用到了 C++ 20 的 `std::constructible_from`，`std::ranges::range_value_t`，這邊就不多做介紹了，可以問一下 ChatGPT 😜

帥吧！
