---
title: "C++ のテンプレート関数でコンテナ型を変換する（std::list<int> → std::vector<double>）"
date: 2023-07-02 00:01:00
tags: [c++, template, vector]
des: "本記事では、テンプレート関数を使って C++ のコンテナを別のコンテナ型へ変換する方法（例：std::list<int> を std::vector<double> に変換）を紹介し、複数のテンプレート技法と考え方を段階的に説明します。"
lang: jp
translation_key: template-for-std-containter-conversion
---

![Cover](https://github.com/tigercosmos/blog/assets/18013815/f695f82e-f849-400d-abf0-9c7ab23ebdd2)
（富士山，下吉田本町通）

## 前書き

この記事は、Raymond Chen の記事「[Reordering C++ template type parameters for usability purposes, and type deduction from the future](https://devblogs.microsoft.com/oldnewthing/20230609-00/?p=108318)」を主に翻訳・整理したものです。いくつか誤っているコードを修正し、さらに自分で調べて分かった点も補足しました。

たとえば「コンテナ（Container）型同士の変換がしたい」という要件を考えてみましょう。`std::list<int>` を `std::vector<double>` に変換したい場合、どう実装すればよいでしょうか？

まずは関数を書きますよね。そして、その関数をより汎用的に使えるようにするためにテンプレート（Template）にします。では、テンプレートをどう使えば目的の機能を実現できるか見ていきましょう。

以下の例はすべて `to_vectorN` という名前で示し、`N` は番号です。

## 作法 0：超・陽春版

直感的な作法はこうです：

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

`decltype` でコンテナの要素（`*c.begin()`）の型を取得できますが、その型は参照や const（例：`const T&`）を含むことがあります。`decay_t` を使うと、それらの属性を落とした素の型 `T` を得られます。

そして新しい `std::vector` を作り、`Container &&c` から Vector へ要素を 1 つずつコピーします。

このコードにより、次のように書けます：

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v = to_vector0(l);
```

ただし、このとき `v` は `std::vector<int>` です。中身を別の数値型にしたい場合はどうしましょう？

実用的な関数であれば、出力したい型を指定できるべきです。つまり `to_vector<T>` の構文で直接出力型を指定したい：

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v = to_vector<double>(l); // 期待する書き方
```

しかし、先ほどの `to_vector0` ではできません。Vector の要素型が入力コンテナの要素型と同じになってしまうからです。

## 作法 1：陽春修正版

OK、それならテンプレート引数を 1 つ増やして、要素型も指定できるようにしましょう。

新しいコードはこうなります：

```cpp
template <typename ElementType, typename Container>
auto to_vector1(Container &&c) {
    std::vector<ElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
```

これで目的は達成できます。`std::list<int>` を `std::vector<double>` に変換できます：

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v1 = to_vector1<double>(l);
```

ただし、この方法だと、要素型が同じ（`std::list<int>` → `std::vector<int>`）場合でも型を明示する必要があります。使う側からすると「なんで同じ型をもう一回書かなきゃいけないの？」となります。

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v1 = to_vector1<int>(l); // なぜ int をわざわざ明示する必要がある？
```

## 作法 2：陽春修正版（改）

次のようには書けません。`Container` が `ElementType` より後に定義されるためです。

```cpp
// コンパイルエラー
template <
    typename ElementType
        = std::decay_t<decltype(*std::declval<Container>().begin())>, // Container が未定義
    typename Container> 
auto to_vector2_wrong(Container &&c) {
    std::vector<ElementType> v;
    std::copy(c.begin(), c.end(), std::back_inserter(v));
    return v;
}
```

ただし順序を入れ替えれば次のように書けます：

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

しかしこれはこれで最悪です。今度は常に `Container` の型を明示しなければなりません：

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v2 = to_vector2<std::list<int>&, double>(l);
```

さらに面倒！

## 作法 3：テンプレート引数をうまく使う版

幸い、この問題は難しくありません。`to_vector1` を修正してみましょう：

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

先ほどの「`Container` が未定義で困る」問題に対して、`ElementType` をいったん `void` にしておき、後で `ActualElementType` で置き換えることで巧妙に解決しています。

`ActualElementType` の意味を平易に言い換えると、「`ElementType` がデフォルトの `void` のままなら、利用者が `ElementType` を指定していないので、入力コンテナの要素型を使う。そうでなければ利用者が指定した `ElementType` を使う」ということです。

これで OK！

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
auto v3_1 = to_vector3(l); // std::vector<int>
auto v3_2 = to_vector3<double>(l); // std::vector<double>
```

## 作法 4：カスタムアロケータ対応版

一般的なコンテナは、カスタムアロケータ（Allocator）を使えるようになっています。

ではカスタムアロケータも対応したい場合はどうしましょう？

たとえば `MyAllocator` が次のようにあるとします（ChatGPT に適当に出してもらった例です）：

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

ここで一気に複雑になります。`MyAllocator` も `to_vector` に渡したいので、使い方としては `to_vector4<ElementType, AllocatorType>(container, allocator)` のようになるはずです。

さきほどの `std::conditional_t<std::is_same_v<...>, ...>` のテクニックを使えば、次のように書けます：

```cpp
template <
    typename ElementType = void,
    typename Allocator = void,
    typename Container>
auto to_vector4(
    Container &&c,
    // Allocator の引数型を決める
    std::conditional_t<std::is_same_v<Allocator, void>, // Allocator がデフォルトか？
        std::allocator<std::conditional_t<std::is_same_v<ElementType, void>, // ElementType がデフォルトか？
            std::decay_t<decltype(*std::declval<Container>().begin())>, ElementType>>, // Container 要素型 または ElementType の std::allocator
    Allocator>  // ユーザ定義の Allocator を使う
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

上の引数部分が少し複雑なので、平易に説明します。

`to_vector4` の第 2 引数はカスタムアロケータの型を受け取りますが、その型はテンプレート引数に応じて推定する必要があります。つまり次のロジックになります：

```
Allocator がデフォルトの場合
    ElementType もデフォルトの場合
        Container の要素型の std::allocator を使う
    else
        ElementType の std::allocator を使う
else
    ユーザ定義の Allocator を使う
```

これなら分かりやすいですよね！

## 作法 5：カスタムアロケータ対応の簡潔版

`to_vector4` はさすがにごちゃごちゃしているので、次のように簡潔化できます：

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

使い方は次のとおりです：

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
MyAllocator<double> al;

auto v5_1 = to_vector5(l);
auto v5_2 = to_vector5<double>(l);
auto v5_3 = to_vector5<double, MyAllocator<double>>(l, al);
```

## 作法 6：関数テンプレートのオーバーロード版

ここまでは「単一の関数でテンプレート引数を操作する」方法でしたが、実は関数テンプレートのオーバーロード（Overloading Function Templates）でも解決できます：

```cpp
// 番号一
template <typename ElementType, typename Allocator = void, typename Container>
constexpr auto to_vector6(Container &&c, Allocator al = {}) {
    return std::vector<ElementType, Allocator>(std::begin(c), std::end(c), al);
}
// 番号二
template <typename ElementType, typename Container> constexpr auto to_vector6(Container &&c) {
    return to_vector6<ElementType>(std::forward<Container>(c), std::allocator<ElementType>{});
}
// 番号三
template <typename Container> constexpr auto to_vector6(Container &&c) {
    return to_vector6<
        std::decay_t<typename std::iterator_traits<decltype(std::begin(std::declval<Container>()))>::value_type>>(
        std::forward<Container>(c));
}
```

かなりすっきりした感じがしませんか？

使い方は同じです：

```cpp
std::list<int> l = {1, 2, 3, 4, 5};
MyAllocator<double> al;

auto v6_1 = to_vector6(l); // 番号三を使用
auto v6_2 = to_vector6<double>(l); // 番号二を使用
auto v6_3 = to_vector6<double, MyAllocator<double>>(l, al); // 番号一を使用
```

## 総整理

みんな忙しい（そして面倒くさい）と思うので、全部まとめておきました。コピペして使ってください 😂

クリックすれば見られます〜

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

C++ テンプレートを扱ういくつかのテクニックを議論しながら、コンテナ型を別のコンテナ型へ変換する方法を段階的に理解しました。作法は色々あり、推論（deduction）の過程もなかなか面白いと思います。

本記事の例は C++17 だけで書けます。さらに面白いのは、C++20 で Concept が導入されたことで、実装の可能性がより広がったことです：

```cpp
template <typename ElementType, typename Container>
    requires std::constructible_from<ElementType, std::ranges::range_value_t<Container>>
auto to_vector(Container&& c) {
    return std::vector<ElementType>(std::begin(c), std::end(c));
}
```

ここでは C++20 の `std::constructible_from` と `std::ranges::range_value_t` を使っています。詳しい説明は省略します。ChatGPT に聞けば OK 😜

かっこいいでしょ！
