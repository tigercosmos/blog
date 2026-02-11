---
title: "C++ で Small Vector を実装する"
date: 2022-06-22 15:00:00
tags: [c++, small vector, vector, optimization]
des: "本記事では C++ における Small Vector の実装を簡単に紹介します。"
lang: jp
translation_key: smallvec
---

`std::vector` は、おそらく C++ で最もよく使われるコンテナです。Chandler Carruth も講演「[Efficiency with Algorithms, Performance with Data Structures (2014)](https://www.youtube.com/watch?v=fHNmRkzxHWs)」で、ほとんどの場合は `std::vector` の利用だけを考えればよい、と述べています。理由は、`std::unordered_map` のように計算量が $O(1)$ に見えるものでも、実際にはメモリアクセスやキャッシュミスの影響を考えると、連続メモリを持つ `std::vector` のほうが有利な場面が多いからです。便宜上、以降は Vector という語で `std::vector` を指します。

Vector を使うとき、要素数の上限がだいたい分かっているなら、`reserve()` でヒープメモリを事前確保できます。これにより、拡張のたびに新しい領域を再確保してコピーするコストを避けられます。ただし、最初にヒープを確保するコスト自体は避けられません。たった 1 回でもアロケーションが入ると、性能を下げる要因になります。

実際には、Vector に入れる要素数が 10 個を超えないケースも多いです。もし Vector が少量の要素しか保持しないことが分かっているなら、インラインメモリ（inline storage）を使って、追加のヒープ確保を避けられます。つまり、オブジェクト自身がスタック上にバッファを持ち、追加のアロケーションなしにそのまま使う方式です。これは Small Object Optimization と呼ばれ、原理は Small String Optimization と基本的に同じです（以前書いた[記事](/post/2022/06/c++/sso/)も参考にしてください）。

「少量要素なら配列（Array）を宣言すればいいのでは？」と思うかもしれません。しかし、`push_back()` のような Vector の API が提供する利便性も同時に欲しいわけです。こうした理由から、少量要素向けに特化した Small Vector は多くのプロジェクトでよく見かけます。以前は「なぜわざわざ Small Vector を作るのか。Vector でいいじゃん」と思っていましたが、要するに「速さをさらに追求したい」からです。

## 簡易版 Small Vector

まずは最も簡単な Small Vector を実装します。基本的なコンストラクタと `push_back` 関数だけを持つものです。

```cpp
template <typename T, size_t N = 10>
class SmallVec
{
public:

    explicit SmallVec(size_t size) : m_size(size)
    {
        if (size > N)
        {
            m_capacity = m_size;
            m_head = new T[m_capacity];
        }
        else
        {
            m_capacity = N;
            m_head = m_data.data();
        }
    }

    void push_back(T const & value)
    {
        if (m_capacity == m_size)
        {
            m_capacity *= 2;
            T *tmp = new T[m_capacity];
            std::copy_n(m_head, m_size, tmp);
            if (m_head != m_data.data())
            {
                delete[] m_head;
            }
            m_head = tmp;
        }
        m_head[m_size++] = value;
    }

private:
    T *m_head = nullptr;
    size_t m_size = 0;
    size_t m_capacity = N;
    std::array<T, N> m_data;
};
```

要素数が `N` 未満なら、`SmallVec` 自身が持つ `m_data` に格納できるため、最初のヒープ確保 1 回分を省けます。これは効果があります。[Quick C++ Benchmark](https://quick-bench.com/) で次のコードを実行して時間を測ってみます：

```cpp
int K = 10; // Small Vector の N を超えない

static void SmallVector(benchmark::State& state) {
  for (auto _ : state) {
    SmallVec<int> sv(0); // inline memory を使用
    for(int i = 0 ; i < K; i++) {
        sv.push_back(i);
    }
  }
}
BENCHMARK(SmallVector);


static void StdVector(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<int> v;
    v.reserve(K); // 新しいヒープメモリを確保
    for(int i = 0 ; i < K; i++) {
        v.push_back(i);
    }
  }
}
BENCHMARK(StdVector);
```

64bit / i3-10100 / GCC 11.2 の環境で次の結果（小さいほど良い）が得られます。少量要素の操作では Small Vector のほうが Vector より効率的であることが確認でき、だいたい 2.6 倍ほど速くなっています。主な理由は、Vector がヒープ確保に余分な時間を使うためです。

![comparison between small vector and std vector](https://user-images.githubusercontent.com/18013815/174789476-7c2693bd-55b1-47cd-a588-2be5c36d8dbc.png)

> Small Vector の性能解析については、楊志璿が[より詳細な議論](https://hackmd.io/@25077667/r1uUxVY59)を提供しています。興味があれば参照してください。

## 完整版 Small Vector

簡易版で性能が良いことは確認できたので、ついでに完全版の Small Vector も練習として実装しました。

完全なコードは記事末尾で展開して確認できます。または [Gist からダウンロード](https://gist.github.com/tigercosmos/4d939e90a350071424567b8ed4d9a378)してください。

コードを簡単に説明すると、操作対象の要素数がデフォルトのインライン容量 `N` を下回るときは内蔵の `std::array` を使い、`N` を超えると `new` で新しいメモリ領域を確保します。注意点として、コピーや Move Semantics の際に、格納先を指す `m_head` の扱いがやや複雑になります。また、ここで実装した Small Vector は基礎型（`int`, `float` など）向けであり、これは Small Vector が使われる典型的なシーンとも一致します。

<details>

```cpp
#include <array>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <iostream>
#include <initializer_list>

template <typename T, size_t N = 10>
class SmallVec
{
public:
    using value_type = T;
    using iterator = T *;
    using const_iterator = T *const;

    SmallVec() = default;

    explicit SmallVec(size_t size) : m_size(size)
    {
        if (size > N)
        {
            m_capacity = m_size;
            m_head = new T[m_capacity];
        }
        else
        {
            m_capacity = N;
            m_head = m_data.data();
        }
    }

    SmallVec(SmallVec const &other) : m_size(other.m_size)
    {
        if (other.m_head == other.m_data.data())
        {
            m_capacity = N;
            m_head = m_data.data();
        }
        else
        {
            m_capacity = m_size;
            m_head = new T[m_capacity];
        }

        std::copy_n(other.m_head, m_size, m_head);
    }

    SmallVec(SmallVec &&other) noexcept : m_size(other.m_size)
    {
        if (m_head == other.m_data.data())
        {
            m_capacity = other.m_capacity;
            m_head = m_data.data();
            std::copy_n(other.m_head, m_size, m_head);
        }
        else
        {
            m_capacity = m_size;
            m_head = other.m_head;

            other.m_capacity = N;
            other.m_size = 0;
            other.m_head = other.m_data.data();
        }

        std::copy_n(other.m_head, m_size, m_head);
    }

    SmallVec(std::initializer_list<T> init_list)
    {
        m_size = init_list.size();
        if (m_size > N)
        {
            m_capacity = m_size;
            m_head = new T[m_capacity];
        }
        else
        {
            m_capacity = N;
            m_head = m_data.data();
        }
        std::copy_n(init_list.begin(), m_size, m_head);
    }

    SmallVec &operator=(SmallVec const &other)
    {
        if (this == &other)
            return *this;

        if (other.m_head == other.m_data.data())
        {
            if (m_head != m_data.data())
            {
                delete[] m_head;
                m_head = m_data.data();
            }
            m_capacity = N;
            m_size = other.m_size;
        }
        else
        {
            if (m_capacity < other.m_size)
            {
                delete[] m_head;
                m_head = nullptr;
            }
            if (m_head == nullptr || m_head == m_data.data())
            {
                m_capacity = other.m_size;
                m_head = new T[m_capacity];
            }
            m_size = other.m_size;
        }

        std::copy_n(other.m_head, m_size, m_head);
        return *this;
    }

    SmallVec &operator=(SmallVec &&other) noexcept
    {
        if (this == &other)
            return *this;

        if (other.m_head == other.m_data.data())
        {
            if (m_head != m_data.data())
            {
                delete[] m_head;
                m_head = m_data.data();
            }
            m_capacity = N;
            m_size = other.m_size;
            std::copy_n(other.m_head, m_size, m_head);
        }
        else
        {
            m_head = other.m_head;
            m_capacity = other.m_capacity;
            m_size = other.m_size;

            other.m_head = other.m_data.data();
            other.m_capacity = N;
            other.size = 0;
        }

        return *this;
    }

    void push_back(T const &value)
    {
        if (m_capacity == m_size)
        {
            m_capacity *= 2;
            T *tmp = new T[m_capacity];
            std::copy_n(m_head, m_size, tmp);
            if (m_head != m_data.data())
            {
                delete[] m_head;
            }
            m_head = tmp;
        }
        m_head[m_size++] = value;
    }

    void pop_back()
    {
        if (m_size == 0)
        {
            throw std::runtime_error("small vector underflow");
        }

        back().~T();
        m_size--;
    }

    T const &operator[](size_t it) const { return m_head[it]; }
    T &operator[](size_t it) { return m_head[it]; }

    size_t size() noexcept { return m_size; }
    size_t capacity() noexcept { return m_capacity; }
    iterator begin() noexcept { return m_head; }
    iterator end() noexcept { return m_head + m_size; }
    const_iterator begin() const noexcept { return m_head; }
    const_iterator end() const noexcept { return m_head + m_size; }

    T const &back() const { return m_head[m_size - 1]; }
    T &back() { return m_head[m_size - 1]; }

    friend std::ostream &operator<<(std::ostream &os, const SmallVec &sv)
    {
        os << '[';
        for (auto v : sv)
        {
            os << v << ' ';
        }
        os << ']';
        return os;
    }

    ~SmallVec()
    {
        if (m_head != m_data.data() && m_head != nullptr)
        {
            delete[] m_head;
        }
    }

private:
    T *m_head = nullptr;
    size_t m_size = 0;
    size_t m_capacity = N;
    std::array<T, N> m_data;
};

int main()
{
    // constructor, stack
    {
        SmallVec<int> sv1;
        SmallVec<int> sv2(sv1);
        SmallVec<int> sv3(std::move(sv1));
        SmallVec<int> sv4 = sv3;
        SmallVec<int> sv5 = std::move(sv4);
    }
    // constructor, heap
    {
        SmallVec<int> sv1(20);
        SmallVec<int> sv2(sv1);
        SmallVec<int> sv3(std::move(sv1));
        SmallVec<int> sv4 = sv3;
        SmallVec<int> sv5 = std::move(sv4);
    }
    // push_back
    {
        SmallVec<int> sv1(3);
        std::cout << sv1 << std::endl;
        // [X, X, X], where X are arbitrary numbers

        for (int i = 0; i < 11; i++)
        {
            sv1.push_back(1);
        }
        std::cout << sv1 << std::endl;
        // [X X X 1 1 1 1 1 1 1 1 1 1 ]
    }
    // pop_back
    {
        SmallVec<int> sv1(20);
        std::cout << "size: " << sv1.size() << ", capacity: " << sv1.capacity() << std::endl;
        // size: 20, capacity: 20
        while (sv1.size())
        {
            sv1.pop_back();
        }
        std::cout << "size: " << sv1.size() << ", capacity: " << sv1.capacity() << std::endl;
        // size: 0, capacity: 20
    }
}
```

</details>
