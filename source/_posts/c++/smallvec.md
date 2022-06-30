---
title: C++ 實作 Small Vector
date: 2022-06-22 15:00:00
tags: [c++, small vector, vector, optimization]
des: "本文簡單介紹 C++ 實作 Small Vector"
---

`std::vector` 應該是所有人使用 C++ 最常使用的容器了，Chandler Carruth 在 「[Efficiency with Algorithms, Performance with Data Structures (2014)](https://www.youtube.com/watch?v=fHNmRkzxHWs)」演講中也提到，絕大多數時候我們基本上只需要考慮使用 `std::vector`，理由是即使像是 `std::unordered_map` 似乎有著 $O(1)$ 的複雜度，但實際上考量到記憶體存取還有 Cache Miss，還不如使用 `std::vector` 的連續記憶體來的好。為了方便，後續都直接用 Vector 指 `std::vector`。

在使用 Vector 的時候，如果我們大概知道數量上限，可以使用 `reserve()` 來事先宣告要使用多少 Heap 記憶體，這樣可以避免重複分配新的記憶體和拷貝，但是我們仍然不能避免第一次使用時分配 Heap 記憶體，只要花一次時間做記憶體分配就會拖慢效能。

事實上，大多數時候我們用 Vector 的時候，很多時候可能數量都不超過 10 個，如果我們早就知道 Vector 只會存取少量元素，那我們應該可以透過用 Inline 記憶體，來避免額外去分配新的 Heap 記憶體。也就是我們讓物件本身擁有一段放在 Stack 記憶體的 Buffer，不需要額外分配記憶體就可以直接使用，這稱為 Small Objects Optimization，原理跟 Small String Optimization 基本上是一樣（可以參考我以前寫的[文章](/post/2022/06/c++/sso/)）。

那你可能會問如果我們只想要少量元素，那為啥不直接宣告 Array 就好了？因為我們同時想要有 Vector 提供的 API 帶來的便利性，像是可以用 `push_back()` 來放元素。於是乎，針對少量元素特製的 Small Vector 就很常可以在各大專案中看到，以前看的時候都不懂為啥要特地弄個 Small Vector 不直接使用 Vector 就好，原因就是我們希望效能快還要再快！

## 簡易版 Small Vector

我們先來實作最簡單版的 Small Vector，只包含一個基礎的建構子還有 `push_back` 函數。

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

如果元素小於 `N` 的話，就可以用 `SmallVec` 本身自帶的 `m_data` 來儲存，就可以省去分配一次 Heap 記憶體的時間。這樣做的成果是有效的，我們用 [Quick C++ Benchmark](https://quick-bench.com/) 執行以下來測時間：

```cpp
int K = 10; // 不超過 Small Vector 的 N

static void SmallVector(benchmark::State& state) {
  for (auto _ : state) {
    SmallVec<int> sv(0); // 使用 inline memory
    for(int i = 0 ; i < K; i++) {
        sv.push_back(i);
    }
  }
}
BENCHMARK(SmallVector);


static void StdVector(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<int> v;
    v.reserve(K); // 分配新的 Heap 記憶體
    for(int i = 0 ; i < K; i++) {
        v.push_back(i);
    }
  }
}
BENCHMARK(StdVector);
```

64bits,i3-10100,GCC11.2 執行環境下得到下圖（越低越好），證實 Small Vector 在少量元素的操作下確實會比 Vector 來的有效率，大概快了 2.6 倍，主要就是因為 Vector 需要多花分配新的 Heap 記憶體的時間。

![comparison between small vector and std vector](https://user-images.githubusercontent.com/18013815/174789476-7c2693bd-55b1-47cd-a588-2be5c36d8dbc.png)

> 關於 Small Vector 效能分析，楊志璿有提供[更詳細的討論](https://hackmd.io/@25077667/r1uUxVY59)，有興趣可以參考！

## 完整版 Small Vector

簡單版的 Small Vector 已經證實了效能確實會比較好，我就順手練習把完整的 Small Vector 也實作出來。

完整的程式碼可以在文末點開檢視，或是去 [Gist 下載](https://gist.github.com/tigercosmos/4d939e90a350071424567b8ed4d9a378)。

簡單解釋一下程式碼，如果操作的元素數量小於預設 inline 的 `N`，就會直接使用內建的 `std::array`，如果超過的時候就會去 `new` 新的記憶體空間。剩下比較值得注意的就是在 Copy 或做 Move Semantics 的時候，指向儲存位置的 `m_head` 的操作稍微複雜一點。另外這邊實作的 Small Vector 僅適用基礎型別（`int`, `float`, etc.），這也符合會使用 Small Vector 的常見情景。

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


