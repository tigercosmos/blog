---
title: "Implementing a Small Vector in C++"
date: 2022-06-22 15:00:00
tags: [c++, small vector, vector, optimization]
des: "This post briefly introduces how to implement a Small Vector in C++."
lang: en
translation_key: smallvec
---

`std::vector` is probably the most frequently used container in C++. Chandler Carruth also mentioned in his talk “[Efficiency with Algorithms, Performance with Data Structures (2014)](https://www.youtube.com/watch?v=fHNmRkzxHWs)” that in the vast majority of cases, we basically only need to consider using `std::vector`. The reason is that even though `std::unordered_map` seems to have $O(1)$ complexity, in practice—considering memory access patterns and cache misses—it may not be as good as the contiguous memory layout of `std::vector`. For convenience, I will use “Vector” to refer to `std::vector` in the rest of this post.

When using Vector, if we roughly know the upper bound on the number of elements, we can call `reserve()` to pre-allocate heap memory. This avoids repeated allocations and copies when the container grows. However, we still cannot avoid the first heap allocation: as long as you allocate once, the allocation cost can slow things down.

In fact, in many situations when we use Vector, the number of elements is often no more than 10. If we know ahead of time that the Vector will store only a small number of elements, we should be able to avoid allocating heap memory by using inline storage. In other words, we let the object itself own a buffer on the stack, so we can use it directly without extra allocations. This is called Small Object Optimization, and the idea is essentially the same as Small String Optimization (you can refer to my previous [post](/post/2022/06/c++/sso/)).

You may ask: if we only want a small number of elements, why not just declare an array? Because we also want the convenience of the Vector API, such as `push_back()`. Therefore, specialized “Small Vector” implementations for small element counts are commonly found in many large projects. In the past, I did not understand why people would implement a Small Vector instead of using Vector directly. The reason is simple: we want performance, and we want it to be even faster!

## A Simple Small Vector

Let’s implement the simplest version of Small Vector first. It only includes a basic constructor and the `push_back` function.

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

If the number of elements is less than `N`, we can store them in `m_data` owned by `SmallVec` itself, saving the cost of one heap allocation. This approach is effective. We can benchmark it with [Quick C++ Benchmark](https://quick-bench.com/) using the following code:

```cpp
int K = 10; // does not exceed Small Vector's N

static void SmallVector(benchmark::State& state) {
  for (auto _ : state) {
    SmallVec<int> sv(0); // use inline memory
    for(int i = 0 ; i < K; i++) {
        sv.push_back(i);
    }
  }
}
BENCHMARK(SmallVector);


static void StdVector(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<int> v;
    v.reserve(K); // allocate new heap memory
    for(int i = 0 ; i < K; i++) {
        v.push_back(i);
    }
  }
}
BENCHMARK(StdVector);
```

On a 64-bit environment (i3-10100, GCC 11.2), we get the following result (lower is better). It confirms that Small Vector is indeed more efficient than Vector for a small number of elements—about 2.6x faster—mainly because Vector spends extra time allocating heap memory.

![comparison between small vector and std vector](https://user-images.githubusercontent.com/18013815/174789476-7c2693bd-55b1-47cd-a588-2be5c36d8dbc.png)

> For Small Vector performance analysis, Yang Zhixuan provides a [more detailed discussion](https://hackmd.io/@25077667/r1uUxVY59) if you are interested.

## A Full Small Vector Implementation

The simple Small Vector already demonstrates better performance, so I went ahead and implemented a more complete Small Vector for practice.

You can view the full code by expanding it at the end of this post, or download it from this [Gist](https://gist.github.com/tigercosmos/4d939e90a350071424567b8ed4d9a378).

To briefly explain the code: if the number of elements is less than the default inline capacity `N`, it uses the built-in `std::array`. If it exceeds `N`, it allocates new memory via `new`. The more subtle part is handling `m_head` during copy operations and move semantics. Also, the Small Vector implemented here is only suitable for basic types (`int`, `float`, etc.), which matches common usage scenarios for Small Vector.

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
