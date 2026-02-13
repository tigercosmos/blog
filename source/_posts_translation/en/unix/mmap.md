---
title: "Using `mmap` to Create Shared Objects"
date: 2019-11-20 11:01:00
tags: [unix, network programming, mmap, shared memory, shared object]
lang: en
translation_key: mmap
---

When you have many processes and want to implement shared memory to handle shared data, you can use shared memory to build the solution. To create shared memory, you can use `mmap` or System V `shmget`. However, according to the Stack Overflow answer “[(How to use shared memory with Linux in C)](https://stackoverflow.com/questions/5656530/how-to-use-shared-memory-with-linux-in-c)”, `shmget` is somewhat outdated, while `mmap` is newer and more flexible.

Shared memory allows us to create a region of memory that can be shared. `mmap` returns a pointer to that region, with type `void *`. If we want to put data into it, we can use `memcpy` to copy objects, strings, or anything else into the shared region. We can also cast the `void *` directly to an object pointer—this way we create a shared object, and different processes can access the object directly.

<!-- more -->

Example implementation:

<pre><code class="bash">$ g++ mmap.cc -std=c++17
</pre></code>

<pre><code class="c++">// mmap.cc

#include &lt;memory&gt;
#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;string.h&gt;
#include &lt;sys/mman.h&gt;
#include &lt;unistd.h&gt;

template &lt;typename T&gt; T *create_shared_memory() {
  // 可讀、可寫
  int protection = PROT_READ | PROT_WRITE;

  // MAP_SHARED 代表分享給其他 process，MAP_ANONYMOUS 讓使其只被自己和 child 可見
  int visibility = MAP_SHARED | MAP_ANONYMOUS;

  // 詳細參數使用見 man mmap
  // 為啥要放 -1 可以看這篇 https://stackoverflow.com/questions/37112682/
  void *ptr = mmap(NULL, sizeof(T), protection, visibility, -1, 0);

  // 將記憶體轉型成 T 物件指標
  return reinterpret_cast&lt;T *&gt;(ptr);
}

struct Bar {
  int a;
  int b;
  Bar(int a, int b) : a(a), b(b) {}
};

struct Foo {
  Bar bar[30];
};

int main() {

  // 建立 Foo * 在 shared memory 中
  auto *foo = create_shared_memory&lt;Foo&gt;();

  auto print = [=]() {
    for (auto i = 0; i &lt; 3; i++) {
      printf("%d: %d, %d\n", i, foo-&gt;bar[i].a, foo-&gt;bar[i].b);
    }
    // 印出 Foo 的地址，檢驗 parent 和 child 是共用
    printf("Foo: %p\nFoo.bar: %p\n---\n", foo, foo-&gt;bar);
  };

  // 初始化
  foo-&gt;bar[0] = Bar(0, 0);
  foo-&gt;bar[1] = Bar(0, 0);
  foo-&gt;bar[2] = Bar(0, 0);
  
  printf("data before fork: \n");
  print();

  if (!fork()) {  // child

    printf("Child read:\n");
    print();

    // 改動 shared object
    foo-&gt;bar[1].a = 2;
    foo-&gt;bar[1].b = 3;

    printf("Child wrote:\n");
    print();

  } else { // parent
    printf("Parent read:\n");
    print();

    sleep(1);

    printf("After 1s, parent read:\n");
    print();
  }
}
</pre></code>

One important caveat: you cannot place STL containers inside a shared object, because pointers created by the container are only valid in the current process; other processes cannot dereference them correctly.

I can think of two approaches to address this. One is to implement a custom allocator for STL containers. The other is to allocate each element (the “small objects” inside the big container) in shared memory first, and then, when initializing the “big object”, insert those elements back into the container one by one.

The second approach looks roughly like this:

<pre><code class="c++">
struct Foo {
  std::vector&lt;std::unique_ptr&lt;Bar&gt;&gt; bars;
};

void main() {
  std::vector&lt;Bar *&gt; bars;
  for(int i = 0; i &lt; 10; i++) {
    auto *bar = create_shared_memory&lt;Bar&gt;();
    bars.push_back(bar);
  }

  if(!fork()) {
    // 生成新的 process 才建立 Foo
    Foo foo1;

    // 把 foo 中的 bar 們一個一個塞回來
    for(size_t i = 0; i &lt; bars.size(); i++) {
      std::unique_ptr&lt;Bar&gt; u_bar;
      u_bar.reset(bars.at(i));
      foo1.push_back(std::move(u_bar));
    }

    // 接下去 foo1 都會另一個 process 的 foo2 共用資料
  } else {
    Foo foo2;

    // 一樣把 bar 塞回來
    // ...
  }
  // ...
}
</pre></code>

