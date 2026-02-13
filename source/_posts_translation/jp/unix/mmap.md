---
title: "`mmap` を使って共有オブジェクトを作る"
date: 2019-11-20 11:01:00
tags: [unix, network programming, mmap, shared memory, shared object]
lang: jp
translation_key: mmap
---

多数のプロセスがあり、共有データを扱うために shared memory（共有メモリ）を実装したい場合、shared memory を使って実現できます。shared memory の作成には `mmap` または System V の `shmget` を使えますが、Stack Overflow の回答「[How to use shared memory with Linux in C](https://stackoverflow.com/questions/5656530/how-to-use-shared-memory-with-linux-in-c)」によると、`shmget` はやや古く、`mmap` のほうが新しく柔軟です。

shared memory を使うと、プロセス間で共有できるメモリ領域を作れます。`mmap` はその領域へのポインタを返し、型は `void *` です。そこにデータを入れたい場合は、`memcpy` でオブジェクトや文字列などを共有領域へコピーできます。また、`void *` をそのままオブジェクトポインタへキャストすれば shared object（共有オブジェクト）を作れます。こうすると別々のプロセスが同じオブジェクトへ直接アクセスできます。

<!-- more -->

実装例：

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

ただし注意点があります。shared object の中に STL コンテナを置くことはできません。コンテナが内部で生成するポインタはそのプロセス内でしか有効ではなく、他のプロセスからは正しく参照できないためです。

これを解決する方法として 2 つ考えられます。1 つは STL コンテナ用の allocator を自作する方法。もう 1 つは、大きなコンテナの要素（小さなオブジェクト）を最初に shared memory に作っておき、大きなオブジェクトを初期化するときにそれらを 1 つずつコンテナへ戻していく方法です。

2 つ目の方法は大まかに以下のようになります：

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

