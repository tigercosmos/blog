---
title: "Reentrancy と Thread-safe を深く理解する"
date: 2021-05-10 06:20:08
tags: [parallel programming, reentrancy, thread-safe, system]
des: "本記事では Reentrancy と Thread-safe を詳しく解説し、コード例も交えて説明します。"
lang: jp
translation_key: reentrant-and-thread-safe-code
---

## 1. 並行と平行

Concurrency（並行）と Parallelism（平行）は非常に似た概念です。前者は「異なる計算が並行に進む」こと、つまり逐次（Sequentially）実行に比べて、ある計算が終わる前に別の計算が開始され得ることを指します。後者は「同じ計算を分割して同時に実行する」ことを指します。

一見すると両者はほとんど同じことを言っているように見えますが、[The Art of Concurrency](https://www.oreilly.com/library/view/the-art-of/9780596802424/) の定義では次のように区別されています：

> A system is said to be concurrent if it can support two or more actions in progress at the same time. A system is said to be parallel if it can support two or more actions executing simultaneously. The key concept and difference between these definitions is the phrase "in progress."

つまり、両者の鍵となる差は “in progress”（進行中）という表現にあります。並行は「複数の動作が進行中である」ことを許容し、平行は「複数の動作がまさに同時に実行されている」ことを意味します。さらに英語の simultaneously には “happening or being done at exactly the same time” という性質があります。

並行化は単一コア（Single Core）でも、タスクを交互に実行することで実現できます。また興味深いことに、並行化はマルチコア上で平行に実行される形でも実現できます。ある意味、両者は似たように見えることもありますが、“in progress” という言葉が両者の違いを的確に表しています。

並行と平行に不慣れな方は、Operating System Concepts 第 4 章「Threads & Concurrency」や、[Operating Systems: Three Easy Pieces](https://pages.cs.wisc.edu/~remzi/OSTEP/) の「Concurrency」章を復習するとよいです。

そして、並行／平行の世界では、プログラムの実行ロジックとデータの正しさを保証するために、Reentrancy と Thread-safe が何を意味するかを理解する必要があります。

![Cover](https://user-images.githubusercontent.com/18013815/117589349-823b5980-b15b-11eb-825d-55307d4c044b.png)

## 2. Reentrancy

計算機科学における [Reentrancy（可重入）](https://zh.wikipedia.org/wiki/%E5%8F%AF%E9%87%8D%E5%85%A5) とは、あるプログラム（またはサブルーチン）のコードが「任意のタイミングで割り込み（Interruption）を受け、OS が別のコードをスケジュールして実行した後、元のコードに戻ってきても正しく動作する」性質を指します。

そもそもなぜ割り込みが起きるのでしょうか。内部の制御フロー（`jump` や `call`）で起きることもあれば、外部イベント（割り込みやシグナル）で起きることもあります。つまり、割り込みは OS の有無とは無関係に起こり得ます。OS がなくても、プログラム自身の挙動で割り込みに似た状況は発生するため、Reentrancy の影響には注意が必要です。

Reentrancy は単一スレッドでも議論できます。たとえば、OS によって中断された後に、そのまま正しく再開できるか？という問題です。言い換えると、中断後に継続して正しく実行できるためには、コードが Reentrant であるべきで、そうでなければ戻ってきたときに結果が壊れる可能性があります。関連して面白い問いとして「[Interrupt handler は Reentrant である必要があるか？](https://stackoverflow.com/questions/18132580/does-an-interrupt-handler-have-to-be-reentrant)」があります。簡単に言うと、ハンドラがネスト（ある割り込み処理中に別の割り込みが入る）しない限り、基本的には心配しなくてよいです。また Linux では、別の割り込みが現在の割り込みを中断しないようにマスクされます。

Reentrancy が重要なのは、並行プログラミング（Concurrent Programming）において、非同期プログラム（Asynchronous Program）がタスク切り替えを行う際に、割り込みによって正しさが壊れないことを保証する必要があるからです。また、再帰（recursive call）を使うとき、通常は Reentrant であることを前提にしており、そうでなければ破綻します。

## 3. Thread-safe

一方で [Thread-safe（スレッド安全）](https://zh.wikipedia.org/zh-tw/%E7%BA%BF%E7%A8%8B%E5%AE%89%E5%85%A8) とは、関数やライブラリがマルチスレッド環境で呼ばれたときに、共有変数（グローバル変数、共有変数）を複数スレッド間で正しく扱い、プログラムの機能が正しく完了する性質を指します。

Thread-safe は平行プログラミングで特に重要です。平行計算では多くのデータが共有されることが多く、共有データを扱うと [Race condition](https://en.wikipedia.org/wiki/Race_condition#Computing) が発生しやすいからです。したがって、各スレッドでデータの read/write を正しく行えるようにすることが鍵になります。

Thread-safe は本質的には Data race を避けることです。その実現方法として Reentrancy を使うこともできますし、Thread-local data（そのスレッドにだけ存在するデータ）、Immutable objects（不変オブジェクト）、Mutex（相互排他）、Atomic operations（原子操作）などを使うこともできます。

## 4. Reentrancy と Thread-safe の関係

ここが本題です。Reentrancy と Thread-safe の関係はどうなっているのでしょうか？

両者は同一ではありませんが、一部重なりがあります。つまり Reentrancy は Thread-safe である場合もあればそうでない場合もあり、逆に Thread-safe も Reentrancy である場合とそうでない場合があります。

以下ではコード例でそれぞれのケースを説明します。

### 4.1 Reentrancy ❌ | Thread-safe ❌

```c
int t;

void swap(int *x, int *y) {
  t = *x;
  *x = *y;
  
  // 這邊可能呼叫 my_func();
  
  *y = t;
}

void my_func() {
  int x = 1, y = 2;
  swap(&x, &y);
}
```

- ❌ Reentrancy
    - `t` が外部にあるため、`swap` の途中で割り込みが入り、別の処理が `t` を変更すると、戻ってきたときの挙動が正しくなくなります。
- ❌ Thread-safe
    - `t` はグローバルです。
    - 別スレッドが `my_func` を呼ぶと、`t` が同じ実行コンテキストに属してしまう可能性があり、挙動は予測不能になります。

### 4.2 Reentrancy ❌ | Thread-safe ✅

```c
#include <threads.h>

// `t` 是每個 thread 自己的
thread_local int t;

void swap(int *x, int *y) {
  t = *x;
  *x = *y;

  // 這邊可能呼叫 my_func();

  *y = t;
}

void my_func() {
  int x = 1, y = 2;
  swap(&x, &y);
}
```

- ❌ Reentrancy
    - `t` はスレッドローカルですが、同じスレッド内でネストした呼び出しなどが起きると、複数回の呼び出しで `t` が書き換えられ得ます。
- ✅ Thread-safe
    - `t` はスレッドごとに独立しており、他スレッドが `t` に影響を与えることはできません。

### 4.3 Reentrancy ✅ | Thread-safe ❌

これは意図的に作った状況ですが、プログラムが複雑だと似た状況が起き得ます。

```c
int t;

void swap(int *x, int *y) {
  int s;
  // 存下全域變數
  s = t;
  
  t = *x;
  *x = *y;

  // `my_func()` 可以在這邊被呼叫

  *y = t;

  // 恢復全域變數
  t = s;
}

void my_func() {
  int x = 1, y = 2;
  swap(&x, &y);
}
```

- ✅ Reentrancy
    - `swap` の前後で `t` は元に戻ります。重要なのは `swap` が外部状態に影響を残さない点です。つまり、変数の変化は `swap` の中に閉じています。
- ❌ Thread-safe
    - `t` はグローバル変数なので、理由は前と同じです。

### 4.4 Reentrancy ✅ | Thread-safe ✅

この例の解決は意外と簡単で、グローバル変数を消すだけです。

```c
void swap(int *x, int *y) {
  int t = *x;
  *x = *y;

  // `my_func()` 執行
  *y = t;
}

void my_func() {
  int x = 1, y = 2;
  swap(&x, &y);
}

```

- ✅ Reentrancy
    - すべてのデータはスタック上にあり、外部から影響を受けません。
- ✅ Thread-safe
    - 共有データがないため Data race が起きません。

## 5. Reentrancy と Thread-safe の原則

上の例を見た上で、Reentrant または Thread-safe なコードを書くには、次の原則を守るとよいです。

Reentrancy：
- static（global）な非定数（non-constant）データを含まないこと。
- static（global）な非定数データのアドレスを返さないこと。
- 呼び出し側（Caller）が提供するデータのみを処理すること（引数で受け取る）。
- 呼び出す関数も Reentrant である必要があること。

Thread-safe：
- 基本的には Race condition を避ければよい
- Lock は友達

## 6. Reentrant / Thread-Safe なライブラリ

Reentrant / Thread-Safe なライブラリは、平行プログラミングや非同期プログラム開発で重要です。

GNU C Library には、MT-Safe（Multi-Thread-Safe）、AS-Safe（Async-Signal-Safe）、AC-Safe（Async-Cancel-Safe）などの安全レベルと、さまざまな非安全レベルがあります。

標準 C ライブラリ関数のうち、`ctime` や `strtok` は Reentrant ではありません。ただし多くの場合、対応する Reentrant 版が用意されており、名前に `_r` サフィックスが付くことが多いです（例：`strtok_r` や `rand_r`）。

また `man` コマンドでも確認できます。たとえば Ubuntu 16 で `man rand_r` を見ると次のような（抜粋）結果になります：

```
ATTRIBUTES
       For an explanation of the terms used in this section, see attributes(7).

       ┌──────────────────────────┬───────────────┬─────────┐
       │Interface                 │ Attribute     │ Value   │
       ├──────────────────────────┼───────────────┼─────────┤
       │rand(), rand_r(), srand() │ Thread safety │ MT-Safe │
       └──────────────────────────┴───────────────┴─────────┘
```

ここから `rand_r` が MT-Safe であることが分かります。これはマルチスレッド（MT）環境でも、期待する機能（ここでは乱数生成）が正しく動作し、マルチスレッド化による不具合（functional safety の破綻）が起きないことを強調します。

ただし、`MT-Safe` は「完全に安全」を意味しません。たとえば MT-Safe な関数を連続して呼び出す場合、状況によっては予期しない挙動が起こり得ます。

では、平行プログラムで MT-Safe ではない関数を使うとどうなるでしょうか。大きく 2 つの可能性があります。1 つ目は、そもそも安全ではないため誤った結果になる可能性があること。2 つ目は、外部状態を奪い合うことで性能が悪化する可能性があることです。たとえば平行プログラムで `rand_r` ではなく `rand` を呼ぶと、乱数生成が非常に遅くなったり（あるいは不正確になったり）することがあります。`rand` の実装には `static` が使われているからです。

## 7. 参考資料

1. cjwind's note. 2017. [Reentrancy and Thread-safety](http://www.cjwind.idv.tw/Reentrancy-and-Thread-safety/)
2. Mike Choi. 2017. [Reentrant and Threadsafe Code](https://deadbeef.me/2017/09/reentrant-threadsafe)
3. IBM. 1997. [AIX Version 4.3 General Programming Concepts: Writing Reentrant and Thread-safe Code](https://sites.ualberta.ca/dept/chemeng/AIX-43/share/man/info/C/a_doc_lib/aixprggd/genprogc/writing_reentrant_thread_safe_code.htm)
4. GNU.ORG. 2021. [POSIX Safety Concepts](https://www.gnu.org/software/libc/manual/html_node/POSIX-Safety-Concepts.html)
