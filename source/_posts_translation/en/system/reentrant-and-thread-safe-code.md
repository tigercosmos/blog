---
title: "Understanding Reentrancy and Thread Safety in Depth"
date: 2021-05-10 06:20:08
tags: [parallel programming, reentrancy, thread-safe, system]
des: "This post explains reentrancy and thread safety in detail, with code examples."
lang: en
translation_key: reentrant-and-thread-safe-code
---

## 1. Concurrency vs Parallelism

Concurrency and parallelism are closely related concepts. Concurrency means different computations make progress in an interleaved way: compared to sequential execution, one computation can start before another finishes. Parallelism means the same computation is split into parts and executed simultaneously.

At first glance they seem similar. Using the definition from [The Art of Concurrency](https://www.oreilly.com/library/view/the-art-of/9780596802424/):

> A system is said to be concurrent if it can support two or more actions in progress at the same time. A system is said to be parallel if it can support two or more actions executing simultaneously. The key concept and difference between these definitions is the phrase "in progress."

In other words, the key difference is whether actions are merely “in progress” (interleaved) versus actually executing at exactly the same time (simultaneously).

Concurrency can be achieved even on a single-core CPU by interleaving tasks. Interestingly, concurrency can also be implemented on multi-core CPUs in a parallel manner. In a sense, concurrency and parallelism can look similar, but the phrase “in progress” captures the distinction well. Also, the word “simultaneously” implies “happening or being done at exactly the same time”.

If you are not familiar with concurrency and parallelism, consider reviewing “Threads & Concurrency” in Chapter 4 of Operating System Concepts, or the “Concurrency” chapter of [Operating Systems: Three Easy Pieces](https://pages.cs.wisc.edu/~remzi/OSTEP/).

In both concurrency and parallelism, to ensure correctness of program logic and data, we need to discuss what reentrancy and thread safety actually mean.

![Cover](https://user-images.githubusercontent.com/18013815/117589349-823b5980-b15b-11eb-825d-55307d4c044b.png)

## 2. Reentrancy

In computer science, [reentrancy](https://zh.wikipedia.org/wiki/%E5%8F%AF%E9%87%8D%E5%85%A5) means that code in a program or subroutine can be “interrupted at any time (interruption), then the operating system schedules and runs other code, and when execution returns to the original code, it still works correctly.”

Why can a program be interrupted? It can be caused by internal behavior such as `jump` or `call`, or by external events such as interrupts or signals. In other words, interrupts can happen regardless of whether an OS exists: even without an OS, the program’s own control flow may introduce reentrant situations. So reentrancy still matters.

Reentrancy can be discussed even in a single-thread scenario: for example, can a program resume correctly after being interrupted by the OS? Put differently, if execution must continue after an interruption, the code should be reentrant—otherwise the result may be wrong after returning. Here is an interesting related question: [Does an interrupt handler have to be reentrant?](https://stackoverflow.com/questions/18132580/does-an-interrupt-handler-have-to-be-reentrant) The short answer is: unless your handler is nested (one handler can interrupt another), you usually do not need to worry. For example, Linux uses masking to prevent another interrupt from interrupting the current interrupt.

Reentrancy is important because in concurrent programming, we need to ensure that asynchronous programs remain correct when switching tasks; interruptions should not break correctness. Also, when using recursion, we expect the code to be reentrant; otherwise it may fail.

## 3. Thread Safety

On the other hand, [thread safety](https://zh.wikipedia.org/zh-tw/%E7%BA%BF%E7%A8%8B%E5%AE%89%E5%85%A8) means that when a function or library is called in a multi-threaded environment, it can correctly handle shared variables (global variables, shared variables) across threads so that the program’s functionality completes correctly.

Thread safety is important in parallel programming because parallel computation often involves shared data. When shared data is accessed concurrently, it is easy to introduce [race conditions](https://en.wikipedia.org/wiki/Race_condition#Computing). Therefore, ensuring correct reads and writes across threads is critical.

So thread safety is fundamentally about avoiding data races. You can achieve it via reentrancy, but also via thread-local data, immutable objects, mutexes, or atomic operations.

## 4. Reentrancy vs Thread Safety

So the key question: what is the relationship between reentrancy and thread safety?

They are not equivalent, but they overlap. Reentrancy may or may not be thread-safe, and thread-safe code may or may not be reentrant.

Below I use code examples to explain the different cases.

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
    - `t` is external. If `swap` is interrupted midway and someone else modifies `t`, then when execution returns, the behavior becomes incorrect.
- ❌ Thread-safe
    - `t` is global.
    - When another thread calls `my_func`, `t` may belong to the same execution context, so the behavior of `t` becomes unpredictable.

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
    - Even though `t` belongs to the thread, within the same thread, nested calls can still change `t` across multiple invocations.
- ✅ Thread-safe
    - Now `t` is per-thread; other threads cannot affect it.

### 4.3 Reentrancy ✅ | Thread-safe ❌

This is a deliberately constructed scenario, but you can imagine it happening in a complex program.

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
    - Before and after `swap`, `t` remains unchanged. The key here is that `swap` does not affect external state; all state changes stay within `swap`.
- ❌ Thread-safe
    - `t` is a global variable, for the same reason as earlier.

### 4.4 Reentrancy ✅ | Thread-safe ✅

The fix in this example is surprisingly simple: remove the global variable.

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
    - All data lives on the stack, so it will not be impacted.
- ✅ Thread-safe
    - There is no shared data, so there is no data race.

## 5. Principles for Reentrancy and Thread Safety

After the examples above, if you want to write reentrant or thread-safe code, you can follow these principles:

Reentrancy:
- Must not contain static (global) non-constant data.
- Must not return the address of static (global) non-constant data.
- Must only operate on data provided by the caller (passed via parameters).
- Any functions it calls must also be reentrant.

Thread safety:
- Fundamentally, avoid race conditions.
- Locks are your good friend.

## 6. Reentrant and Thread-Safe Libraries

Reentrant and thread-safe libraries are important when we write parallel code or develop asynchronous programs.

In the GNU C Library, there are safety levels such as MT-Safe (Multi-Thread-Safe), AS-Safe (Async-Signal-Safe), AC-Safe (Async-Cancel-Safe), and various unsafe levels.

Some standard C library functions, such as `ctime` and `strtok`, are not reentrant. But they often have a corresponding reentrant version whose name typically has an `_r` suffix, such as `strtok_r` or `rand_r`.

In fact, we can also check via `man`. For example, on Ubuntu 16, I ran `man rand_r` and got the following snippet:

```
ATTRIBUTES
       For an explanation of the terms used in this section, see attributes(7).

       ┌──────────────────────────┬───────────────┬─────────┐
       │Interface                 │ Attribute     │ Value   │
       ├──────────────────────────┼───────────────┼─────────┤
       │rand(), rand_r(), srand() │ Thread safety │ MT-Safe │
       └──────────────────────────┴───────────────┴─────────┘
```

We can see `rand_r` is MT-Safe, which means we can use it in parallel programs. MT-Safe emphasizes that in a multi-threaded environment, the expected functionality (here, random number generation) is still correct, and there should not be additional functional safety bugs introduced by multi-threading.

However, note that MT-Safe does not mean “completely safe” in all situations. For example, behavior may still be surprising when repeatedly calling MT-Safe functions in certain patterns.

You might ask: what if I use a non-MT-Safe function in a parallel program? Two things can happen. First, you may get incorrect results, because it is not safe. Second, performance may degrade, because it may contend on external state. For example, if you call `rand` (instead of `rand_r`) in a parallel program, you may find random number generation becomes extremely slow (and may even be incorrect), because `rand` uses `static` state internally.

## 7. References

1. cjwind's note. 2017. [Reentrancy and Thread-safety](http://www.cjwind.idv.tw/Reentrancy-and-Thread-safety/)
2. Mike Choi. 2017. [Reentrant and Threadsafe Code](https://deadbeef.me/2017/09/reentrant-threadsafe)
3. IBM. 1997. [AIX Version 4.3 General Programming Concepts: Writing Reentrant and Thread-safe Code](https://sites.ualberta.ca/dept/chemeng/AIX-43/share/man/info/C/a_doc_lib/aixprggd/genprogc/writing_reentrant_thread_safe_code.htm)
4. GNU.ORG. 2021. [POSIX Safety Concepts](https://www.gnu.org/software/libc/manual/html_node/POSIX-Safety-Concepts.html)
