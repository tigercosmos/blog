---
title: "Achieving Lock-Free Programming with Compare-and-Swap"
date: 2020-10-28 02:25:00
tags: [compare and swap, lock free, atomic, parallel programming]
des: "This post introduces the principles of Compare-and-Swap (CAS) and experimentally shows that CAS can incur less overhead than locks."
lang: en
translation_key: cas-lock-free
---

## 1. Introduction

We all know that locks have many drawbacks. For example, locks introduce extra overhead, and incorrect usage can lead to deadlocks. Therefore, we generally want to avoid locks whenever possible. A lock-free programming style can help us reduce the use of locks, and it avoids extra overhead while handling critical sections.

One lock-free approach is to use [Compare and Swap (CAS)](https://en.wikipedia.org/wiki/Compare-and-swap). Since CAS is an atomic instruction, its cost is small, and it can ensure data safety in multi-threaded scenarios.

This post introduces the principles of Compare-and-Swap and experimentally demonstrates that the overhead of CAS can be smaller than that of locks.

## 2. Compare-and-Swap Pseudocode

CAS is typically supported by hardware, and the compiler can invoke the corresponding intrinsic. If we write CAS as a pseudocode function, it looks like this:

```c++
bool CAS(int* p, int old, int new) {
    if *p ≠ old {
        return false;
    }
    *p ← new
    return true;
}
```

As you can see, the CAS instruction has three parameters: the pointer to the variable to compare, `*p`; the old value expected at that address, `old`; and the new value to update to, `new`.

In practice, CAS is commonly used in a pattern like this:

```c
int old, new;

do {
    old = *p;
    new = NEW_VALUE;
} while(!CAS(*p, old, new));
```

The loop wrapped around CAS can be regarded as a critical section. When executing the CAS instruction, CAS checks whether the value of `*p` is the same as `old`. If they match, it means no other thread has modified `*p` during the execution window, so it is safe to update `*p` to `new`. Otherwise, if `*p` differs from `old`, it means someone has changed `*p` in the meantime, so this iteration is discarded and we retry the loop—hoping the next attempt will not be interfered with by other threads.

With CAS, we can achieve lock-free behavior because we do not need to acquire any lock. However, this is not block-free, because the CAS loop may fail and retry. In practice, this situation is rare—usually just one or two retries—so we can still achieve thread safety with low overhead.

## 3. Compare-and-Swap Example

### 3.1 Sum Serial Version

Here is a very simple sum program. It just keeps adding.

```c
#include <stdio.h>

int main() {

    int sum = 0;

    for(int i = 0; i < 10000000; i++) {

        for(int i = 0; i < 500; i++) {} // Pretend there is a task taking some time

        sum += 3; // Intentionally make it two instructions
        sum -= 2;
    }

    printf("sum = %d\n", sum);
}
```

Execution time:

```shell
$ gcc test.c; time ./a.out
sum = 10000000

real    0m7.548s
```

### 3.2 Sum OpenMP Multi-Thread without Lock

Next, we modify it into a multi-threaded program using OpenMP.

```c
#include <stdio.h>

int main()
{
    int sum = 0;

#pragma omp parallel for shared(sum)
    for (int i = 0; i < 10000000; i++)
    {
        for (int i = 0; i < 500; i++){}
        sum += 3;
        sum -= 2;
    }

    printf("sum = %d\n", sum);
}
```

```shell
$ gcc test.c -fopenmp; time ./a.out
sum = 9120084

real    0m2.035s
```

Since my machine has 4 threads, you can see the speed is roughly 4× faster. However, we also see the result is incorrect: instead of the expected 10000000, we get 9120084. This is because we did not use any locking, so threads can observe stale values. (This issue becomes even more obvious as the numbers grow.)

### 3.3 Sum OpenMP Multi-Thread with Lock

So we add a lock.

```c
#include <stdio.h>

int main()
{
    int sum = 0;

#pragma omp parallel for shared(sum)
    for (int i = 0; i < 10000000; i++)
    {
        for (int i = 0; i < 500; i++){}
#pragma omp critical
        {
            sum += 3;
            sum -= 2;
        }
    }

    printf("sum = %d\n", sum);
}
```

```shell
$ gcc test.c -fopenmp; time ./a.out
sum = 10000000

real    0m2.116s
```

Now the result is correct, but the time is slightly longer. This shows that locks do introduce overhead.

### 3.4 Sum OpenMP Multi-Thread with Lock-Free

Next, we use CAS to implement a lock-free approach. Since I am using GCC, I can use GCC’s `__sync_bool_compare_and_swap` API. Of course, you can also use the APIs provided by `std::atomic`.

```c
#include <stdio.h>

int main()
{
    int sum = 0;
    int current, next;

#pragma omp parallel for shared(sum) private(current, next)
    for (int i = 0; i < 10000000; i++)
    {
        for (int i = 0; i < 500; i++) {}
        do
        {
            current = sum;
            next = current;
            next += 3;
            next -= 2;
        } while (!__sync_bool_compare_and_swap(&sum, current, next));
    }

    printf("sum = %d\n", sum);
}
```

```shell
$ gcc test.c -fopenmp; time ./a.out
sum = 10000000

real    0m2.099s
user    0m8.348s
sys     0m0.000s
```

We can see that CAS is slightly slower than the version without locks, but still faster than the locked version (2.099s vs 2.116s). As mentioned earlier, locks have overhead, and the overhead becomes more visible when locks are used frequently.

Next, I inserted a counter into the code to see how many times CAS failed in total. The number I got was 262408. That is, among the original 10000000 critical sections, CAS retried 262408 times in total, accounting for 2.6% of all attempts. The number of times it failed two or more times in a row was about 2800, with a probability of 0.028%.

If we do not use CAS, we effectively pay the overhead of 1e7 lock acquisitions. With CAS, we only need to retry with a probability of 2.6%, so it is roughly (1.026 * 1e7) CAS attempts versus 1e7 lock attempts. Because CAS is an atomic instruction and costs fewer cycles than locks, CAS wins in the end.

## 4. Conclusion

Locks can protect our data, but they come with a cost. So we should use them sparingly. In some scenarios, if we can adopt a lock-free style—or use atomic operations—we can significantly reduce the overhead of our programs.
