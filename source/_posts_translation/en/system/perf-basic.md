---
title: "Performance Analysis with perf on Linux (Beginner)"
date: 2020-08-29 00:20:08
tags: [linux, perf, 效能分析,]
des: "This post introduces perf, a performance profiling tool on Linux. Using a simple program example, it demonstrates how to analyze a program with perf and how a profiler helps you find the root cause more easily."
lang: en
translation_key: perf-basic
---

## Introduction

With a profiler, we can learn more about how software runs—such as memory usage, CPU cycles, cache misses, I/O time, and more. This information is extremely helpful for locating performance bottlenecks. The ultimate goal of performance analysis is to find what slows the program down and maximize performance.

This post introduces [perf](http://www.brendangregg.com/perf.html) on Linux. Using a simple program example, I will demonstrate how to analyze a program with perf, and you will see that using a profiling tool can make it much easier to identify the root cause. This post references Gabriel Krisman Bertaz’s [Performance analysis in Linux](https://www.collabora.com/news-and-blog/blog/2017/03/21/performance-analysis-in-linux/)。
<!-- more -->

You can also watch my tutorial video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/Mba2ONCA0kI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## A Branch Prediction Example

There is a very popular Stack Overflow question: “[Why is processing a sorted array faster than processing an unsorted array?](https://stackoverflow.com/questions/11227809/)”.

The code is as follows:

`test.cc`:

```c++
#include <algorithm>
#include <ctime>
#include <iostream>

int main()
{
    // 測試用陣列
    const int arr_len = 32768;
    int data[arr_len];

    for (int c = 0; c < arr_len; ++c)
        data[c] = std::rand() % 256;

    // std::sort(data, data + arr_len); // 是否排序
    
    long long sum = 0;

    for (int i = 0; i < 30000; ++i)
    {
        for (int c = 0; c < arr_len; ++c)
        {
            if (data[c] >= 128) { // 故意選 256 一半  
                sum += data[c];
            }
        }
    }

    std::cout << "sum = " << sum << std::endl;
}
```

First, compile the unsorted version:

```shell
$ g++ test.cc -o unsort
```

Then uncomment the `sort` line and compile again:

```shell
$ g++ test.cc -o sort
```

Now let’s compare runtime:

```shell
$ time ./unsort
real    0m5.671s

$ time ./sort
real    0m1.932s
```

The question is: after sorting `data`, the program becomes faster, as the experiment shows. We know sorting is $O(NlogN)$, so it should be “slower” than not sorting and running the loop directly (which looks like $O(N)$). But in practice, sorting makes it faster.

In conclusion, we know this happens because the CPU performs **branch prediction**. Informally, if the `if` condition was `true` last time, the CPU may guess it will be `true` next time. The CPU can speculatively execute based on this guess; if it guesses correctly, the program runs faster. But if it guesses wrong, the speculative work is discarded, which wastes time—this is called a branch miss (see “computer architecture” for details). Therefore, branch prediction is a double-edged sword: if the condition tends to have the same outcome repeatedly, you get speedups; if it flips unpredictably, the CPU mispredicts repeatedly and the program slows down. In the code above, the sorted version is faster because mispredictions happen only once—around where `data` crosses `128`. Before that point, the condition is always `false`, and after that point it is always `true`.

## The perf Profiling Tool

Finding the root cause in a piece of code is usually not easy. Even in the simple example above, if we analyze it purely from an algorithmic angle, we will go in the wrong direction—the real issue is about computer architecture. If a simple piece of code can already mislead us, imagine a large program with all kinds of issues: algorithms, caches, CPU instructions, network, I/O, and so on. In such cases, we need tools to help analyze programs.

Linux provides many tools:

![Linux 分析工具](https://user-images.githubusercontent.com/18013815/91632981-533ee680-ea17-11ea-90f8-06676583ea52.png)

In this post, I focus on perf. Using the example above, I will demonstrate how to use perf to find the issue when we do not yet know the slowdown is caused by branch misses.

You can install perf on Ubuntu with:

```shell
$ sudo apt install linux-tools-$(uname -r) linux-tools-generic
```

Or you can build perf from the Linux kernel source:

```shell
$ sudo apt install flex bison libelf-dev libunwind-dev libaudit-dev libslang2-dev libdw-dev
$ git clone https://github.com/torvalds/linux --depth=1
$ cd linux/tools/perf/
$ make
$ make install
$ sudo cp perf /usr/bin
$ perf
```

After installing perf, you may need to adjust permissions. By default, perf may not have enough privileges:

```shell
$ sudo su # As Root
$ sysctl -w kernel.perf_event_paranoid=-1
$ echo 0 > /proc/sys/kernel/kptr_restrict
$ exit
```

## Using perf

Next, for perf to work well, we compile the program with debug information using the `-g3` flag.

Still using `test.cc`, first compile the unsorted version:

```shell
$ g++ test.cc -g3 -o unsort
```

Then compile the sorted version:

```shell
$ g++ test.cc -g3 -o sort
```

### perf record

Now we want to know why `./unsort` is slower. We can use `perf record` to record execution information:

```shell
$ perf record ./unsort
```

perf records data into `perf.data`, and other perf commands can read this record file.

### perf annotate

We can use `perf annotate` to inspect the results:

```shell
$ perf annotate
```

![perf annotate](https://user-images.githubusercontent.com/18013815/91634468-44f6c780-ea23-11ea-9bf5-cd14907e22e8.png)

perf automatically jumps to the hottest region. As shown above, the left side is the time percentage, and the right side shows the source code alongside the corresponding assembly. You can move with the up/down arrow keys, or press `h` to see help.

In fact, from the assembly-time percentages you can already spot the clue. Usually, we look for what takes the most time, and then investigate why. The key here is the two lines `d8` and `cf`. `addl` corresponds to `sum += data[c]`, so these two paths represent the branch-prediction “correct guess” path and the “mispredict” path.

In this image, the arrow marks the branch-prediction path where the CPU **predicts correctly**. You can see line `d8` is almost 0.0%.
<img src="https://user-images.githubusercontent.com/18013815/91635364-93f42b00-ea2a-11ea-89b8-19075dbc67fc.png" alt="branch prediction 猜對" width=70%>

In this image, the arrow marks the branch-prediction path where the CPU **predicts incorrectly**. You can see line `cf` is about 27.7%.
<img src="https://user-images.githubusercontent.com/18013815/91635373-9eaec000-ea2a-11ea-8347-1f2386373a57.png" alt="branch prediction 猜錯" width=70%>

So we can clearly see that the program wastes a lot of time due to branch misses.

We can also “peek” at the `./sort` result:

```shell
$ perf record ./sort && perf annotate
```

<img src="https://user-images.githubusercontent.com/18013815/91636294-01578a00-ea32-11ea-888d-d46b2b65163c.png" alt="sort version's branch prediction" width=70%>

Because there are almost no branch misses, you can see the `addl` instructions at `ee` and `f7` barely take any time.

### perf stat

Reading assembly directly can be time-consuming. If you want a higher-level overview, you can use `perf stat`.

```shell
# 未排序版本
$ perf stat ./unsort
sum = 94479480000

 Performance counter stats for './unsort':

          5,671.51 msec task-clock                #    1.000 CPUs utilized
                24      context-switches          #    0.004 K/sec
                 0      cpu-migrations            #    0.000 K/sec
               147      page-faults               #    0.026 K/sec
    20,366,870,320      cycles                    #    3.591 GHz
    11,328,534,095      instructions              #    0.56  insn per cycle
     2,951,455,487      branches                  #  520.401 M/sec
       467,676,925      branch-misses             #   15.85% of all branches

       5.671777216 seconds time elapsed

       5.671781000 seconds user
       0.000000000 seconds sys

# 排序版本
$ perf stat ./sort
sum = 94479480000

 Performance counter stats for './sort':

          1,927.09 msec task-clock                #    1.000 CPUs utilized
                 6      context-switches          #    0.003 K/sec
                 0      cpu-migrations            #    0.000 K/sec
               146      page-faults               #    0.076 K/sec
     6,917,745,957      cycles                    #    3.590 GHz
    11,345,543,927      instructions              #    1.64  insn per cycle
     2,954,388,946      branches                  # 1533.084 M/sec
           268,192      branch-misses             #    0.01% of all branches

       1.927654198 seconds time elapsed

       1.927349000 seconds user
       0.000000000 seconds sys
```

`perf stat` shows aggregated statistics. High context switches, page faults, or branch misses often indicate performance issues that need optimization.

For example, in `unsort`, branch misses are especially high (the sorted version is almost 0). Then we can look for conditional branches in the original program. Combined with the time percentages from `perf annotate`, we can quickly pinpoint the problematic area. Also, from the cycle counts, we can see the two versions differ by about 3×.

> For more perf usage, see Brendan Gregg’s “[perf Examples](http://www.brendangregg.com/perf.html)”. Also, this HackMD [note](https://hackmd.io/@1IzBzEXXRsmj6-nLXZ9opw/HkBl5kCSU) is pretty good.

## Conclusion

This post introduced basic perf usage and demonstrated how to observe performance and identify likely root causes with a simple program.

We often need performance analysis when a program runs slowly, but finding the root cause is not easy. A slow program can be caused by algorithms and data structures, OS system calls, or processor architecture. As the example in this post shows, algorithmic complexity does not necessarily represent real-world runtime; we also need to consider the OS and hardware architecture. Using performance analysis tools well helps us find issues faster.
