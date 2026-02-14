---
title: "A Simple Pthreads Parallelization Example and Performance Analysis"
date: 2020-07-02 23:00:00
tags: [c, pthread, parallel programming, 效能分析, 平行化]
des: "POSIX threads (Pthreads) let us write parallel programs in C/C++. This post provides a simple example of using Pthreads to parallelize the computation of π, covering how to create threads, configure them, and use a mutex. Finally, it includes a brief performance analysis."
lang: en
translation_key: simple-pthread-usage
---

## Introduction

[POSIX threads](https://zh.wikipedia.org/zh-tw/POSIX%E7%BA%BF%E7%A8%8B) (Pthreads) let us write parallel programs in C/C++. `pthread` is a set of well-defined API functions. We simply call APIs that start with `pthread_`, and the underlying parallelization mechanism is handled for us.

There are many classic scenarios where parallelization makes sense. In general, as long as you have a loop and the work inside each iteration has low dependency, it can often be parallelized. My favorite example is computing π, and this post will also use π computation as the example.

## Computing π with a single thread

First, let’s see how to compute π using only one thread:

```c
// pi_single_thread.c

#include <stdio.h>

static long num_steps = 1e9;

int main()
{
    double x, pi, sum = 0.0;
    double step = 1.0 / num_steps;
    for (int i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    printf("%.10lf\n", pi);
}
```

The output looks like this:

```shell
$ gcc pi_single_thread.c && ./a.out
3.1415926536
```

As you can see, this example has only one loop. The work done in `sum` can be split easily and independently, so it’s a great fit for parallelization.

## Parallelizing π computation with pthread

Now let’s rewrite the code above using pthread. The code is as follows:

```c
// pi_multi_thread.c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUMTHRDS 4
#define MAGNIFICATION 1e9

typedef struct
{
   int thread_id;
   int start;
   int end;
   double *pi;
} Arg; // the parameter type passed into each thread

pthread_t callThd[NUMTHRDS]; // declare pthreads
pthread_mutex_t mutexsum;    // pthread mutex

// the task each thread performs
void *count_pi(void *arg)
{

   Arg *data = (Arg *)arg;
   int thread_id = data->thread_id;
   int start = data->start;
   int end = data->end;
   double *pi = data->pi;

   // split the original π computation into multiple parts
   double x;
   double local_pi = 0;
   double step = 1 / MAGNIFICATION;
   for (int i = start; i < end; i++)
   {
      x = (i + 0.5) * step;
      local_pi += 4 / (1 + x * x);
   }

   local_pi *= step;

   // **** critical section ****
   // allow only one thread to access at a time
   pthread_mutex_lock(&mutexsum);
   // add the partial π into the final π
   *pi += local_pi;
   pthread_mutex_unlock(&mutexsum);
   // *****************

   printf("Thread %d did %d to %d:  local Pi=%lf global Pi=%.10lf\n", thread_id, start,
          end, local_pi, *pi);

   pthread_exit((void *)0);
}

int main(int argc, char *argv[])
{
   // initialize the mutex
   pthread_mutex_init(&mutexsum, NULL);

   // set pthread attributes to be joinable
   pthread_attr_t attr;
   pthread_attr_init(&attr);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

   // π shared by all threads
   // since multiple threads need to access it, use a pointer
   double *pi = malloc(sizeof(*pi));
   *pi = 0;

   int part = MAGNIFICATION / NUMTHRDS;

   Arg arg[NUMTHRDS]; // arguments passed to each thread
   for (int i = 0; i < NUMTHRDS; i++)
   {
      // set arguments
      arg[i].thread_id = i;
      arg[i].start = part * i;
      arg[i].end = part * (i + 1);
      arg[i].pi = pi; // shared π pointer

      // create a thread: run count_pi and pass &arg[i]
      pthread_create(&callThd[i], &attr, count_pi, (void *)&arg[i]);
   }

   // destroy attribute object
   pthread_attr_destroy(&attr);

   void *status;
   for (int i = 0; i < NUMTHRDS; i++)
   {
      // wait for each thread to finish
      pthread_join(callThd[i], &status);
   }

   // all threads finished; print π
   printf("Pi =  %.10lf \n", *pi);

   // destroy the mutex
   pthread_mutex_destroy(&mutexsum);
   // exit
   pthread_exit(NULL);
}
```

The output looks like this:

```shell
$ gcc pi_multi_thread.c  -lpthread && ./a.out
Thread 3 did 750000000 to 1000000000:  local Pi=0.567588 global Pi=0.5675882184
Thread 2 did 500000000 to 750000000:  local Pi=0.719414 global Pi=1.2870022176
Thread 1 did 250000000 to 500000000:  local Pi=0.874676 global Pi=2.1616780011
Thread 0 did 0 to 250000000:  local Pi=0.979915 global Pi=3.1415926536
Pi =  3.1415926536
```

## Performance analysis

Here is a small experiment. I measured the time difference between the single-thread version and the multi-thread version on a VM running Ubuntu 20, using an AMD Ryzen 7 2700X Eight-Core Processor.

The test code uses the `pi_single_thread.c` and `pi_multi_thread.c` shown above.

Compiled with GCC 7.5 using `-O2`, the results are:

| Thread  | Time(s)  |
|---|---|
|  1 | 3.1113  |
|  2 | 1.531  |
|  4 |  0.817 |
|  8 |  0.489 |
| 16 |  0.345 |

![Time-Threads](https://user-images.githubusercontent.com/18013815/86372410-b11fae00-bcb4-11ea-9c25-5db81e9a9d55.png)

Going from 1 thread to 2 threads cuts the time in half immediately. But from 8 threads to 16 threads, the time decreases only a little. That’s reasonable: as the number of threads increases, the cost of synchronization and memory access also increases.

Also, let’s look at `perf stat` for 1 thread, 8 threads, and 16 threads:

| Thread  | CPU Usage  | Page Fault |
|---|---|---|
|  1 | 0.998  | 52 |
|  8 |  7.353 | 86 |
| 16 |  13.439 | 105 |

Since my machine has only 16 threads, when running with 16 threads the CPU usage is still only 13.4/16, which makes the actual runtime longer than expected. You can also see that the number of page faults increases significantly.

Next, let’s see which part of the code takes the most time.

![perf code time](https://user-images.githubusercontent.com/18013815/86374997-c2b68500-bcb7-11ea-9db0-3f257c09d460.png)

From the figure above, unsurprisingly, most of the time cost is spent on the two key lines that compute π:

```c
x = (i + 0.5) * step;
local_pi += 4 / (1 + x * x);
```

The `movapd` instruction takes a bit too long. Under `-O2`, storing data to memory at this step is reasonable. But if you compile with `-O3`, you may get better register allocation and this instruction might not appear.

## Conclusion

This post provides a simple example of using Pthreads to parallelize π computation, covering how to create threads, configure them, and add a mutex. Finally, it includes a brief performance analysis.

