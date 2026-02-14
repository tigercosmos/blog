---
title: "Converting Pthreads to JavaScript with Emscripten and Performance Analysis"
date: 2020-07-07 18:00:00
tags: [JavaScript, web worker, nodejs, c, pthread, parallel programming, browser, browsers, 效能分析, 平行化]
des: "This post explains how to use Emscripten to convert C/C++ Pthreads into Web Workers and WebAssembly, and compares performance across (1) native C code, (2) Emscripten-generated JS/WASM, and (3) a Web Worker implementation written directly in JavaScript. With -O3, native C is about 30% faster than the Pthread-to-WASM version, and the Pthread-to-WASM version is roughly comparable to pure JavaScript Web Workers."
lang: en
translation_key: emscripten-pthread-to-js
---

## Introduction

Emscripten is a tool that compiles C/C++ to WebAssembly. Under the hood it goes through LLVM. It supports compiling Pthreads: it turns them into JavaScript Web Workers plus WebAssembly. It can even translate OpenGL to WebGL, allowing programs to run in a browser with performance close to native.

The focus of this post is exactly that: converting Pthreads into Web Workers + WebAssembly. I will take an example program and try the conversion in practice. Finding a good benchmark program is not easy, so I wrote a small parallel Pthread program to compute π as the test case.

I will first walk through how to use Emscripten to convert Pthreads to JS. While following the official documentation, I ran into several pitfalls; I also record them here so you don’t fall into the same holes. Then I will compare performance for (1) native C code, (2) Emscripten-generated JS/WASM, and (3) a Web Worker implementation written directly in JavaScript.
<!-- more -->

## Pthread example program

The example is a small program that uses Pthreads to compute π in parallel. I chose it for a few reasons: when writing a parallel program, the most important things to verify are whether it can create threads, whether it can use shared memory, whether it can lock, and whether it can wait for other threads. That is basically the core of a parallel program, and this π example is sufficient for what I need.

To save space, for more details on Pthreads you can refer to my earlier post: “[A simple Pthreads parallelization example and performance analysis](/post/2020/07/simple-pthread-usage/)”.

pi.c:
```c
// pi.c
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
} Arg;

pthread_t callThd[NUMTHRDS];
pthread_mutex_t mutexsum;

void *count_pi(void *arg)
{

   Arg *data = (Arg *)arg;
   int thread_id = data->thread_id;
   int start = data->start;
   int end = data->end;
   double *pi = data->pi;

   double x;
   double local_pi = 0;
   double step = 1 / MAGNIFICATION;
   for (int i = start; i < end; i++)
   {
      x = (i + 0.5) * step;
      local_pi += 4 / (1 + x * x);
   }

   local_pi *= step;

   pthread_mutex_lock(&mutexsum);
   *pi += local_pi;
   pthread_mutex_unlock(&mutexsum);

   printf("Thread %d did %d to %d:  local Pi=%lf global Pi=%.10lf\n", thread_id, start,
          end, local_pi, *pi);

   pthread_exit((void *)0);
}

int main(int argc, char *argv[])
{
   pthread_mutex_init(&mutexsum, NULL);

   pthread_attr_t attr;
   pthread_attr_init(&attr);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

   double *pi = malloc(sizeof(*pi));
   *pi = 0;

   int part = MAGNIFICATION / NUMTHRDS;

   Arg arg[NUMTHRDS];
   for (int i = 0; i < NUMTHRDS; i++)
   {
      arg[i].thread_id = i;
      arg[i].start = part * i;
      arg[i].end = part * (i + 1);
      arg[i].pi = pi;
      pthread_create(&callThd[i], &attr, count_pi, (void *)&arg[i]);
   }

   pthread_attr_destroy(&attr);

   void *status;
   for (int i = 0; i < NUMTHRDS; i++)
   {
      pthread_join(callThd[i], &status);
   }

   printf("Pi =  %.10lf \n", *pi);

   free(pi);

   pthread_mutex_destroy(&mutexsum);
   pthread_exit(NULL);
}
```

## Download Emscripten

To use Emscripten, you first need to download its GitHub repo. Make sure you have Git installed on your machine.

```shell
# fetch the emsdk repo
git clone https://github.com/emscripten-core/emsdk.git

# enter the directory
cd emsdk
```

Then follow these steps:

```shell
# install the latest SDK
./emsdk install latest

# activate the latest SDK
./emsdk activate latest

# set up environment variables
source ./emsdk_env.sh
```

Once installed and activated, every time you want to use Emscripten later, you just need to enter `emsdk` and run `source ./emsdk_env.sh`.

If you do not want to set up the environment every time, you can consider adding `source ./emsdk_env.sh` to `.bashrc`. However, this can override the NodeJS path, so I do not really recommend it.

## Emscripten basics

`emcc` or `em++` is the Emscripten frontend (roughly like how Clang is to LLVM). That explanation is a bit abstract; in practice, running this command compiles C/C++ into WebAssembly (WASM).

A simple demo:

```c
// hello.c
#include <stdio.h>

int main() {
  printf("hello, world!\n");
  return 0;
}
```

Compile:

```shell
$ emcc hello.c
```

This produces `a.out.js` and `a.out.wasm`. This is because today, both browsers and NodeJS still require JavaScript to bootstrap WASM.

```shell
$ emcc hello.c -o hello.html
```

You can also output an HTML file, which you can open in a browser as an example. The page contains a virtual terminal, and you can see the original program output in the terminal inside the web page.

Note that if you output HTML, you must serve it via a local server. This is because browsers do not allow `file://` XHR requests. Also, Emscripten has additional requirements on HTTP headers and file types, which can be fairly annoying. For details, check the Emscripten docs.

For more usage instructions, see the [Emscripten Tutorial](https://emscripten.org/docs/getting_started/Tutorial.html#emscripten-tutorial).

## Compiling Pthreads with Emscripten

Now let’s compile `pi.c`. According to the official documentation, to compile Pthreads you need to add `-s USE_PTHREADS=1`.

```shell
$ emcc pi.c -s USE_PTHREADS=1
```

Let’s see what happens with this output (NodeJS needs `--experimental-wasm-threads --experimental-wasm-bulk-memory`):

```
$ node  --experimental-wasm-threads --experimental-wasm-bulk-memory a.out.js

// nothing happens at all
```

Yes—it's stuck.

After asking on GitHub, I learned that this is due to how Emscripten works (I don’t fully understand the details either). See the [issue](https://github.com/emscripten-core/emscripten/issues/11543#issuecomment-654317178) I filed. In short, there are three ways to fix it:

1. Compile with `-s PROXY_TO_PTHREAD`
2. Compile with `-s PTHREAD_POOL_SIZE=N` where N > 0
3. Replace `main()` with `emscripten_set_main_loop()`

We will compile with:

```shell
$ emcc pi.c  -s USE_PTHREADS=1  -s PTHREAD_POOL_SIZE=4
```

Now it runs correctly!

```
$ node  --experimental-wasm-threads --experimental-wasm-bulk-memory a.out.js
Thread 1 did 250000000 to 500000000:  local Pi=0.874676 global Pi=0.8746757835
Thread 2 did 500000000 to 750000000:  local Pi=0.719414 global Pi=1.5940897827
Thread 0 did 0 to 250000000:  local Pi=0.979915 global Pi=2.5740044352
Thread 3 did 750000000 to 1000000000:  local Pi=0.567588 global Pi=3.1415926536
Pi =  3.1415926536」」
```

When I first tried this, almost none of the basic Pthread constructs worked, and it really shocked me. Honestly, I think the documentation has serious problems. Even the developers opened an [issue](https://github.com/emscripten-core/emscripten/issues/11554) saying it needs improvement.

## Performance analysis

Next, I want to measure the same parallel π-computation logic under three scenarios: (1) native pthread, (2) pthread compiled to WASM, and (3) Web Workers written in JavaScript.

For (1) and (2), I use `pi.c` from this post and the WASM generated by Emscripten. For (3), I use the JavaScript Web Worker code from my earlier post “[
Evaluation of Web Worker for Parallel Programming with Browsers, NodeJS and Deno](/post/2020/06/js/web-worker-evaluation/#NodeJS)”. To save space, I will not repost the code for (3) here. Interested readers can click into that post.

The three implementations are logically the same: the total number of loop iterations is the same, and they spawn the same number of threads. The experiments are run on Windows 10 WSL 1 (Ubuntu 20.04) with an AMD Ryzen 7 2700X 3.7 GHz 8-core CPU (using only 4 threads), NodeJS v12.18, and emcc v1.39. Both gcc and emcc use the default -O2 optimization. This set of experiments can also be compared with the earlier [Evaluation of Web Worker](/post/2020/06/js/web-worker-evaluation/) post (you can treat the NodeJS result as a baseline).

The only difference is that implementation (3) uses integers to represent π. This is because if you want to combine SharedArrayBuffer with Atomics (locking), the buffer must be an integer typed array. If you need floating-point, you would have to encode it manually. In this experiment, I do not do floating-point encoding.


| Case  | Time(s)  |
|---|---|
|  pthread | 0.751  |
|  em2wasm | 1.174  |
|  js |  0.486 |

And then I found that the JS version was dramatically faster. That felt unbelievable.

Thinking about it, maybe integer arithmetic is much faster than floating-point arithmetic, and the precision is also different. So to be fair, I changed `double* pi` in `pi.c` to `unsigned* pi`, so the C logic also computes π in a scaled integer form and divides it back at the end. I call that version `pi2.c`. JavaScript uses `double` for floating-point calculations, so this matches the precision of the original C `double` version. Surprisingly, `pi2.c` was about the same speed, and even slightly slower at 0.77s.

Then I thought: maybe mutex locking is slower. In JS I used Atomics, so I changed `pi2.c` to use `atomic_fetch_add_explicit` for locking, and called that `pi3.c`. The result was 0.755s—basically no difference. But that also makes sense: mutex locking is indeed slower, but the difference becomes obvious only when magnified a lot more.

Then I used `perf` to look at the C code:

```shell
       │     local_pi += 4 / (1 + x * x);
       │       movsd     -0x8(%rbp),%xmm0
  0.02 │       mulsd     -0x8(%rbp),%xmm0
  0.02 │       movsd     _IO_stdin_used+0x60,%xmm1
  0.02 │       addsd     %xmm1,%xmm0
  0.13 │       movsd     _IO_stdin_used+0x68,%xmm1
       │       divsd     %xmm0,%xmm1
 13.20 │       movapd    %xmm1,%xmm0
  0.07 │       movsd     -0x28(%rbp),%xmm1
 34.67 │       addsd     %xmm1,%xmm0
 42.64 │       movsd     %xmm0,-0x28(%rbp) 
```

Aha! I forgot that in my post “[A simple Pthreads parallelization example and performance analysis](/post/2020/07/simple-pthread-usage/)” I already pointed out this issue: for some reason, -O2 spends a lot of time on memory operations, but -O3 does not.

So I quickly compiled `pi3.c` with -O3 and reran it:

```shell
$ time gcc pi3.c -lpthread -g -O3
real    0m0.177s
$ time ./a.out
real    0m0.350s
```

The runtime alone is 0.350s, which is finally faster than JS. But JS is JIT. So if you also count the C compile time (0.177s), the total becomes 0.527s—still slower than JS at 0.486s. I have to admit it: V8, you win. How on earth do you compile this?!

So I reran the benchmark using `pi2.c` (intentionally using pthread mutex), and using -O3 for both gcc and emcc:

| Case  | Time(s)  |
|---|---|
|  pthread | 0.346  |
|  em2wasm | 0.525  |
|  js |  0.504 |

This result finally matches expectations. Much better!

You can see that the performance of “Pthread → Web Worker + WASM” is only slightly slower than writing Web Workers directly in JS. Still, V8 is truly impressive. When gcc and emcc use -O3, their compile time is long, but V8’s compile+run time is still faster than gcc’s compile+run time. My guess is that this is because the program is very small. For larger programs, the performance gap between C and JS should become more significant.

## Conclusion

This post explained how to use Emscripten to convert C/C++ Pthreads into Web Workers and WebAssembly, and compared performance across (1) native C code, (2) Emscripten-generated JS/WASM, and (3) Web Workers written directly in JavaScript. With -O3, native C is about 30% faster than the Pthread-to-WASM version, and the Pthread-to-WASM version is roughly comparable to pure JavaScript Web Workers.
