---
title: 使用 Emscripten 將 Pthread 轉成 JavaScript 與效能分析
date: 2020-07-07 18:00:00
tags: [JavaScript, web worker, nodejs, c, pthread, parallel programming, browser, browsers, 效能分析, 平行化]
des: "本文介紹如何使用 Emscripten 去將 C/C++ Pthread 轉換成 Web Worker 和 WebAssembly，並對 (1)原生 C Code (2)Emscripten 轉換的 JS (3)直接用 JS 寫的 Web Worker，三種情境作效能分析。發現在開 O3 優化的情況下，Native C 比 Pthread 轉換的 WASM 快 30% 左右，並且用 Pthread 轉換的 WASM 和直接用 Web Worker 的 JS 差不多。"
lang: zh
translation_key: emscripten-pthread-to-js
---

## 簡介

Emscripten 是可以將 C/C++ 轉換成 WebAssembly 的工具，背後是透過 LLVM 轉換，支援轉換 Pthread，會轉成 JavaScript 的 Web Worker 加 WebAssembly，甚至可以將 OpenGL 轉成 WebGL，讓程式在網頁上跑還能有接近原生程式的效能。

其中 Pthread 轉成 Web Worker 加 WebAssembly 的部分，就是本文要介紹的重點。我會拿一個範例程式來實際轉換看看，不過要找一個好的測試程式不容易，所以我寫一支 Pthread 計算 PI 的平行程式，用來做轉換測試。

本文先介紹如何使用 Emscripten 將 Pthread 轉 JS，照著官網教學做的過程採到一些坑，順便也記錄下來，免得大家又落坑了。隨後會分析 (1)原生 C Code (2)Emscripten 轉換的 JS (3)直接用 JS 寫的 Web Worker，三種情境下的效能差異。
<!-- more -->

## Pthread 範例程式

範例程式是用 Pthread 平行計算 PI 的小程式，使用這個程式有幾個原因，寫平行程式最重要是測他能不能開新的執行緒，能不能使用共享記憶體，能不能上鎖，以及能不能等待其他執行緒，基本上一個平行程式的概念差不多就這樣，而這個計算 PI 的 Pthread 程式剛好足夠滿足我需求。

為了節省篇幅，Pthread 詳細可以參考之前的文章「[簡易 Pthreads 平行化範例與效能分析](/post/2020/07/simple-pthread-usage/)」。

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

## Emscripten 下載

要用 Emscripten 要先下載他的 Github Repo，不過你電腦要記得先裝 Git。

```shell
# 取得 emsdk repo
git clone https://github.com/emscripten-core/emsdk.git

# 進去資料夾
cd emsdk
```

接著照著步驟

```shell
# 下載最新的 sdk
./emsdk install latest

# 啟用最新的 sdk
./emsdk activate latest

# 加入環境變數
source ./emsdk_env.sh
```

一旦安裝啟用過，之後每次要使用 Emscripten，就要進來 `emsdk` 執行 `source ./emsdk_env.sh` 即可。

如果你不想要每次都要重新啟動環境變數，你可以考慮將 `.bashrc` 加入 `source ./emsdk_env.sh`，不過這個設定會蓋掉 NodeJS 的位置，所以不是很建議。

## Emscripten 入門

`emcc` 或 `em++` 是 Emscripten 的前端程式 (相當於 Clang 之於 LLVM)，不過這樣說有點複雜，就是跑這個程式就可以編譯 C/C++ 成 WebAssembly (WASM)。

簡單示範：

```c
// hello.c
#include <stdio.h>

int main() {
  printf("hello, world!\n");
  return 0;
}
```

編譯：

```shell
$ emcc hello.c
```

這樣就會生成 `a.out.js` 和 `a.out.wasm`，之所以是這樣是因為 WASM 目前不管是瀏覽器或是 NodeJS 都需要由 JS 來啟動。

```shell
$ emcc hello.c -o hello.html
```

你也可以指定輸出成 HTML，那就會是一個可以用瀏覽器開的範例，網頁檔裡面會有個虛擬的終端機，可以看到原本的程式碼在網頁裡的終端機執行。

這邊要注意，如果輸出成 HTML，必須先開一個 Local Server 來開啟網頁，這是因為瀏覽器不支援 `file://` XHR 請求，同時 Emscripten 對於 HTTP Header 的 File Type 還有特別要求，其實還頗麻煩。這部分細節請上 Emscripten 官網查詢。

更多使用教學可以參考 [Emscripten 的 Tutorial](https://emscripten.org/docs/getting_started/Tutorial.html#emscripten-tutorial)。

## Emscripten 編譯 Pthread

現在我們要來編譯 `pi.c`，根據官網說法編譯 Pthread 要在後面加上 `-s USE_PTHREADS=1` 參數。

```shell
$ emcc pi.c -s USE_PTHREADS=1
```

讓我們來看看這樣會有啥結果 (NodeJS 要加上 `--experimental-wasm-threads --experimental-wasm-bulk-memory`)：

```
$ node  --experimental-wasm-threads --experimental-wasm-bulk-memory a.out.js

// 完全沒動靜
```

沒錯，他就這樣卡住了。

上 Github 問才知道，原來 Emscripten 的機制問題 (細節我也沒很懂)，參考我提的 [issue](https://github.com/emscripten-core/emscripten/issues/11543#issuecomment-654317178)，總之有三個方法可以解決：

1. 編譯加上 `-s PROXY_TO_PTHREAD`
2. 編譯加上 `-s PTHREAD_POOL_SIZE=N` 其中 N > 0
3. 將 `main()` 用 `emscripten_set_main_loop()` 取代

我們用以下編譯：

```shell
$ emcc pi.c  -s USE_PTHREADS=1  -s PTHREAD_POOL_SIZE=4
```

這下能正常執行了！

```
$ node  --experimental-wasm-threads --experimental-wasm-bulk-memory a.out.js
Thread 1 did 250000000 to 500000000:  local Pi=0.874676 global Pi=0.8746757835
Thread 2 did 500000000 to 750000000:  local Pi=0.719414 global Pi=1.5940897827
Thread 0 did 0 to 250000000:  local Pi=0.979915 global Pi=2.5740044352
Thread 3 did 750000000 to 1000000000:  local Pi=0.567588 global Pi=3.1415926536
Pi =  3.1415926536」」
```

我當初試 Pthread 幾乎所有基本語法都跑不動，真是嚇傻了。老實說我覺得是文件很有問題，開發成員也提出 [issue](https://github.com/emscripten-core/emscripten/issues/11554) 指出需要改進。

## 效能分析

接下來要測試，同一個計算 PI 的平行化邏輯，使用 (1) 原生 pthread (2) pthread 轉 WASM (3) 用 JS 寫 Web Worker，這三種情況下效能差異。

其中 (1)、(2) 的程式碼使用本文的 `pi.c` 和 Emscripten 轉出來的 WASM，(3) 使用我先前文章「[
Evaluation of Web Worker for Parallel Programming with Browsers, NodeJS and Deno](/post/2020/06/js/web-worker-evaluation/#NodeJS)」中的 JS Web Worker 程式碼。為節省篇幅，這邊不重複 PO (3) 的程式碼，有興趣讀者可以點進文章查看。

基本上三種程式碼的邏輯是一樣的，控制迭代的總迴圈一樣，並且開相同數量的 thread，在 Windows 10 WSL 1 的 Ubuntu 20.04 中，以 AMD Ryzen 7 2700X 3.7 GHz 八核處理器 (只開 4 執行緒) 中做實驗，NodeJS v12.18 和 emcc v1.39。gcc 和 emcc 都用預設的 O2 優化。本文實驗也可以跟先前的 [Evaluation of Web Worker](/post/2020/06/js/web-worker-evaluation/) 文章一起比較 (可以用 NodeJS 執行結果當基準)。

唯一差異是，(3) 的程式碼是使用整數型態來處理 PI，這是因為 SharedArrayBuffer 要搭配 Atomic (上鎖) 的話，只能是整數型別的 Buffer。如果要處理浮點數型別的話，只能採用編碼處理，但這邊實驗不做型別轉換的編碼處理。


| Case  | Time(s)  |
|---|---|
|  pthread | 0.751  |
|  em2wasm | 1.174  |
|  js |  0.486 |

然後發現 JS 版本竟然快超多，簡直不可思議！

後來想想也許是整數型別運算比浮點數運算快很多，此外精度也不一樣。所以為了公平起見，我將 `pi.c` 原本 `double* pi` 改成 `unsigned* pi`，讓 C 的邏輯一樣是先放大計算 PI 最後再除回去，這個版本為 `pi2.c`。JS 計算浮點數時都是 `double`，因此原本 C 的 `double` 精度是一致的。結果沒想到改過的 `pi2.c` 的速度還是差不多，反而還慢了一點點跑出 0.77 S。

接著我又想，也許是 Mutex 上鎖比較慢，JS 中是使用 Atomic，我又將 `pi2.c` 改成用 `atomic_fetch_add_explicit` 來上鎖，為 `pi3.c`。想到結果跑出 0.755 S，意思是幾乎沒差，不過想想也是，上鎖速度的確 Mutex 會慢一點，但是也是要放大很多倍的時候才會比較顯著。

後來我用 perf 看 C 程式碼的結果：

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

啊哈！我都忘記自己在「[簡易 Pthreads 平行化範例與效能分析](/post/2020/07/simple-pthread-usage/)」文章中就點出這個問題，O2 不知道為啥在處理記憶體部分很花時間，但 O3 就沒這問題。

於是我趕緊用 O3 編譯 `pi3.c` 跑看看：

```shell
$ time gcc pi3.c -lpthread -g -O3
real    0m0.177s
$ time ./a.out
real    0m0.350s
```

可以看到只看執行時間只花 0.350s，總算是比 JS 快了，但 JS 是 JIT，於是如果把 C 的編譯時間 0.177s 也算進去的話，總共是 0.527s，比 JS 的 0.486 還慢！我真的服了，V8 算你厲害，你到底是怎麼編譯的！

所以重新跑一次結果，採用 `pi2.c` (故意測 Pthread 的 Mutex)，gcc 和 emcc 都用 O3 優化：

| Case  | Time(s)  |
|---|---|
|  pthread | 0.346  |
|  em2wasm | 0.525  |
|  js |  0.504 |

這結果總算符合預期了！舒服！

可以發現 Pthread 轉 Web Worker + WASM 的效能只比我直接用 JS 寫慢一點點而已。不過 V8 真的很厲害，當 gcc 和 emcc 開 O3 時，編譯時間都很久，但 V8 卻是編譯加執行時間還比 gcc 編譯加執行時間快。推測是因為程式很小隻，如果是比較大型的程式，C 和 JS 的效能差異就會比較顯著了。

## 結論

本文介紹如何使用 Emscripten 去將 C/C++ Pthread 轉換成 Web Worker 和 WebAssembly，並對 (1)原生 C Code (2)Emscripten 轉換的 JS (3)直接用 JS 寫的 Web Worker，三種情境作效能分析。發現在開 O3 優化的情況下，Native C 比 Pthread 轉換的 WASM 快 30% 左右，並且用 Pthread 轉換的 WASM 和直接用 Web Worker 的 JS 差不多。
