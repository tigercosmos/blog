---
title: 簡易 Pthreads 平行化範例與效能分析
date: 2020-07-02 23:00:00
tags: [c, pthread, parallel programming, 效能分析, 平行化]
des: "POSIX 執行緒 (Pthreads) 讓我們能用 C/C++ 寫出平行程式。本文提供一個簡單的 Pthread 平行化計算 PI 的範例，了解怎樣用 Pthread 建立執行緒，給予設定，並加上互斥鎖。最後簡單做了效能分析。"
lang: zh
translation_key: simple-pthread-usage
---

## 簡介

[POSIX 執行緒](https://zh.wikipedia.org/zh-tw/POSIX%E7%BA%BF%E7%A8%8B) (Pthreads) 讓我們能用 C/C++ 寫出平行程式。pthread 是一套定義好的 API 函式庫，我們只需呼叫 `pthread_` 開頭的 API，背後就會幫我們完成平行化的機制。

可以平行化的經典情境有很多，基本上只要有迴圈並且執行內容相依性很低，就可以做平行化。我最喜歡用的範例是計算 PI，本文也會以算 PI 為範例。

## 單緒計算 PI 

先來看如果只用一個執行緒怎樣算 PI：

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

執行結果如下：

```shell
$ gcc pi_single_thread.c && ./a.out
3.1415926536
```

可以看到這範例中就只有一個迴圈，`sum` 執行的動作是很容易可以被獨立切開的，因此很適合做平行化。

## 用 pthread 平行化計算 PI

我們將上面的程式碼以 pthread 改寫，程式碼如下：

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
} Arg; // 傳入 thread 的參數型別

pthread_t callThd[NUMTHRDS]; // 宣告建立 pthread
pthread_mutex_t mutexsum;    // pthread 互斥鎖

// 每個 thread 要做的任務
void *count_pi(void *arg)
{

   Arg *data = (Arg *)arg;
   int thread_id = data->thread_id;
   int start = data->start;
   int end = data->end;
   double *pi = data->pi;

   // 將原本的 PI 算法切成好幾份
   double x;
   double local_pi = 0;
   double step = 1 / MAGNIFICATION;
   for (int i = start; i < end; i++)
   {
      x = (i + 0.5) * step;
      local_pi += 4 / (1 + x * x);
   }

   local_pi *= step;

   // **** 關鍵區域 ****
   // 一次只允許一個 thread 存取
   pthread_mutex_lock(&mutexsum);
   // 將部分的 PI 加進最後的 PI
   *pi += local_pi;
   pthread_mutex_unlock(&mutexsum);
   // *****************

   printf("Thread %d did %d to %d:  local Pi=%lf global Pi=%.10lf\n", thread_id, start,
          end, local_pi, *pi);

   pthread_exit((void *)0);
}

int main(int argc, char *argv[])
{
   // 初始化互斥鎖
   pthread_mutex_init(&mutexsum, NULL);

   // 設定 pthread 性質是要能 join
   pthread_attr_t attr;
   pthread_attr_init(&attr);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

   // 每個 thread 都可以存取的 PI
   // 因為不同 thread 都要能存取，故用指標
   double *pi = malloc(sizeof(*pi));
   *pi = 0;

   int part = MAGNIFICATION / NUMTHRDS;

   Arg arg[NUMTHRDS]; // 每個 thread 傳入的參數
   for (int i = 0; i < NUMTHRDS; i++)
   {
      // 設定傳入參數
      arg[i].thread_id = i;
      arg[i].start = part * i;
      arg[i].end = part * (i + 1);
      arg[i].pi = pi; // PI 的指標，所有 thread 共用

      // 建立一個 thread，執行 count_pi 任務，傳入 arg[i] 指標參數
      pthread_create(&callThd[i], &attr, count_pi, (void *)&arg[i]);
   }

   // 回收性質設定
   pthread_attr_destroy(&attr);

   void *status;
   for (int i = 0; i < NUMTHRDS; i++)
   {
      // 等待每一個 thread 執行完畢
      pthread_join(callThd[i], &status);
   }

   // 所有 thread 執行完畢，印出 PI
   printf("Pi =  %.10lf \n", *pi);

   // 回收互斥鎖
   pthread_mutex_destroy(&mutexsum);
   // 離開
   pthread_exit(NULL);
}
```

執行結果如下：

```shell
$ gcc pi_multi_thread.c  -lpthread && ./a.out
Thread 3 did 750000000 to 1000000000:  local Pi=0.567588 global Pi=0.5675882184
Thread 2 did 500000000 to 750000000:  local Pi=0.719414 global Pi=1.2870022176
Thread 1 did 250000000 to 500000000:  local Pi=0.874676 global Pi=2.1616780011
Thread 0 did 0 to 250000000:  local Pi=0.979915 global Pi=3.1415926536
Pi =  3.1415926536
```

## 效能分析

以下做個小實驗，使用 AMD Ryzen 7 2700X Eight-Core Processor 在 VM Ubuntu 20 下測試單緒版本和多緒版本時間差異。

測試碼使用上面的 `pi_single_thread.c` 和 `pi_multi_thread.c`。

以 GCC 7.5 O2 優化編譯，測試結果如下：

| Thread  | Time(s)  |
|---|---|
|  1 | 3.1113  |
|  2 | 1.531  |
|  4 |  0.817 |
|  8 |  0.489 |
| 16 |  0.345 |

![Time-Threads](https://user-images.githubusercontent.com/18013815/86372410-b11fae00-bcb4-11ea-9c25-5db81e9a9d55.png)

從 1 緒到 2 緒直接時間減半，但從 8 緒到 16 緒時間只減少了一點，這滿合理的，因為執行緒一多，要處理資料同步還有記憶體的成本就增加了。

此外觀察一下 1 緒、8 緒、16 緒的 `perf stat`：

| Thread  | CPU Usage  | Page Fault |
|---|---|---|
|  1 | 0.998  | 52 |
|  8 |  7.353 | 86 |
| 16 |  13.439 | 105 |

因為我電腦就只有 16 緒，所以開 16 緒 CPU 使用率也只有 13.4/16，這讓實際運算時間又比預期更久。此外也可以看到 Page Fault 數量增加許多。

再來看看哪段程式碼跑最久。

![perf code time](https://user-images.githubusercontent.com/18013815/86374997-c2b68500-bcb7-11ea-9db0-3f257c09d460.png)

從上圖可以看到，不意外地，大部分時間成本都花在計算 PI 的關鍵兩行程式碼：

```c
x = (i + 0.5) * step;
local_pi += 4 / (1 + x * x);
```

`movapd` 這個指令花的時間有點過久，在 O2 編譯的情況下這一步將資料存進記憶體非常合理，但若是用 O3 編譯時會有更好的暫存器配置，就不會有這指令了。

## 結論

本文提供一個簡單的 Pthread 平行化計算 PI 的範例，了解怎樣用 Pthread 建立執行緒，給予設定，並加上互斥鎖。最後簡單做了效能分析。
