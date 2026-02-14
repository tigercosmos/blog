---
title: "Pthreads による簡単な並列化例と性能分析"
date: 2020-07-02 23:00:00
tags: [c, pthread, parallel programming, 效能分析, 平行化]
des: "POSIX スレッド（Pthreads）を使うと、C/C++ で並列プログラムを書けます。本記事では π を計算する簡単な例を使い、Pthreads でスレッドを作成して設定し、ミューテックスを追加する方法を説明します。最後に簡単な性能分析も行います。"
lang: jp
translation_key: simple-pthread-usage
---

## はじめに

[POSIX スレッド](https://zh.wikipedia.org/zh-tw/POSIX%E7%BA%BF%E7%A8%8B)（Pthreads）を使うと、C/C++ で並列プログラムを書けます。pthread は定義済みの API 関数群で、`pthread_` で始まる API を呼び出すだけで、背後で並列化の仕組みを提供してくれます。

並列化できる典型的な場面はたくさんあります。基本的にはループがあり、各反復の実行内容の依存性が低ければ、並列化できる可能性が高いです。私がいちばん好きな例は π の計算で、本記事でも π を計算する例を使います。

## 単一スレッドで π を計算する

まずは 1 スレッドで π を計算する方法を見てみます。

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

実行結果は次の通りです。

```shell
$ gcc pi_single_thread.c && ./a.out
3.1415926536
```

この例には 1 つのループしかありません。`sum` の計算は独立に分割しやすいため、並列化に適しています。

## pthread で π 計算を並列化する

では、上のコードを pthread で書き換えます。コードは次の通りです。

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
} Arg; // thread に渡す引数型

pthread_t callThd[NUMTHRDS]; // pthread の配列
pthread_mutex_t mutexsum;    // pthread ミューテックス

// 各 thread が実行する処理
void *count_pi(void *arg)
{

   Arg *data = (Arg *)arg;
   int thread_id = data->thread_id;
   int start = data->start;
   int end = data->end;
   double *pi = data->pi;

   // 元の π 計算を複数パートに分割
   double x;
   double local_pi = 0;
   double step = 1 / MAGNIFICATION;
   for (int i = start; i < end; i++)
   {
      x = (i + 0.5) * step;
      local_pi += 4 / (1 + x * x);
   }

   local_pi *= step;

   // **** クリティカルセクション ****
   // 同時に 1 thread だけがアクセスできるようにする
   pthread_mutex_lock(&mutexsum);
   // 部分 π を全体 π に加算
   *pi += local_pi;
   pthread_mutex_unlock(&mutexsum);
   // *****************

   printf("Thread %d did %d to %d:  local Pi=%lf global Pi=%.10lf\n", thread_id, start,
          end, local_pi, *pi);

   pthread_exit((void *)0);
}

int main(int argc, char *argv[])
{
   // ミューテックス初期化
   pthread_mutex_init(&mutexsum, NULL);

   // thread 属性を join 可能にする
   pthread_attr_t attr;
   pthread_attr_init(&attr);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

   // 全 thread で共有する π
   // 複数 thread からアクセスされるためポインタを使う
   double *pi = malloc(sizeof(*pi));
   *pi = 0;

   int part = MAGNIFICATION / NUMTHRDS;

   Arg arg[NUMTHRDS]; // 各 thread に渡す引数
   for (int i = 0; i < NUMTHRDS; i++)
   {
      // 引数設定
      arg[i].thread_id = i;
      arg[i].start = part * i;
      arg[i].end = part * (i + 1);
      arg[i].pi = pi; // 全 thread で共有する π のポインタ

      // thread を作成し、count_pi を実行。引数として &arg[i] を渡す
      pthread_create(&callThd[i], &attr, count_pi, (void *)&arg[i]);
   }

   // 属性を破棄
   pthread_attr_destroy(&attr);

   void *status;
   for (int i = 0; i < NUMTHRDS; i++)
   {
      // 各 thread の終了を待つ
      pthread_join(callThd[i], &status);
   }

   // 全 thread が終了したので π を表示
   printf("Pi =  %.10lf \n", *pi);

   // ミューテックスを破棄
   pthread_mutex_destroy(&mutexsum);
   // 終了
   pthread_exit(NULL);
}
```

実行結果は次の通りです。

```shell
$ gcc pi_multi_thread.c  -lpthread && ./a.out
Thread 3 did 750000000 to 1000000000:  local Pi=0.567588 global Pi=0.5675882184
Thread 2 did 500000000 to 750000000:  local Pi=0.719414 global Pi=1.2870022176
Thread 1 did 250000000 to 500000000:  local Pi=0.874676 global Pi=2.1616780011
Thread 0 did 0 to 250000000:  local Pi=0.979915 global Pi=3.1415926536
Pi =  3.1415926536
```

## 性能分析

小さな実験をしてみます。AMD Ryzen 7 2700X Eight-Core Processor を使い、VM 上の Ubuntu 20 で単一スレッド版とマルチスレッド版の時間差を測定しました。

テストコードは上の `pi_single_thread.c` と `pi_multi_thread.c` を使います。

GCC 7.5 を `-O2` で最適化してコンパイルし、結果は次の通りです。

| Thread  | Time(s)  |
|---|---|
|  1 | 3.1113  |
|  2 | 1.531  |
|  4 |  0.817 |
|  8 |  0.489 |
| 16 |  0.345 |

![Time-Threads](https://user-images.githubusercontent.com/18013815/86372410-b11fae00-bcb4-11ea-9c25-5db81e9a9d55.png)

1 スレッドから 2 スレッドでは時間がほぼ半分になりますが、8 スレッドから 16 スレッドでは少ししか減っていません。これは合理的で、スレッドが増えるほど同期やメモリアクセスのコストが増えるからです。

次に、1 スレッド／8 スレッド／16 スレッドの `perf stat` を見ます。

| Thread  | CPU Usage  | Page Fault |
|---|---|---|
|  1 | 0.998  | 52 |
|  8 |  7.353 | 86 |
| 16 |  13.439 | 105 |

私のマシンは 16 スレッドしかないので、16 スレッドで回しても CPU 使用率は 13.4/16 しかありません。そのため実際の実行時間が予想より長くなります。また Page Fault 数も増えていることが分かります。

次に、どのコードが最も時間を消費しているかを見てみます。

![perf code time](https://user-images.githubusercontent.com/18013815/86374997-c2b68500-bcb7-11ea-9db0-3f257c09d460.png)

上の図から分かるように、ほとんどの時間は π を計算する重要な 2 行に費やされています。

```c
x = (i + 0.5) * step;
local_pi += 4 / (1 + x * x);
```

`movapd` 命令に時間がかかりすぎています。`-O2` の場合、このステップでデータをメモリに格納するのは合理的ですが、`-O3` でコンパイルするとより良いレジスタ割り当てが得られ、この命令が消えることがあります。

## 結論

本記事では、Pthreads を使って π の計算を並列化する簡単な例を示し、スレッドの作成、設定、ミューテックスの追加方法を説明しました。最後に簡単な性能分析も行いました。

