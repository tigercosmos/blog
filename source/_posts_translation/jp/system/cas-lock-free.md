---
title: "Compare-and-Swap でロックフリーを実現する"
date: 2020-10-28 02:25:00
tags: [compare and swap, lock free, atomic, parallel programming]
des: "本記事では Compare-and-Swap（CAS）の原理を紹介し、CAS のオーバーヘッドがロックより小さくなり得ることを実験で示します。"
lang: jp
translation_key: cas-lock-free
---

## 1. はじめに

ロックには多くの欠点があります。たとえばロックは余計なオーバーヘッドを生み、使い方を誤るとデッドロックを引き起こします。そのため、可能な限りロックを避けたいところです。ロックフリーな書き方を採用すると、ロックの使用を減らせるだけでなく、クリティカルセクションを扱う際の余計なオーバーヘッドも避けられます。

ロックフリーの方法の一つとして、[Compare and Swap (CAS)](https://en.wikipedia.org/wiki/Compare-and-swap) を使う手法があります。CAS はアトミック命令であるためコストが小さく、同時にマルチスレッド環境でもデータの安全性を確保できます。

本記事では Compare-and-Swap の原理を紹介し、CAS のオーバーヘッドがロックより小さくなり得ることを実験で示します。

## 2. Compare-and-Swap の擬似コード

CAS は一般にハードウェアでサポートされ、コンパイラから対応する intrinsic を呼び出せます。CAS を擬似コードの関数として書くと、次のようになります。

```c++
bool CAS(int* p, int old, int new) {
    if *p ≠ old {
        return false;
    }
    *p ← new
    return true;
}
```

CAS 命令には 3 つの引数があります。1 つ目は比較対象の変数ポインタ `*p`、2 つ目はそのポインタが指す「期待する古い値」`old`、3 つ目は更新したい新しい値 `new` です。

CAS の典型的な使い方は、次のような形になります。

```c
int old, new;

do {
    old = *p;
    new = NEW_VALUE;
} while(!CAS(*p, old, new));
```

CAS を囲むループはクリティカルセクションと見なせます。CAS 命令を実行するとき、CAS は `*p` の値が `old` と同じかどうかを確認します。同じであれば、実行中に他のスレッドが `*p` を変更していないことを意味するため、安心して `*p` を `new` に更新できます。逆に `*p` が `old` と異なる場合は、すでに誰かが `*p` を変更していることを意味します。そのため、この反復は破棄してループをやり直し、次は他のスレッドから干渉されないことを期待します。

このように CAS を使うことでロックを一切取らずにロックフリーを実現できます。ただし CAS が失敗してリトライする可能性があるため、ブロックフリーではありません。しかし実際には失敗しても 1〜2 回程度で済むことが多く、スレッドセーフかつ低オーバーヘッドを実現できます。

## 3. Compare-and-Swap の例

### 3.1 逐次版の合計

まずはとても簡単な合計プログラムを書きます。ひたすら加算するだけです。

```c
#include <stdio.h>

int main() {

    int sum = 0;

    for(int i = 0; i < 10000000; i++) {

        for(int i = 0; i < 500; i++) {} // 何かの処理があって時間がかかる体にする

        sum += 3; // わざと 2 命令にする
        sum -= 2;
    }

    printf("sum = %d\n", sum);
}
```

実行時間：

```shell
$ gcc test.c; time ./a.out
sum = 10000000

real    0m7.548s
```

### 3.2 ロックなし OpenMP マルチスレッド版

次に OpenMP を使ってマルチスレッド化します。

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

私のマシンは 4 thread なので、速度はおよそ 4 倍になっています。しかし結果が正しくありません。期待した 10000000 ではなく 9120084 になっています。これはロックを取っていないため、スレッドが古い値を読み取ってしまうことがあるからです。（数が大きいほどこの問題はさらに顕著になります）

### 3.3 ロックあり OpenMP マルチスレッド版

そこでロックを追加します。

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

結果は正しくなりましたが、時間が少し伸びています。つまりロックにはオーバーヘッドがあることが分かります。

### 3.4 ロックフリー版（CAS）

次に CAS を使ったロックフリー方式にします。私は GCC を使っているので、GCC の `__sync_bool_compare_and_swap` API を利用できます。もちろん `std::atomic` が提供する API を使っても構いません。

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

CAS の時間はロックなし版よりはわずかに長いものの、ロックあり版よりは速いことが分かります（2.099s vs 2.116s）。先ほど述べた通りロックにはオーバーヘッドがあり、ロックが頻繁になるほど影響は大きくなります。

さらにコードにカウンタを入れ、CAS が合計で何回失敗したかを数えました。得られた回数は 262408 でした。つまり、もともと 10000000 回あるクリティカルセクションのうち、CAS は合計 262408 回リトライしており、全体の 2.6% を占めます。連続して 2 回以上失敗した回数は約 2800 回で、確率は 0.028% でした。

CAS を使わない場合、実質的に 1e7 回分のロックのオーバーヘッドを支払うことになります。一方 CAS を使う場合、リトライが必要になる確率は 2.6% なので、概算では (1.026 * 1e7) 回の CAS と 1e7 回のロックを比較することになります。CAS はアトミック命令であり、ロックよりも少ないサイクルで済むため、最終的には CAS が勝ちます。

## 4. 結論

ロックはデータを保護できますが、その代償があります。したがって、使えるならなるべく少なく使うべきです。状況によってはロックフリーな書き方やアトミック操作を採用することで、プログラムのオーバーヘッドを大きく下げられます。
