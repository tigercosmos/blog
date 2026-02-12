---
title: "使用 Compare and Swap 做到 Lock Free"
date: 2020-10-28 02:25:00
tags: [compare and swap, lock free, atomic, parallel programming]
des: "本文介紹 Compare and Swap 原理，並實驗證明 CAS 的 Overhead 比 Lock 還小。"
lang: zh
translation_key: cas-lock-free
---

## 1. 簡介

我們都知道 Lock 有很多缺點，例如 Lock 會造成多餘的 Overhead，使用不慎也會造成 Dead Lock，所以希望能盡量避免使用 Lock。於是使用 Lock Free 的程式寫法就可以幫我們減少使用 Lock，並且在處理 Critical Section 不會有多餘的 Overhead。

一種 Lock Free 的方法可以使用 [Compare and Swap (CAS)](https://en.wikipedia.org/wiki/Compare-and-swap) 的技術，由於他是 Atomic Instruction，所以它的代價很小，同時可以確保多執行緒時候可以資料是安全的。

本文介紹 Compare and Swap 原理，並實驗證明 CAS 的 Overhead 比 Lock 還小。

## 2. Compare and Swap Pseudocode

CAS 一般來說是由硬體支援，然後編譯器可以呼叫對應的 Instrinsic，如果將 CAS 寫成虛擬碼函數的話會長這樣：

```c++
bool CAS(int* p, int old, int new) {
    if *p ≠ old {
        return false;
    }
    *p ← new
    return true;
}
```

可以看到說 CAS 指令的有三個參數，第一個是要比較的變數指標 `*p`，第二個是該指標原本舊的數值 `old`，以及該指標應該要更新的新數值 `new`。

一般使用 CAS 的方法大概會是以下情境：

```c
int old, new;

do {
    old = *p;
    new = NEW_VALUE;
} while(!CAS(*p, old, new));
```

CAS 包住的迴圈可以視為 Critical Section，當執行到 CAS 指令時，CAS 會去看 `*p` 的值和 `old` 是否一樣，一樣的話代表在執行期間沒有其他 Thread 去改動過 `*p`，於是就可以安心將 `*p` 更新成 `new` 的數值。反之，如果 `*p` 的數值和 `old` 不一樣，代表期間已經有人動過 `*p`，那麼這次迴圈數值作罷，再重新跑一次迴圈，並希望這次不會受到別的 Thread 干擾。

可以看到藉由 CAS 的方法我們可以做到 Lock Free，因為我們不需要上任何鎖，但她卻不是 Block Free，因為 While 的 CAS 有機會失敗重跑，但這種情況很低，最多發生一兩次，所以我們可以做到 Thread Safe 且低 Overhead。

## 3. Compare and Swap Example

### 3.1 Sum Serial Version

這邊我寫了一個很簡單的 Sum 程式，可以看到就是一直往上加。

```c
#include <stdio.h>

int main() {

    int sum = 0;

    for(int i = 0; i < 10000000; i++) {

        for(int i = 0; i < 500; i++) {} // 假裝有任務要跑一段時間

        sum += 3; // 故意讓他有兩個指令
        sum -= 2;
    }

    printf("sum = %d\n", sum);
}
```

執行時間：

```shell
$ gcc test.c; time ./a.out
sum = 10000000

real    0m7.548s
```

### 3.2 Sum OpenMP Multi-Thread without Lock

接著我們使用 OpenMP 來改成 Multi-thread 的程式。

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

因為我用的機器有 4 thread，所以可以看到速度差不多快了四倍。但我們發現答案不對，不是預期要的 10000000 而是 9120084，這是因為我們沒有上鎖的關係，所以會有 Thread 拿到舊資料的情況。(數字越大時這種意外會越明顯)


### 3.3 Sum OpenMP Multi-Thread with Lock

於是我幫程式加上了 Lock。

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

可以看到答案就對了，但是所花的時間稍微變長了一點點，所以我們可以知道 Lock 是有 Overhead 的。

### 3.4 Sum OpenMP Multi-Thread with Lock-Free

接下來我們使用 CAS 的 Lock Free 方法，因為我是用 GCC，所以可以使用 GCC 的 `__sync_bool_compare_and_swap` API 來做到。當然你也可以使用 `std::atomic` 裡面提供的 API。

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

我們發現 CAS 的時間雖然比沒用 Lock 還要久一點點，但卻比上 Lock 還要快 (2.099s vs 2.116s)，如同先前所說的，Lock 是會有 Overhead 的，而且當 Lock 越頻繁時會越明顯。

接下來我又在程式碼中插入一個 Counter，看 CAS 總共失敗多少次，那我得到的答案是 262408。也就是說，在原本 10000000 個 Critical Section 中，
CAS 總共重跑了 262408 次，佔所有次數中的 2.6%。連續失敗兩次以上的次數約為 2800，機率為 0.028%。

如果不使用 CAS，我們等於有 1e7 個 Lock 的 Overhead，但使用 CAS 的情況下，我們僅有 2.6% 的機率會需要重跑，所以總共約是 (1.026 * 1e7) 個 CAS 對比 1e7 個 Lock。因為 CAS 是 atomic instruction，比 Lock 少了不少 cycles，所以最後會是 CAS 獲勝。

## 4. 結論

Lock 可以保護我們的資料，但他會有一些代價，所以我們能少用就少用，一些情境如果可以用 Lock Free 的寫法，或是使用 Atomic 的操作的話，可以大大降低我們程式的 Overhead。
