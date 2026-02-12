---
title: 深入理解 Reentrancy 和 Thread-safe
date: 2021-05-10 06:20:08
tags: [parallel programming, reentrancy, thread-safe, system]
des: "本文詳細介紹 Reentrancy 和 Thread-safe，並搭配程式碼範例解說。"
lang: zh
translation_key: reentrant-and-thread-safe-code
---

## 1. 並行與平行

Concurrency (並行) 和 Parallelism (平行) 是非常像的概念。前者是說不同的計算並行地執行，相較於序列化 (Sequentially) 執行，在一個計算結束之前另一個就開始了；後者則是讓同個計算工作同時分工一起執行。

乍看之下這兩者要講的概念其實很像，採用 [The Art of Concurrency](https://www.oreilly.com/library/view/the-art-of/9780596802424/) 的定義如下：

> A system is said to be concurrent if it can support two or more actions in progress at the same time. A system is said to be parallel if it can support two or more actions executing simultaneously. The key concept and difference between these definitions is the phrase "in progress."
>
> 一個系統稱做並行是當他可以同時執行兩個行進中的任務。一個系統稱為平行是當它可以同時執行多個任務。兩者關鍵差異是「行進中」。

並行化可以在單核處理器 (Single Core) 上可以藉由交錯地執行任務來辦到，有趣的是並行化也可以在多核處理器上以平行的方式執行。某種程度上並行跟平行做的事情滿像的，但「行進中」精闢點出兩者區別，並且英文中 simultaneously 有 "happening or being done at exactly the same time" 的特性。

對於平行和並行不熟的同學可以考慮複習 Operating System Concepts 第四章「Threads & Concurrency」，或是 [Operating Systems: Three Easy Pieces](https://pages.cs.wisc.edu/~remzi/OSTEP/) 的 「Concurrency」 章節。

而在並行與平行中，為了確保程式執行邏輯正確性和資料正確性，就會需要討論 Reentrancy 和 Thread-safe 分別是啥玩意。

![Cover](https://user-images.githubusercontent.com/18013815/117589349-823b5980-b15b-11eb-825d-55307d4c044b.png)

## 2. Reentrancy 

在電腦科學中，[Reentrancy (可重入)](https://zh.wikipedia.org/wiki/%E5%8F%AF%E9%87%8D%E5%85%A5) 代表一個程式或子程式裡面的程式碼可以「在任意時刻發生中斷 (Interruption)，然後作業系統排程 (Schedule) 執行另外一段程式碼，當再次回來執行這段程式碼時卻不會出錯」。

至於為甚麼程式會甚麼會有中斷，有可能是內部自己的行為像是 `jump` 或 `call`，抑或是外部行為像是 Interrupt 或 Signal。也就是說，在程式執行的過程中，中斷的發生無關乎作業系統 (OS) 是否存在，沒有 OS 情況下，自身的行為也可能會引發中斷，所以還是要注意 Reentrancy 影響。

Reentrancy 在單執行緒的形況下就可以做討論，像是程式是否能在被 OS Interrupt 後，再次直接恢復執行？換句話說，如果被中斷後要能繼續執行，應該要是 Reentrant Code，否則回來的時候答案就錯了。這邊還有個有趣的問題，那就是 [Interrupt Handler 需要是 Reentrant 嗎](https://stackoverflow.com/questions/18132580/does-an-interrupt-handler-have-to-be-reentrant)？簡單的答案是，除非你的 Handler 是巢狀的 (一個接一個呼叫)，否則是不需要擔心，另外像是 Linux 會有遮罩阻止另一個 Interrupt 打斷目前的 Interrupt。

Reentrancy 非常重要是因為在 Concurrent Programming (並行程式設計) 開發中，我們需要確保異步程式 (Asynchronous Program) 在做任務切換的時候，中斷不會影響我們執行正確性。此外當我們使用遞迴的時候，預期他就是 Reentrant 否則會出錯。

## 3. Thread-safe

另一方面， [Thread-safe (執行緒安全)](https://zh.wikipedia.org/zh-tw/%E7%BA%BF%E7%A8%8B%E5%AE%89%E5%85%A8)是指某個函數、函數庫在多執行緒環境中被呼叫時，能夠正確地處理多個執行緒之間的公用變數 (全域變數、共享變數)，使程式功能正確完成。

Thread-safe 在平行程式中很重要，因為平行計算的過程中，往往很多資料是共用的，這樣處理資料的時候，很容易讓資料產生 [Race Conditions](https://en.wikipedia.org/wiki/Race_condition#Computing)，因此如何確保資料在每個 Thread 都有被正確進行讀寫就很關鍵。

所以 Thread-safe 基本上就是在避免資料發生 Data Racing，實現機制可以使用 Reentrancy，但也可以使用 Thread-Local Data (只存在自己執行緒的資料)、不可改變的物件 (Immutable Objects)、互斥鎖 (Mutex)、原子化操作 (Atomic Operations)。

## 4. Reentrancy VS Thread-safe

所以重點來了，那 Reentrancy 與 Thread-safe 關係是甚麼呢？

兩者其實互不等於對方，但卻有部分交集，亦即 Reentrancy 可以是也可以不是 Thread-safe，反之 Thread-safe 對 Reentrancy 也是如此。

以下就直接用程式碼範例來做不同情況下的解釋。

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
    - `t` 放在外部，如果 `swap` 執行到一半，接著另一個人修改了 `t`，那回來繼續執行時，行為就不正確。
- ❌ Thread-safe
    - `t` 是 Global
    - 當其他 Thread 呼叫 `my_func` 時，`t` 可能剛好屬於同個 Context (程式執行環境)，則 `t` 的行為就會難以預測。

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
    - 雖然 `t` 是 Thread 自己的，但同個 Thead 中可能在多次呼叫中改變 `t`。
- ✅ Thread-safe
    - `t` 現在是 Thread 自己的，別的 Thread 不可能去影響 `t`。

### 4.3 Reentrancy ✅ | Thread-safe ❌

以下為刻意製造的情境，但可以想成程式很複雜的情況下會有的狀況。

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
    - `swap` 執行前後 `t` 都會保持不變。注意這邊關鍵其實是因為 `swap` 不會對外部造成影響，亦即所有變數變化只保留在 `swap` 裡面。
- ❌ Thread-safe
    - `t` 是全域變數，理由跟前面一樣。


### 4.4 Reentrancy ✅ | Thread-safe ✅

這個例子中的解決方法意外的容易，把全域變數拿掉就沒事了。

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
    - 所有數據都存在 Stack 中，因此不會被影響。
- ✅ Thread-safe
    - 沒有共享的資料，不會有 Data Racing。


## 5. Reentrancy 與 Thread-safe 的原則

看過上面的範例之後，所以我們如果要寫出 Reentrancy 或 Thread-safe 可以遵守以下原則。

Reentrancy：
- 不能含有 static (global) 非常量 (non-constant) 數據。
- 不能返回 static (global) 非常量數據的地址。
- 只能處理由呼叫者 (Caller) 提供的數據。(透過參數傳入)
- 呼叫的函數也必需也是 Reentrant。

Thread-safe：
- 基本上只要避免 Race Condition 就好
- Lock 是你的好朋友

## 6. Reentrant 與 Thread-Safe 函式庫

Reentrant 與 Thread-Safe 函式庫在我們寫平行程式或開發異步程式的時候很重要。

在 GNU C 函式庫中，有分以下安全 (Safety) 等級：MT-Safe (Multi-Thread-Safe)、AS-Safe (Async-Signal-Safe)、AC-Safe (Async-Cancel-Safe) 以及各種不安全的等級。

一些 C 標準函式庫的函式，像是 `ctime` 和 `strtok` 就不是 Reentrant。但通常他們會有一個對應的 Reentrant 版本，名稱通常會是 `_r` 後綴，例如 `strtok_r` 或 `rand_r`。

事實上我們也可以透過 `man` 命令 (Command) 來確認，例如在 Ubuntu 16 上我查 `man rand_r`，得到以下結果 (片段)：

```
ATTRIBUTES
       For an explanation of the terms used in this section, see attributes(7).

       ┌──────────────────────────┬───────────────┬─────────┐
       │Interface                 │ Attribute     │ Value   │
       ├──────────────────────────┼───────────────┼─────────┤
       │rand(), rand_r(), srand() │ Thread safety │ MT-Safe │
       └──────────────────────────┴───────────────┴─────────┘
```

可以看到我們查到 `rand_r` 是 MT-Safe，意味著我們可以用他在平行程式中。MT-Safe 強調在多執行緒 (MT) 的環境中，原本預期的功能 (如這邊的亂數產生器) 仍可正確地展現，且不該存在因為放到多執行緒而有的錯誤 (即 functional safety)。

所以要注意的是，`MT-Safe` 不代表程式碼就是完全安全的，例如當連續呼叫 MT-Safe 的程式的時候，行為可能會是無法預期的。

那你可能就要問了，那如果我在平行程式上用非 MT-Safe 的函數呢？那可能會有兩種可能，一種是你得到錯的答案，因為他就不是安全的，第二種是你可能效率會比較差，因為他可能會不斷去搶外部變數。舉例來說，假設你在平行程式去呼叫 `rand` 而非 `rand_r`，那你將會發現亂數產生的速度會極慢 (甚至會不正確)，因為 `rand` 的實作中，就有使用 `static`。

## 7. 參考資料

1. cjwind's note. 2017. [Reentrancy and Thread-safety](http://www.cjwind.idv.tw/Reentrancy-and-Thread-safety/)
2. Mike Choi. 2017. [Reentrant and Threadsafe Code](https://deadbeef.me/2017/09/reentrant-threadsafe)
3. IBM. 1997. [AIX Version 4.3 General Programming Concepts: Writing Reentrant and Thread-safe Code](https://sites.ualberta.ca/dept/chemeng/AIX-43/share/man/info/C/a_doc_lib/aixprggd/genprogc/writing_reentrant_thread_safe_code.htm)
4. GNU.ORG. 2021. [POSIX Safety Concepts](https://www.gnu.org/software/libc/manual/html_node/POSIX-Safety-Concepts.html)
