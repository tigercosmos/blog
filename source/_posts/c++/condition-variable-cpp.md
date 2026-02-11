---
title: Condition Variable in C++ 一個簡單的範例
date: 2023-06-25 3:00:00
tags: [c++, condition variable]
des: "本文提供了一個簡單的範例，介紹如何用 C++ 寫 Condition Variable"
lang: zh
translation_key: condition-variable-cpp
---

## 前言

我第一次聽到 [Condition Variable（條件變數）](https://zh.wikipedia.org/wiki/%E7%9B%A3%E8%A6%96%E5%99%A8_(%E7%A8%8B%E5%BA%8F%E5%90%8C%E6%AD%A5%E5%8C%96)#%E6%A2%9D%E4%BB%B6%E8%AE%8A%E6%95%B8(Condition_Variable)) 是在台大資工作業系統課上，老實說當時我對電腦概念還非常模糊，糊里糊塗修完課後就忘記了這玩意（當時覺得老師教不好，為啥就是聽不懂，也許是慧根不夠）。

在交大資工所即使當了平行計算助教，很意外的還是沒用過，我現在也覺得挺意外的（？）題外話，平行計算本來就跟並行運算有些區別就是了，另外就是這兩門學問都博大精深，我到現在都不敢說自己「會」，即便我有一篇跟平行計算有關的[論文](https://link.springer.com/chapter/10.1007/978-3-031-20891-1_13)上 LNCS，連工作面試時別人隨便考個平行計算都被電，常常讓我感慨學無止盡且艱難。

扯遠了，這幾天工作上要寫一個很經典的[生產者消費者問題](https://zh.wikipedia.org/zh-tw/%E7%94%9F%E4%BA%A7%E8%80%85%E6%B6%88%E8%B4%B9%E8%80%85%E9%97%AE%E9%A2%98)（Producer-Consumer Problem），問了一下同事我要處理的問題，他就說「阿這不就是 Condition Variable 嗎，去看一下應該會有幫助」，當然他是用英文跟我講啦。我覺得滿滿的慚愧，好歹學電腦也好幾年了，竟然到現在才實際去寫 Condition Variable，查了一下怎麼寫算是滿快就搞定，巧的是短期就有機會用了兩次。

廢話講完了，這篇文章不打算詳細解釋 Condition Variable 是什麼，可以詳閱恐龍書（作業系統概論）或是 Wikipedia 看一下。我也不會教很高深的實作技巧，這篇就是一個簡單的範例告訴你 C++ `std::condition_variable` 可以這樣用。

皆さん、始めましょう！

![Cover，長野地獄谷雪猴](https://github.com/tigercosmos/blog/assets/18013815/2f3ae415-8d97-42eb-bb3c-9d5a49146590)
（長野地獄谷雪猴）

## Condition Variable 範例

說到 Condition Variable，簡單來說他的概念是讓多執行緒的程式中，使一個或多個執行緒去等待一個共享記憶體的資料變化，比方說 A 執行緒改變全域變數 `FLAG` 從 0 變 1，B 執行緒發現 `FLAG` 變成 1 的時候就開始執行任務。

前面提到 Condition Variable 可以用在生產者消費者問題上，這個情況下，生產者會產生一堆「事件」或「資料」，而消費者則會在「事件發生」或是「資料產生」時去作對應的事情。


要用 C++ 實作這個問題，首先我們會需要以下兩個變數放在全域：

```cpp
std::condition_variable g_cond;
std::mutex g_mutex;
```

> 在此範例中我簡單把他放在全域變數，但實作上只需要讓生產者 P 和消費者 Q 都能存取的就好，例如 P 和 Q 都在同一個 Class，上面兩個變數可以是 Class 的成員變數即可。


### 生產者

先來看看生產者的程式碼：

```cpp
void run_producer_thread(std::queue<std::string> &queue) {
    for (;;) {
        auto word = generateRandomString(); // 任意資料
        std::lock_guard<std::mutex> lock{g_mutex}; // 鎖住 queue 以免同時被 consumer 讀取
        queue.push(word);
        g_cond.notify_one(); // 通知 consumer
    }
}
```

在這個例子中，生產者把資料丟入佇列（Queue），這邊我們用到了 `g_mutex` 來確保 `queue` 不會同時被生產者讀取。

`g_cond.notify_one` 會負責發送一個「我好了！」的訊號給消費者。

### 消費者

再來看看消費者的程式碼：

```cpp
void run_consumer_thread(std::queue<std::string> &queue) {
    for (;;) {
        std::unique_lock<std::mutex> lock(g_mutex);        // 這邊的 lock 還不會真的上鎖
        g_cond.wait(lock, [&] { return !queue.empty(); }); // 當進行檢查 queue 不為空時，lock 會上鎖

        auto word = queue.front(); // 把資料拷貝出來，才能盡快解鎖
        queue.pop();

        lock.unlock(); // 解除 g_mutex 的鎖
    }
}
```

注意這邊用到 `std::unique_lock`，他的[功能很多](https://en.cppreference.com/w/cpp/thread/unique_lock)，包含以下「deferred locking, time-constrained attempts at locking, recursive locking, transfer of lock ownership, and use with condition variables」（抱歉我懶 :P）。

在這個範例下，白話解釋就是「晚一點才會用上你」的一種鎖，所以宣告當下並不會真的上鎖。

`g_cond.wait` 會一直做等待（或稱做阻塞，Blocking），並且當收到 `notify_one` 時，觸發進行條件檢查。這邊的檢查是確認 `queue` 是否不為空（亦即有資料進來）。如果 `g_cond` 的檢查得到 `true`，則取消阻塞，反之繼續進行等待。

在每次做檢查之前，**`lock` 的所有權會被取得**，才能進行條件檢查（換句話說，`lock` 先上鎖，才檢查 `!queue.empty()`）

檢查完如果**未達成**條件 `lock` 就會解鎖，反之就會繼續保持上鎖狀態，程式接著執行 `g_cond.wait` 之後的程式碼。

接者生產者就可以把資料從 `queue` 中拿出來。

然後進入下一個 for 循環，`g_cond.wait` 觸發開始等待。

> 值得注意的是，在實際事件發生之前，`g_cond.wait` 可能會因為作業系統底層的機制而多次**解除阻塞**，稱做[虛假喚醒（Spurious Wakeup）](https://en.wikipedia.org/wiki/Spurious_wakeup)。在每次喚醒時，無論是虛驚一場或是真的來自 `notify_one` 通知，`lock` 都會上鎖來做檢查條件。

### 完整範例程式碼

完整範例程式碼如下，強烈建議大家實際跑跑看，看看結果。

```cpp
// g++ example.cpp -std=c++17 -lpthread

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>

std::condition_variable g_cond; // 用來做通知
std::mutex g_mutex;             // 用來保護資料

// 產生隨機字串，只是個 Helper 函數，不是本範例重點
std::string generateRandomString() {
    std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(0, characters.length() - 1);

    std::string result;
    int randomLength = distribution(generator);
    for (int i = 0; i < randomLength; ++i) {
        int randomIndex = distribution(generator);
        result += characters[randomIndex];
    }

    return result;
}

void run_producer_thread(std::queue<std::string> &queue) {
    for (;;) {
        std::cout << "P: === This is producer thread ===" << std::endl << std::flush;

        auto word = generateRandomString();

        std::lock_guard<std::mutex> lock{g_mutex}; // 鎖住 queue 以免同時被 consumer 讀取
        queue.push(word);
        g_cond.notify_one(); // 通知 consumer
        std::cout << "P: producer just push a word: `" << word << "`" << std::endl << std::flush;

        std::cout << "P: producer sleep for 1500 ms" << std::endl << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // 睡 1500 ms 方便觀察
    }
}

void run_consumer_thread(std::queue<std::string> &queue) {
    for (;;) {
        std::cout << "C: == This is consumer thread ==" << std::endl << std::flush;
        std::unique_lock<std::mutex> lock(g_mutex);        // 這邊的 lock 用來下一行檢查 queue 狀態
        g_cond.wait(lock, [&] { return !queue.empty(); }); // 當 queue 不為空時

        auto word = queue.front(); // 把資料拷貝出來，才能盡快解鎖
        queue.pop();

        lock.unlock(); // 解除 g_mutex 的鎖

        std::cout << "C: Consumer get a word `" << word << "`" << std::endl << std::flush;

        // 可以試試看取消註解來看影響，有什麼特別的情況，為什麼會這樣？
        // std::cout << "C: consumer sleep for 1500 ms" << std::endl << std::flush;
        // std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // 睡 1500 ms 方便觀察
    }
}

int main() {
    std::queue<std::string> queue;

    auto producer_thread = std::thread([&queue]() { run_producer_thread(queue); });
    auto consumer_thread = std::thread([&queue]() { run_consumer_thread(queue); });

    producer_thread.join();
    consumer_thread.join();
}
```

## 結論

總結，Condition Variable 是多執行緒程式開發中確保執行緒之間的同步和資料傳遞。使用時要小心，上鎖的方式與時機，一不小心可能就會 Dead Lock。實際應用場景包含生產者消費者問題、讀者寫者問題（Readers-Writers Problem）、執行緒同步、任務協調、工作佇列等等。
