---
title: "A Simple Example of Condition Variables in C++"
date: 2023-06-25 3:00:00
tags: [c++, condition variable]
des: "This post provides a simple example of using std::condition_variable in C++."
lang: en
translation_key: condition-variable-cpp
---

## Preface

The first time I heard about [Condition Variables](https://zh.wikipedia.org/wiki/%E7%9B%A3%E8%A6%96%E5%99%A8_(%E7%A8%8B%E5%BA%8F%E5%90%8C%E6%AD%A5%E5%8C%96)#%E6%A2%9D%E4%BB%B6%E8%AE%8A%E6%95%B8(Condition_Variable)) was in the Operating Systems course at NTU. To be honest, back then my understanding of computer fundamentals was still very fuzzy. I somehow finished the class, then promptly forgot about it (at the time, I felt like the instructor was not good—why couldn’t I understand? Maybe I just didn’t have the talent).

Even during grad school at NCTU, I surprisingly still never used it—even while being a teaching assistant for parallel computing (which I still find kind of surprising). As an aside, parallel computing is not exactly the same as concurrency. And both areas are extremely deep; even today I don’t dare claim I “know” them. Despite having a [paper](https://link.springer.com/chapter/10.1007/978-3-031-20891-1_13) on LNCS related to parallel computing, I still get destroyed in interviews when someone casually asks parallel computing questions. It often reminds me how endless (and difficult) learning can be.

Anyway, in the last few days at work I needed to implement the classic [Producer–Consumer Problem](https://zh.wikipedia.org/zh-tw/%E7%94%9F%E4%BA%A7%E8%80%85%E6%B6%88%E8%B4%B9%E8%80%85%E9%97%AE%E9%A2%98). I asked a colleague about it, and he said, “Isn’t that just a condition variable? Go take a look—it should help.” (Of course, he said it in English.) I felt pretty ashamed. I’ve been studying CS for years, yet only now am I actually writing condition variables in real code. After checking how to do it, it turned out to be fairly quick. Coincidentally, I got to use it twice in a short period.

Enough rambling. This post won’t explain in detail what a condition variable is—you can read a proper OS textbook or Wikipedia for that. I also won’t cover advanced implementation tricks. This is just a simple example showing how you can use C++ `std::condition_variable` like this.

Alright, let’s get started!

![Cover, Jigokudani Snow Monkeys in Nagano](https://github.com/tigercosmos/blog/assets/18013815/2f3ae415-8d97-42eb-bb3c-9d5a49146590)
(Jigokudani Snow Monkeys in Nagano)

## Condition Variable Example

Speaking of condition variables: conceptually, they allow one or more threads in a multithreaded program to wait for a change in shared state. For example, thread A updates a global variable `FLAG` from 0 to 1, and thread B starts doing its work after it observes `FLAG` becoming 1.

As mentioned earlier, condition variables are often used for the producer–consumer problem. In this scenario, the producer generates “events” or “data”, and the consumer reacts when “an event happens” or “data is produced”.

To implement this in C++, we first need the following two global variables:

```cpp
std::condition_variable g_cond;
std::mutex g_mutex;
```

> In this example I simply put them in global scope, but in practice you only need them to be accessible by the producer P and the consumer Q. For instance, if P and Q are in the same class, these two variables can be class members.

### Producer

Let’s look at the producer code first:

```cpp
void run_producer_thread(std::queue<std::string> &queue) {
    for (;;) {
        auto word = generateRandomString(); // arbitrary data
        std::lock_guard<std::mutex> lock{g_mutex}; // lock the queue to avoid concurrent access by the consumer
        queue.push(word);
        g_cond.notify_one(); // notify the consumer
    }
}
```

In this example, the producer pushes data into a queue. Here we use `g_mutex` to ensure the `queue` won’t be accessed concurrently.

`g_cond.notify_one` is responsible for sending a “I’m ready!” signal to the consumer.

### Consumer

Now let’s look at the consumer code:

```cpp
void run_consumer_thread(std::queue<std::string> &queue) {
    for (;;) {
        std::unique_lock<std::mutex> lock(g_mutex);        // this lock does not necessarily lock immediately in all scenarios
        g_cond.wait(lock, [&] { return !queue.empty(); }); // during the predicate check, the lock will be held

        auto word = queue.front(); // copy the data out so we can unlock ASAP
        queue.pop();

        lock.unlock(); // release g_mutex
    }
}
```

Notice that we use `std::unique_lock`. It has [many features](https://en.cppreference.com/w/cpp/thread/unique_lock), including “deferred locking, time-constrained attempts at locking, recursive locking, transfer of lock ownership, and use with condition variables” (sorry, I’m lazy :P).

In plain terms for this example: it’s a lock that you can use in a more flexible way, and it works with condition variables.

`g_cond.wait` keeps waiting (i.e., blocking). When it receives `notify_one`, it wakes up and evaluates the condition. Here the condition checks whether `queue` is non-empty (i.e., data has arrived). If the predicate evaluates to `true`, it stops blocking; otherwise, it continues waiting.

Before every predicate evaluation, **the ownership of `lock` is acquired**, so the condition can be checked (in other words: it locks first, then checks `!queue.empty()`).

If the condition is **not satisfied**, the lock is released and it goes back to waiting. If it is satisfied, it continues executing the code after `g_cond.wait` while keeping the lock held.

Then the consumer can take the data out of the `queue`.

After that, it enters the next `for` iteration, and `g_cond.wait` starts waiting again.

> Note that before the actual event occurs, `g_cond.wait` can be woken up multiple times due to OS-level mechanisms, which is called a [Spurious Wakeup](https://en.wikipedia.org/wiki/Spurious_wakeup). On every wakeup—whether it is spurious or triggered by `notify_one`—the `lock` will be acquired to check the predicate.

### Full Example Code

Here is the full example code. I highly recommend running it yourself and observing the output.

```cpp
// g++ example.cpp -std=c++17 -lpthread

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>

std::condition_variable g_cond; // used for notifications
std::mutex g_mutex;             // protects the shared data

// Generate a random string. Just a helper function; not the focus of this example.
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

        std::lock_guard<std::mutex> lock{g_mutex}; // lock the queue to avoid concurrent access by the consumer
        queue.push(word);
        g_cond.notify_one(); // notify the consumer
        std::cout << "P: producer just push a word: `" << word << "`" << std::endl << std::flush;

        std::cout << "P: producer sleep for 1500 ms" << std::endl << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // sleep for 1500 ms for easier observation
    }
}

void run_consumer_thread(std::queue<std::string> &queue) {
    for (;;) {
        std::cout << "C: == This is consumer thread ==" << std::endl << std::flush;
        std::unique_lock<std::mutex> lock(g_mutex);        // the lock is used in the next line to check queue state
        g_cond.wait(lock, [&] { return !queue.empty(); }); // wait until the queue is not empty

        auto word = queue.front(); // copy the data out so we can unlock ASAP
        queue.pop();

        lock.unlock(); // release g_mutex

        std::cout << "C: Consumer get a word `" << word << "`" << std::endl << std::flush;

        // Try uncommenting these lines and observe the impact. What special situation happens, and why?
        // std::cout << "C: consumer sleep for 1500 ms" << std::endl << std::flush;
        // std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // sleep for 1500 ms for easier observation
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

## Conclusion

In summary, condition variables are used in multithreaded programming to ensure synchronization and data transfer between threads. You need to be careful about how and when you lock; otherwise you can easily end up with a deadlock. Common use cases include the producer–consumer problem, the readers–writers problem, thread synchronization, task coordination, work queues, and so on.
