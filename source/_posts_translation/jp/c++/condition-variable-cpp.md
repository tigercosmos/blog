---
title: "C++ の Condition Variable：シンプルな例"
date: 2023-06-25 3:00:00
tags: [c++, condition variable]
des: "本記事では、C++ の std::condition_variable を使ったシンプルな例を紹介します。"
lang: jp
translation_key: condition-variable-cpp
---

## 前書き

[Condition Variable（条件変数）](https://zh.wikipedia.org/wiki/%E7%9B%A3%E8%A6%96%E5%99%A8_(%E7%A8%8B%E5%BA%8F%E5%90%8C%E6%AD%A5%E5%8C%96)#%E6%A2%9D%E4%BB%B6%E8%AE%8A%E6%95%B8(Condition_Variable)) を初めて聞いたのは、台大（NTU）のオペレーティングシステム（OS）講義でした。正直当時はコンピュータの概念がまだかなり曖昧で、なんとなく受講して単位を取った後はこの話をすっかり忘れていました（当時は「先生の教え方が悪いから理解できないんだ」と思っていましたが、たぶん自分の理解力が足りなかっただけかもしれません）。

交大（NCTU）の大学院に進んで、並列計算の TA をしていたときですら、意外にも使ったことがありませんでした（今でも不思議です）。余談ですが、並列計算（parallel computing）と並行処理（concurrency）は似ているようで少し違いますし、どちらもとても奥が深い分野です。今でも自分が「分かっている」とは言えません。並列計算に関する [LNCS の論文](https://link.springer.com/chapter/10.1007/978-3-031-20891-1_13) があっても、面接で少し突っ込まれるだけで普通にボコボコにされます。学びには終わりがなく、しかも難しいものだと痛感します。

話を戻します。ここ数日、仕事でとても典型的な [Producer–Consumer Problem（生産者・消費者問題）](https://zh.wikipedia.org/zh-tw/%E7%94%9F%E4%BA%A7%E8%80%85%E6%B6%88%E8%B4%B9%E8%80%85%E9%97%AE%E9%A2%98) を書く必要がありました。扱う課題について同僚に聞いてみたところ、「それって Condition Variable じゃない？ちょっと見れば役に立つと思うよ」と言われました（もちろん英語で、ですが）。長年 CS を学んできたのに、いまになって初めて実務で Condition Variable を書くのは情けない気持ちになりました。とはいえ、書き方を調べたら意外とすぐ実装できましたし、短期間で 2 回使う機会があったのも偶然です。

前置きはここまで。この投稿では Condition Variable の概念を詳しく解説しません。恐竜本（OS の教科書）や Wikipedia を参照してください。高度な実装テクニックも扱いません。あくまで「C++ の `std::condition_variable` はこういうふうに使える」というシンプルな例です。

皆さん、始めましょう！

![Cover，長野地獄谷雪猴](https://github.com/tigercosmos/blog/assets/18013815/2f3ae415-8d97-42eb-bb3c-9d5a49146590)
（長野地獄谷雪猴）

## Condition Variable の例

Condition Variable をざっくり説明すると、マルチスレッドプログラムにおいて、共有メモリ上の状態変化を待つための仕組みです。たとえばスレッド A がグローバル変数 `FLAG` を 0 から 1 に変更し、スレッド B は `FLAG` が 1 になったことを検知したタイミングで処理を開始します。

先ほど触れたように、Condition Variable は生産者・消費者問題でよく使われます。この場合、生産者が「イベント」や「データ」を生成し、消費者は「イベントが発生した」または「データが生成された」タイミングで対応する処理を行います。

これを C++ で実装するために、まず次の 2 つの変数をグローバルに用意します：

```cpp
std::condition_variable g_cond;
std::mutex g_mutex;
```

> この例では簡単のためグローバル変数にしていますが、実装上は「生産者 P と消費者 Q の両方からアクセスできる」ことが重要です。たとえば P と Q が同じクラス内にあるなら、上の 2 つをクラスのメンバ変数にしても問題ありません。

### 生産者（Producer）

まずは生産者側のコードです：

```cpp
void run_producer_thread(std::queue<std::string> &queue) {
    for (;;) {
        auto word = generateRandomString(); // 任意データ
        std::lock_guard<std::mutex> lock{g_mutex}; // consumer が同時に読まないよう queue をロック
        queue.push(word);
        g_cond.notify_one(); // consumer に通知
    }
}
```

この例では、生産者がデータをキュー（Queue）に入れます。ここで `g_mutex` を使い、`queue` が消費者によって同時アクセスされないようにしています。

`g_cond.notify_one` は消費者に対して「準備できた！」という通知を送ります。

### 消費者（Consumer）

次に消費者側のコードです：

```cpp
void run_consumer_thread(std::queue<std::string> &queue) {
    for (;;) {
        std::unique_lock<std::mutex> lock(g_mutex);        // このロックは柔軟に扱える
        g_cond.wait(lock, [&] { return !queue.empty(); }); // 条件チェック中はロックが保持される

        auto word = queue.front(); // できるだけ早く unlock したいので先にコピーする
        queue.pop();

        lock.unlock(); // g_mutex のロックを解除
    }
}
```

ここでは `std::unique_lock` を使っています。`unique_lock` は [多くの機能](https://en.cppreference.com/w/cpp/thread/unique_lock)（deferred locking、時間制限付き lock、再帰 lock、所有権の移譲、Condition Variable との連携など）を持ちます（ごめんなさい、全部説明するのは面倒です :P）。

この例に限って言えば、「後で柔軟に使えるロックで、Condition Variable と一緒に使える」という理解で十分です。

`g_cond.wait` は待機（ブロック）し続け、`notify_one` を受け取ったときに条件の評価を行います。ここでは `queue` が空でない（データが来た）ことを確認しています。条件が `true` ならブロック解除され、`false` なら待機を継続します。

毎回の条件評価の前に、**`lock` の所有権が取得**されます。つまり、先にロックしてから `!queue.empty()` をチェックします。

条件が **満たされない** 場合はロックを解放して待機へ戻り、満たされる場合はロックを保持したまま `g_cond.wait` 以降の処理へ進みます。

その後、消費者は `queue` からデータを取り出せます。

そして次の `for` ループへ進み、再び `g_cond.wait` による待機が始まります。

> 注意点として、実際にイベントが起きる前でも、OS の内部機構により `g_cond.wait` が複数回 **解除（wakeup）** されることがあります。これは [Spurious Wakeup（虚偽の起床）](https://en.wikipedia.org/wiki/Spurious_wakeup) と呼ばれます。虚偽か `notify_one` によるものかに関わらず、起床するたびに `lock` を取得して条件を確認します。

### 完全なサンプルコード

完全なサンプルコードは次のとおりです。実際に動かして、結果を観察することを強くおすすめします。

```cpp
// g++ example.cpp -std=c++17 -lpthread

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>

std::condition_variable g_cond; // 通知に使う
std::mutex g_mutex;             // データ保護に使う

// 乱数文字列を生成する。単なるヘルパー関数で、この例の本質ではない
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

        std::lock_guard<std::mutex> lock{g_mutex}; // consumer が同時に読まないよう queue をロック
        queue.push(word);
        g_cond.notify_one(); // consumer に通知
        std::cout << "P: producer just push a word: `" << word << "`" << std::endl << std::flush;

        std::cout << "P: producer sleep for 1500 ms" << std::endl << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // 観察しやすいように 1500 ms 眠る
    }
}

void run_consumer_thread(std::queue<std::string> &queue) {
    for (;;) {
        std::cout << "C: == This is consumer thread ==" << std::endl << std::flush;
        std::unique_lock<std::mutex> lock(g_mutex);        // 次行で queue の状態を確認するための lock
        g_cond.wait(lock, [&] { return !queue.empty(); }); // queue が空でないことを待つ

        auto word = queue.front(); // できるだけ早く unlock したいので先にコピーする
        queue.pop();

        lock.unlock(); // g_mutex のロックを解除

        std::cout << "C: Consumer get a word `" << word << "`" << std::endl << std::flush;

        // コメントアウトを外して影響を見てみてください。何が起きて、なぜそうなるでしょうか？
        // std::cout << "C: consumer sleep for 1500 ms" << std::endl << std::flush;
        // std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // 観察しやすいように 1500 ms 眠る
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

まとめると、Condition Variable はマルチスレッド開発において、スレッド間の同期やデータ受け渡しを実現するために使われます。ロックの取り方やタイミングを誤ると簡単に Dead Lock になるため注意が必要です。実際の適用例としては、生産者・消費者問題、Readers–Writers Problem（読者・書き手問題）、スレッド同期、タスク協調、ワークキューなどが挙げられます。
