---
title: "Linux で perf を使った性能分析（入門）"
date: 2020-08-29 00:20:08
tags: [linux, perf, 效能分析,]
des: "本記事では Linux の性能分析ツール perf を紹介します。簡単なプログラム例を通じて、perf でプログラムを分析する方法と、プロファイラを使うことで根本原因をより簡単に見つけられることを示します。"
lang: jp
translation_key: perf-basic
---

## イントロダクション

性能分析ツール（Profiler）を使うと、ソフトウェアの実行に関するより多くの情報を得られます。たとえば、使用メモリ量、CPU サイクル、キャッシュミス、I/O 処理時間などです。これらの情報は、プログラムの性能ボトルネックを見つける上で非常に役立ちます。どこがプログラムを遅くしているのかを見つけ、性能を最大化することが性能分析の最大の目的です。

本記事では Linux の性能分析ツール [perf](http://www.brendangregg.com/perf.html) を紹介します。簡単なプログラム例を使って perf による分析手順を示し、分析ツールを使うと問題の根本原因をより見つけやすくなることを確認します。本文は Gabriel Krisman Bertaz の [Performance analysis in Linux](https://www.collabora.com/news-and-blog/blog/2017/03/21/performance-analysis-in-linux/) を参考にしています。
<!-- more -->

私の解説動画も合わせてどうぞ：

<iframe width="560" height="315" src="https://www.youtube.com/embed/Mba2ONCA0kI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Branch Prediction の例

Stack Overflow に「[Why is processing a sorted array faster than processing an unsorted array?](https://stackoverflow.com/questions/11227809/)」という有名な質問があります。

そのコードは次の通りです：

`test.cc`：

```c++
#include <algorithm>
#include <ctime>
#include <iostream>

int main()
{
    // 測試用陣列
    const int arr_len = 32768;
    int data[arr_len];

    for (int c = 0; c < arr_len; ++c)
        data[c] = std::rand() % 256;

    // std::sort(data, data + arr_len); // 是否排序
    
    long long sum = 0;

    for (int i = 0; i < 30000; ++i)
    {
        for (int c = 0; c < arr_len; ++c)
        {
            if (data[c] >= 128) { // 故意選 256 一半  
                sum += data[c];
            }
        }
    }

    std::cout << "sum = " << sum << std::endl;
}
```

まずは未ソート版をコンパイルします：

```shell
$ g++ test.cc -o unsort
```

次に `sort` の行のコメントを外して、もう一度コンパイルします：

```shell
$ g++ test.cc -o sort
```

実行時間を見てみます：

```shell
$ time ./unsort
real    0m5.671s

$ time ./sort
real    0m1.932s
```

問題の趣旨は、`data` をソートすると上のコードが速くなる、という点です。実験結果もその通りになっています。ソートの計算量は $O(NlogN)$ なので、ソートせずにそのまま回す $O(N)$ より遅いはず……と思いがちですが、結果は逆です。

結論として、これは CPU の **Branch Prediction**（分岐予測）が原因です。ざっくり言うと、前回 `if` が `true` だったら次も `true` だと先に予測し、CPU はその予測に基づいて先行実行（投機実行）します。予測が当たれば速くなりますが、外れた場合は先行実行した内容をすべて捨てる必要があり、かえって時間を浪費します。これを Branch Miss（詳細は「計算機組織」など）と呼びます。つまり Branch Prediction は諸刃の剣で、条件分岐の結果が揃いやすければ加速しますが、結果がコロコロ変わると予測ミスが増えて遅くなります。上のコードでソート版が速いのは、予測ミスが `data` が `128` 付近を跨ぐ一箇所でしか起きないためです。それ以前はずっと `false`、それ以後はずっと `true` になります。

## perf 性能分析ツール

プログラムの問題箇所を見つけるのは簡単ではありません。上の例でも、アルゴリズムの観点から分析すると的外れになりがちで、実際の問題は計算機組織の原理にあります。単純なコードですら原因を見失う可能性があるのに、巨大なプログラムになると、アルゴリズム、メモリキャッシュ、CPU 命令、ネットワーク、I/O など、さまざまな要素が絡み合います。そこで分析ツールが必要になります。

Linux には多くのツールがあります：

![Linux 分析工具](https://user-images.githubusercontent.com/18013815/91632981-533ee680-ea17-11ea-90f8-06676583ea52.png)

本記事では perf に絞って紹介し、上のプログラムを使って「まだ Branch Miss が原因だと分かっていない状況で」perf を使って問題を見つける流れを示します。

Ubuntu では次のコマンドで perf をインストールできます：

```shell
$ sudo apt install linux-tools-$(uname -r) linux-tools-generic
```

あるいは Linux Kernel から perf をビルドして使うこともできます：

```shell
$ sudo apt install flex bison libelf-dev libunwind-dev libaudit-dev libslang2-dev libdw-dev
$ git clone https://github.com/torvalds/linux --depth=1
$ cd linux/tools/perf/
$ make
$ make install
$ sudo cp perf /usr/bin
$ perf
```

perf をインストールしたあと、権限設定が必要な場合があります。デフォルト設定だと perf の権限が不足することが多いです：

```shell
$ sudo su # As Root
$ sysctl -w kernel.perf_event_paranoid=-1
$ echo 0 > /proc/sys/kernel/kptr_restrict
$ exit
```

## perf の使い方

次に、perf が情報を取れるように、`-g3` でデバッグ情報を付けてビルドします。

同じ `test.cc` を使い、まず未ソート版：

```shell
$ g++ test.cc -g3 -o unsort
```

次にソート版：

```shell
$ g++ test.cc -g3 -o sort
```

### perf record

`./unsort` がなぜ遅いのかを知りたいので、`perf record` で実行情報を記録します。

```shell
$ perf record ./unsort
```

これにより `./unsort` の実行データが `perf.data` に保存され、他の perf コマンドでこの記録ファイルを読み取れます。

### perf annotate

結果を見るには `perf annotate` を使います：

```shell
$ perf annotate
```

![perf annotate](https://user-images.githubusercontent.com/18013815/91634468-44f6c780-ea23-11ea-9bf5-cd14907e22e8.png)

perf は自動的に時間を多く使っている箇所にジャンプします。上図のように、左が実行時間割合、右がソースコードに対応するアセンブリです。上下キーで移動でき、`h` を押すと操作説明が出ます。

アセンブリの時間割合を見るだけでも手がかりが得られます。一般には、最も時間がかかっている箇所を見つけ、そこから原因を調べます。ここで重要なのは `d8` と `cf` の 2 行です。`addl` は `sum += data[c]` に相当するので、この 2 行は「Branch Prediction が当たった経路」と「外れた経路」を表しています。

次の画像で矢印が示しているのは Branch Prediction が **当たった** 経路です。`d8` 行の割合はほぼ 0.0% です。
<img src="https://user-images.githubusercontent.com/18013815/91635364-93f42b00-ea2a-11ea-89b8-19075dbc67fc.png" alt="branch prediction 猜對" width=70%>

次の画像で矢印が示しているのは Branch Prediction が **外れた** 経路です。`cf` 行の割合は 27.7% 近くあります。
<img src="https://user-images.githubusercontent.com/18013815/91635373-9eaec000-ea2a-11ea-8347-1f2386373a57.png" alt="branch prediction 猜錯" width=70%>

つまり、Branch Miss によって多くの時間を無駄にしていることが分かります。

ついでに `./sort` の結果も見てみましょう：

```shell
$ perf record ./sort && perf annotate
```

<img src="https://user-images.githubusercontent.com/18013815/91636294-01578a00-ea32-11ea-888d-d46b2b65163c.png" alt="sort version's branch prediction" width=70%>

Branch Miss がほとんど起きないため、`ee` と `f7` の `addl` はほとんど時間を占めていないことが観察できます。

### perf stat

アセンブリを直接読むのは時間がかかるので、全体像を掴みたいときは `perf stat` を使うとよいです。

```shell
# 未排序版本
$ perf stat ./unsort
sum = 94479480000

 Performance counter stats for './unsort':

          5,671.51 msec task-clock                #    1.000 CPUs utilized
                24      context-switches          #    0.004 K/sec
                 0      cpu-migrations            #    0.000 K/sec
               147      page-faults               #    0.026 K/sec
    20,366,870,320      cycles                    #    3.591 GHz
    11,328,534,095      instructions              #    0.56  insn per cycle
     2,951,455,487      branches                  #  520.401 M/sec
       467,676,925      branch-misses             #   15.85% of all branches

       5.671777216 seconds time elapsed

       5.671781000 seconds user
       0.000000000 seconds sys

# 排序版本
$ perf stat ./sort
sum = 94479480000

 Performance counter stats for './sort':

          1,927.09 msec task-clock                #    1.000 CPUs utilized
                 6      context-switches          #    0.003 K/sec
                 0      cpu-migrations            #    0.000 K/sec
               146      page-faults               #    0.076 K/sec
     6,917,745,957      cycles                    #    3.590 GHz
    11,345,543,927      instructions              #    1.64  insn per cycle
     2,954,388,946      branches                  # 1533.084 M/sec
           268,192      branch-misses             #    0.01% of all branches

       1.927654198 seconds time elapsed

       1.927349000 seconds user
       0.000000000 seconds sys
```

`perf stat` は統計情報を直接表示してくれます。Context Switch、Page Fault、Branch Miss が高い場合は、プログラムの性能がまだ最適化の余地があることを示唆します。

`unsort` を例にすると、Branch Miss が特に高い（ソート版はほぼ 0）ことが分かります。そこで元のコードの条件分岐を確認し、さらに `perf annotate` の時間割合と合わせれば、問題箇所を素早く特定できます。また Cycle 数からも、2 つのバージョンで約 3 倍の差があることが分かります。

> perf の詳細な使い方は Brendan Gregg の「[perf Examples](http://www.brendangregg.com/perf.html)」が参考になります。また、この HackMD の[メモ](https://hackmd.io/@1IzBzEXXRsmj6-nLXZ9opw/HkBl5kCSU)も良いです。

## 結論

本記事では perf の基本的な使い方を紹介し、簡単なサンプルプログラムを通して、性能を観察して問題の原因を見つける流れを示しました。

プログラムの性能が悪いと分析が必要になりますが、原因を見つけるのは簡単ではありません。性能低下はアルゴリズムやデータ構造が原因かもしれませんし、OS の System Call が原因かもしれませんし、プロセッサアーキテクチャが原因かもしれません。本記事の例が示す通り、アルゴリズムの計算量だけでは実際の速度を説明できず、OS やハードウェアも考慮する必要があります。性能分析ツールを上手く使うことで、より速く問題点を見つけられます。
