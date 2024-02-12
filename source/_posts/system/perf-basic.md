---
title: 在 Linux 上使用 Perf 做效能分析(入門篇)
date: 2020-08-29 00:20:08
tags: [linux, perf, 效能分析,]
des: "本文將介紹 Linux 上的 perf 效能分析工具，藉由一個簡單的程式範例，示範如何使用 perf 去分析一隻程式，我們將會發現使用分析工具時能更輕易的發現問題根源。"
---

## 簡介

透過效能分析工具 (Profiler)，我們可以得知更多關於軟體的運行資訊，像是花了多少記憶體、多少 CPU Cycles、多少 Cache Misses、I/O 處理時間等等，這些資訊對我們去找到程式效能瓶頸很有幫助。想辦法找到那裡讓程式變慢，進而最大化效能，便是我們做效能分析的最大目的。

本文將介紹 Linux 上的 [perf](http://www.brendangregg.com/perf.html) 效能分析工具，藉由一個簡單的程式範例，示範如何使用 perf 去分析一隻程式，我們將會發現使用分析工具時能更輕易的發現問題根源。本文參考 Gabriel Krisman Bertaz 寫的 [Performance analysis in Linux](https://www.collabora.com/news-and-blog/blog/2017/03/21/performance-analysis-in-linux/)。
<!-- more -->

本文可以搭配我的教學影片：

<iframe width="560" height="315" src="https://www.youtube.com/embed/Mba2ONCA0kI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## 一個 Branch Prediction 的範例

Stack Overflow 上有一個很火的問題「[Why is processing a sorted array faster than processing an unsorted array?](https://stackoverflow.com/questions/11227809/)」。

問題的 Code 如下：

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

首先我們先編譯未排序版本：

```shell
$ g++ test.cc -o unsort
```

接著我們把 `sort` 那行取消註解，再編譯一次：

```shell
$ g++ test.cc -o sort
```

先來看看執行時間：

```shell
$ time ./unsort
real    0m5.671s

$ time ./sort
real    0m1.932s
```

問題大意是說 `data` 如果排列過後，上面這段程式碼反而更快，如同我們實驗結果。我們知道排序的複雜度是 $O(NlogN)$，所以應該會比沒排直接跑的 $O(N)$ 還慢，但結果是排序後反而更快。

就結論來說，我們知道這個結果是因為 CPU 會做 **Branch Prediction**。白話來說就是上次如果 `if` 是 `true`，下次就先猜也是 `true`，CPU 可以藉由先猜來偷跑，猜對了的話就可以跑更快；但相對的猜錯的話偷跑的東西通通要丟掉，反而更浪費時間，稱為 Branch Miss(詳細原理可以參考「計算機組織」)。所以 Branch Prediction 算是一種雙面刃，如果可以一直讓條件判斷有相同結果就可以進而加速程式，反之判斷一直反覆不定就會導致一直「猜測」錯誤而變慢，因此在上面程式碼中排列過的版本反而比較快，因為猜測錯誤只會發生一次，就是在 `data` 剛好在 `128` 附近的位置，在那之前全部會是 `false`，而後都會是 `true`。

## Perf 效能分析工具

想要找到一段程式碼的問題通常不容易，就以上面的程式範例來說，假設我們朝演算法去分析就會走錯路，實際問題其實在計算機組織的原理。光是一段簡單的程式碼就讓我們可能找不到原因，更甭說碰到一個大的程式，裡面有各種問題存在，可能是演算法、記憶體快取、CPU 指令、網路連線、I/O 等等，這時我們需要一個分析程式來幫助我們。

Linux 上其實有很多工具可以使用：

![Linux 分析工具](https://user-images.githubusercontent.com/18013815/91632981-533ee680-ea17-11ea-90f8-06676583ea52.png)

不過本文將針對 perf 做介紹，並用上面程式來示範假設我們還不知道問題是因為 Branch Miss，如何用 perf 找到問題。

你可以用下面指令在 Ubuntu 上裝 perf：

```shell
$ sudo apt install linux-tools-$(uname -r) linux-tools-generic
```

或是你也可以考慮自己從 Linux Kernel 編譯 perf 來用：

```shell
$ sudo apt install flex bison libelf-dev libunwind-dev libaudit-dev libslang2-dev libdw-dev
$ git clone https://github.com/torvalds/linux --depth=1
$ cd linux/tools/perf/
$ make
$ make install
$ sudo cp perf /usr/bin
$ perf
```

安裝完 perf 後，你可能會需要設定系統權限，預設應該會使 perf 權限不足：

```shell
$ sudo su # As Root
$ sysctl -w kernel.perf_event_paranoid=-1
$ echo 0 > /proc/sys/kernel/kptr_restrict
$ exit
```

## 使用 perf

接著我們要測試的程式，為了讓 perf 能用，我們要加上 `-g3` 參數開啟除錯模式。

一樣是編譯 `test.cc`，首先是沒排序：

```shell
$ g++ test.cc -g3 -o unsort
```

接著是編譯有排序版本：

```shell
$ g++ test.cc -g3 -o sort
```

### Perf Record

我們現在想要知道為甚麼 `./unsort` 跑得比較慢，我們可以透過 `perf record` 來記錄程式執行的資訊。

```shell
$ perf record ./unsort
```

這樣 perf 會將 `./unsort` 跑的資料記錄在 `perf.data` 中，perf 其他指令可以用來讀取這個紀錄檔。

### Perf Annotate

我們可以用 `perf annotate` 來看結果：

```shell
$ perf annotate
```

![perf annotate](https://user-images.githubusercontent.com/18013815/91634468-44f6c780-ea23-11ea-9bf5-cd14907e22e8.png)

perf 會自動跳到花費比較多的區塊，如上圖所示，左邊是執行時間比例，右邊是程式碼對照的 Assembly Code。你可以用上下方向鍵移動，或用 `h` 來看操作說明。

其實從這個 Assembly 時間比例就可以看出端倪，通常我們會去看哪邊花最多時間，然後去研究背後原因。這邊的關鍵是 `d8` 和 `cf` 這兩行，`addl` 其實就是在做 `sum += data[c]`，所以這兩行分別代表 Branch Prediction 猜對和猜錯的路徑。

這張圖「箭頭」標註的是 Branch Prediction **猜對**的路徑，可以看到 `d8` 行占比幾乎是 0.0%。
<img src="https://user-images.githubusercontent.com/18013815/91635364-93f42b00-ea2a-11ea-89b8-19075dbc67fc.png" alt="branch prediction 猜對" width=70%>

這張圖「箭頭」標註的是 Branch Prediction **猜錯**的路徑，可以看到 `cf` 行占比幾乎是 27.7%。
<img src="https://user-images.githubusercontent.com/18013815/91635373-9eaec000-ea2a-11ea-8347-1f2386373a57.png" alt="branch prediction 猜錯" width=70%>

所以其實就可以發現整隻程式因為 Branch Misses 浪費很多時間。

這邊我們可以偷偷看一下 `./sort` 的結果：

```shell
$ perf record ./sort && perf annotate
```
<img src="https://user-images.githubusercontent.com/18013815/91636294-01578a00-ea32-11ea-888d-d46b2b65163c.png" alt="sort version's branch prediction" width=70%>

因為不會有 Branch Miss，可以觀察到 `ee` 和 `f7` 的 `addl` 基本上沒占多少時間。

### Perf Stat

直接看 Assembly 其實滿花時間的，如果想要直接「掌握大局」，可以考慮用 `perf stat`。

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

`perf stat` 可以直接看到統計資料，如果有很高的 Context Switch、Page Fault、Branch Miss 都代表程式本身效能有待優化。

以 `unsort` 為例可以看到 Branch Miss 特別高 (排序版本會幾乎是 0)，這時我們就可以去看原本的程式哪邊有條件判斷，然後根據 Annotate 的時間比例，就可以快速找到問題點。另外從 Cycle 上我們也可以發現兩個版本差了三倍之多。

> 更多 perf 的用法可以參考 Brendan Gregg 的「[perf Examples](http://www.brendangregg.com/perf.html)」。另外這個 HackMD 的[筆記](https://hackmd.io/@1IzBzEXXRsmj6-nLXZ9opw/HkBl5kCSU)也挺不錯的。

## 結論

本文介紹 perf 的簡單用法，用簡單的範例程式示範如何去觀察效能並找出可能的問題原因。

我們常常因為程式效能不佳而需要分析效能，但找到問題的過程往往不容易，一段程式碼效能不佳可能是演算法與資料結構的問題，可能是作業系統 System Call 導致，也可能是因為處理器架構的關係。如同本文的程式範例，說明了演算法的複雜度不代表真實跑出來的速度，往往還需要去考慮作業系統或是硬體架構。善用效能分析工具才能讓我們更快找到問題點。
