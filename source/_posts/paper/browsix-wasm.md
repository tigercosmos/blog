---
title: "[論文] 好像沒有那麼快：WebAssembly vs. Native Code 程式碼效能分析"
date: 2019-12-18 00:00:00
tags: [paper, WebAssembly, browser, CPU SPEC, 效能分析, 平行化]
---
**Paper**: Not So Fast: Analyzing the Performance of WebAssembly vs. Native Code
**Author**: Abhinav Jangda, Bobby Powers, Emery D. Berger, and Arjun Guha, University of Massachusetts Amherst
**Source**: [USENIX ATC 2019](https://www.usenix.org/conference/atc19/presentation/jangda)

---

## 簡介

作者們先前開發了 [BROWSIX](https://github.com/plasma-umass/browsix)，可以讓 Unix 程式直接轉成 JavaScript 並跑在瀏覽器。BROWSIX 基本上就是把程式碼轉成 JavaScript 並且用 Browser API 來模擬 Unix System Call。此篇作者更進一步，開發 BROWSIX-WASM（fork 自 [Emscripten](https://github.com/emscripten-core/emscripten)）將程式碼轉成 WebAssembly，一來解決原本 JavaScript 效能不佳問題，同時也能可以拿來和 Native Code 在 SPEC CPU suite of Benchmark 上做效能比較。實驗結果 WebAssembly 在現代瀏覽器（Chrome & Firefox）上慢 Native Code 大約 50%，並深入探討為什麼會這樣。

整體來說我覺得是很有趣的一篇論文，整理了重點並加入一些我個人的見解。

<!-- more --> 
 
原論文摘要：

> All major web browsers now support WebAssembly, a low-level bytecode intended to serve as a compilation target for code written in languages like C and C++. A key goal of WebAssembly is performance parity with native code; previous work reports near parity, with many applications compiled to WebAssembly running on average 10% slower than native code. However, this evaluation was limited to a suite of scientific kernels, each consisting of roughly 100 lines of code. Running more substantial applications was not possible because compiling code to WebAssembly is only part of the puzzle: standard Unix APIs are not available in the web browser environment. To address this challenge, we build Browsix-Wasm , a significant extension to Browsix that, for the first time, makes it possible to run unmodified WebAssembly-compiled Unix applications directly inside the browser. We then use Browsix-Wasm to conduct the first large-scale evaluation of the performance of WebAssembly vs. native. Across the SPEC CPU suite of benchmarks, we find a substantial performance gap: applications compiled to WebAssembly run slower by an average of 45% (Firefox) to 55% (Chrome), with peak slowdowns of 2.08× (Firefox) and 2.5× (Chrome). We identify the causes of this performance degradation, some of which are due to missing optimizations and code generation issues, while others are inherent to the WebAssembly platform.

## WebAssembly

WebAssembly（WASM）在瀏覽器中是一個獨立模組，他也可以是獨立的程式。詳細介紹可以看 Lin Clark 寫的[介紹](https://hacks.mozilla.org/2018/10/webassemblys-post-mvp-future/)。一個 WASM 模組由 4GB 的線性記憶體、function table、WASM code 所組成。具備 [stack machine](https://en.wikipedia.org/wiki/Stack_machine)、手動管理記憶體、Structed control flow、與 JS 協作、Type safe 等特性。

## 從 BROWSIX 到 BROWSIX-Wasm

BROWSIX 可以處理 socket、pipe、file system 等 System Call（[Source Code](https://github.com/plasma-umass/browsix-emscripten/blob/browsix-wasm/src/library_browsix.js)），並將每個 System Call 都丟給一個 WebWorker。原先 BROWSIX 因為是 JavaScript，每個 Process(WebWorker 用來模擬 process) 記憶體都放在一個 Buffer 並和 kernel(一個 `SharedArrayBuffer`)分享用來做溝通，這樣會有個很大問題，因為 Chrome 中給 JS 的最大 Heap 空間大概是 2.2 GB，假設每個 Process 各吃掉 500 MB，那就只能最多有四個 Process 做 Concurrency。同時 WebAssembly 也不支援 `SharedArrayBuffer`，再者 WASM 的 WebWorker 不支援 Atomic 指令（原先 BROWSIX 用來等待 System Call）。因此原本 BROWSIX 的 Kernel-process 溝通的方式若改成 WebAssembly 實作時則不可行。

為了解決這個問題，BROWSIX 修改了 Emscripten Runtime（C++ 轉 WASM 工具）稱之 BROWSIX-Wasm，讓每個 process 產生一個是 `SharedArrayBuffer` 的隨機 64 MB Buffer 用來當介面，WASM 的資料都寫到這個隨機 Buffer，然後傳訊息告訴 Kernel 去讀這個 Buffer，因為是透過 `SharedArrayBuffer` 所以可以用原先的溝通方法。所以關係就是 WASM <-> 隨機 Buffer <-> Kernel。

其實 Emscripten 似乎已經有支援 pthread 了，可以參考[這篇文件](https://emscripten.org/docs/porting/pthreads.html)，也許是實作方式有差，我還沒深入去研究細節。

論文中特別提到 BROWSIX 處理 BrowserFS 是直接開一段記憶體空間，每次有 Append 時就開更大的空間並將原本的資料拷貝過去，這方法有點類似 C++ STD 中 Vector 的作法，不過這樣在小量資料時會有嚴重的負荷，所以他們讓 Buffer 每次增加都至少 4KB，可以有效優化小量增加資料時的效能。

## BROWSIX-SPEC

![browsix-spec arch](https://user-images.githubusercontent.com/18013815/71056747-606fba00-2195-11ea-852e-8252067f801b.png)
（論文圖 2）

為了跟 Native Code 比效能，作者建立 BROWSIX-SPEC 測試沙盒，可以用瀏覽器去跑 SPEC CPU2006、SPEC CPU2017 和 PolyBenchC，並且不需要去改動 Benchmark 任何程式碼。

## 測試評估

![figure 3](https://user-images.githubusercontent.com/18013815/71236893-0f012f80-233b-11ea-845d-b74188981302.png)
（論文圖 3）

左邊這張圖是 PolybenchC 的測試結果，水平線是 Native code 基準為 1。這個 benchmark 主要是測 Small scientific kernels，所以不見得對大型軟體有直接意義，但可以證明說 BROWSIX-Wasm 的負擔不會比 Native code 多。

右邊那張圖示直接跑 SPEC CPU 的結果，水平線是 Native Code 基準為 1，然後幾乎所有的測試項目不管在 Chrome 和 Firefox 上 WASM 都比較慢，整體平均大約會慢 1.5x。

作者也調整 BROWSIX-Wasm 讓他可以轉成 `asm.js`（WASM 誕生前的技術） 和 WASM 比較，基本上 WASM 一定贏（不然要 WASM 幹嘛），事實也是如此，所以這邊略過附圖。

## 案例研究

再來就是整篇論文最精彩的部分了，個案分析一個 Matrix 相乘程式碼 `matmul.c`。比較 Clang 將 Matrix C code 轉成 `x86-64` 組語，和 Chrome 將 Matrix WASM（從 C 轉成）轉成 `x86-64` 組語，兩者的差異。

論文圖 `7.a matmul.c`:

<pre><code class="c">void matmul (int C[NI][NJ], int A[NI][NK], int B[NK][NJ]) {
  for (int i = 0; i < NI; i++) {
    for (int k = 0; k < NK; k++) {
      for (int j = 0; j < NJ; j++) {  // 原論文這邊有打錯 j 成 k
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
</code></pre>

X86-84 Code，論文圖 `7.b` 和 `7.c`：

<div>
  <div style="display: inline-block; width:48%; vertical-align:top;">
    <img width="100%" alt="llvm-x86" src="https://user-images.githubusercontent.com/18013815/71238200-7a003580-233e-11ea-8779-a28b61ba7443.png">
  </div>
  <div style="display: inline-block; width:48%;">
    <img alt="chrome-x86-1" src="https://user-images.githubusercontent.com/18013815/71238202-7a98cc00-233e-11ea-8c72-a77fb12e2b0f.png">
    <img alt="chrome-x86-2" src="https://user-images.githubusercontent.com/18013815/71238205-7bc9f900-233e-11ea-8619-c97f201b66ec.png">
  </div>
<div>

以 Matrix 這個案例，觀察到幾個現象：

- Chrome 沒有有效利用 Memory addressing。例如 Clang 在 `7b` 的 14 行只用一行，Chrome 卻在處理 `ecx` 上花了三行。
- `r13` 和 `r10` 被 V8 佔用，Chrome 可用的 Register 比較少
- Chrome 比較多 brach。例如 `7b` 的 9 和 15 行，Clang 藉由翻轉 counter 來減少條件檢查（這邊用到 x86 的特性，[add](https://c9x.me/x86/html/file_module_x86_id_5.html) 會去改 ZF flag，讓 jne 可以跳出）。Chrome 為了避免 Register spill 所以必須 jump，例如 `7c` 的 7 到 9 行。

## 效能分析

平均來說 Chrome 和 Firefox 都會產生 Native Code 兩倍左右的 Instruction retired（就是執行多少行），而越少會越快，所以 Native Code 的測試結果才比較快。

### Register

Chrome 和 Firefox 都保留至少兩個 Register 給原本的 JS 引擎使用，因此可用的 Register 就比較少。

此外 Chrome 和 Firefox 在處理 Register 上效率比較差，會比 Clang 多用幾個 Register。因為兩者採用 Linear scan register allocator，好處是編譯比較快，壞處是不是最佳優化的結果。反之，Clang 採用 Greedy graph-coloring register allocator，優化程度比較高，但缺點是編譯比較慢。

另外 `x86-64` 有提供 Operand 兩種模式。一種是 Register mode，所有操作透過 Register；而另一種是 Memory address mode，可以讓 Operand 直接在指定記憶體位置做 read 和 write。後者可以避免不必要的 Register 壓力，但 Chrome 的編譯器並沒有採用，造成額外的 Register 壓力。

### Branch

在 Matrix 案例中，可以發現 Chrome 就會多產生 Branch。此外 WASM 必須確保程式沒有超過最大 Stack，所以會多產生額外的比較和 Conditional jump instruction。此外 WASM 會做函數型別檢查，也會增加額外的比較。

### Cache

因為比較差的 Register allocator，造成 Instruction 數量比較多，進而導致 L1 cache miss 數量增加，以致於 CPU cycle 數量增多。回到 SPEC CPU 測試結果（見上方圖 3），發現 `429.mcf` 測試中，儘管生成 Instruction 比 Native Code 多，但竟然跑得比 Native Code 還快，作者歸納出是因為該筆測試只有一個含有大部分 Instruction 的主要的迴圈，所以幾乎沒有 L1 Cache miss。反之 `450.soplex` 這比測試中，因為有很多 Virtual function 導致許多 Indirect function call，所以 Chrome 和 Firefox 的 L1 Cache miss 多了四倍之多。

## 結果與討論

總結，WASM 因為不成熟的 Register allocator 和 Code generator，以致於效能上比較差。但是這可能是一個 Tradeoff，因為 Clang 會花比較多時間去編譯和優化，而瀏覽器需要即時快速的編譯和執行。

這篇論文中漏掉討論的部分是，像在 Firefox 中，WASM 有兩個引擎處理，一個是針對 Stream 去編譯，另一個是拿到所有 Code 後去優化編譯。當後者編譯完成後，會直接取代前者。在這篇論文中，無法得知 Firefox 的實驗數據是根據那個引擎的結果。

WebAssembly 比 Native Code 慢有幾個先天限制，WASM 必須檢查 Stack overflow、Indirect call 檢查、JS 引擎保留 Register，這些都導致比較大的 Code 數量，就會跑得比較慢。未來也許可以把檢查放在編譯時檢查，就能省下不少執行時負擔，此外 WASM 終究需要和 JS 互動，彼此耦合的負擔如何降到最低也是可研究方向。
