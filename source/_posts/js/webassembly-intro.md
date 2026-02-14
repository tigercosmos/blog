---
title: 完整介紹 WebAssembly 使用方式
date: 2020-08-17 00:05:30
tags: [JavaScript, WebAssembly, browsers,]
des: "本文完整介紹 WebAssembly 所有的 JavaScript API 的操作方式，並實際示範如何從 C Code 生成 WASM，並在網頁上讓 JavaScript 去使用 WASM。"
lang: zh
translation_key: webassembly-intro
---

## 1. 簡介

先來說說 WebAssembly 是甚麼，為甚麼我們需要這玩意？

如今網頁技術已經無所不在，人們上網時間佔一天好幾個小時，人們透過網路享受各種服務，各種應用服務也漸漸都轉移到網頁上，桌面程式或手機程式也越來越多採用網頁程式 (Web App) 或混合式程式 (Hybrid App)，可以說網頁技術已經主宰了世界。

網頁是採用 JavaScript (JS) 語言，隨著瀏覽器進步、JS 引擎發展，我們似乎已經達到優化效能的瓶頸，JS 不管怎樣快，由於是直譯式語言，會一行一行讀程式碼同時進行編譯，這種模式鐵定會比編譯式語言像是 C++ 還要慢。那你會問，為啥網頁不用 C++ 這種編譯式語言？這是因為不管是事先編好，編譯出來的執行檔會很大，這樣傳給瀏覽器反而會造成網路傳輸花太多時間，或是瀏覽器拿到檔案才開始編譯，那要等他全部編完才能跑也會花太多時間。所以直譯式語言的 JS 先傳給瀏覽器，由於是原始檔檔案不會很大傳輸時間就還行，然後拿到檔案後一行一行編譯，先編譯好的部分可以先跑，也不會讓速度過慢。

不管如何，JS 效能已經到上限了，人們開始想要怎樣讓網頁可以在瀏覽器上跑得更快，於是就有 WebAssembly (WASM) 的誕生了，WASM 是一個低階類似組合語言的語言，其效能可以接近原生 (Native) 的程式，像是 C++ 或 Rust 跑出來的效能。開發網頁時，JS 和 WASM 會搭配使用，概念簡單來說是一般程式邏輯一樣給 JS 跑，但把部分負擔很重的程式碼改成事先編譯的 WASM，這樣就可以讓效能更加進化。同時享有 JS 可以快速啟動的好處，也享有 WASM 在運算複雜的程式碼上有高效能的表現。

接下來本文將會介紹如何使用 JS 搭配 WASM 進行開發。

<!-- more -->
## 2. WebAssembly API 介紹

WebAssembly 主要物件有：

1. **載入/初始化 WebAssembly：** `WebAssembly.compile()`/`WebAssembly.instantiate()` 函數。
2. **建立 WebAssembly 的記憶體緩衝 (Memory Buffer)/紀錄表 (Table)：**`WebAssembly.Memory()`/`WebAssembly.Table()` 建構子。
3. **處理 WebAssembly 錯誤：** `WebAssembly.CompileError()`/`WebAssembly.LinkError()`/`WebAssembly.RuntimeError()` 建構子。

### 2.1 JS 編譯 WASM

WASM 是一個編譯好的 Binary 檔案，會從 C++ 或 Rust 編譯過來，怎麼生成 WASM 後面會說明。這邊我們先假設已經有一個編譯好的 WASM 檔案 `*.wasm`。

首先我們先在瀏覽器下載 `*.wasm` 原始檔，然後在瀏覽器上「再次」編譯這個 WASM 檔，WASM 雖然已經是編譯過的 Binary，但其實他算是一種 IR (白話就是編譯到一半的產物)，所以拿到 WASM 原始檔後還是要把它編譯成最後產物。

編譯 WASM 我們有以下幾種選擇：

- 用 `WebAssembly.compile()`
- 用 `WebAssembly.compileStreaming()`
- 用 `WebAssembly.Module` 的建構子

差別在於 `compile()` 和 `compileStreaming()` 為異步 (async)，其中 `compilerStreaming()` 顧名思義是將 Stream 進行編譯 (邊下載邊編譯)；`Module` 建構子則為同步 (sync)，。三種編譯完都會產生 `WebAssembly.Module` 物件。

`WebAssembly.Module` 物件代表瀏覽器已經編譯處理好，接下來可以將物件傳給 Web Worker 使用或是重複初始化。

#### 2.1.1 WebAssembly.compile()

```js
Promise<WebAssembly.Module> WebAssembly.compile(bufferSource);
```

`WebAssembly.compile()` 示例：

```js
const worker = new Worker("wasm_worker.js");

// 先抓 WASM 檔案
fetch('simple.wasm')
    .then(response =>
        response.arrayBuffer()
    // 編譯 bytes
    ).then(bytes =>
        // 同步編譯
        WebAssembly.compile(bytes)
    // 將 Module 傳給 worker
    ).then(mod =>
        worker.postMessage(mod)
    );
```

#### 2.1.2 WebAssembly.compileStreaming()

```js
Promise<WebAssembly.Module> WebAssembly.compileStreaming(source);
```

`WebAssembly.compileStreaming()` 示例：


```js
// 異步邊下載邊編譯 WASM 檔案
WebAssembly.compileStreaming(fetch('simple.wasm'))
    .then(module => {
        // 得到 WebAssembly.Module
    })
```

要注意的是用 `compileStreaming` 的話，Server 的 HTTP Request Header 必須將 WASM 檔案標註 `Application/wasm` 才行。

#### 2.2.3 WebAssembly.Module 建構子

```js
new WebAssembly.Module(bufferSource);
```

`WebAssembly.Module` 建構子示例：

```js
fetch('simple.wasm').then(response =>
  response.arrayBuffer()
).then(bytes => {
  let mod = new WebAssembly.Module(bytes);
})
```

先將 WASM 檔案編譯成 `WebAssembly.Module` 物件之後，可以等等再初始化，或是將 `Module` 傳給 Worker 使用。

### 2.2 初始化 WebAssembly

我們必須將 `WebAssembly.Module` 物件初始化成 `WebAssembly.Instance` 物件後才可以正式使用。

要生成 `Instance` 物件我們可以用：

- `WebAssembly.instantiate()`
- `WebAssembly.instantiateStreaming()`
- `WebAssembly.Instance` 建構子

三種方式來建構，一樣前兩者是異步而後者是同步。

#### 2.2.1 WebAssembly.instantiate()

```js
Promise<WebAssembly.Instance> WebAssembly.instantiate(module, importObject);
Promise<ResultObject> WebAssembly.instantiate(bufferSource, importObject);
```

`instantiate()` 可以吃 WASM binary 或是編譯過的 `Module`，意思是你如果不事先用 `compile()` 跑也沒關係，主要是對程式碼安排有更大彈性。`importObject` 是將一些函數、變數、物件引入 WASM 裡面。

**關於 `importObject` 的使用方式，到「2.3」的時候再介紹。**

如果是用 `Module` 當參數則回傳 `Instance`，若是傳入 WASM 則回傳 `ResultObject {instance: Instance, module: Module}`。

`WebAssembly.instantiate()` 示例：

```js
fetch('simple.wasm').then(response =>
  response.arrayBuffer()
).then(bytes => {
  let mod = new WebAssembly.Module(bytes);
  let instance = new WebAssembly.Instance(mod, importObject);
  instance.exports.exported_func(); // 呼叫 WASM 的 exported_func
})
```

WASM 裡面的物件可以透過 `instance.exports` 去取得。

#### 2.2.2 WebAssembly.instantiateStreaming()

```js
Promise<ResultObject> WebAssembly.instantiateStreaming(bytes, importObject);
```

`WebAssembly.instantiateStreaming()` 只能吃 WASM Stream，`importObject` 是要引入的物件，然後會產生 `ResultObject {instance: Instance, module: Module}`。

示例：

```js
WebAssembly.instantiateStreaming(fetch('simple.wasm'), importObject)
    .then(obj => obj.instance.exports.exported_func());
```

這邊一樣，用 `instantiateStreaming` 的話，Server 的 HTTP Request Header 必須將 WASM 檔案標註 `Application/wasm` 才行。

#### 2.2.3 WebAssembly.Instance 建構子

```js
new WebAssembly.Instance(module, importObject);
```

注意用建構子時只能放編譯好的 `Module`，`importObject` 則和前面一樣。要注意的是，用建構子是同步，意思是會把 Thread 卡住，而且初始化通常很花時間，除非必要不然用前面介紹的異步方法比較好。

```js
fetch('simple.wasm').then(response =>
  response.arrayBuffer()
).then(bytes => {
  // 先取得 Module
  let mod = new WebAssembly.Module(bytes);
  // 用 Module 初始化得到 Instance
  let instance = new WebAssembly.Instance(mod, importObject);
  instance.exports.exported_func();
})
```

### 2.3 WebAssembly Memory

目前 WASM 必須由 JS 來啟動，WASM 跑完的結果自然也要透過 JS 去取得，為此透過 `WebAssembly.Memory` 讓 JS 和 WASM 共同使用記憶體區塊，兩邊都可以做讀取。

```js
const memory = new WebAssembly.Memory({initial:10, maximum:100, shared: true});
```

`WebAssembly.Memory` 本身其實就是 `ArrayBuffer` 或 `SharedArrayBuffer`，可以由 `memory.buffer` 直接去操作 Raw Memory。

宣告 `WebAssembly.Memory` 總共會有三個參數，分別是 `initial`、`maximum` 和 `shared`，代表初始記憶體大小，記憶體上限大小，兩者都以 64 kB 為單位 (一個記憶體分頁 (Page Size))，最後 `shared` 為是否是 Shared Memory。

之所以有分出使大小和上限值，是因為 `WebAssembly.Memory` 允許動態擴增 (Resize)，使用：

```js
memory.grow(number);
```

便可以改變記憶體大小，`grow()` 一樣是以 64kB 為單位。`grow` 原則上不會有太大 Overheads，因為 WASM 記憶體原理是管理記憶體分頁，會有個 `manifest` 做管理，[定義](https://webassembly.github.io/spec/core/exec/runtime.html#syntax-meminst)如下：

$$
\begin{split}\begin{array}{llll}
{\mathit{meminst}} &::=&
  \\{ {\mathsf{data}}\ {\mathit{vec(bytes)}},\ {\mathsf{max}}\ {\mathit{u32}}^? \\} \\
\end{array}\end{split}
$$

不過要注意的是，雖然底層操作是 Page 所以不會有 Overheads，但是每次做 Resize 時，不管是 `ArrayBuffer` 或 `SharedArrayBuffer` 都會產生新的物件，而原本舊的物件則會被 detached。

[示例](https://mdn.github.io/webassembly-examples/js-api-examples/memory.html)：

```js
WebAssembly.instantiateStreaming(
  fetch('memory.wasm'), 
  { js: { mem: memory } } // 代表 WASM 宣告引入 js.mem
).then(obj => {
    let i32 = new Uint32Array(memory.buffer);
    for (let i = 0; i < 10; i++) {
        i32[i] = i;
    }
    let sum = obj.instance.exports.accumulate(0, 10);
    console.log(sum);
});
```

解釋上面這段 Code 之前我們先看 `memory.wasm` 是甚麼？

`memory.wasm` 轉成 WAT 格式 (可識讀格式)：
```
(module
  (memory (import "js" "mem") 1)
  (func (export "accumulate") (param $ptr i32) (param $len i32) (result i32)
    (local $end i32)
    (local $sum i32)
    (local.set $end (i32.add (local.get $ptr) (i32.mul (local.get $len) (i32.const 4))))
    (block $break (loop $top
      (br_if $break (i32.eq (local.get $ptr) (local.get $end)))
      (local.set $sum (i32.add (local.get $sum)
                               (i32.load (local.get $ptr))))
        (local.set $ptr (i32.add (local.get $ptr) (i32.const 4)))
        (br $top)
    ))
    (local.get $sum)
  )
)
```

其中，這行代表 WASM 一開始要引入 `js.mem` 來用，所以我們在 `importOject` 裡面要定義 `js.mem` 並設成 `WebAssembly.Memory` 物件。

```
(memory (import "js" "mem") 1)
```

接著 WASM 裡面定義了一個 `accumulate` 函數，會將傳入的陣列做相加後回傳。

```
(func (export "accumulate") (param $ptr i32) (param $len i32) (result i32)
```

所以 JS 範例，先在 JS 中透過 `i32` 去寫入 `memory.buffer`，此時 WASM 裡面的 `js.mem` 就有數值。 

```js
let i32 = new Uint32Array(memory.buffer);
```

再透過 `instance.exports.accumulate()` 去執行 WASM 裡面的 `accumulate()`，就能得到答案。 

```js
let sum = obj.instance.exports.accumulate(0, 10);
```

### 2.4 WebAssembly Table

不同於 WebAssembly Memory 是 JS 和 WASM 之間共享資料，`WebAssembly.Table` 是將 WASM 內部的函數包裝成一個 WASM table，可以讓 JS 或 WASM 去存取或改變 table 裡面所儲存的函數參考 (Function Reference)。(白話來說，就是一個可以抓 WASM 裡面有啥函數的表)

```js
const table = new WebAssembly.Table({
                element: "anyfunc", // 表格物件型別，目前只能是「任意函數」
                initial: Number, // 多少個元素
                maximum: Number? // Optional，表可以擴展的最大值
              });
```

我們可以透過 `table.get(index)` 來取得元素，`table.set(index)` 設定元素，和用 `table.grow(number)` 擴增 Table。

示例：

```js
const tbl = new WebAssembly.Table({initial: 2, element: "anyfunc"});
console.log(tbl.length);  // "2"
// 此時此刻，table 還是空的
console.log(tbl.get(0));  // "null" 
console.log(tbl.get(1));  // "null"


const importObj = {js: {tbl: tbl}};
WebAssembly.instantiateStreaming(fetch('table2.wasm'), importObject)
  .then(function(obj) {
    // 表格已經和 WASM 同步
    console.log(tbl.get(0)()); // 呼叫 table 第 0 個元素代表的函數
    console.log(tbl.get(1)());  // 呼叫 table 第 1 個元素代表的函數
  });
```

`tbl.get(0)()` 代表先取得函數 `tbl.get(0)` 之後再呼叫他 `()`。

我們可以看 `table.wasm` 長怎樣：

```
(module
    (import "js" "tbl" (table 2 anyfunc))
    (func $f42 (result i32) i32.const 42)
    (func $f83 (result i32) i32.const 83)
    (elem (i32.const 0) $f42 $f83)
)
```

其實就是引入 `js.tbl` 宣告為 `table`，然後將兩個函數參考當元素填入。

### 2.5 WebAssembly Global

`WebAssembly.Global` 是一個 `Global` 物件，可以同時給 JS 和多個 `Module` 存取，最大處是他可以達到不同 `Module` 之間動態連結 (Dynamic Linking) 的功用。

WASM 可以由 C++ 等語言編譯而來，如同我們編譯 C++ 時可以將不同 `cpp` 檔案做連結，WASM 也能做到一樣的事，就會透過 `Global` 來做到，Emscripten 這類的 WASM 編譯器編譯出來的 WASM 即是透過這個方法。

```js
new WebAssembly.Global(descriptor {value, mutable}, value);
```

第一個參數 `descriptor` 的 `value` 代表型別，`mutable` 代表是否可改動。第二個參數 `value` 代表變數的初始值，如果只填入 `0` 代表填入預設值。

[示例](https://mdn.github.io/webassembly-examples/js-api-examples/global.html)：

```js
const global = new WebAssembly.Global(
  {value:'i32', mutable:true}, // 可變的 i32
   0 // 填入預設值
);

WebAssembly.instantiateStreaming(fetch('global.wasm'), { js: { global } })
  .then(({instance}) => {
      global.value = 42; // 用 JS 設為 42
      instance.exports.incGlobal(); // incGlobal 是 WASM 的函數，可以加一，所以現在是 43
      assertEq(global.value, 43); // 確認是 43 無誤
  });
```


### 2.6 WebAssembly Error

WASM 定義了 3 種錯誤，分別為 `WebAssembly.CompileError`、`WebAssembly.LinkError`、`WebAssembly.RuntimeError`。

```js
new WebAssembly.CompileError(message, fileName, lineNumber)
new WebAssembly.LinkError(message, fileName, lineNumber)
new WebAssembly.RuntimeError(message, fileName, lineNumber)
```

三種用法都一樣，示例：

```js
try {
  throw new WebAssembly.CompileError('Hello', 'someFile', 10);
} catch (e) {
  console.log(e instanceof CompileError); // true
  console.log(e.message);                 // "Hello"
  console.log(e.name);                    // "CompileError"
  console.log(e.fileName);                // "someFile"
  console.log(e.lineNumber);              // 10
  console.log(e.columnNumber);            // 0
  console.log(e.stack);                   // returns the location where the code was run
}
```

## 3. 應用

### 3.1 簡易 C 函數

```c
// square.c
int square(int n) { 
   return n*n; 
}
```

用 Emscripten 編譯：

```shell
$ emcc square.c -s SIDE_MODULE -o square.wasm
```

Emscripten 使用方式可以參照我之前的[介紹文](/post/2020/07/js/emscripten-pthread-to-js/#Emscripten-%E4%B8%8B%E8%BC%89)。注意我們一定要加上 `-s SIDE_MODULE` 代表這個 WASM 不是 Runtime，然後指定 `-o *.wasm` 輸出 WASM，不然 Emscripten 預設是會輸出 JS + WASM。

編譯出來的 WASM 轉成 WAT 長這樣：

```
$ ./wasm2wat square.wasm
(module
  (type (;0;) (func))
  (type (;1;) (func (param i32) (result i32)))
  (func (;0;) (type 0)
    nop)
  (func (;1;) (type 1) (param i32) (result i32)
    local.get 0
    local.get 0
    i32.mul)
  (global (;0;) i32 (i32.const 0))
  (export "__wasm_apply_relocs" (func 0))
  (export "square" (func 1))
  (export "__dso_handle" (global 0))
  (export "__post_instantiate" (func 0)))
```

可以看到關鍵是 `(export "square" (func 1))`，然後有一些不相干的我們可以忽略。

接著我們寫一個網頁 `square.html`：

```html
<!-- square.html -->
<script>
    (async () => {
        const res = await fetch("square.wasm");
        const wasmFile = await res.arrayBuffer();
        const module = await WebAssembly.compile(wasmFile);
        const instance = await WebAssembly.instantiate(module);

        const square = instance.exports.square(13);
        console.log("The square of 13 = " + square);
    })();
</script>
```

將 `square.html` 和 `square.wasm` 放在同一個目錄，用 HTTP Server 伺服開啟網頁 (因為要能下載 WASM)，打開 Console 就可以看到結果了。

### 3.2 C 函數：使用 WebAssembly.Memory

這個範例其實跟「2.3」做的事情一樣，只是示範從 C 開始的流程。

```c
// accumulate.c
int arr[];

int accumulate(int start, int end) { 
   int sum = 0;
   for(int i = start; i < end; i++) {
      sum += arr[i];
   }
   return sum;
}
```

```shell
$  emcc accumulate.c  -O3  -s SIDE_MODULE  -o accumulate.wasm
```

轉成 WAT：

```
(module
  (type (;0;) (func))
  (type (;1;) (func (result i32)))
  (type (;2;) (func (param i32 i32) (result i32)))
  (import "env" "g$arr" (func (;0;) (type 1)))
  (import "env" "__memory_base" (global (;0;) i32))
  (import "env" "memory" (memory (;0;) 0))
  // 省略
```

從上面可以看到 `accumulate.wasm` 要引入 `env.__memory_base`、`env.memory`、`env.g$arr`，所以我們在 JS 裡面要先宣告。

`accumulate.html` 網頁程式碼如下：

```html
<!-- accumulate.html -->
<script>
    const memory = new WebAssembly.Memory({
        initial: 1,
    });

    const importObj = {
        // 根據 WASM 來宣告
        env: {
            memory: memory,
            __memory_base: 0,
            g$arr: () => {}
        }
    };

    (async () => {
        const res = await fetch("accumulate.wasm");
        const wasmFile = await res.arrayBuffer();
        const module = await WebAssembly.compile(wasmFile);
        const instance = await WebAssembly.instantiate(module, importObj);

        const arr = new Uint32Array(memory.buffer);
        for (let i = 0; i < 10; i++) {
            arr[i] = i;
        }

        const sum = instance.exports.accumulate(0, 10);
        console.log("accumulate from 0 to 10: " + sum);
    })();
</script>
```

由於 WASM 要求要有 `env.memory`，所以我們先宣告 `WebAssembly.Memory` 給他使用，`env.__memory_base` 則是記憶體要從哪開始讀。

由於原始的 C Code 只有全域 `arr[]`，所以 `env.memory` 其實就是給 `arr[]` 使用。最後不知道為啥 Emscripten 會弄出 `env.g$arr`，感覺是沒用的東西所以放個空函數。

### 3.3 Pthread 轉 JS + WASM

透過 Emscripten 我們可以輕鬆將 Pthread 程式轉成網頁的 Web Worker + SharedArrayBuffer + WebAssembly，便可以在網頁上執行平行程式。

細節可以參考我之前的文章「[使用 Emscripten 將 Pthread 轉成 JavaScript 與效能分析](/post/2020/07/js/emscripten-pthread-to-js)」和其續集「[Pthread 轉 WASM: Merge Sort](/post/2020/08/js/emscripten-pthread-to-js-2/)」

## 4. 結論

本文完整介紹 WebAssembly 所有的 JavaScript API 的操作方式，並實際示範如何從 C Code 生成 WASM，並在網頁上讓 JavaScript 去使用 WASM。

WebAssembly 勢必是未來網頁趨勢，因為隨著大家設備越來越好，我們對效能追求的程度就越高，同時 WASM 還持續在進化當中，像是以後可以直接用 WASM 開 Thread 或是用 WASM 執行 SIMD 指令，同時 WASM 也被應用在不只網頁領域，嵌入式裝置和雲服務也都開始嘗試使用，可以說 WASM 未來的發展值得期待。

我總覺得網路上看到的文章在思路上總欠缺甚麼，因此我重新整理過網路資源，以我認為最有邏輯的方式將整個 WASM 概念講過，希望有幫助到大家。

## 5. 參考資料

- [MDN: WebAssembly](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly)
- [WebAssembly JS API SPEC](https://webassembly.github.io/spec/js-api/#webassembly-namespace)
- [WebAssembly Execution SPEC](https://webassembly.github.io/spec/core/exec/index.html)
- [WebAssembly Standalone](https://github.com/emscripten-core/emscripten/wiki/WebAssembly-Standalone)
