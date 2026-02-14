---
title: "WebAssembly の使い方を徹底解説"
date: 2020-08-17 00:05:30
tags: [JavaScript, WebAssembly, browsers,]
des: "本記事では WebAssembly の JavaScript API を網羅的に紹介し、C コードから WASM を生成して、Web ページ上で JavaScript から WASM を利用する方法を実例で示します。"
lang: jp
translation_key: webassembly-intro
---

## 1. 概要

まずは WebAssembly とは何か、そしてなぜ必要なのかから整理しましょう。

いまや Web 技術はあらゆる場面に浸透しています。人々は 1 日のうち何時間もインターネットを利用し、さまざまなサービスを享受しています。多くのアプリケーションが Web に移行し、デスクトップアプリやモバイルアプリでも Web アプリ（Web App）やハイブリッドアプリ（Hybrid App）を採用するケースが増えています。言い換えるなら、Web 技術が世界を支配していると言っても過言ではありません。

Web の中心にある言語は JavaScript（JS）です。ブラウザや JS エンジンの進化によって性能は向上してきましたが、最適化には限界があります。JS はインタプリタ型言語であり、実行時にコードを 1 行ずつ読み込みながらコンパイルして実行します。このモデルは、C++ のようなコンパイル型言語より遅くなりがちです。

では、なぜ Web で C++ のようなコンパイル型言語をそのまま使わないのでしょうか。事前にコンパイルして配布する場合、生成物（実行ファイル）が大きくなりやすく、ブラウザへ配信するネットワーク転送に時間がかかります。逆に、ブラウザが受け取ってからコンパイルする場合は、コンパイル完了まで待つ必要があり、それも時間がかかります。JS はソースが比較的小さく、転送時間を抑えやすい上に、ブラウザが逐次的にコンパイルして「先にコンパイルできた部分から実行する」こともできるため、体感の遅さを抑えやすいという事情があります。

それでも、JS の性能には上限があります。そこで「ブラウザ上でより高速に動作させたい」という要求から誕生したのが WebAssembly（WASM）です。WASM は低レベルでアセンブリに近い形式であり、C++ や Rust のようなネイティブ（Native）コードに近い性能を狙えます。Web 開発では JS と WASM を組み合わせて使います。イメージとしては、一般的なアプリのロジックは JS で動かしつつ、計算負荷が高い部分を事前にコンパイルされた WASM に置き換えることで性能を向上させます。つまり、JS の「起動が速い」という利点と、WASM の「重い計算を高速に実行できる」という利点を両取りするアプローチです。

以降では、JS と WASM を組み合わせた開発方法を紹介します。

<!-- more -->
## 2. WebAssembly API 紹介

WebAssembly の主要なオブジェクトは次の通りです：

1. **WebAssembly のロード／初期化：** `WebAssembly.compile()` / `WebAssembly.instantiate()` 関数。
2. **WebAssembly のメモリバッファ（Memory Buffer）／テーブル（Table）の作成：** `WebAssembly.Memory()` / `WebAssembly.Table()` コンストラクタ。
3. **WebAssembly のエラー処理：** `WebAssembly.CompileError()` / `WebAssembly.LinkError()` / `WebAssembly.RuntimeError()` コンストラクタ。

### 2.1 JS で WASM をコンパイルする

WASM はコンパイル済みのバイナリで、通常は C++ や Rust から生成されます。WASM の生成方法は後ほど説明します。ここでは、すでに `*.wasm` が用意されていると仮定します。

まずブラウザで `*.wasm` をダウンロードし、ブラウザ上でその WASM を「再度」コンパイルします。WASM はすでにコンパイル済みのバイナリですが、実際には IR（平たく言えばコンパイル途中の生成物）に近い側面があり、取得したバイト列はブラウザ内部で最終形へ変換されます。

WASM のコンパイル方法は次の選択肢があります：

- `WebAssembly.compile()` を使う
- `WebAssembly.compileStreaming()` を使う
- `WebAssembly.Module` のコンストラクタを使う

`compile()` と `compileStreaming()` は非同期（async）で、`compileStreaming()` はストリームをコンパイル（ダウンロードしながらコンパイル）します。`Module` コンストラクタは同期（sync）です。いずれもコンパイル後は `WebAssembly.Module` オブジェクトになります。

`WebAssembly.Module` は「ブラウザ側でコンパイルが完了した状態」を表し、この後 Web Worker に渡したり、繰り返しインスタンス化したりできます。

#### 2.1.1 WebAssembly.compile()

```js
Promise<WebAssembly.Module> WebAssembly.compile(bufferSource);
```

`WebAssembly.compile()` の例：

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

`WebAssembly.compileStreaming()` の例：

```js
// 異步邊下載邊編譯 WASM 檔案
WebAssembly.compileStreaming(fetch('simple.wasm'))
    .then(module => {
        // 得到 WebAssembly.Module
    })
```

`compileStreaming` を使う場合、サーバの HTTP レスポンスヘッダで WASM ファイルを `Application/wasm` として返す必要があります。

#### 2.2.3 WebAssembly.Module 建構子

```js
new WebAssembly.Module(bufferSource);
```

`WebAssembly.Module` コンストラクタの例：

```js
fetch('simple.wasm').then(response =>
  response.arrayBuffer()
).then(bytes => {
  let mod = new WebAssembly.Module(bytes);
})
```

WASM ファイルを `WebAssembly.Module` にコンパイルした後は、後でインスタンス化することもできますし、`Module` を Worker に渡して利用することもできます。

### 2.2 WebAssembly をインスタンス化する

`WebAssembly.Module` を `WebAssembly.Instance` に初期化してはじめて、実際に利用できます。

`Instance` を生成する方法は次の通りです：

- `WebAssembly.instantiate()`
- `WebAssembly.instantiateStreaming()`
- `WebAssembly.Instance` コンストラクタ

最初の 2 つは非同期で、コンストラクタは同期です。

#### 2.2.1 WebAssembly.instantiate()

```js
Promise<WebAssembly.Instance> WebAssembly.instantiate(module, importObject);
Promise<ResultObject> WebAssembly.instantiate(bufferSource, importObject);
```

`instantiate()` は WASM バイナリ（bytes）でも、コンパイル済み `Module` でも受け取れます。つまり、必ずしも事前に `compile()` を呼ぶ必要はなく、コードの組み立てに柔軟性があります。`importObject` は WASM に関数・変数・オブジェクトなどを取り込むためのものです。

**`importObject` の使い方は「2.3」で説明します。**

引数が `Module` の場合は `Instance` を返し、WASM bytes の場合は `ResultObject {instance: Instance, module: Module}` を返します。

`WebAssembly.instantiate()` の例：

```js
fetch('simple.wasm').then(response =>
  response.arrayBuffer()
).then(bytes => {
  let mod = new WebAssembly.Module(bytes);
  let instance = new WebAssembly.Instance(mod, importObject);
  instance.exports.exported_func(); // 呼叫 WASM 的 exported_func
})
```

WASM の export は `instance.exports` から取得できます。

#### 2.2.2 WebAssembly.instantiateStreaming()

```js
Promise<ResultObject> WebAssembly.instantiateStreaming(bytes, importObject);
```

`WebAssembly.instantiateStreaming()` は WASM のストリームのみを受け取り、`importObject` を取り込んだ上で `ResultObject {instance: Instance, module: Module}` を生成します。

例：

```js
WebAssembly.instantiateStreaming(fetch('simple.wasm'), importObject)
    .then(obj => obj.instance.exports.exported_func());
```

ここでも `instantiateStreaming` を使う場合、サーバの HTTP レスポンスヘッダで WASM を `Application/wasm` として返す必要があります。

#### 2.2.3 WebAssembly.Instance 建構子

```js
new WebAssembly.Instance(module, importObject);
```

コンストラクタを使う場合、引数にできるのはコンパイル済みの `Module` のみです。`importObject` は同様です。コンストラクタは同期であり、スレッドをブロックします。また初期化は通常コストが高いので、必要がなければ前述の非同期方式の方が望ましいです。

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

現時点では WASM は JS で起動する必要があり、WASM の実行結果も JS を通じて取得することが一般的です。そのため `WebAssembly.Memory` を用いて、JS と WASM が同じメモリ領域を共有し、双方から読み書きできるようにします。

```js
const memory = new WebAssembly.Memory({initial:10, maximum:100, shared: true});
```

`WebAssembly.Memory` は実体としては `ArrayBuffer` もしくは `SharedArrayBuffer` であり、`memory.buffer` を通して raw memory を直接操作できます。

`WebAssembly.Memory` のパラメータは `initial`、`maximum`、`shared` の 3 つです。`initial` と `maximum` はそれぞれ初期メモリサイズと上限メモリサイズを表し、どちらも 64 kB（1 ページ）単位です。`shared` は Shared Memory かどうかです。

初期サイズと上限が分かれているのは、`WebAssembly.Memory` が動的な拡張（Resize）を許可しているためです。次で拡張できます：

```js
memory.grow(number);
```

`grow()` も 64 kB 単位で増やします。WASM メモリはページ単位で管理され、`manifest` のような情報で管理されるため、原理的には大きなオーバーヘッドにはなりにくいです。[定義](https://webassembly.github.io/spec/core/exec/runtime.html#syntax-meminst)は次の通りです：

$$
\begin{split}\begin{array}{llll}
{\mathit{meminst}} &::=&
  \\{ {\mathsf{data}}\ {\mathit{vec(bytes)}},\ {\mathsf{max}}\ {\mathit{u32}}^? \\} \\\\
\end{array}\end{split}
$$

ただし注意点として、内部はページ操作であっても、Resize のたびに `ArrayBuffer`／`SharedArrayBuffer` は新しいオブジェクトが生成され、古いオブジェクトは detached になります。

[例](https://mdn.github.io/webassembly-examples/js-api-examples/memory.html)：

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

このコードを説明する前に、`memory.wasm` が何かを見てみます。

`memory.wasm` を WAT（可読形式）に変換すると次の通りです：
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

この行は、WASM が `js.mem` を import して使うことを意味します。したがって `importOject` に `js.mem` を定義し、`WebAssembly.Memory` オブジェクトを渡す必要があります。

```
(memory (import "js" "mem") 1)
```

次に、WASM 側で `accumulate` 関数を定義しており、渡された配列を加算して返します。

```
(func (export "accumulate") (param $ptr i32) (param $len i32) (result i32)
```

したがって JS の例では、まず `i32` を通じて `memory.buffer` に値を書き込みます。この時点で WASM 内部の `js.mem` には値が入っています。

```js
let i32 = new Uint32Array(memory.buffer);
```

その後 `instance.exports.accumulate()` を呼び出して WASM 側の `accumulate()` を実行すれば、答えが得られます。

```js
let sum = obj.instance.exports.accumulate(0, 10);
```

### 2.4 WebAssembly Table

WebAssembly Memory が JS と WASM の間で「データ」を共有するのに対して、`WebAssembly.Table` は WASM 内部の関数を参照（Function Reference）として保持する WASM table を提供します。JS または WASM が table に格納された関数参照にアクセスしたり、変更したりできます。（平たく言えば「WASM にどんな関数があるかを取り出せる表」）

```js
const table = new WebAssembly.Table({
                element: "anyfunc", // 表格物件型別，目前只能是「任意函數」
                initial: Number, // 多少個元素
                maximum: Number? // Optional，表可以擴展的最大值
              });
```

`table.get(index)` で要素を取得し、`table.set(index)` で要素を設定し、`table.grow(number)` で table を拡張できます。

例：

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

`tbl.get(0)()` は「関数 `tbl.get(0)` を取得してから `()` で呼び出す」ことを意味します。

`table.wasm` は次のようになっています：

```
(module
    (import "js" "tbl" (table 2 anyfunc))
    (func $f42 (result i32) i32.const 42)
    (func $f83 (result i32) i32.const 83)
    (elem (i32.const 0) $f42 $f83)
)
```

要するに `js.tbl` を `table` として import し、2 つの関数参照を要素として埋め込んでいます。

### 2.5 WebAssembly Global

`WebAssembly.Global` は `Global` オブジェクトで、JS と複数の `Module` から同時にアクセスできます。最大の利点は、異なる `Module` 間の動的リンク（Dynamic Linking）を実現できることです。

WASM は C++ などからコンパイルできます。C++ をコンパイルする際には複数の `cpp` ファイルをリンクできますが、WASM でも同様のことができ、その仕組みとして `Global` を利用します。Emscripten のような WASM コンパイラが生成する WASM も、この方法を利用しています。

```js
new WebAssembly.Global(descriptor {value, mutable}, value);
```

第 1 引数 `descriptor` の `value` は型、`mutable` は変更可能かどうかを表します。第 2 引数 `value` は初期値で、`0` のみを渡すとデフォルト値が使われます。

[例](https://mdn.github.io/webassembly-examples/js-api-examples/global.html)：

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

WASM には 3 種類のエラーが定義されています：`WebAssembly.CompileError`、`WebAssembly.LinkError`、`WebAssembly.RuntimeError` です。

```js
new WebAssembly.CompileError(message, fileName, lineNumber)
new WebAssembly.LinkError(message, fileName, lineNumber)
new WebAssembly.RuntimeError(message, fileName, lineNumber)
```

使い方は 3 つとも同じです。例：

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

## 3. 応用

### 3.1 簡単な C 関数

```c
// square.c
int square(int n) { 
   return n*n; 
}
```

Emscripten でコンパイル：

```shell
$ emcc square.c -s SIDE_MODULE -o square.wasm
```

Emscripten の使い方は、以前の[紹介記事](/post/2020/07/js/emscripten-pthread-to-js/#Emscripten-%E4%B8%8B%E8%BC%89)を参照してください。ここでは必ず `-s SIDE_MODULE` を付けて「この WASM は Runtime ではない」ことを示し、`-o *.wasm` で WASM のみを出力します。そうしないと、Emscripten はデフォルトで JS + WASM を出力します。

コンパイルした WASM を WAT に変換すると次の通りです：

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

重要なのは `(export "square" (func 1))` であり、それ以外の関係ない部分は無視して構いません。

次に Web ページ `square.html` を書きます：

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

`square.html` と `square.wasm` を同じディレクトリに置き、HTTP サーバで配信して（WASM をダウンロードできる必要があるため）ページを開きます。Console を開けば結果を確認できます。

### 3.2 C 関数：WebAssembly.Memory を使う

この例は「2.3」で行ったことと本質的には同じですが、C から始める手順を示します。

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

WAT に変換：

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

上記から、`accumulate.wasm` は `env.__memory_base`、`env.memory`、`env.g$arr` を import しているので、JS 側で先に定義する必要があります。

`accumulate.html` の Web コードは次の通りです：

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

WASM 側が `env.memory` を要求するため、先に `WebAssembly.Memory` を用意して渡します。`env.__memory_base` はどの位置から読み出すかを表します。

元の C コードではグローバル `arr[]` しかないため、`env.memory` は実質的に `arr[]` のためのメモリです。最後に、なぜか Emscripten が `env.g$arr` を生成します。用途不明ですが、不要そうなので空関数を渡しています。

### 3.3 Pthread を JS + WASM に変換する

Emscripten を使えば、Pthread プログラムを Web Worker + SharedArrayBuffer + WebAssembly に比較的簡単に変換でき、ブラウザ上で平行プログラムを実行できます。

詳細は以前の記事「[使用 Emscripten 將 Pthread 轉成 JavaScript 與效能分析](/post/2020/07/js/emscripten-pthread-to-js)」および、その続編「[Pthread 轉 WASM: Merge Sort](/post/2020/08/js/emscripten-pthread-to-js-2/)」を参照してください。

## 4. 結論

本記事では WebAssembly の JavaScript API を網羅的に紹介し、C コードから WASM を生成して、Web ページ上で JavaScript から WASM を利用する方法を実例で示しました。

WebAssembly は Web の将来の大きなトレンドになるはずです。デバイス性能が上がるにつれて、私たちの性能要求も高くなります。また WASM 自体も進化を続けており、将来的には WASM から直接スレッドを起動したり、SIMD 命令を実行したりできるようになるでしょう。さらに WASM は Web 以外の領域にも広がっており、組み込みデバイスやクラウドサービスでも試行が始まっています。WASM の今後の発展は非常に楽しみです。

私は、ネット上の記事は思考の流れがどこか欠けていることが多いと感じています。そこで、オンラインリソースを整理し、私が最も論理的だと思う順序で WASM の概念をまとめ直しました。少しでも役に立てば幸いです。

## 5. 参考資料

- [MDN: WebAssembly](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly)
- [WebAssembly JS API SPEC](https://webassembly.github.io/spec/js-api/#webassembly-namespace)
- [WebAssembly Execution SPEC](https://webassembly.github.io/spec/core/exec/index.html)
- [WebAssembly Standalone](https://github.com/emscripten-core/emscripten/wiki/WebAssembly-Standalone)

