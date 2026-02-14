---
title: "A Comprehensive Guide to Using WebAssembly"
date: 2020-08-17 00:05:30
tags: [JavaScript, WebAssembly, browsers,]
des: "This post provides a comprehensive overview of the JavaScript APIs for WebAssembly, and demonstrates how to generate WASM from C code and use it from JavaScript in a web page."
lang: en
translation_key: webassembly-intro
---

## 1. Introduction

Let’s start with the basics: what is WebAssembly, and why do we need it?

Today, web technology is everywhere. People spend hours online every day, enjoying all kinds of services. More and more applications are moving to the web, and desktop or mobile apps increasingly adopt web-based implementations (Web Apps) or hybrid architectures (Hybrid Apps). In a sense, web technology has taken over the world.

The web runs on JavaScript (JS). As browsers and JS engines have evolved, we seem to have reached a bottleneck for performance optimization. No matter how fast JS becomes, because it is an interpreted language, it reads and compiles code line by line at runtime. This execution model is inevitably slower than compiled languages such as C++.

Then you might ask: why not use compiled languages like C++ directly on the web? Because whether you compile ahead of time or compile in the browser, there are trade-offs. Ahead-of-time compilation typically produces large binaries; shipping those to the browser can waste network transfer time. If you ship source-like representations and compile in the browser, you may still pay a significant compile time before you can run. With JS, the source files are relatively small (so transfer is acceptable), and the browser can compile incrementally as it reads, so earlier compiled parts can start running sooner without making the experience unbearably slow.

Either way, JS performance has limits. People wanted a way to make web applications run faster in browsers, and that is where WebAssembly (WASM) comes in. WASM is a low-level, assembly-like format whose performance can approach native code—similar to what you get from C++ or Rust. In web development, JS and WASM are used together. The basic idea is: keep general application logic in JS, but move compute-heavy parts into precompiled WASM to significantly improve performance. This way, you get both fast startup (JS) and high performance for heavy computation (WASM).

Next, this post introduces how to develop using JS together with WASM.

<!-- more -->
## 2. WebAssembly JavaScript API overview

The main WebAssembly objects are:

1. **Load/initialize WebAssembly:** the `WebAssembly.compile()` / `WebAssembly.instantiate()` functions.
2. **Create WebAssembly memory buffers (Memory) / tables (Table):** the `WebAssembly.Memory()` / `WebAssembly.Table()` constructors.
3. **Handle WebAssembly errors:** the `WebAssembly.CompileError()` / `WebAssembly.LinkError()` / `WebAssembly.RuntimeError()` constructors.

### 2.1 Compiling WASM in JavaScript

WASM is a compiled binary, usually produced from C++ or Rust. I will explain how to generate WASM later. For now, assume we already have a compiled `*.wasm` file.

First, the browser downloads the `*.wasm` file, and then “compiles” it again in the browser. Even though WASM is already a compiled binary, it is closer to an IR (informally: a mid-stage compilation artifact), so after obtaining the raw bytes, the browser still compiles it into the final internal representation.

To compile WASM, we have the following options:

- Use `WebAssembly.compile()`
- Use `WebAssembly.compileStreaming()`
- Use the `WebAssembly.Module` constructor

The difference is that `compile()` and `compileStreaming()` are asynchronous, and `compileStreaming()` compiles a stream while downloading. The `Module` constructor is synchronous. All three result in a `WebAssembly.Module` object.

A `WebAssembly.Module` indicates that the browser has finished compiling, and you can then pass the module to a Web Worker or instantiate it multiple times.

#### 2.1.1 WebAssembly.compile()

```js
Promise<WebAssembly.Module> WebAssembly.compile(bufferSource);
```

Example for `WebAssembly.compile()`:

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

Example for `WebAssembly.compileStreaming()`:

```js
// 異步邊下載邊編譯 WASM 檔案
WebAssembly.compileStreaming(fetch('simple.wasm'))
    .then(module => {
        // 得到 WebAssembly.Module
    })
```

Note: when using `compileStreaming`, the server’s HTTP response header must mark the WASM file as `Application/wasm`.

#### 2.2.3 WebAssembly.Module constructor

```js
new WebAssembly.Module(bufferSource);
```

Example for the `WebAssembly.Module` constructor:

```js
fetch('simple.wasm').then(response =>
  response.arrayBuffer()
).then(bytes => {
  let mod = new WebAssembly.Module(bytes);
})
```

After compiling the WASM file into a `WebAssembly.Module`, you can instantiate it later, or pass the `Module` to a worker.

### 2.2 Instantiating WebAssembly

You need to instantiate a `WebAssembly.Module` into a `WebAssembly.Instance` before you can actually use it.

To produce an `Instance`, you can use:

- `WebAssembly.instantiate()`
- `WebAssembly.instantiateStreaming()`
- the `WebAssembly.Instance` constructor

The first two are asynchronous, while the constructor is synchronous.

#### 2.2.1 WebAssembly.instantiate()

```js
Promise<WebAssembly.Instance> WebAssembly.instantiate(module, importObject);
Promise<ResultObject> WebAssembly.instantiate(bufferSource, importObject);
```

`instantiate()` can take either WASM binary bytes or a compiled `Module`. In other words, you do not have to call `compile()` first; this gives you more flexibility in how you structure code. `importObject` is used to import functions, variables, and objects into the WASM module.

**I will explain how to use `importObject` in section “2.3”.**

If you pass a `Module`, it returns an `Instance`. If you pass WASM bytes, it returns `ResultObject {instance: Instance, module: Module}`.

Example for `WebAssembly.instantiate()`:

```js
fetch('simple.wasm').then(response =>
  response.arrayBuffer()
).then(bytes => {
  let mod = new WebAssembly.Module(bytes);
  let instance = new WebAssembly.Instance(mod, importObject);
  instance.exports.exported_func(); // 呼叫 WASM 的 exported_func
})
```

You can access exported items via `instance.exports`.

#### 2.2.2 WebAssembly.instantiateStreaming()

```js
Promise<ResultObject> WebAssembly.instantiateStreaming(bytes, importObject);
```

`WebAssembly.instantiateStreaming()` only accepts a WASM stream. `importObject` is the imported object, and it produces `ResultObject {instance: Instance, module: Module}`.

Example:

```js
WebAssembly.instantiateStreaming(fetch('simple.wasm'), importObject)
    .then(obj => obj.instance.exports.exported_func());
```

Similarly, when using `instantiateStreaming`, the server must mark the WASM response as `Application/wasm`.

#### 2.2.3 WebAssembly.Instance constructor

```js
new WebAssembly.Instance(module, importObject);
```

When using the constructor, you can only pass a compiled `Module`. `importObject` is the same as above. Note that the constructor is synchronous: it blocks the thread. Instantiation is also often expensive, so unless you have a reason, the asynchronous methods are usually better.

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

Currently, WASM must be bootstrapped from JS, and naturally the results produced by WASM are also consumed from JS. For this reason, `WebAssembly.Memory` provides a shared memory region that both JS and WASM can access.

```js
const memory = new WebAssembly.Memory({initial:10, maximum:100, shared: true});
```

`WebAssembly.Memory` is essentially an `ArrayBuffer` or `SharedArrayBuffer`. You can operate on raw memory directly via `memory.buffer`.

`WebAssembly.Memory` has three parameters: `initial`, `maximum`, and `shared`. `initial` and `maximum` represent initial and maximum memory size, both measured in 64 kB pages (the memory page size). `shared` indicates whether this is shared memory.

There is both an initial size and a maximum size because `WebAssembly.Memory` can be resized dynamically via:

```js
memory.grow(number);
```

This changes the memory size; `grow()` is also measured in 64 kB pages. In principle, `grow` should not have a large overhead because WASM memory is managed in pages and uses a manifest-like structure. The [definition](https://webassembly.github.io/spec/core/exec/runtime.html#syntax-meminst) is:

$$
\begin{split}\begin{array}{llll}
{\mathit{meminst}} &::=&
  \\{ {\mathsf{data}}\ {\mathit{vec(bytes)}},\ {\mathsf{max}}\ {\mathit{u32}}^? \\} \\\\
\end{array}\end{split}
$$

However, note that while the underlying operation is page-based and therefore does not have large overhead, every resize creates a new `ArrayBuffer` / `SharedArrayBuffer` object, and the old object is detached.

Example ([source](https://mdn.github.io/webassembly-examples/js-api-examples/memory.html)):

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

Before explaining the JS snippet above, let’s look at what `memory.wasm` is.

Convert `memory.wasm` to WAT (human-readable format):
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

This line means WASM imports `js.mem`, so in `importObject` we must define `js.mem` and set it to a `WebAssembly.Memory` object:

```
(memory (import "js" "mem") 1)
```

Then WASM defines an `accumulate` function that sums an input array and returns the result:

```
(func (export "accumulate") (param $ptr i32) (param $len i32) (result i32)
```

So in the JS example, we first write values into `memory.buffer` via `i32`. At that point, `js.mem` inside WASM contains those values.

```js
let i32 = new Uint32Array(memory.buffer);
```

Then we execute WASM’s `accumulate()` via `instance.exports.accumulate()`, and we can get the answer:

```js
let sum = obj.instance.exports.accumulate(0, 10);
```

### 2.4 WebAssembly Table

Unlike WebAssembly Memory, which shares data between JS and WASM, `WebAssembly.Table` wraps internal WASM functions into a table that can be accessed or modified by JS or WASM. It stores function references (Function Reference). (Informally: a table you can use to see which functions exist in WASM.)

```js
const table = new WebAssembly.Table({
                element: "anyfunc", // 表格物件型別，目前只能是「任意函數」
                initial: Number, // 多少個元素
                maximum: Number? // Optional，表可以擴展的最大值
              });
```

You can use `table.get(index)` to get an element, `table.set(index)` to set an element, and `table.grow(number)` to grow the table.

Example:

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

`tbl.get(0)()` means you first get the function reference `tbl.get(0)`, and then call it with `()`.

Let’s see what `table.wasm` looks like:

```
(module
    (import "js" "tbl" (table 2 anyfunc))
    (func $f42 (result i32) i32.const 42)
    (func $f83 (result i32) i32.const 83)
    (elem (i32.const 0) $f42 $f83)
)
```

It imports `js.tbl` as a `table`, and then fills it with references to two functions.

### 2.5 WebAssembly Global

`WebAssembly.Global` is a `Global` object that can be accessed by JS and by multiple WASM modules. Its biggest value is that it enables dynamic linking between different modules.

WASM can be compiled from languages such as C++. When compiling C++, we can link multiple `cpp` files together. WASM can do something similar, and it uses `Global` to achieve that. WASM compilers such as Emscripten produce WASM that relies on this mechanism.

```js
new WebAssembly.Global(descriptor {value, mutable}, value);
```

In the first argument `descriptor`, `value` is the type, and `mutable` indicates whether it can be modified. The second argument `value` is the initial value. If you only pass `0`, it uses the default value.

Example ([source](https://mdn.github.io/webassembly-examples/js-api-examples/global.html)):

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

WASM defines three error types: `WebAssembly.CompileError`, `WebAssembly.LinkError`, and `WebAssembly.RuntimeError`.

```js
new WebAssembly.CompileError(message, fileName, lineNumber)
new WebAssembly.LinkError(message, fileName, lineNumber)
new WebAssembly.RuntimeError(message, fileName, lineNumber)
```

They are used in the same way. Example:

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

## 3. Applications

### 3.1 A simple C function

```c
// square.c
int square(int n) { 
   return n*n; 
}
```

Compile with Emscripten:

```shell
$ emcc square.c -s SIDE_MODULE -o square.wasm
```

For Emscripten usage, you can refer to my earlier [introduction](/post/2020/07/js/emscripten-pthread-to-js/#Emscripten-%E4%B8%8B%E8%BC%89). Note that we must add `-s SIDE_MODULE` to indicate this WASM is not a runtime, and specify `-o *.wasm` to output only WASM; otherwise, Emscripten outputs JS + WASM by default.

Convert the compiled WASM to WAT:

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

The key part is `(export "square" (func 1))`. The other unrelated exports can be ignored.

Next, write a web page `square.html`:

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

Put `square.html` and `square.wasm` in the same directory, serve them via an HTTP server (so the WASM can be fetched), and open the page. You will see the result in the console.

### 3.2 A C function using WebAssembly.Memory

This example is essentially the same as what we did in “2.3”, except it demonstrates the flow starting from C.

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

Convert to WAT:

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

From this, we can see `accumulate.wasm` imports `env.__memory_base`, `env.memory`, and `env.g$arr`, so we must define them in JS first.

The web page code for `accumulate.html` is:

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

Because WASM requires `env.memory`, we first declare a `WebAssembly.Memory` object. `env.__memory_base` indicates where in memory to begin reading.

In the original C code, we only have the global `arr[]`, so `env.memory` is effectively the backing storage for `arr[]`. Finally, for some reason Emscripten generates `env.g$arr`; it seems unused, so we provide an empty function.

### 3.3 Compiling Pthreads to JS + WASM

With Emscripten, we can easily compile Pthread programs into Web Workers + SharedArrayBuffer + WebAssembly, and run parallel programs in the browser.

For details, see my earlier post “[Converting Pthreads to JavaScript with Emscripten and Performance Analysis](/post/2020/07/js/emscripten-pthread-to-js)” and its follow-up “[Pthread to WASM: Merge Sort](/post/2020/08/js/emscripten-pthread-to-js-2/)”.

## 4. Conclusion

This post provided a comprehensive overview of the JavaScript APIs for WebAssembly, and demonstrated how to generate WASM from C code and use it from JavaScript in a web page.

WebAssembly is very likely the future trend of the web. As devices become more powerful, our expectations for performance increase. At the same time, WASM itself continues to evolve—such as supporting threads directly in WASM in the future, or enabling SIMD instructions. WASM is also expanding beyond the web: embedded devices and cloud services are beginning to adopt it as well. Overall, WASM’s future is worth looking forward to.

I often feel that many online articles miss something in terms of reasoning and structure. So I reorganized online resources and presented the WASM concepts in what I think is the most logical order. I hope it helps.

## 5. References

- [MDN: WebAssembly](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly)
- [WebAssembly JS API SPEC](https://webassembly.github.io/spec/js-api/#webassembly-namespace)
- [WebAssembly Execution SPEC](https://webassembly.github.io/spec/core/exec/index.html)
- [WebAssembly Standalone](https://github.com/emscripten-core/emscripten/wiki/WebAssembly-Standalone)

