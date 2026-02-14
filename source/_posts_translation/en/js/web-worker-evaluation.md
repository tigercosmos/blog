---
title: "Evaluation of Web Worker for Parallel Programming with Browsers, NodeJS and Deno"
date: 2020-06-26 00:00:00
tags: [JavaScript, web worker, nodejs, deno, parallel programming, browser, browsers, 效能分析, 平行化]
des: "This post briefly evaluates how major browsers, NodeJS, and Deno support Web Workers for parallel programming, and compares differences in usage. The results show that Chromium-based browsers provide good support, it is unfortunate that Firefox does not, and among JavaScript runtimes NodeJS works as expected while Deno still needs time to catch up."
lang: en
translation_key: web-worker-evaluation
---

## Introduction

This post briefly evaluates how major browsers, NodeJS, and Deno support Web Workers for writing parallel programs, and compares differences in usage.

For a deeper introduction to Web Workers, see my earlier post: “[JavaScript parallelism with Web Worker, SharedArrayBuffer, and Atomics](https://tigercosmos.xyz/post/2020/02/web/js-parallel-worker-sharedarraybuffer/)”. I will skip the basics here.

All experiments in this post are run on Windows 10, using an AMD Ryzen 7 2700X 3.7 GHz 8-core CPU (but only enabling 4 threads). I test the same parallel program in Chrome, Edge, Firefox, NodeJS, and Deno, and compare the results.

## Browsers

The web test code is as follows:

```html
<!-- pi.html -->
<script id="worker" type="app/worker">
    addEventListener('message', function (e) {
        const data = e.data;
        const thread = data.thread;
        const start = data.start;
        const end = data.end;
        const u32Arr = new Uint32Array(data.buf);
        const step = data.step;
        const magnification = 1e9;
    
        let x;
        let sum = 0.0;
    
        for (let i = start; i < end; i++) {
            x = (i + 0.5) * step;
            sum = sum + 4.0 / (1.0 + x * x);
        }
    
        sum = sum * step;
    
        Atomics.add(u32Arr, 0, sum * magnification | 0); // (C)
        Atomics.add(u32Arr, 1, 1); // (D)
    
        if (Atomics.load(u32Arr, 1) === thread) { // (E)
            const pi = u32Arr[0] / magnification;
            console.log("PI is", pi);
        }

        close();

    }, false);
</script>
<script>
    const thread = 4;

    const num_steps = 1e9;
    const step = 1.0/num_steps;
    const part = num_steps / thread;

    // [0] = pi, [1] = barrier
    const u32Buf = 
      new SharedArrayBuffer(2 * Uint32Array.BYTES_PER_ELEMENT); // (A)
    const u32Arr = new Uint32Array(u32Buf);
    u32Arr[0] = 0;
    u32Arr[1] = 0;

    for (let i = 0; i < thread; i++) { // (B)
        const blob = new Blob([document.querySelector('#worker').textContent]);
        const url = window.URL.createObjectURL(blob);
        const worker = new Worker(url);

        worker.postMessage({
            thread: thread,
            start: part * i,
            end: part * (i + 1),
            step: step,
            buf: u32Buf,
        });
    }
</script>
```


### Chrome & Edge

Chrome results:

![Chrome Result](https://user-images.githubusercontent.com/18013815/85740272-12cf9d80-b734-11ea-9da9-4f95adc32700.png)

Edge results:

![Edge Result](https://user-images.githubusercontent.com/18013815/85740443-3692e380-b734-11ea-94df-cc88978d2df8.png)

Since both are Chromium-based, the results are similar. (Even the developer tools are the same 😂)

Both start at around 40ms and finish at around 580ms.

### Firefox

In Firefox, SharedArrayBuffer is disabled by default. The setting `javascript.options.shared_memory` is `false` by default, and even after setting it to `true` it still cannot be used. Firefox reports:

“`TypeError: The WebAssembly.Memory object cannot be serialized. The Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy HTTP headers will enable this in the future.`”

For details, see this issue in Emscripten: [issue](https://github.com/emscripten-core/emscripten/issues/10014).

As a result, you cannot run this parallel program on Firefox. At best, you can use Web Workers for independent tasks. After some investigation, I concluded that using SharedArrayBuffer is too difficult, so I gave up.

## NodeJS

Web Workers are an API. They are not part of the JavaScript engine; they are something the runtime needs to implement. Therefore, NodeJS implements the API differently from browsers.

In practice, the main differences are that the main program needs to import `Worker` from `worker_threads`, and inside a worker you need to use `parentPort`.

Test code:

main.js
```js
const {
    Worker
  } = require('worker_threads');

const thread = 4;

const num_steps = 1e9;
const step = 1.0/num_steps;
const part = num_steps / thread;

// [0] = pi, [1] = barrier
const u32Buf = 
  new SharedArrayBuffer(2 * Uint32Array.BYTES_PER_ELEMENT); // (A)
const u32Arr = new Uint32Array(u32Buf);
u32Arr[0] = 0;
u32Arr[1] = 0;

for (let i = 0; i < thread; i++) { // (B)
    const worker = new Worker('./worker.js');

    worker.postMessage({
        thread: thread,
        start: part * i,
        end: part * (i + 1),
        step: step,
        buf: u32Buf,
    });
}
```

worker.js
```js
const { parentPort } = require('worker_threads')

parentPort.onmessage = function (e) {
    const data = e.data;
    const thread = data.thread;
    const start = data.start;
    const end = data.end;
    const u32Arr = new Uint32Array(data.buf);
    const step = data.step;
    const magnification = 1e9;

    let x;
    let sum = 0.0;

    for (let i = start; i < end; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }

    sum = sum * step;

    Atomics.add(u32Arr, 0, sum * magnification | 0); // (C)
    Atomics.add(u32Arr, 1, 1); // (D)

    if (Atomics.load(u32Arr, 1) === thread) { // (E)
        const pi = u32Arr[0] / magnification;
        console.log("PI is", pi);
    }

    parentPort.close();
};
```

Running with Node v12.14.1 yields:

```shell
$ Measure-Command {node main.js}  

Milliseconds      : 465
Ticks             : 4659527
TotalMilliseconds : 465.9527
```

This is quite surprising, because it is about 100 ms faster than in browsers. Note that the engine is V8 in both cases, and in the browser test I can confirm via performance profiling that nothing else is consuming CPU resources. Even if we subtract worker startup time in the browser, it still looks significantly slower than Node.

My guess is that the difference comes from the runtime-level implementation of Web Workers. As mentioned earlier, Web Workers are implemented by the runtime, not by V8 itself.

## Deno

Unfortunately, as of the time this post was published, Deno v1.1.1 does not fully support Web Workers. For example, it does not support SharedArrayBuffer, and Workers only support the module type.

For more details, see the issue I filed: “[SharedArrayBuffer not works #6433](https://github.com/denoland/deno/issues/6433)”.

Test code:

main.js
```js
const u32Buf = new SharedArrayBuffer(2 * Uint32Array.BYTES_PER_ELEMENT);
const u32Arr = new Uint32Array(u32Buf);

const worker = new Worker(new URL("./worker.js", import.meta.url).href, { type: "module" });

u32Arr[0] = 1

worker.postMessage({
    buf: u32Buf,
});
```

worker.js
```js
self.onmessage = function (e) {
    const data = e.data;
    const u32Arr = new Uint32Array(data.buf);

    console.log(u32Arr[0])
};
```

We would expect `u32Arr[0]` to become `1`, but because it is not supported, we get `undefined`:

```shell
$ deno run --allow-read main.js
undefined
```

For the implementation status of Web Workers in Deno, you can also refer to the project’s [source code](https://github.com/denoland/deno/search?p=2&q=Worker&unscoped_q=Worker).

## Conclusion

I am very optimistic about Web Workers. In an era where everyone is chasing performance, web experiences should be fast and efficient—so parallel processing with multiple threads should be the natural direction. At the same time, Web Workers are not only for browsers. In the “everything is JavaScript” era, cross-platform is commonplace, so runtimes like NodeJS and Deno should also keep up.

From these experiments, it is clear that Chromium-based browsers support Web Workers for parallel programming quite well. It is unfortunate that Firefox does not. As for JavaScript runtimes, NodeJS—as the veteran—naturally supports it, while Deno—as a rising newcomer—still needs more time.
