---
title: Evaluation of Web Worker for Parallel Programming with Browsers, NodeJS and Deno
date: 2020-06-26 00:00:00
tags: [JavaScript, web worker, nodejs, deno, parallel programming, browser, browsers, 效能分析, 平行化]
des: "本篇簡單評估各家瀏覽器、NodeJS 和 Deno 對 Web Worker 在開發平行程式上支援程度和使用差異。實驗結果來看，顯然 Chromium 為底的瀏覽器都支援不錯，但對 Firefox 不支援感到可惜，至於 JavaScript 中的 Runtime，NodeJS 身為元老自然支援，Deno 為新起之秀在支援上還要再等等。"
lang: zh
translation_key: web-worker-evaluation
---

## 簡介

本篇簡單評估各家瀏覽器、NodeJS 和 Deno 對 Web Worker 在開發平行程式上支援程度和使用差異。

關於 Web Worker 的深入介紹可以看我之前寫的「[JavaScript 平行化使用 Web Worker、SharedArrayBuffer、Atomics](https://tigercosmos.xyz/post/2020/02/web/js-parallel-worker-sharedarraybuffer/)」，本篇將略過基本介紹。

本文將在 Windows 10 平台中，以 AMD Ryzen 7 2700X 3.7 GHz 八核處理器 (但只開 4 執行緒) 中做實驗，測試同一段平行程式在 Chrome、Edge、Firefox、NodeJS、Deno 中的表現差異。

## 瀏覽器

網頁測試碼如下：

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

Chrome 效能結果：

![Chrome Result](https://user-images.githubusercontent.com/18013815/85740272-12cf9d80-b734-11ea-9da9-4f95adc32700.png)

Edge 效能結果：

![Edge Result](https://user-images.githubusercontent.com/18013815/85740443-3692e380-b734-11ea-94df-cc88978d2df8.png)

因為都是基於 Chromium，所以表現差不多。(開發工具也一樣😂)

啟動時間都差不多在 40ms，結束時間都差不多 580 ms。

### Firefox

Firefox 對 SharedArrayBuffer 預設是關閉，其設定值 `javascript.options.shared_memory` 預設是 `false`，即使調成 `true` 一樣還是不能使用。會吐出「`TypeError: The WebAssembly.Memory object cannot be serialized. The Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy HTTP headers will enable this in the future.`」細節可以參照 Emscripten 中的[這條 Issue](https://github.com/emscripten-core/emscripten/issues/10014)。

所以在 Firefox 下無法進行平行化程式，大概只能用 Web Worker 跑獨立作業，稍微研究後認為要使用 SharedArrayBuffer 困難重重，直接放棄。

## NodeJS

Web Worker 是 API，跟 JavaScript 引擎無關，屬於 Runtime 自己要處理的東西，所以 NodeJS 實現 API 的方式跟瀏覽器有差異。

使用上，主要差異是主程序要引入 `worker_threads` 函式庫中的 `Worker`，以及 Worker 裡面要呼叫 `parentPort`。

測試碼如下：

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

用 Node v12.14.1 執行結果如下：

```shell
$ Measure-Command {node main.js}  

Milliseconds      : 465
Ticks             : 4659527
TotalMilliseconds : 465.9527
```

這結果滿令人意外的，因為整整比瀏覽器執行還少了 100 ms，要知道背後引擎都是 V8，且瀏覽器測試時可以從效能分析看到沒有其他部分占用 CPU 資源，就算扣掉瀏覽器中全部 Worker 啟動的時間，還是比 Node 多很多。

猜測是在實現 Web Worker 上的差異。先前提過 Web Worker 的實作是建立在 Runtime 上，也就是跟 V8 無關。

## Deno

很遺憾，截至本文發布為止，Deno V1.1.1 並不完整支援 Web Worker，像是其不支援 SharedArrayBuffer，Worker 也還只支援 Module Type。

更進一步，可以參考我提出的 Issue「[SharedArrayBuffer not works #6433](https://github.com/denoland/deno/issues/6433)」。

測試碼如下：

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

執行結果預期 `u32Arr[0]` 會變成 `1`，但因為不支援所以得到 `undefined`。 

```shell
$ deno run --allow-read main.js
undefined
```

關於 Deno 在 Web Worker 上實現的進度可以參考其專案的[原始碼](https://github.com/denoland/deno/search?p=2&q=Worker&unscoped_q=Worker)。

## 結論

我對 Web Worker 是非常期待的，在追逐效能的時代，網頁體驗應該高速有效率，那麼開多個執行緒的平行處理也就理所當然。同時 Web Worker 不單只是瀏覽器專用，萬物皆 JavaScript 的這年頭，跨平台也是稀鬆平常，所以 NodeJS 和 Deno 這類的 Runtime 必然也得上緊發條。

實驗結果來看 Web Worker 開發平行程式可行性，顯然 Chromium 為底的瀏覽器都支援不錯，但對 Firefox 不支援感到可惜，至於 JavaScript 中的 Runtime，NodeJS 身為元老自然支援，Deno 為新起之秀在支援上還要再等等。
