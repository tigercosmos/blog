---
title: "ブラウザ・NodeJS・Deno における Web Worker の平行プログラミング対応評価"
date: 2020-06-26 00:00:00
tags: [JavaScript, web worker, nodejs, deno, parallel programming, browser, browsers, 效能分析, 平行化]
des: "本記事では、主要ブラウザ、NodeJS、Deno が平行プログラミング用途で Web Worker をどの程度サポートしているか、そして利用上の違いを簡単に評価します。結果として Chromium 系ブラウザは良好に対応している一方、Firefox が対応していない点は残念であり、JavaScript の Runtime では NodeJS は問題なく動作するものの、Deno はまだ追いつくまで時間が必要です。"
lang: jp
translation_key: web-worker-evaluation
---

## 概要

本記事では、主要ブラウザ、NodeJS、Deno が平行プログラミング用途で Web Worker をどの程度サポートしているか、そして利用上の違いを簡単に評価します。

Web Worker の詳細な紹介については、以前書いた記事「[JavaScript 平行化使用 Web Worker、SharedArrayBuffer、Atomics](https://tigercosmos.xyz/post/2020/02/web/js-parallel-worker-sharedarraybuffer/)」を参照してください。本記事では基本説明は省略します。

実験は Windows 10 環境で行い、CPU は AMD Ryzen 7 2700X 3.7 GHz の 8 コア（ただしスレッドは 4 のみ使用）です。同一の平行プログラムを Chrome、Edge、Firefox、NodeJS、Deno で実行し、結果の差を比較します。

## ブラウザ

Web 側のテストコードは次の通りです：

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

Chrome の結果：

![Chrome Result](https://user-images.githubusercontent.com/18013815/85740272-12cf9d80-b734-11ea-9da9-4f95adc32700.png)

Edge の結果：

![Edge Result](https://user-images.githubusercontent.com/18013815/85740443-3692e380-b734-11ea-94df-cc88978d2df8.png)

どちらも Chromium ベースなので、結果はほぼ同じです。（開発ツールも同じです😂）

起動はだいたい 40ms、終了はだいたい 580ms でした。

### Firefox

Firefox では SharedArrayBuffer がデフォルトで無効になっています。設定 `javascript.options.shared_memory` は既定で `false` であり、`true` に変更しても利用できません。次のようなエラーが出ます：

「`TypeError: The WebAssembly.Memory object cannot be serialized. The Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy HTTP headers will enable this in the future.`」

詳細は Emscripten のこの [Issue](https://github.com/emscripten-core/emscripten/issues/10014) を参照してください。

そのため Firefox ではこの平行プログラムを動かせません。Web Worker を独立作業に使うことはできても、SharedArrayBuffer を使うのはハードルが高すぎると判断して、ここでは断念しました。

## NodeJS

Web Worker は API であり、JavaScript エンジンそのものとは無関係で、Runtime が実装する機能です。そのため NodeJS の API 実装はブラウザとは異なります。

実際の違いとしては、メイン側で `worker_threads` ライブラリから `Worker` を読み込む必要があること、そして Worker 側で `parentPort` を使う必要があることです。

テストコードは次の通りです：

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

Node v12.14.1 での実行結果は次の通りです：

```shell
$ Measure-Command {node main.js}  

Milliseconds      : 465
Ticks             : 4659527
TotalMilliseconds : 465.9527
```

これはかなり意外です。ブラウザ実行より 100ms ほど速いからです。背後のエンジンはどちらも V8 であり、ブラウザ側でも性能分析を見る限り他の処理が CPU を消費しているわけではありません。ブラウザで Worker 起動にかかる時間を差し引いても、Node の方がかなり速いように見えます。

原因としては、Web Worker の実装差（Runtime 側の差）だと推測しています。先に述べた通り、Web Worker の実装は V8 ではなく Runtime が担います。

## Deno

残念ながら、本記事の公開時点では Deno v1.1.1 は Web Worker を完全にはサポートしていません。たとえば SharedArrayBuffer をサポートしておらず、Worker も Module Type のみ対応です。

さらに詳しくは、私が提出した Issue「[SharedArrayBuffer not works #6433](https://github.com/denoland/deno/issues/6433)」を参照してください。

テストコードは次の通りです：

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

本来は `u32Arr[0]` が `1` になることを期待しますが、未対応のため `undefined` になります：

```shell
$ deno run --allow-read main.js
undefined
```

Deno の Web Worker 実装の進捗については、プロジェクトの [ソースコード](https://github.com/denoland/deno/search?p=2&q=Worker&unscoped_q=Worker) も参考になります。

## 結論

私は Web Worker に大きく期待しています。性能を追い求める時代において、Web 体験は高速かつ効率的であるべきであり、複数スレッドによる平行処理は自然な流れです。さらに Web Worker はブラウザ専用というわけではありません。「万物 JavaScript」の時代ではクロスプラットフォームは当たり前であり、NodeJS や Deno のような Runtime も当然対応を進める必要があります。

実験結果を見ると、Chromium ベースのブラウザは Web Worker を使った平行プログラミングに十分対応していることが分かります。一方で Firefox が対応していない点は残念です。JavaScript の Runtime では、元老である NodeJS は自然にサポートしている一方、新進気鋭の Deno はまだ時間が必要です。

