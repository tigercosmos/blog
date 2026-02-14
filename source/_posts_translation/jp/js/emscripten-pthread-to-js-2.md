---
title: "Emscripten で Pthread を JavaScript に変換し、性能を分析する (2) — Merge Sort"
date: 2020-08-10 08:00:00
tags: [JavaScript, web worker, nodejs, c, pthread, parallel programming, browser, browsers, 效能分析, 平行化, algorithm, 演算法]
des: "本記事では、Merge Sort を題材に (1) Pthread 版 (2) Emscripten による Pthread→JS+WASM 版 (3) 純 JavaScript 版の性能を比較します。結果として Pthread 版が最速で、次に Emscripten 変換の WASM が続きます。Emscripten では配列長が小さい場合は通常モードが速く、配列長が大きい場合は Proxy モードが有利であり、最後に純 JS 版が最も遅いという結論でした。"
lang: jp
translation_key: emscripten-pthread-to-js-2
---

## 1. 概要
 
本記事は前編「[使用 Emscripten 將 Pthread 轉成 JavaScript 與效能分析](/post/2020/07/js/emscripten-pthread-to-js/)」の続きです。Emscripten を使って Pthread を JS に変換する別ケースを試し、今回は Merge Sort の挙動を検証します。

## 2. Merge Sort

[Merge Sort](https://en.wikipedia.org/wiki/Merge_sort) の計算量はもともと $\Theta(nlogn)$ です。Recursion 部分を平行化すれば $\Theta(n)$ に縮められ、さらに merge（合併）の部分まで平行化できれば $\Theta({n \over (log n)^2})$ まで最適化できます。

<img alt="merge sort image" src="https://user-images.githubusercontent.com/18013815/89739372-a37af680-dab2-11ea-9f27-2718fd2535c4.png" width=50%>

(source: [Wikipedia](https://en.wikipedia.org/wiki/Merge_sort) )


### 2.1 Pthread 版 Merge Sort

以下の Pthread 版 Merge Sort の C++ サンプルコードは [Geeksforgeeks](https://www.geeksforgeeks.org/merge-sort/) から引用しています。このサンプルは Recursion の部分のみを平行化しており、merge（合併）の部分は単一スレッドで実行します。コメントはすべて中国語に翻訳してあります。

`merge_sort.cc`:
```c++
#include <iostream> 
#include <pthread.h> 
#include <time.h> 

// 陣列元素數量
#define MAX 10000

// 幾個 threads 
#define THREAD_MAX 4 

using namespace std; 

// 用來算 merge sort 的陣列 
int arr[MAX]; 
int part = 0; 

// 用來合併兩個已經排好的部份的函數
void merge(int low, int mid, int high) 
{ 
    // 暫時儲存左邊部分和右邊部分的陣列
	int* left = new int[mid - low + 1]; 
	int* right = new int[high - mid]; 

	// n1 是左邊部分的長度，n2 是右邊部分的長度
	int n1 = mid - low + 1, n2 = high - mid, i, j; 

	// 將左邊部分從 arr 複製到 left 
	for (i = 0; i < n1; i++) 
		left[i] = arr[i + low]; 

	// 將右邊部分從 arr 複製到 right
	for (i = 0; i < n2; i++) 
		right[i] = arr[i + mid + 1]; 

	int k = low; 
	i = j = 0; 

	// 從左到右遞升合併進 arr
	while (i < n1 && j < n2) { 
		if (left[i] <= right[j]) 
			arr[k++] = left[i++]; 
		else
			arr[k++] = right[j++]; 
	} 

	// 插入左邊剩下的部分
	while (i < n1) { 
		arr[k++] = left[i++]; 
	} 

	// 插入右邊剩下的部分
	while (j < n2) { 
		arr[k++] = right[j++]; 
	} 
} 

// merge sort 函數
void merge_sort(int low, int high) 
{ 
	// 取得陣列中位數
	int mid = low + (high - low) / 2; 
	if (low < high) { 

		// 遞迴左半邊
		merge_sort(low, mid); 

		// 遞迴右半邊
		merge_sort(mid + 1, high); 

		// 合併
		merge(low, mid, high); 
	} 
} 

// thread 用的函數
void* merge_sort(void* arg) 
{ 
	// 4 個 thread 的哪一個
	int thread_part = part++; 

	// 每個 thread 的範圍
	int low = thread_part * (MAX / 4); 
	int high = (thread_part + 1) * (MAX / 4) - 1; 

	int mid = low + (high - low) / 2; 
	if (low < high) { 
		merge_sort(low, mid); 
		merge_sort(mid + 1, high); 
		merge(low, mid, high); 
	} 
} 

// 起始點
int main() 
{ 
	// 隨便填入數值
	for (int i = 0; i < MAX; i++) 
		arr[i] = rand() % 100; 

	// t1 and t2 用來計算時間，分別是開始和結束
	clock_t t1, t2; 

	t1 = clock(); 
	pthread_t threads[THREAD_MAX]; 

	// 建立 4 個 threads 
	for (int i = 0; i < THREAD_MAX; i++) 
		pthread_create(&threads[i], NULL, merge_sort, (void*)NULL); 

	// Join 4 個 Thread 
	for (int i = 0; i < 4; i++) 
		pthread_join(threads[i], NULL); 

	// 將 4 個部分的結果合併
	merge(0, (MAX / 2 - 1) / 2, MAX / 2 - 1); 
	merge(MAX / 2, MAX/2 + (MAX-1-MAX/2)/2, MAX - 1); 
	merge(0, (MAX - 1)/2, MAX - 1); 

	t2 = clock(); 

	// 列印結果，註解掉防止 I/O 造成負擔
	/* 
        cout << "Sorted array: "; 
        for (int i = 0; i < MAX; i++) 
            cout << arr[i] << " "; 
    */

	// 列印執行時間
	cout << "Time taken: " << (t2 - t1) / 
			(double)CLOCKS_PER_SEC << endl; 

	return 0; 
}
```

このコードを実行します：
```shell
$ g++ merge_sort.cc -lpthread
$ ./a.out
```

### 2.2 Pthread→JavaScript 版 Merge Sort

Emscripten の概要と使い方は [前編](/post/2020/07/js/emscripten-pthread-to-js/#Emscripten-%E4%B8%8B%E8%BC%89) を参照してください。本記事では 2 つの実験を行います。1 つは元の `main()` を JS の Main Thread 上で実行するもの、もう 1 つは `main()` を Worker で駆動するように変更するものです。

後者は Emscripten では Proxy モードと呼ばれます。理由は、JS の Main Thread は通常 Web の UI とイベントも処理する必要があるためです。Proxy を使わない場合、通常の Pthread のロジックでは、Main Thread が `pthread_join` やロック領域に到達するとそのままブロックしてしまいます。画面がフリーズするだけでなく、ブラウザが「ページが応答していません」といった警告を出す可能性もあり、Web 設計上これは致命的です。

Proxy モードを使うには `PROXY_TO_PTHREAD=1` を設定します。さらに、Merge Sort の配列は大きくなり得るので、既定のメモリ制限を突破するため `ALLOW_MEMORY_GROWTH=1` を指定します。

Proxy モード：
```shell
$ em++ merge_sort.cc \
    -s USE_PTHREADS=1 \
    -s PROXY_TO_PTHREAD=1 \
    -s PTHREAD_POOL_SIZE=4 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -O3 \
    -o merge_sort_proxy.html
```

通常モード：
```shell
$ em++ merge_sort.cc \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE=4 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -O3 \
    -o merge_sort_no_proxy.html
```

### 2.3 JavaScript 版 Merge Sort

C++ 版のロジックに沿って、JavaScript 版を作り直しました。

違いとして、`arr` は元々グローバル変数でしたが、JavaScript 版では SharedArrayBuffer を使う必要があります。

また Join に相当する部分は、Worker が Message で完了通知を行い、SharedArrayBuffer と Atomics で Join のバリアを実装します（1 回に 1 だけ加算し、4 になったら全 Worker が完了したことを意味します）。

さらに小技として、C++ の int 除算は小数点以下が自動で切り捨てられますが、JS の数値はすべて Float64 です。小数部分を切り捨てる最速の方法は `|0` で、例えば ` 5/2 | 0` は `2` になります。

以下を HTML ファイルとして保存すれば、ブラウザで直接開くだけで動きます：

```html
<!-- merge-sort.html -->
<div id="content"></div>

<script id="worker" type="app/worker">
    let arr;

    // merge function for merging two parts 
    function merge(low, mid, high) {
        const left = new Array(mid - low + 1);
        const right = new Array(high - mid);

        const n1 = mid - low + 1;
        const n2 = high - mid;
        let i, j;

        for (i = 0; i < n1; i++)
            left[i] = arr[i + low];

        for (i = 0; i < n2; i++)
            right[i] = arr[i + mid + 1];

        let k = low;
        i = j = 0;

        while (i < n1 && j < n2) {
            if (left[i] <= right[j])
                arr[k++] = left[i++];
            else
                arr[k++] = right[j++];
        }

        while (i < n1) {
            arr[k++] = left[i++];
        }

        while (j < n2) {
            arr[k++] = right[j++];
        }
    }

    function merge_sort(low, high) {
        const mid = (low + (high - low) / 2 | 0 );
        if (low < high) {

            merge_sort(low, mid);

            merge_sort(mid + 1, high);

            merge(low, mid, high);
        }
    }

    addEventListener('message', function (e) {
        const low = e.data.low;
        const high = e.data.high;
        arr = new Int32Array(e.data.buf);

        const mid = (low + (high - low) / 2) | 0;

        if (low < high) {
            merge_sort(low, mid);
            merge_sort(mid + 1, high);
            merge(low, mid, high);
        }

        postMessage("done");
    }, false);
</script>
<script>
    function merge(low, mid, high) {
        console.log(arr)
        const left = new Array(mid - low + 1);
        const right = new Array(high - mid);

        const n1 = mid - low + 1;
        const n2 = high - mid;
        let i, j;

        for (i = 0; i < n1; i++)
            left[i] = arr[i + low];

        for (i = 0; i < n2; i++)
            right[i] = arr[i + mid + 1];

        let k = low;
        i = j = 0;

        while (i < n1 && j < n2) {
            if (left[i] <= right[j])
                arr[k++] = left[i++];
            else
                arr[k++] = right[j++];
        }

        while (i < n1) {
            arr[k++] = left[i++];
        }

        while (j < n2) {
            arr[k++] = right[j++];
        }
    }

    const first_time = (new Date()).getTime();
    const thread = 4;
    const arrMax = 1e5;

    const buf = new SharedArrayBuffer(arrMax * Int32Array.BYTES_PER_ELEMENT);
    const arr = new Int32Array(buf);

    for (const i in arr) {
        arr[i] = Math.random() * 10000 | 0;
    }
    document.getElementById("content").innerHTML = "origin: " + arr + "<br/>";

    const barrierBuf = new SharedArrayBuffer(1 * Int32Array.BYTES_PER_ELEMENT);
    const barrier = new Int32Array(barrierBuf);

    for (let i = 0; i < thread; i++) {
        const blob = new Blob([document.querySelector('#worker').textContent]);
        const url = window.URL.createObjectURL(blob);
        const worker = new Worker(url);

        const low = i * (arrMax / thread);
        const high = (i + 1) * (arrMax / thread) - 1;

        worker.postMessage({
            low: low,
            high: high,
            buf: buf,
        });

        worker.onmessage = e => {
            if (e.data == "done") {
                Atomics.add(barrier, 0, 1);

                if (Atomics.load(barrier, 0) === thread) {

                    merge(0, ((arrMax / 2 | 0) - 1) / 2 | 0, (arrMax / 2 | 0) - 1);
                    merge(arrMax / 2, arrMax / 2 | 0 + 
                        (arrMax - 1 - (arrMax / 2 | 0) | 0 / 2) | 0, arrMax - 1);
                    merge(0, (arrMax - 1) / 2 | 0, arrMax - 1);

                    const last_time = (new Date()).getTime();
                    const content = "new: " + arr + "<br/>" +
                        "time: " + (last_time - first_time) / 1000 + "s"
                    document.getElementById("content").innerHTML += content;
                }
            }
        };
    }
</script>
```

## 3. 性能分析

性能実験では、(1) Pthread を Emscripten Proxy モード + -O3 で JS + WASM に変換した場合、(2) Pthread を Emscripten 通常モード + -O3 で JS + WASM に変換した場合、(3) Pthread を g++ -O3 でコンパイルした場合、(4) Pthread を g++ -O2 でコンパイルした場合、(5) Chrome 上で純 JavaScript 版を実行した場合、これらの性能差を測ります。実験環境は AMD Ryzen 7 2700X 3.7 GHz（8 コアだが 4 スレッドのみ使用）、emcc v1.39、g++ v7.5、Chrome 84 です。

| Array Length  | Em Proxy | Em No Proxy | C++ O3 | C++ O2 | JS |
|---|---|---|---|---|---|---|
|  100000 | 0.056s  | 0.062s | 0.027s | 0.032s | 0.214s |
|  10000 | 0.031s  | 0.009s | 0.002s | 0.003s | 0.052s |
|  1000 |  0.029s | 0.002s | 0.001s | 0.001s | 0.023s |

上表の時間を log 変換し反転すると、次の図になります。同じ配列長で比較する場合、異なる手法の中で値が高いほど速いことを意味します。配列長が異なる場合でも、同じ手法の中で値が高いほど速いことを意味します。

![result graph](https://user-images.githubusercontent.com/18013815/89744481-eacbac00-dadf-11ea-9513-825905cba9d4.png)

Pthread が最速なのは想定通りです。一方で -O3 と -O2 の差はそれほど大きくなく、前編の π の結果とは大きく異なります。

また興味深い現象として、配列長に関係なく、各手法において 1e3 と 1e4 の差は小さく、1e4 から 1e5 ほど劇的ではありません。配列長が 1e4 以下では、オーバーヘッドが別の箇所で支配的になっている可能性があると推測しています。

Emscripten Proxy モードは配列数が小さいときにオーバーヘッドが大きいようで、1e3 と 1e4 では通常モードに対して大きく負けています。しかし配列数が 1e5 に達すると、Proxy モードの方が通常モードより速くなります。

最後に、純 JS 版は配列長に関わらず他の方法に負けています。ただしこれは自然で、C++ や WASM は元々 JS より速いからです。ただ 1e3 と 1e4 では差が小さく、1e5 になると JS は C++ の 8 倍遅く、WASM の 6 倍遅くなります。

## 4. 結論

本記事では、Merge Sort を題材に (1) Pthread 版 (2) Emscripten による Pthread→JS+WASM 版 (3) 純 JavaScript 版の性能を比較しました。結果として Pthread 版が最速で、次に Emscripten 変換の WASM が続きます。Emscripten では配列長が小さい場合は通常モードが速く、配列長が大きい場合は Proxy モードが有利であり、最後に純 JS 版が最も遅いという結論でした。

