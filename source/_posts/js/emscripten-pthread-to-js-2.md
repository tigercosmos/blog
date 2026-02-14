---
title: 使用 Emscripten 將 Pthread 轉成 JavaScript 與效能分析 (2) — Merge Sort
date: 2020-08-10 08:00:00
tags: [JavaScript, web worker, nodejs, c, pthread, parallel programming, browser, browsers, 效能分析, 平行化, algorithm, 演算法]
des: "本文實驗 Pthread、Pthread 轉 JS + WASM 和純 JS，執行 Merge Sort 的效能。結果 Merge Sort 以 Pthread 版本跑最快，Emscripten 轉換的 WASM 其次，其中又以一般模式在陣列長度小時比較快，而 Proxy 模式在陣列長度大時有優勢，最後純 JS 的版本執行速度最慢。"
lang: zh
translation_key: emscripten-pthread-to-js-2
---

## 1. 簡介
 
本文接續前篇「[使用 Emscripten 將 Pthread 轉成 JavaScript 與效能分析](/post/2020/07/js/emscripten-pthread-to-js/)」，對用 Emscripten 將 Pthread 轉 JS 做另一個 Case 的測試，這次將檢驗 Merge Sort 的表現。

## 2. Merge Sort 

[Merge Sort](https://en.wikipedia.org/wiki/Merge_sort) 複雜度原本是 $\Theta(nlogn)$，如果將 Recursion 的部分做平行化之後可以縮減到 $\Theta(n)$，若是合併 Sort 也平行化可以優化到 $\Theta({n \over (log n)^2})$。

<img alt="merge sort image" src="https://user-images.githubusercontent.com/18013815/89739372-a37af680-dab2-11ea-9f27-2718fd2535c4.png" width=50%>

(source: [Wikipedia](https://en.wikipedia.org/wiki/Merge_sort) )


### 2.1 Pthread 版本 Merge Sort

以下 Merge Sort 的 Pthread 範例 C++ Code 來自 [Geeksforgeeks](https://www.geeksforgeeks.org/merge-sort/)，這個範例 Code 僅將 Recursion 做平行化，合併 Sort 的部分為單緒執行。我已經將註解都翻譯成中文。

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

執行這段 code：
```shell
$ g++ merge_sort.cc -lpthread
$ ./a.out
```

### 2.2 Pthread 轉 JavaScript 版本 Merge Sort

Emscripten 的介紹和使用方法請看[前篇](/post/2020/07/js/emscripten-pthread-to-js/#Emscripten-%E4%B8%8B%E8%BC%89)。本文將進行兩個實驗，一個是原本的 `main()` 執行在 JS 的 Main Thread，另一個則是將 `main()` 改成由一個 Worker 驅動。

後者在 Emscripten 中稱為 Proxy 模式，之所以要這樣是因為 JS 的 Main Thread 通常還要負責網頁的 UI 和 Event。如果沒有用 Proxy，照一般 Pthread 的邏輯，Main Thread 只要出現 `pthread_join` 或是有上 Lock 的地方，Main Thread 就會直接卡住，不僅畫面會卡住，瀏覽器還可能跳出網頁沒有回應的警告，這在網頁設計中是很致命的。

如果要用 Proxy 模式，要設定 `PROXY_TO_PTHREAD=1`，此外因為我們 Merge Sort 的陣列可能會很大，要設定 `ALLOW_MEMORY_GROWTH=1` 突破記憶體預設限制。

Proxy 模式：
```shell
$ em++ merge_sort.cc \
    -s USE_PTHREADS=1 \
    -s PROXY_TO_PTHREAD=1 \
    -s PTHREAD_POOL_SIZE=4 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -O3 \
    -o merge_sort_proxy.html
```

一般模式：
```shell
$ em++ merge_sort.cc \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE=4 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -O3 \
    -o merge_sort_no_proxy.html
```

### 2.3 JavaScript 版本 Merge Sort

依照 C++ 版本邏輯，我重新刻了一個 JavaScript 版本。

差別在於 `arr` 原本是全域變數，現在必須使用 SharedArrayBuffer。

然後 Join 的部分改成 Worker 用 Message 去通知，並用 SharedArrayBuffer 和 Atomics 來實現 Join 的 Barrier (一次只能加 1，加到 4 代表所有 Worker 都做完)。

此外這邊用到一個小技巧，因為 C++ 中 int 型別除法會自動捨棄小數，但 JS 中所有數值都是 Float64，所以要將小樹捨棄，速度最快的作法是 `|0`，例如 ` 5/2 | 0` 會得到 `2`。

以下儲存成一個 HTML 檔案，直接用瀏覽器打開就能跑了：

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

## 3. 效能分析

效能實驗分別測試 (1) Pthread 用 Emscripten Proxy 模式 O3 轉 JS + WASM  (2) Pthread 用Emscripten 一般模式 O3 轉 JS + WASM (3) Pthread g++ O3 (4) Pthread g++ O2 (5) Chrome 跑純 JavaScript 版本，這幾種情況下效能差異。實驗環境為 AMD Ryzen 7 2700X 3.7 GHz 八核處理器 (只開 4 執行緒)，emcc v1.39、g++ v7.5、Chrome 84。

| Array Length  | Em Proxy | Em No Proxy | C++ O3 | C++ O2 | JS |
|---|---|---|---|---|---|---|
|  100000 | 0.056s  | 0.062s | 0.027s | 0.032s | 0.214s |
|  10000 | 0.031s  | 0.009s | 0.002s | 0.003s | 0.052s |
|  1000 |  0.029s | 0.002s | 0.001s | 0.001s | 0.023s |

將上表時間取 log 並反轉可以得到下圖。同個長度下，不同方法之間越高代表越快。不同長度，同個方法下，越高代表越快。

![result graph](https://user-images.githubusercontent.com/18013815/89744481-eacbac00-dadf-11ea-9513-825905cba9d4.png)

Pthread 不意外是最快的，不過 O3 和 O2 相差沒很多，這個結果跟前篇文章 PI 的結果大相逕庭。

有個滿特別的現象是，不管長度如何，不同方法的 1e3 和 1e4 之間相差很少，不像 1e4 到 1e5 這樣劇烈，推測是長度在 1e4 以下 Overheads 會發生在其他地方。

看起來 Emscripten Proxy 模式在陣列數量小的時候 Overheads 比較大，在 1e3 和 1e4 都輸給 Emscripten 一般模式不少，不過在數量達 1e5 的時候，Proxy 模式反而跑得比一般模式快。

最後 JS 版本不管長度如何都輸給其他方法，不過這也很合理，畢竟不管是 C++ 或 WASM 本來就會比 JS 快。不過 1e3 和 1e4 的時候輸比較少，到了 1e5 的時候 JS 直接慢 C++ 8x 和慢 WASM 6x。

## 4. 結論

本文實驗 Pthread、Pthread 轉 JS + WASM 和純 JS，執行 Merge Sort 的效能。結果 Merge Sort 以 Pthread 版本跑最快，Emscripten 轉換的 WASM 其次，其中又以一般模式在陣列長度小時比較快，而 Proxy 模式在陣列長度大時有優勢，最後純 JS 的版本執行速度最慢。
