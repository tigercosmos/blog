---
title: "Converting Pthreads to JavaScript with Emscripten and Performance Analysis (2) — Merge Sort"
date: 2020-08-10 08:00:00
tags: [JavaScript, web worker, nodejs, c, pthread, parallel programming, browser, browsers, 效能分析, 平行化, algorithm, 演算法]
des: "This post benchmarks Merge Sort across (1) native Pthreads, (2) Pthreads compiled to JS + WASM via Emscripten, and (3) a pure JavaScript implementation. The results show that the Pthread version is fastest, Emscripten-generated WASM comes next; among the Emscripten modes, the normal mode is faster for small arrays while Proxy mode has an advantage for large arrays; and the pure JS version is the slowest."
lang: en
translation_key: emscripten-pthread-to-js-2
---

## 1. Introduction
 
This post continues the previous one: “[Converting Pthreads to JavaScript with Emscripten and Performance Analysis](/post/2020/07/js/emscripten-pthread-to-js/)”. Here, I run another case study for “Pthread → JS” using Emscripten, and this time I evaluate Merge Sort performance.

## 2. Merge Sort

[Merge Sort](https://en.wikipedia.org/wiki/Merge_sort) has time complexity $\Theta(nlogn)$. If you parallelize the recursion part, it can be reduced to $\Theta(n)$. If you also parallelize the merge step, it can be further optimized to $\Theta({n \over (log n)^2})$.

<img alt="merge sort image" src="https://user-images.githubusercontent.com/18013815/89739372-a37af680-dab2-11ea-9f27-2718fd2535c4.png" width=50%>

(source: [Wikipedia](https://en.wikipedia.org/wiki/Merge_sort) )


### 2.1 Pthread version of Merge Sort

The following Pthread-based Merge Sort C++ example code is from [Geeksforgeeks](https://www.geeksforgeeks.org/merge-sort/). This example only parallelizes the recursion part; the merge step runs single-threaded. I have translated all the comments into Chinese.

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

Run this code:
```shell
$ g++ merge_sort.cc -lpthread
$ ./a.out
```

### 2.2 Merge Sort compiled from Pthreads to JavaScript

For Emscripten introduction and usage, see the [previous post](/post/2020/07/js/emscripten-pthread-to-js/#Emscripten-%E4%B8%8B%E8%BC%89). In this post, I run two experiments: one where the original `main()` runs on the JS main thread, and another where `main()` is driven by a worker.

The latter is called Proxy mode in Emscripten. The reason is that the JS main thread typically also handles web UI and events. Without Proxy mode, under normal Pthread logic, whenever the main thread hits `pthread_join` or any locked region, it will block. Not only does the UI freeze, the browser may also show “page unresponsive” warnings. This is fatal for web applications.

To enable Proxy mode, set `PROXY_TO_PTHREAD=1`. In addition, because the Merge Sort array can be large, set `ALLOW_MEMORY_GROWTH=1` to break through the default memory limit.

Proxy mode:
```shell
$ em++ merge_sort.cc \
    -s USE_PTHREADS=1 \
    -s PROXY_TO_PTHREAD=1 \
    -s PTHREAD_POOL_SIZE=4 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -O3 \
    -o merge_sort_proxy.html
```

Normal mode:
```shell
$ em++ merge_sort.cc \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE=4 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -O3 \
    -o merge_sort_no_proxy.html
```

### 2.3 JavaScript version of Merge Sort

Following the C++ logic, I reimplemented a JavaScript version.

The key difference is that `arr` was a global variable in the C++ version; now it must use SharedArrayBuffer.

Also, for the “join” behavior, I changed it so that workers notify completion via messages, and then I use SharedArrayBuffer + Atomics to implement a join barrier (it can only be incremented by 1; when it reaches 4, it means all workers have finished).

There is also a small trick here: in C++, integer division truncates automatically. But in JS, all numbers are Float64. The fastest way to truncate is using `|0`; for example, ` 5/2 | 0` yields `2`.

Save the following as an HTML file. You can open it directly in the browser and it will run:

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

## 3. Performance analysis

The benchmark compares (1) Pthreads compiled with Emscripten in Proxy mode with -O3 to JS + WASM, (2) Pthreads compiled with Emscripten in normal mode with -O3 to JS + WASM, (3) Pthreads compiled with g++ -O3, (4) Pthreads compiled with g++ -O2, and (5) the pure JavaScript version running in Chrome. The experiment environment is an AMD Ryzen 7 2700X 3.7 GHz 8-core CPU (using only 4 threads), emcc v1.39, g++ v7.5, and Chrome 84.

| Array Length  | Em Proxy | Em No Proxy | C++ O3 | C++ O2 | JS |
|---|---|---|---|---|---|---|  
|  100000 | 0.056s  | 0.062s | 0.027s | 0.032s | 0.214s |
|  10000 | 0.031s  | 0.009s | 0.002s | 0.003s | 0.052s |
|  1000 |  0.029s | 0.002s | 0.001s | 0.001s | 0.023s |

If we take the log of the times and invert them, we get the following figure. For the same length, a higher value means faster across different methods. Across different lengths for the same method, a higher value also means faster.

![result graph](https://user-images.githubusercontent.com/18013815/89744481-eacbac00-dadf-11ea-9513-825905cba9d4.png)

As expected, the Pthread implementation is the fastest. The difference between -O3 and -O2 is not that large here, which is very different from the π results in the previous post.

One interesting phenomenon is that regardless of length, the gap between 1e3 and 1e4 is small across methods, unlike the more dramatic change from 1e4 to 1e5. My guess is that when the length is below 1e4, overhead is dominated by other factors.

It seems Proxy mode has higher overhead for small arrays: at 1e3 and 1e4 it loses by a noticeable margin to Emscripten normal mode. However, when the size reaches 1e5, Proxy mode becomes faster than normal mode.

Finally, the pure JS version loses to other methods at all lengths. This is reasonable: C++ and WASM are generally faster than JS. However, at 1e3 and 1e4 the gap is smaller, while at 1e5 JS becomes much slower—about 8x slower than C++ and 6x slower than WASM.

## 4. Conclusion

This post benchmarks Merge Sort across (1) native Pthreads, (2) Pthreads compiled to JS + WASM via Emscripten, and (3) a pure JavaScript implementation. The results show that the Pthread version is fastest, Emscripten-generated WASM comes next; among the Emscripten modes, the normal mode is faster for small arrays while Proxy mode has an advantage for large arrays; and the pure JS version is the slowest.
