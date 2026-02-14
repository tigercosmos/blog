---
title: JavaScript 遍歷 Array 的四種方法：for、for-in、for-of、forEach()
date: 2021-06-12 07:00:00
tags: [JavaScript, array, 效能分析, 陣列]
des: "本文介紹遍歷 Array 元素的四種方法，分別為 `for`、`for-in`、`forEach` 和 `for-of`，並在最後做簡單的效能實驗。"
lang: zh
translation_key: js-array-for-methods
---

## 1. 簡介

長話短說，要遍歷 JS Array 裡面的所有元素，可以直接使用以下語法。

`for` 迴圈:

```js
for (let index=0; index < someArray.length; index++) {
  const elem = someArray[index];
  // ···
}
```

`for-in` 迴圈:
```js
for (const key in someArray) {
  console.log(key);
}
```

Array 中的 `.forEach()`:
```js
someArray.forEach((elem, index) => {
  console.log(elem, index);
});
```


`for-of` 迴圈:
```js
for (const elem of someArray) {
  console.log(elem);
}
```

在開始之前，如果對 Array 不熟悉的話，可以看我之前寫的文章：「[JS Array 入門](https://tigercosmos.xyz/post/2018/11/master_js/array/)」。

有個非常重要的觀念是 JS 中萬物皆物件，這個特性會大大影響我們觀察 Array 遍歷的行為。

接下來會詳細介紹不同方法的差異。

![Cover Image](https://user-images.githubusercontent.com/18013815/121756482-c485fb00-cb4c-11eb-867e-244dda88e32b.png)

## 2. `for` 迴圈語法

ES1 就有的語法，這是最直觀的語法，所有語言都是這樣寫，似乎也是理所當然。

```js
const arr = ['a', 'b', 'c'];
arr.prop = 'property value';

for (let index=0; index < arr.length; index++) {
  const elem = arr[index];
  console.log(index, elem);
}

// Output:
// 0, 'a'
// 1, 'b'
// 2, 'c'
```

有時候要存取 Array 裡的物件要指定 Index 會有點冗長。

不過它的好處是起始、結束、間隔都可以自訂。

就文章最後的實驗結果，`for` 的表現也是所有遍歷方法中最快的，可以說是樸實無華卻一鳴驚人。

## 3. `for-in` 迴圈語法

`for-in` 也是 ES1 就有，他會遍歷物件 (Object) 中的 Keys。

注意在一般情況下，Keys 就是 Arrays 的 Index，但如果有特別賦予 Array 屬性 (Property) 的話，就會讓遍歷行為多拜訪這些屬性。

```js
const arr = ['a', 'b', 'c'];
arr.prop = 'property value';

for (const key in arr) {
  console.log(key);
}

// Output:
// '0'
// '1'
// '2'
// 'prop'
```

所以不太建議用 `for-in` 來遍歷 Array，因為它看 Keys，可能會有意外的行為。

此外因為是 Keys，所以 Array 的 Index 會是字串而非數字。

它會遍歷所有可以枚舉的屬性，包含自身跟繼承的，這可能會是不錯的好處。

## 4. Array 自帶的 `forEach()` 函數

`Array.prototype.forEach()` 是 ES5 有的語法，時至今日其實滿常見的。它讓遍歷行為被包裝成一個 Callback 函數，語法有 Functional Programming 的味道。

```js
const arr = ['a', 'b', 'c'];
arr.prop = 'property value';

arr.forEach((elem, index) => {
  console.log(elem, index);
});

// Output:
// 'a', 0
// 'b', 1
// 'c', 2

arr.forEach(elem => {
  console.log(elem);
});

// Output:
// 'a'
// 'b'
// 'c'
```

有兩種寫法，端看你要不要 Index 資訊。

這種語法的好處是我們可以直接得到 Array 裡面的元素，看起來會比較乾淨，不過 Callback 的元素修改是不會影響到原本的 Array。

此外這方法缺點是 Callback 裡不能有 `await`，且不能提前中斷。

如果你真的很想要提前中斷的話，你可以使用 `Array.prototype.some()`、`Array.prototype.every()` 函數幫助你操作，這邊就不多作介紹。我自己是不建議使用，畢竟我看這麼多專案還沒怎麼看過人用，而且一般情況要中斷的話用 `for` 的語法會比較清楚。

## 5. `for-of` 迴圈語法

ES6 有了 `for-of` 語法，這大概就是結合 `forEach` 可以直接得到元素的好處，外加 `for` 可以用 `break`、`continue` 做到更彈性操作。

```js
const arr = ['a', 'b', 'c'];
arr.prop = 'property value';

for (const elem of arr) {
  console.log(elem);
}
// Output:
// 'a'
// 'b'
// 'c'
```

現在你可以在 `for-of` 中直接得到元素了，並且裡面可以使用 `await`，你也可以提早中斷迴圈。

當你只需要直接取得元素，**不需要 Index 資訊時**，這應該是最棒的寫法了。但要注意的是，`for-of` 中的物件操作是不會影響到原本 Array。

當然如果你真的很想要 Index 的話可以這樣寫：

```js
const arr = ['chocolate', 'vanilla', 'strawberry'];

for (const index of arr.keys()) {
  console.log(index);
}
// Output:
// 0
// 1
// 2
```

雖然這樣寫來取得 Index 有點冗就是了。

如果同時想要 Array 的 Index & Value 怎辦？

```js
const arr = ['chocolate', 'vanilla', 'strawberry'];

for (const [index, value] of arr.entries()) {
  console.log(index, value);
}
// Output:
// 0, 'chocolate'
// 1, 'vanilla'
// 2, 'strawberry'
```

這種寫法也滿少見的，就是證明行得通而已。

另外 `for-of` 也可以直接用來遍歷物件，這種寫法跟 Python 有點神似：

```js
const myMap = new Map()
  .set(false, 'no')
  .set(true, 'yes');

for (const [key, value] of myMap) {
  console.log(key, value);
}

// Output:
// false, 'no'
// true, 'yes'
```

## 6. 效能大比拚

既然我們有這麼多種方法，那哪一種寫法效能最好呢？

我準備了兩種測資，一種是 Array 裡面都是單純的數字而已，另一種是 Array 裡面放的都是物件。

測試方法很簡單，就是遍歷所有元素，存取裡的數值，然後看哪個方法速度最快。

這只是非常簡單的測試，可能不夠周全，但大概可以給我們一點概念。

### 6.1 Array 裡面都是數字

不囉嗦，直接上 Code：

```js
// test1.js

const { performance } = require('perf_hooks');

// 建立資料
const arr = []
for (let i = 0; i < 100000; i++) {
    arr.push(i);
}


let sum;
let time_marker;

// ===
time_marker = performance.now();
sum = 0;
for (let j = 0; j < 50; j++) // 放大 50 倍
    for (let i = 0; i < 100000; i++) {
        sum += arr[i];
    }
console.log("for", performance.now() - time_marker);

// ===
time_marker = performance.now();
sum = 0;
for (let j = 0; j < 50; j++)
    for (const i in arr) {
        sum += arr[i];
    }
console.log("for-in", performance.now() - time_marker);

// ===
time_marker = performance.now();
sum = 0;
for (let j = 0; j < 50; j++)
    arr.forEach(v => {
        sum += v;
    });
console.log("forEach", performance.now() - time_marker);

// ===
time_marker = performance.now();
sum = 0;
for (let j = 0; j < 50; j++)
    for (const v of arr) {
        sum += v;
    }
console.log("for of", performance.now() - time_marker);
```

跑出來結果是

```
$ node .\test.js
for 41.0147999972105
for-in 541.3449000120163
forEach 91.50919999182224
for of 49.270000010728836
```

其實滿合理的，`for` 迴圈在本身都是處理數字情況下，編譯器是非常好做優化的，相對的其他語法編譯出來的指令 (Instruction) 就會比較複雜。`for-in` 特別慢應該是因為它是將 Index 當作 String 在處理，這樣的話中間的消耗其實十分大，此外他可能需要去枚舉所有繼承的屬性。

### 6.2 Array 裡面都是物件

我將 Code 稍作修改：

```js
// test2.js

const { performance } = require('perf_hooks');

// 建立資料
const arr = []
for (let i = 0; i < 100000; i++) {
    arr.push({
        a: 1,
        b: 2
    });
}


let sum;
let time_marker;

// ===
time_marker = performance.now();
sum = 0;
for (let j = 0; j < 50; j++) // 放大 50 倍
    for (let i = 0; i < 100000; i++) {
        sum += arr[i].a;
    }
console.log("for", performance.now() - time_marker);

// ===
time_marker = performance.now();
sum = 0;
for (let j = 0; j < 50; j++)
    for (const i in arr) {
        sum += arr[i].a;
    }
console.log("for-in", performance.now() - time_marker);

// ===
time_marker = performance.now();
sum = 0;
for (let j = 0; j < 50; j++)
    arr.forEach(v => {
        sum += v.a;
    });
console.log("forEach", performance.now() - time_marker);

// ===
time_marker = performance.now();
sum = 0;
for (let j = 0; j < 50; j++)
    for (const v of arr) {
        sum += v.a;
    }
console.log("for of", performance.now() - time_marker);
```

讓我們來看看結果：

```
$ node .\test2.js
for 18.817900002002716
for-in 438.9100999981165
forEach 56.67289999127388
for of 25.645999997854233
```

這個結果就十分意外了，`for` 依舊效能最好，其次是 `for-of`，而 `for-in` 依舊是最差的，我想理由大概跟前面例子差不多。

這意味著，不管遍歷甚麼，似乎 `for` 都比較容易得到比較好的最佳化，至於為甚麼我不清楚，可能需要去分析 V8 編譯出來的 IR 才能比較好判斷。

不過由於我的實驗有點簡陋，也有可能是在這個情況下他表現比較好，但我們大概能有個底，`for` 或 `for-of` 應該是比較適合拿來做物件的遍歷。

至於 `for-of` 都稍慢 `for`，可以猜測是 `for-of` 在迴圈中會對物件產生 Shallow Copy，因此多了額外的開銷。

真實情況下，如果你很在乎效能，建議都實做看看，最後選效能最好的那個寫法就好。

## 7. 結論

本文介紹遍歷 Array 元素的多種方法，分別為 `for`、`for-in`、`forEach` 和 `for-of`，並在最後做簡單的效能實驗。

範例大量參考 Axel Rauschmayer 文章，Axel 本身算是 JS 專家，他在文章中提倡 `for-of` 是最好的寫法，但我並不同意，如同最後的效能測試，我們可以發現一般的 `for` 表現都是最好的。所以在一般情況下為了速度我們有理由繼續用 `for`，即使會寫出沒那麼帥氣的程式語法。但如果這點效能差異是允許的，你也可以考慮使用其他遍歷 Array 的方法，有時候好閱讀比效能好還重要。

## 8. 參考資料

- [Looping over Arrays: for vs. for-in vs. .forEach() vs. for-of](https://2ality.com/2021/01/looping-over-arrays.html)
