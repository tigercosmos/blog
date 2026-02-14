---
title: "JavaScript で配列を走査する 4 つの方法：for、for-in、for-of、forEach()"
date: 2021-06-12 07:00:00
tags: [JavaScript, array, 效能分析, 陣列]
des: "本記事では、配列要素を走査する 4 つの方法（`for`、`for-in`、`forEach`、`for-of`）を紹介し、最後に簡単な性能実験を行います。"
lang: jp
translation_key: js-array-for-methods
---

## 1. 概要

手短に言うと、JavaScript の配列（Array）の全要素を走査したい場合は、次のような構文をそのまま使えます。

`for` ループ:

```js
for (let index=0; index < someArray.length; index++) {
  const elem = someArray[index];
  // ···
}
```

`for-in` ループ:
```js
for (const key in someArray) {
  console.log(key);
}
```

配列の `.forEach()`:
```js
someArray.forEach((elem, index) => {
  console.log(elem, index);
});
```


`for-of` ループ:
```js
for (const elem of someArray) {
  console.log(elem);
}
```

本題に入る前に、配列に慣れていない場合は以前書いた記事「[JS Array 入門](https://tigercosmos.xyz/post/2018/11/master_js/array/)」を先に読むと分かりやすいと思います。

非常に重要な前提として、JavaScript では「万物はオブジェクト」です。この特性は、配列走査の挙動を観察する際に大きく影響します。

以降では、それぞれの方法の違いを詳しく説明します。

![Cover Image](https://user-images.githubusercontent.com/18013815/121756482-c485fb00-cb4c-11eb-867e-244dda88e32b.png)

## 2. `for` ループの構文

ES1 から存在する構文で、最も直感的です。ほとんどの言語がこの形式なので、「当然こう書くよね」という感じがあります。

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

要素にアクセスするために毎回 index を指定するのは、少し冗長に感じることもあります。

一方で、開始・終了・ステップを自由に制御できるのが利点です。

記事の最後にある実験結果を見ると、`for` は走査方法の中でも最速でした。地味ですが強いです。

## 3. `for-in` ループの構文

`for-in` も ES1 から存在し、オブジェクト（Object）のキー（Keys）を走査します。

通常、配列に対してはキーは index になります。ただし、配列に独自のプロパティ（Property）を付与すると、そのプロパティも走査対象に含まれます。

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

そのため、配列の走査に `for-in` を使うのはあまりおすすめしません。キーを走査する以上、意図しない挙動になり得るからです。

また、キーなので配列の index は数値ではなく文字列になります。

列挙可能なすべてのプロパティ（自身のものと継承したものを含む）を走査するため、ケースによっては利点にもなります。

## 4. 配列が持つ `forEach()` 関数

`Array.prototype.forEach()` は ES5 で導入された構文で、現在では非常によく見かけます。走査処理をコールバックに包むため、関数型プログラミングの雰囲気があります。

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

index 情報が必要かどうかで、よく使われる書き方が 2 つあります。

この構文の良い点は、配列要素を直接受け取れるのでコードがすっきりしやすいことです。一方で、コールバック内で要素を変更しても、元の配列には反映されません。

また欠点として、コールバック内では `await` が使えず、途中で早期終了（break）できません。

どうしても途中で終了したい場合は `Array.prototype.some()` や `Array.prototype.every()` を利用できますが、ここでは詳しく触れません。個人的にはあまりおすすめしません。多くのプロジェクトを見てきましたが、この用途で使っているのはあまり見かけませんし、一般に早期終了が必要なら `for` の方が意図が明確です。

## 5. `for-of` ループの構文

ES6 で `for-of` が追加されました。`forEach()` のように要素を直接取得でき、さらに `for` と同様に `break` や `continue` が使えるため、より柔軟に書けます。

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

`for-of` では要素を直接取得でき、`await` の利用やループの早期終了も可能です。

要素だけが欲しくて **index 情報が不要な場合**、これは最も扱いやすい書き方です。ただし、`for-of` 内でのオブジェクト操作は元の配列には影響しません。

index がどうしても欲しければ、次のように書けます：

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

とはいえ、やや冗長です。

配列の index と value の両方が欲しい場合はどうでしょう？

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

この書き方も一般的ではなく、「できる」ということの確認に近いです。

また `for-of` は map の走査にも使えます。Python に少し似た雰囲気があります：

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

## 6. 性能比較

方法が複数ある以上、どれが最も速いのでしょうか？

今回は 2 種類の測定データを用意しました。1 つは配列が単純な数値だけで構成される場合、もう 1 つは配列にオブジェクトが入っている場合です。

方法は単純で、全要素を走査して値を読み出し、どの方法が最も速いかを測ります。

あくまで簡易的なベンチマークであり網羅的ではありませんが、ざっくりした感覚は掴めるはずです。

### 6.1 配列が数値だけの場合

余計な説明は省いて、まずコードです：

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

結果は次の通りです：

```
$ node .\test.js
for 41.0147999972105
for-in 541.3449000120163
forEach 91.50919999182224
for of 49.270000010728836
```

これはかなり納得できる結果です。数値だけを扱う場合、`for` ループはエンジン側で最適化しやすいです。一方、他の構文はコンパイル後の命令（Instruction）が複雑になりがちです。`for-in` が特に遅いのは、index を文字列として扱うためオーバーヘッドが大きいこと、さらに継承したプロパティも列挙する可能性があることが理由だと思います。

### 6.2 配列がオブジェクトの場合

コードを少し変更します：

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

結果を見てみましょう：

```
$ node .\test2.js
for 18.817900002002716
for-in 438.9100999981165
forEach 56.67289999127388
for of 25.645999997854233
```

少し意外ですが、`for` が依然として最速で、次が `for-of`、`for-in` が最も遅いという結果になりました。理由は前の例と同様だと思います。

つまり、何を走査する場合でも `for` は最適化が効きやすい傾向があるように見えます。なぜそうなるのかは私にも分かりませんが、V8 が生成する IR を分析しないと、より確かな判断は難しいでしょう。

ただし、この実験はかなり簡略化しています。状況によっては結果が変わる可能性もあります。それでも、ざっくり言えばオブジェクトを走査するなら `for` か `for-of` が無難だと思います。

また、`for-of` が `for` よりわずかに遅いのは、`for-of` がループ中にオブジェクトのシャローコピーを行い、その分のオーバーヘッドが増えているからだと推測できます。

実運用で性能が重要なら、実際に計測して最速の書き方を選ぶのが確実です。

## 7. 結論

本記事では、配列要素を走査する複数の方法（`for`、`for-in`、`forEach`、`for-of`）を紹介し、最後に簡単な性能実験を行いました。

例の多くは Axel Rauschmayer の記事を大いに参考にしています。Axel は JavaScript の専門家として知られており、彼の記事では `for-of` が最良の書き方だと述べています。しかし私は同意しません。性能テストの結果から分かるように、通常の `for` が最も良いパフォーマンスを示しました。したがって、一般的な状況で速度を重視するなら、多少格好良くなくても `for` を使い続ける理由があります。

もちろん、この程度の差が許容できるなら、他の方法を選んでも問題ありません。場合によっては、性能より読みやすさの方が重要なこともあります。

## 8. 参考資料

- [Looping over Arrays: for vs. for-in vs. .forEach() vs. for-of](https://2ality.com/2021/01/looping-over-arrays.html)

