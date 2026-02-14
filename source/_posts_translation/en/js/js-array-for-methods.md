---
title: "Four Ways to Iterate Over Arrays in JavaScript: for, for-in, for-of, forEach()"
date: 2021-06-12 07:00:00
tags: [JavaScript, array, 效能分析, 陣列]
des: "This post introduces four ways to iterate over array elements: `for`, `for-in`, `forEach`, and `for-of`, and ends with a simple performance experiment."
lang: en
translation_key: js-array-for-methods
---

## 1. Introduction

To keep it short: if you want to iterate through all elements in a JavaScript array, you can directly use the following syntaxes.

`for` loop:

```js
for (let index=0; index < someArray.length; index++) {
  const elem = someArray[index];
  // ···
}
```

`for-in` loop:
```js
for (const key in someArray) {
  console.log(key);
}
```

`.forEach()` on an array:
```js
someArray.forEach((elem, index) => {
  console.log(elem, index);
});
```


`for-of` loop:
```js
for (const elem of someArray) {
  console.log(elem);
}
```

Before we start, if you are not familiar with arrays, you can read my earlier post: “[JS Array Basics](https://tigercosmos.xyz/post/2018/11/master_js/array/)”.

One very important idea is that in JavaScript, everything is an object. This characteristic significantly affects how array iteration behaves.

Next, I will go through the differences between these approaches in more detail.

![Cover Image](https://user-images.githubusercontent.com/18013815/121756482-c485fb00-cb4c-11eb-867e-244dda88e32b.png)

## 2. `for` loop syntax

This syntax has existed since ES1. It is the most straightforward form—pretty much every language looks like this—so it feels natural.

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

Sometimes, having to specify an index to access elements can feel a bit verbose.

However, its advantage is that you can fully customize the start, end, and step.

According to the experiment results at the end of this post, `for` is also the fastest among these iteration methods—plain and simple, but surprisingly strong.

## 3. `for-in` loop syntax

`for-in` has also existed since ES1. It iterates over the keys of an object.

In typical cases, those keys are the array indices. But if you assign custom properties to an array, the iteration will also visit those properties.

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

Therefore, I generally do not recommend using `for-in` to iterate over arrays. Because it iterates over keys, you can run into unexpected behavior.

Also, because these are keys, array indices are strings rather than numbers.

It enumerates all enumerable properties, including both own and inherited ones, which can be an advantage in some cases.

## 4. The built-in `forEach()` function on arrays

`Array.prototype.forEach()` was introduced in ES5 and is very common today. It wraps iteration into a callback function, giving it a functional-programming flavor.

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

There are two common forms—depending on whether you need index information.

The nice part of this style is that you directly get the elements, and the code often looks cleaner. However, modifying elements inside the callback does not affect the original array.

In addition, the downsides are that you cannot use `await` inside the callback, and you cannot break out early.

If you really need early termination, you can use `Array.prototype.some()` or `Array.prototype.every()` to help. I won’t go into that here. Personally, I do not recommend it—across many projects, I rarely see people use these for control flow; and in most cases, if you need to break early, a `for` loop is clearer.

## 5. `for-of` loop syntax

ES6 introduced `for-of`. You can think of it as combining the “directly get elements” benefit of `forEach()` with the flexibility of `for` (you can use `break` and `continue`).

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

With `for-of`, you can directly get elements, use `await`, and terminate the loop early.

When you only need elements—**and do not need index information**—this is arguably the cleanest approach. But note that operations on objects inside `for-of` do not affect the original array.

If you really want indices, you can do this:

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

It is admittedly a bit verbose.

What if you want both the index and the value?

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

This pattern is also uncommon; it mainly demonstrates that it works.

`for-of` can also be used to iterate over maps, which looks a bit like Python:

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

## 6. Performance comparison

Now that we have several methods, which one performs best?

I prepared two test datasets: one where the array contains only numbers, and another where the array contains objects.

The method is simple: iterate over all elements, access the values, and measure which approach is fastest.

This is a very simple benchmark and may not be comprehensive, but it should give us a rough idea.

### 6.1 An array of numbers

No more talk—here is the code:

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

The result is:

```
$ node .\test.js
for 41.0147999972105
for-in 541.3449000120163
forEach 91.50919999182224
for of 49.270000010728836
```

This is fairly reasonable. When you are dealing with pure numbers, `for` loops are easy for the engine to optimize. In contrast, other syntaxes may compile down to more complex instructions. `for-in` is especially slow likely because it treats indices as strings, which adds significant overhead, and it may also enumerate inherited properties.

### 6.2 An array of objects

I modified the code slightly:

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

Let’s look at the results:

```
$ node .\test2.js
for 18.817900002002716
for-in 438.9100999981165
forEach 56.67289999127388
for of 25.645999997854233
```

This is somewhat surprising. `for` is still the fastest, followed by `for-of`, and `for-in` is still the worst. I think the reason is similar to the previous case.

This suggests that regardless of what you are iterating over, `for` tends to receive better optimization. As for why, I’m not sure—you would likely need to inspect the IR generated by V8 to make a more confident judgment.

That said, my benchmark is quite rough. It is also possible that the performance depends on the specific scenario. Still, the takeaway is that `for` or `for-of` is usually a better choice for iterating over objects.

As for why `for-of` is consistently a bit slower than `for`, one hypothesis is that `for-of` performs a shallow copy of objects during iteration, introducing additional overhead.

In real-world code, if performance matters, it is best to prototype and measure, then pick the fastest approach for your case.

## 7. Conclusion

This post introduced multiple ways to iterate over array elements: `for`, `for-in`, `forEach`, and `for-of`, and finished with a simple performance experiment.

Many examples are heavily inspired by an article by Axel Rauschmayer. Axel is a well-known JavaScript expert, and in his article he advocates that `for-of` is the best approach. However, I disagree. As shown in the performance tests, a plain `for` loop performs best. Therefore, in general situations where speed matters, we have a good reason to keep using `for`—even if it does not look as “fancy”.

Of course, if the performance difference is acceptable, you can choose other iteration methods. Sometimes readability is more important than squeezing out the last bit of performance.

## 8. References

- [Looping over Arrays: for vs. for-in vs. .forEach() vs. for-of](https://2ality.com/2021/01/looping-over-arrays.html)

