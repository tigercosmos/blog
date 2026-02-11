---
title:  紅黑樹（Red Black Tree）介紹
date: 2019-11-27 00:00:00
tags: [algorithm, red black tree, 演算法, 紅黑樹]
lang: zh
slug: red-black-tree
translation_key: red-black-tree
---

## 紅黑樹簡介

紅黑樹是一種特別的資料結構，他具有跟 AVL Tree 一樣可以自動平衡的功能。

一個紅黑樹具有一下特性：

- 每個 node 只有黑與紅
- Root 必須是黑
- 每個葉子（leaf）是 `NIL` 且為黑
- 如果一個 node 是紅，那他的 child 都會是黑
- 每一條從 root 到 leaf 的路徑所包含的黑色 node 數量都一樣

遵循這樣的規則，可以確保在增加和刪除節點時，保持著樹的平衡狀態。紅黑數保證當 node 數量是 `n` 時，他的高度最多不會超過 `2log(n+1)`。

在現實生活中，常常用來實作 set 和 dictionary，C++ 中的 `std::set` 和 `std::map` 背後正是紅黑樹。

<!-- more --> 

下圖為一紅黑樹例子：
![red black tree](https://user-images.githubusercontent.com/18013815/69704764-131ca180-112f-11ea-9897-ea561e87fc35.png)

每條路徑所包含的黑色 node 數量皆為三（算上 `NIL` 的話，皆為四），每個紅色節點必定是黑色 child，root 也是黑色。
這張圖中沒有畫上 `NIL`，`NIL` 代表的意義其實就是，在實作 node 時，left 和 right 必須放一個 `nullptr`。

紅黑樹的規則很清楚了，接下來就是如何遵循規則進行插入和刪除。
這部分真的很複雜，我直接看「Introduce to Algorithm」也看不懂在寫啥。

我找了很久，發現以下推薦的印度人的影片解釋得很清楚，非常建議直接看影片去瞭解：

## 插入

可以觀看這個：

<iframe width="560" height="315" src="https://www.youtube.com/embed/UaLIHuR1t8Q" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

- 每次插入必定是紅色
- uncle 是紅色，那基本上就是要去改變顏色
- uncle 是黑，就需要進行左旋或右旋

## 刪除

可以觀看這個：

<iframe width="560" height="315" src="https://www.youtube.com/embed/CTvfzU_uNKE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

- 刪除中間節點要找最近的繼承者，先把自己取代成繼承者數值，再去處理原本繼承者
  - 如果繼承者 node 是紅帶有兩個 `NIL`，直接刪掉補上就好，基本上所有規則都不會違背。
    - 每條路徑黑色數量保持
    - root 還是黑色
    - 不可能有紅 child 的狀況
  - 如果是繼承者是黑節點，繼承者剛好有紅 child，那刪掉繼承者，補上紅 child，紅 child 轉成黑。所有規則都會繼續遵守。
- 其他狀況都會與規則衝突，就會出現「雙黑」或是「紅黑」帶有雙重屬性節點，並衍生身 6 種狀況，根據六種狀況的配套一一去解。
