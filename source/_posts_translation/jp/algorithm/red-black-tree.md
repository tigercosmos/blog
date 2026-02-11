---
title: "赤黒木（Red-Black Tree, RBT）の紹介"
date: 2019-11-27 00:00:00
tags: [algorithm, red black tree, data structures]
lang: jp
slug: red-black-tree
translation_key: red-black-tree
---

## 赤黒木の概要

赤黒木（Red-Black Tree）は特殊なデータ構造で、AVL 木と同様に自動で平衡を保つ機能を持ちます。

赤黒木には以下の性質があります：

- 各ノードは黒または赤のどちらか
- 根（root）は黒
- すべての葉（leaf）は `NIL` であり黒
- あるノードが赤なら、その子はすべて黒
- root から leaf までの各パスに含まれる黒ノード数は等しい

これらの規則に従うことで、ノードの挿入や削除を行っても木の平衡状態を保てます。赤黒木は、ノード数が `n` のとき木の高さが最大でも `2log(n+1)` を超えないことが保証されます。

現実世界では集合（set）や辞書（dictionary）の実装に使われることが多く、C++ の `std::set` や `std::map` は内部的に赤黒木で実装されています。

<!-- more --> 

以下は赤黒木の例です：
![red black tree](https://user-images.githubusercontent.com/18013815/69704764-131ca180-112f-11ea-9897-ea561e87fc35.png)

各パスに含まれる黒ノード数はすべて 3（`NIL` を含めるなら 4）で、赤ノードは必ず黒い子を持ち、root も黒です。
この図には `NIL` は描かれていません。`NIL` は実装上の意味としては、ノードの left と right に `nullptr` を入れることを指します。

規則が分かったところで、次はこれらの規則を守りつつ挿入・削除を行う方法です。
この部分は本当に複雑で、「Introduction to Algorithms」を読んでも私は理解できませんでした。

探し回った結果、以下のインドの方の動画がとても分かりやすかったので、動画で理解することを強くおすすめします：

## 挿入

こちら：

<iframe width="560" height="315" src="https://www.youtube.com/embed/UaLIHuR1t8Q" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

- 挿入するノードは必ず赤
- uncle が赤なら、基本的には色を変える（recolor）
- uncle が黒なら、左回転（left rotation）または右回転（right rotation）が必要

## 削除

こちら：

<iframe width="560" height="315" src="https://www.youtube.com/embed/CTvfzU_uNKE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

- 中間ノードを削除する場合は、最も近い後継（successor）を見つけて値を置き換え、その後に元の後継ノードを処理する
  - 後継ノードが赤で `NIL` を 2 つ持つ場合は、単に削除して `NIL` を補えばよく、基本的に規則に反しない
    - 各パスの黒ノード数が維持される
    - root は黒のまま
    - 赤い子を持つ赤ノードが発生しない
  - 後継が黒で、かつ赤い子を持つ場合は、後継を削除して赤い子で置き換え、その子を黒に変える。これで規則は維持される
- その他のケースでは規則と衝突し、「double black」や「red-black」のような二重属性のノードが現れる。その結果 6 つのケースに分かれ、それぞれに対応する処理を行う

