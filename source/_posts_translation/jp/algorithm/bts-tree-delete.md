---
title: "二分探索木のノード削除を図解する（LeetCode 450: Delete Node in a BST）"
date: 2025-02-05 01:00:00
tags: [algorithm, binary search tree, leetcode 450]
des: "本記事では、二分探索木（BST）のノード削除を図解で説明し、完全なコード例もあわせて紹介します。"
lang: jp
translation_key: bts-tree-delete
---

## 問題

二分探索木（binary search tree, BST）が与えられたとき、指定された値を持つノードを見つけて削除します。LeetCode 450: Delete Node in a BST に対応する問題です。

たとえば次の木が与えられたとき、値が「3」のノードを削除します：
![Image](https://github.com/user-attachments/assets/d6a7afce-c772-41b0-a651-76b3fa4649cf)

図の左側の木は `[5, 3, 6, 2, 4, null, 7]` で、削除対象として「3」を選びます。

削除後、右側の木は `[5, 4, 6, 2, null, null, 7]` となり、「3」が取り除かれていることが分かります。

> 注意：本記事では木の構造を「level-order（レベル順）」で表記します。つまり上のレベルから順に、左から右へ並べます。子が存在しない場合は `null` と表記します。

## 背景

解法に入る前に、BST の性質を復習します。各ノードについて、左部分木の値は必ず自分より小さく、右部分木の値は必ず自分より大きくなります。削除操作でも、この性質を保つ必要があります。

## 解法の方針

ノード削除は 3 つのケースに分けられます：

- 対象ノードに子がまったくない
- 対象ノードに子が 1 つだけある（左または右）
- 対象ノードに左右 2 つの子がある

以下、それぞれを説明します。

### ケース 1：子がない

対象ノードに子がない場合は、単純に取り除くだけでよいです。

<img width="80%" alt="ケース 1" src="https://github.com/user-attachments/assets/2d79c090-b1f5-4877-9a2f-afb939ebc29a" />

Input: `[5, 2, 8, 1, 3]`, target = 8  
Output: `[5, 2, null, null, 1, 3]`

### ケース 2：子が 1 つ

対象ノードに子が 1 つだけある場合は、対象ノードを削除し、その子ノードを親に直接つなぎ替えます。

<img width="80%" alt="ケース 2" src="https://github.com/user-attachments/assets/a0a803ef-74cc-45c3-a07a-731e832542d8" />

Input: `[5, 2, 8, 1, 3, null, 10]`, target = 8  
Output: `[5, 2, 10, 1, 3]`

### ケース 3：子が 2 つ

最後のケースは最も厄介です。BST の性質を保つため、削除した位置は妥当な値で埋め直す必要があります。埋め直しは左部分木から行っても右部分木から行ってもよく、どちらも正しい方法です。

#### 左部分木から埋め直す

左部分木から埋め直す場合、左部分木の最大値を使います。BST の性質上、これは左部分木の「最も右端のノード」（右へ右へと辿った先）です。

下の例では左部分木が `[2, 1, 3]` なので、最も右端の「3」を使って置き換えます。

<img width="100%" alt="ケース 3-1" src="https://github.com/user-attachments/assets/94e6ec25-0e06-45ee-b470-f6b77b3808f6" />

Input: `[5, 2, 8, 1, 3, 6, 10]`, target = 5  
Output: `[3, 2, 8, 1, null, 6, 10]`

#### 右部分木から埋め直す

右部分木から埋め直す場合、右部分木の最小値を使います。BST の性質上、これは右部分木の「最も左端のノード」（左へ左へと辿った先）です。

下の例では右部分木が `[8, 6, 10]` なので、最も左端の「6」を使って置き換えます。

<img width="100%" alt="ケース 3-2" src="https://github.com/user-attachments/assets/ca72b984-a132-4062-8021-898efd772e9a" />

Input: `[5, 2, 8, 1, 3, 6, 10]`, target = 5  
Output: `[6, 2, 8, 1, 3, null, 10]`

## コード例

```cpp
TreeNode *deleteNode(TreeNode *root, int key) {
    // 空の木、または null に到達した場合
    if (!root) {
        return nullptr;
    }

    // 現在ノードの値が key より大きい -> 左部分木へ
    if (root->val > key) {
        root->left = deleteNode(root->left, key);
    }
    // 現在ノードの値が key より小さい -> 右部分木へ
    else if (root->val < key) {
        root->right = deleteNode(root->right, key);
    }
    // 現在ノードの値が key と等しい
    else {
        // ケース 1：子がないので、そのまま削除
        if (!root->left && !root->right) {
            return nullptr;
        }
        // ケース 2：左が空で右がある -> 右の子をつなぐ
        if (!root->left && root->right) {
            return root->right;
        }
        // ケース 2：右が空で左がある -> 左の子をつなぐ
        if (!root->right && root->left) {
            return root->left;
        }

        // ケース 3：左右の部分木がある。
        // 左部分木の最右端ノード、または右部分木の最左端ノードを探す。
        // 以下では左部分木の最右端ノードを探す例を示す。
        TreeNode *current = root->left;
        TreeNode *localParent = root;
        while (current->right) {
            localParent = current;
            current = current->right;
        }

        // `current` が左部分木の最右端ノードで、`localParent` はその親
        // `current` の値を `root` にコピーし、`current` を削除する
        root->val = current->val;

        // `current` は最右端なので、`current->right` は必ず null
        // `current->left` を `localParent` に接続し直すだけでよい
        if (localParent->left == current) { // `current` は `localParent` の左の子
            localParent->left = current->left;
        } else { // `current` は `localParent` の右の子
            localParent->right = current->left;
        }
    }

    return root;
}
```

