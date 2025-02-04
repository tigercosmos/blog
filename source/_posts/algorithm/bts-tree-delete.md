---
title:  圖解刪除二元搜尋樹的節點（LeetCode 450 Delete Node in a BST）
date: 2025-02-05 01:00:00
tags: [algorithm, binary search tree, 演算法, 二元搜尋樹]
des: "本文以圖解說明如何刪除二元搜尋樹的節點，並提供完整程式碼範例。"
---

## 題幹

給定一個二元搜尋樹（binary search tree, BST）結構，找到指定數值的節點並刪除。此題可對照 LeetCode 450 Delete Node in a BST。

例如給定以下結構，刪除數值為「3」的節點：
![Image](https://github.com/user-attachments/assets/d6a7afce-c772-41b0-a651-76b3fa4649cf)

根據圖例，左邊結構為 `[5, 3, 6, 2, 4, null, 7]`，並且我們選定「3」來刪除。

刪除完之後，得到右邊的結構 `[5, 4, 6, 2, null, null, 7]`，可以看到「3」已經被移除。

> 注意，本文會以「level-order」來標註樹的結構，也就是從上層一層一層從左往右，如果沒有子節點則標註 null。

## 背景知識

在開始解題之前，我們先複習 BST 定義，即每個節點的子節點中，左子節點一定比自己小，而右子節點則一定比自己大。所以當我們進行刪除時，一樣得保持這樣的結構。

## 解題思路

刪除節點時，可以分三種情況：

- 目標節點完全沒有子節點
- 目標節點只有左邊或右邊有子節點
- 目標節點兩邊都有子節點

以下我們針對此三種情況個別來討論。

### 第一種情況

當目標節點完全沒有子節點時，其實只需要直接移除就可以了。

<img width="80%" alt="第一種情況" src="https://github.com/user-attachments/assets/2d79c090-b1f5-4877-9a2f-afb939ebc29a" />

Input： `[5, 2, 8, 1, 3]`, target = 8
Output： `[5, 2, null, null, 1, 3]`

### 第二種情況

當目標節點只有左邊或右邊有子節點時，我們將目標節點刪除，並把它的左邊或右邊子節點直接往上接。

<img width="80%" alt="第二種情況" src="https://github.com/user-attachments/assets/a0a803ef-74cc-45c3-a07a-731e832542d8" />

Input： `[5, 2, 8, 1, 3, null, 10]`, target = 8
Output： `[5, 2, 10, 1, 3]`

### 第三種情況

最後一種情況，目標節點兩邊都有子節點，也是最棘手的部分，因為根據 BST 定義，我們刪除節點後必須保持樹的平衡，所以被刪除的節點的位置，有可能會從左邊的子樹中遞補，也有可能從右邊子樹中遞補，兩種都是合法的作法。

#### 從左邊遞補

從左邊遞補時，取當前節點左子樹中最大的值來頂替，根據 BST 定義，就是不斷地往左子樹中找最右邊的節點。

在下圖範例中，左子樹為 `[2, 1, 3]`，取最右邊的「3」來頂替。

<img width="100%" alt="第三種情況-1" src="https://github.com/user-attachments/assets/94e6ec25-0e06-45ee-b470-f6b77b3808f6" />

Input： `[5, 2, 8, 1, 3, 6, 10]`, target = 5
Output： `[3, 2, 8, 1, null, 6, 10]`

#### 從右邊遞補

從右邊遞補時，取當前節點右子樹中最小的值來頂替，根據 BST 定義，就是不斷地往右子樹中找最左邊的節點。

在下圖範例中，右子樹為 `[8, 6, 10]`，取最左邊的「6」來頂替。

<img width="100%" alt="第三種情況-2" src="https://github.com/user-attachments/assets/ca72b984-a132-4062-8021-898efd772e9a" />

Input： `[5, 2, 8, 1, 3, 6, 10]`, target = 5
Output： `[6, 2, 8, 1, 3, null, 10]`

## 程式碼範例

```cpp
TreeNode *deleteNode(TreeNode *root, int key) {
    // 空的樹，或是末端
    if (!root) {
        return nullptr;
    }

    // 當前節點的值大於 key，往左子樹找
    if (root->val > key) {
        root->left = deleteNode(root->left, key);
    }
    // 當前節點的值小於 key，往右子樹找
    else if (root->val < key) {
        root->right = deleteNode(root->right, key);
    } else if (root->val < key) {
        root->right = deleteNode(root->right, key);
    }
    // 當前節點的值等於 key
    else {
        // 第一種情況，直接刪掉
        if (!root->left && !root->right) {
            return nullptr;
        }
        // 第二種情況，左子樹為空，右子樹不為空，將右子樹接上來
        if (!root->left && root->right) {
            return root->right;
        }
        // 第二種情況，右子樹為空，左子樹不為空，將左子樹接上來
        if (!root->right && root->left) {
            return root->left;
        }

        // 第三種情況，左右子樹都不為空，找左子樹的最右邊節點，或是右子樹的最左邊節點
        // 以下示範找左子樹的最右邊節點
        TreeNode *current = root->left;
        TreeNode *localParent = root;
        while (current->right) {
            localParent = current;
            current = current->right;
        }

        // 此時 current 為左子樹的最右邊節點，localParent 為 current 的 parent
        // 將 current 的值複製到 root 上，並刪除 current
        root->val = current->val;

        // 由於 current 是最右邊節點，所以 current->right 一定是空
        // 只需將 current->left 接到 localParent
        if (localParent->left == current) { // current 是 localParent 的左子樹
            localParent->left = current->left;
        } else { // current 是 localParent 的右子樹
            localParent->right = current->left;
        }
    }

    return root;
}
```
