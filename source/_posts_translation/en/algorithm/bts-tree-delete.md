---
title: "Deleting a Node in a Binary Search Tree (LeetCode 450: Delete Node in a BST)"
date: 2025-02-05 01:00:00
tags: [algorithm, binary search tree, leetcode 450]
des: "This post explains how to delete a node in a binary search tree with diagrams, and provides a complete code example."
lang: en
translation_key: bts-tree-delete
---

## Problem Statement

Given a binary search tree (BST), find the node with the specified value and delete it. This corresponds to LeetCode 450: Delete Node in a BST.

For example, given the following tree, delete the node with value “3”:
![Image](https://github.com/user-attachments/assets/d6a7afce-c772-41b0-a651-76b3fa4649cf)

In the diagram above, the tree on the left is `[5, 3, 6, 2, 4, null, 7]`, and we choose “3” to delete.

After deletion, we get the tree on the right: `[5, 4, 6, 2, null, null, 7]`. You can see that “3” has been removed.

> Note: In this post, I describe trees in “level-order”, i.e., level by level from top to bottom, and from left to right. If a node is missing, I use `null`.

## Background

Before solving the problem, let’s quickly review the BST property: for every node, all values in its left subtree must be smaller than the node’s value, and all values in its right subtree must be larger. When we delete a node, we must preserve this property.

## Approach

When deleting a node, there are three cases:

- The target node has no children.
- The target node has exactly one child (either left or right).
- The target node has two children.

Let’s go through them one by one.

### Case 1: No Children

If the target node has no children, we can simply remove it.

<img width="80%" alt="Case 1" src="https://github.com/user-attachments/assets/2d79c090-b1f5-4877-9a2f-afb939ebc29a" />

Input: `[5, 2, 8, 1, 3]`, target = 8  
Output: `[5, 2, null, null, 1, 3]`

### Case 2: One Child

If the target node has only one child, we delete the target node and connect its child directly to the target node’s parent.

<img width="80%" alt="Case 2" src="https://github.com/user-attachments/assets/a0a803ef-74cc-45c3-a07a-731e832542d8" />

Input: `[5, 2, 8, 1, 3, null, 10]`, target = 8  
Output: `[5, 2, 10, 1, 3]`

### Case 3: Two Children

This is the most tricky case. After deleting a node, we must preserve the BST property, so the deleted node’s position must be filled by a valid replacement. The replacement can come from either the left subtree or the right subtree; both are valid approaches.

#### Replace from the Left Subtree

If we replace from the left subtree, we use the maximum value in the target node’s left subtree. By the BST property, this is the rightmost node in the left subtree (keep going right).

In the example below, the left subtree is `[2, 1, 3]`, so we use the rightmost value “3” as the replacement.

<img width="100%" alt="Case 3-1" src="https://github.com/user-attachments/assets/94e6ec25-0e06-45ee-b470-f6b77b3808f6" />

Input: `[5, 2, 8, 1, 3, 6, 10]`, target = 5  
Output: `[3, 2, 8, 1, null, 6, 10]`

#### Replace from the Right Subtree

If we replace from the right subtree, we use the minimum value in the target node’s right subtree. By the BST property, this is the leftmost node in the right subtree (keep going left).

In the example below, the right subtree is `[8, 6, 10]`, so we use the leftmost value “6” as the replacement.

<img width="100%" alt="Case 3-2" src="https://github.com/user-attachments/assets/ca72b984-a132-4062-8021-898efd772e9a" />

Input: `[5, 2, 8, 1, 3, 6, 10]`, target = 5  
Output: `[6, 2, 8, 1, 3, null, 10]`

## Code Example

```cpp
TreeNode *deleteNode(TreeNode *root, int key) {
    // Empty tree, or reached a null child.
    if (!root) {
        return nullptr;
    }

    // Current node value is greater than key: search in the left subtree.
    if (root->val > key) {
        root->left = deleteNode(root->left, key);
    }
    // Current node value is smaller than key: search in the right subtree.
    else if (root->val < key) {
        root->right = deleteNode(root->right, key);
    }
    // Current node value equals key.
    else {
        // Case 1: no children, delete directly.
        if (!root->left && !root->right) {
            return nullptr;
        }
        // Case 2: left subtree is empty, right subtree exists: connect right child.
        if (!root->left && root->right) {
            return root->right;
        }
        // Case 2: right subtree is empty, left subtree exists: connect left child.
        if (!root->right && root->left) {
            return root->left;
        }

        // Case 3: both subtrees exist. Find either:
        // - the rightmost node of the left subtree, or
        // - the leftmost node of the right subtree.
        // Below demonstrates finding the rightmost node in the left subtree.
        TreeNode *current = root->left;
        TreeNode *localParent = root;
        while (current->right) {
            localParent = current;
            current = current->right;
        }

        // Now `current` is the rightmost node of the left subtree, and `localParent` is its parent.
        // Copy `current`'s value to `root`, and then delete `current`.
        root->val = current->val;

        // Since `current` is the rightmost node, `current->right` must be null.
        // We only need to connect `current->left` back to `localParent`.
        if (localParent->left == current) { // `current` is the left child of `localParent`.
            localParent->left = current->left;
        } else { // `current` is the right child of `localParent`.
            localParent->right = current->left;
        }
    }

    return root;
}
```

