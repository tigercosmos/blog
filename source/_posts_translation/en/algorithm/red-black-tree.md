---
title: "Red-Black Tree (RBT) Introduction"
date: 2019-11-27 00:00:00
tags: [algorithm, red black tree, data structures]
lang: en
slug: red-black-tree
translation_key: red-black-tree
---

## Red-Black Tree Overview

A red-black tree is a special data structure that can automatically balance itself, similar to an AVL tree.

A red-black tree has the following properties:

- Each node is either red or black
- The root must be black
- Every leaf (`NIL`) is black
- If a node is red, its children must be black
- Every path from the root to a leaf has the same number of black nodes

Following these rules ensures the tree stays balanced during insertions and deletions. A red-black tree guarantees that when the number of nodes is `n`, the height is at most `2log(n+1)`.

In practice, it is often used to implement sets and dictionaries. In C++, `std::set` and `std::map` are backed by red-black trees.

<!-- more -->

Below is an example red-black tree:
![red black tree](https://user-images.githubusercontent.com/18013815/69704764-131ca180-112f-11ea-9897-ea561e87fc35.png)

Each path contains three black nodes (four if you count `NIL`), every red node has black children, and the root is black.
The diagram does not draw `NIL`. In practice, `NIL` means when you implement a node, the left and right pointers must be `nullptr`.

Now that the rules are clear, the next step is how to follow them during insertion and deletion.
This part is complex. I even struggled to understand the explanations in "Introduction to Algorithms".

I searched for a long time and found the following videos to be very clear. I highly recommend watching them:

## Insertion

You can watch this one:

<iframe width="560" height="315" src="https://www.youtube.com/embed/UaLIHuR1t8Q" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

- Every inserted node starts as red
- If the uncle is red, recolor
- If the uncle is black, perform a left or right rotation

## Deletion

You can watch this one:

<iframe width="560" height="315" src="https://www.youtube.com/embed/CTvfzU_uNKE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

- When deleting an internal node, find the nearest successor, replace the current node with the successor's value, then handle the original successor node
  - If the successor node is red with two `NIL` children, remove it directly. All rules remain valid.
    - Every path keeps the same number of black nodes
    - The root stays black
    - There is no red node with red children
  - If the successor is black and has one red child, remove the successor, replace it with the red child, then recolor the child to black. All rules still hold.
- Other cases conflict with the rules, producing "double black" or "red-black" nodes with two attributes. There are six cases to resolve in order.
