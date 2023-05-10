---
title: "深入淺出介紹我的碩士研究—Offworker: An Offloading Framework for Parallel Web Applications" 
date: 2022-08-29 06:20:08
tags: [research, offloading, JavaScript, system]
des: "本文是我的研究「Offworker: An Offloading Framework for Parallel Web Applications」的中文介紹"
---

## 簡介

＜[Offworker: An Offloading Framework for Parallel Web Applications](https://etd.lib.nctu.edu.tw/cgi-bin/gs32/tugsweb.cgi?o=dnctucdr&s=id=%22GT0708560050%22.&switchlang=en)＞是我的碩士論文，以此研究投稿到 The 23rd International Conference on Web Information Systems and Engineering ([WISE '22](https://link.springer.com/conference/wise)) 通過 Full Paper 的審查被接受。

這個研究先後大概花了快兩年完成，至於碩士期間怎麼研究的過程，可以參考「[交大觀察與心得](/tags/%E4%BA%A4%E5%A4%A7%E8%A7%80%E5%AF%9F%E8%88%87%E5%BF%83%E5%BE%97/)」系列文章。

## 摘要



這邊提供論文中的中文摘要，有興趣可以點開看。

<details>
<h3> 論文中文摘要 </h3>
由於行動裝置越來越普及以及無線通訊技術越來越進步，越來越多的應用程式正在從傳統的桌面應用軟體移轉成網頁應用程式。Web Worker API 因而被制訂，它讓應用程式可以將計算繁重的工作從應用程式的主執行序卸載到其他的工作執行序（或稱做 Web Worker），使得主執行序可以專注處理使用者介面和互動，進而改善使用者操作體驗。先前的研究證實透過將 Web Worker 分配到遠端的伺服器來卸載計算繁重的工作可以改善網頁應用程式的效能，但是那些研究的實作可能會有潛在的安全性漏洞，同時他們也並未考慮一些應用程式可能會用多個 Web Worker 來實現並行化或平行化的程式。在這篇論文中，我們實現了一個卸載框架（稱做 Offworker），可以透通地卸載並行的 Web Worker 到邊緣節點或雲端伺服器，同時也提供 Web Worker 更安全的執行環境。我們設計了一套基準測試集（稱做 Rodinia-JS）來評估我們的框架，這是由 Rodinia 平行化基準測試集改寫成 JavaScript 的版本。實驗證實 Offworker 在將 Web Worker 從行動裝置卸載到伺服器之後，可以有效的改進平行程式效能（最多達 4.8 倍加速），而 Offworker 在計算密集型的應用程式中，僅比原生執行多了 12.1% 幾何平均的間接費用。我們相信 Offworker 提供平行化網頁應用程式一個有前景且安全的計算卸載解決方案。
</details>