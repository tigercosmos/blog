---
title: "程式設計原來不只有寫 CODE！銜接學校與職場的五堂軟體開發實習課"
date: 2024-12-18 00:00:00
tags: [劉安齊, 程式設計原來不只有寫 CODE, 資工基本素養, 軟體開發, 軟體職涯, 軟體工程師, 書籍]
des: "「程式設計原來不只有寫 CODE！銜接學校與職場的五堂軟體開發實習課」書籍官方介紹頁面，包含書本的詳細資訊，以及其它延伸資料。"
layout: books
---

<style>
    .book-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
    }
    .book-image {
        flex: 1 1 30%;
        max-width: 30%;
    }
    .book-details {
        flex: 1 1 70%;
        max-width: 70%;
        font-size: 1rem;
    }
    .book-details ul {
        margin: 10% 10% 0 10%;
        list-style-type: none;
        padding: 0;
    }
    @media (max-width: 768px) {
        .book-image, .book-details {
            flex: 1 1 100%;
            max-width: 100%;
        }
    }

    .book-buttons {
      display: flex;
      justify-content: center;
      gap: 10px;
      margin-top: 20px;
    }
    .book-buttons button {
      padding: 10px 20px;
      font-size: 1rem;
      cursor: pointer;
      border: none;
      background-color: #5c8db7;
      color: white;
      border-radius: 5px;
      transition: background-color 0.3s ease;
    }
    .book-buttons button:hover {
      background-color:rgb(51, 81, 108);
    }
    .book-preview-button {
      margin: 0;
    }
    .book-preview-button button {
      font-size: 0.8rem;
    }
    .book-preview-button button:hover {
      font-size: 0.8rem;
    }
    #previewDialog {
      width: 60%;
      height: 85%;
      max-height: min-content;
      display: none;
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background-color: white;
      padding: 0 20px;
      border: 1px solid #ccc;
      border-radius: 10px;
      box-shadow: 0 0 10px rgba(0,0,0,0.1);
      z-index: 100;
    }
    @media (max-width: 768px) {
        #previewDialog {
            width: 100%;
            padding: 0 5%;
            height: auto;
        }
    }

    .overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent black */
      display: none; /* Hidden by default */
      justify-content: center;
      align-items: center;
      z-index: 1000; /* Make sure it is on top of other elements */
    }
</style>

<h2 class="book_title_h2">
    <span class="book_title_left">程式設計原來</span>
    <span class="book_title_right">不只有寫 CODE！</span>
</h2>
<h3 class="book_title_h3">銜接學校與職場的五堂軟體開發實習課</h3>

<div class="book-container">
    <div class="book-image">
        <img src="https://raw.githubusercontent.com/tigercosmos/beyond-just-coding-book/refs/heads/master/book_picture.jpg" width="100%" alt="「程式設計原來不只有寫 CODE！銜接學校與職場的五堂軟體開發實習課」書籍封面">
    </div>
    <div class="book-details">
        <ul>
            <li><strong>書名：</strong> 程式設計原來不只有寫 CODE！銜接學校與職場的五堂軟體開發實習課</li>
            <li><strong>英文書名：</strong> Beyond Just Coding: Five Essential Lessons from Classroom to Career in Software Development</li>
            <li><strong>作者：</strong> 劉安齊 Liu, An-Chi</li>
            <li><strong>出版社：</strong> 博碩文化</li>
            <li><strong>出版日期：</strong> 2025-01-03</li>
            <li><strong>ISBN：</strong> 6264140341</li>
            <li><strong>ISBN-13：</strong> 9786264140348</li>
            <li><strong>定價：</strong> NT$ 700</li>
            <li><strong>GitHub repo：</strong> <a href="https://github.com/tigercosmos/beyond-just-coding-book">tigercosmos/beyond-just-coding-book</a></li>
            <li><strong>購買連結：</strong> <a href="https://www.tenlong.com.tw/products/9786264140348">天瓏書局</a> （最便宜） </li>
            <li><strong>海外購買連結：</strong> <a href="https://www.books.com.tw/products/0011008985">博客來</a></li>
            <li><strong>電子書：</strong> 預計 2025/3/31 上市</li>
            <li><strong>勘誤表：</strong> <a href="#勘誤表">連結</a></li>
        </ul>
    </div>
</div>

<div class="book-buttons">
  <button onclick="openPreview()">預覽書籍頁面</button>
  <button onclick="window.open('https://www.tenlong.com.tw/products/9786264140348', '_blank')">線上購買</button>
  <button onclick="location.href='#目錄'">線上搶先看</button>
</div>

<script>
  function openPreview() {
    document.getElementById('overlay').style.display = 'flex';
    document.getElementById('previewDialog').style.display = 'block';
  }

  function closePreview() {
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('previewDialog').style.display = 'none';
  }
</script>

<div class="overlay" id="overlay">
  <div id="previewDialog">
    <div id="previewPages" style="overflow-y:scroll;height:90%;">
      <img src="/img/beyond-just-coding-preview-1.jpg" alt="預覽頁面" id="previewImage" style="width: 100%; display:block; margin:auto;">
    </div>
    <div class="book-buttons book-preview-button">
      <button onclick="previousPage()">上一頁</button>
      <button onclick="nextPage()">下一頁</button>
    </div>
    <script>
      function previousPage() {
        let previewImage = document.getElementById('previewImage');
        let parts = previewImage.src.split("-");
        let tmp = parts[parts.length - 1];
        let pageNumber = parseInt(tmp.split(".")[0]);
        if (pageNumber > 1) {
          pageNumber -= 1;
        }
        previewImage.src = '/img/beyond-just-coding-preview-' + pageNumber+ '.jpg';
      }
      function nextPage() {
        let previewImage = document.getElementById('previewImage');
        const parts = previewImage.src.split("-");
        let tmp = parts[parts.length - 1];
        let pageNumber = parseInt(tmp.split(".")[0]);
        if (pageNumber < 15) {
          pageNumber += 1;
        }
        previewImage.src = '/img/beyond-just-coding-preview-' + pageNumber + '.jpg';
      }
    </script>
    <button onclick="closePreview()" style="position:absolute; top:10px; right:30px;">X</button>
  </div>
</div>


## 簡介

成為優秀的程式設計師，可以從基本的資工素養開始培養起！

一個資工系所畢業的學生該具備什麼技能？一個半路出家的工程師需要具備什麼能力？
除了基礎程式設計與專業科目知識之外，本書透過情境式的故事帶領讀者了解成為優秀
程式設計師所必備的技能與素養，原來程式設計不只有寫 CODE！

## 專業推薦


> 「對於想要進入軟體開發領域的學習者來說，無論是否是本科出身，這本書都將是一個彌足珍貴的指南。」 —— 游逸平（國立陽明交通大學 資訊工程學系 副教授）


> 「本書十分清楚地說明了寫程式不只是寫程式，還要掌握眾多的基本技能才能讓你成為一位稱職的軟體工程師。」 —— 陳永昱（新思科技 首席工程師）


## 內容簡介

就讀大學的小悅進入微中子科技公司實習，她將跟著導師齊哥學習各種程式開發的知識與技能，逐步探索軟體工程師的真實世界。從寫程式碼到解決實際問題，小悅將面臨程式設計、除錯、測試、團隊合作以及專案管理等各種挑戰，學會如何成為一位獨當一面的程式設計師。在這五堂課中，小悅將不斷精進技術，並且培養出職場必備的專業素養與實戰經驗，而讀者將跟著小悅的腳步一同學習。這本書不僅適合資訊、理工相關科系的學生，也為所有即將踏入職場或剛進入職場的程式開發者提供了一條充滿啟發的修煉之旅。

## 書本特色

- **身歷其境的軟體實習旅程**：本書帶領讀者進入程式設計的真實職場世界，從實習生小悅的視角出發，透過與導師齊哥的互動，模擬在軟體公司的實習過程中會遇到的各種學習與挑戰。
- **扎實的技能訓練**：從開發環境的搭建到高效率系統操作、程式碼閱讀與除錯，再到團隊協作和專案品質管理，書中涵蓋了成為一位全方位工程師所需的五大核心能力，幫助讀者紮實地掌握業界必備技能。
- **注重實戰與實用工具**：透過範例與實作教學，讀者將學會使用各種重要的開發工具，如 Git、Vim、SSH、GDB 等，並掌握軟體開發中除錯丶分析、版本控制丶自動化測試、CI/CD 等專業技能。
- **專業知識延伸與解惑**：針對電腦系統、程式效能分析、網路配置等專業知識，本書以簡潔易懂的方式帶領讀者逐步進入核心技術領域，提供在學校課堂中難以學到的實用知識。
- **專為實習生與初階工程師設計**：不僅是一本技術書，更是一本指導實習生、初階工程師如何在真實環境中成長的指南。除了專業技術，書中也融入了職場溝通、團隊合作、學習心法等實務技巧。

## 適合對象

- 正在尋找程式開發實習機會的各科類學生
- 即將步入職場成為軟體工程師的準畢業生
- 從其他領域轉行至軟體開發的工程師
- 資訊、理工等相關科系的學生
- 對程式設計充滿興趣，想提升自我的讀者

## 目錄

> 有連結的章節為公開內容，將不定期公開部分書籍內容，敬請期待！

- [序](/post/2024/12/beyond-just-coding-book/preface/)
- 推薦序（交大資工副教授游逸平）
- 推薦序（新思科技首席工程師陳永昱）
- 致謝
- [新員報到](/post/2024/12/beyond-just-coding-book/newcomer/)
- Chapter 1 程式開發環境
  - 1.1 作業系統
    - 1.1.1 Linux
    - 1.1.2 Windows
    - 1.1.3 macOS
  - 1.2 編輯器
    - 1.2.1 學習盲打
    - 1.2.2 Visual Studio Code
    - [1.2.3 Vim](/post/2024/12/beyond-just-coding-book/vim/)
- Chapter 2 系統操作
  - 2.1 Shell
    - 2.1.1 Shell 的功用
    - 2.1.2 Shell 的原理
    - 2.1.3 Shell 中使用 Pipe 和重新導向
    - 2.1.4 常用 Shell 命令和命令組合技
    - 2.1.5 環境變數
    - 2.1.6 實作簡易 Shell
  - 2.2 系統操作與資源管理
    - 2.2.1 系統資源
    - 2.2.2 檔案系統與磁碟管理
    - 2.2.3 網路配置與診斷
  - 2.3 SSH 連線
    - 2.3.1 SSH 連線
    - 2.3.2 SSH 設定檔
    - 2.3.3 自己建立一個 SSH 伺服器
    - 2.3.4 常見 SSH 使用方式
    - 2.3.5 SSH 相關命令
    - 2.3.6 小結
- Chapter 3 程式碼閱讀、除錯、追蹤與分析
  - 3.1 如何有效率去閱讀程式碼
    - 3.1.1 了解不同程式專案的性質
    - 3.1.2 認識程式專案
    - 3.1.3 從上至下閱讀
    - 3.1.4 從下而上閱讀
    - 3.1.5 處理多型
    - 3.1.6 文件化發現
    - 3.1.7 從測試程式碼理解程式
    - 3.1.8 查詢程式碼改動記錄
    - 3.1.9 編譯與執行原始碼
  - 3.2 除錯器
    - 3.2.1 使用 GDB 分析 C++ 程式
    - 3.2.2 使用 PDB 分析 Python 程式
  - 3.3 分析程式執行效能與行為
    - 3.3.1 使用 perf 分析程式效能
    - 3.3.2 [使用 tcpdump & Wireshark 分析網路行為](/post/2025/01/beyond-just-coding-book/tcpdump/)
- Chapter 4 多人協作開發
  - 4.1 程式碼版本控制今生今世
  - 4.2 Git 工具使用教學
    - 4.2.1 設定 Git 和 GitHub
    - 4.2.2 Git 專案初始化
    - 4.2.3 Git 提交程式碼修改
    - 4.2.4 使用 VSCode 的 Git 整合功能
    - 4.2.5 Git 分支
    - 4.2.6 Git 分支合併與變更基底
    - 4.2.7 解決合併或變更基底的衝突
  - 4.3 GitHub 平台操作
    - 4.3.1 GitHub Issue 介紹
    - 4.3.2 如何寫好的 Issue
    - 4.3.3 Pull Request 介紹
    - 4.3.4 如何發一個好的 Pull Request
    - 4.3.5 程式碼審查流程
  - 4.4 貢獻開源專案
- Chapter 5 程式專案的品質維護管理
  - 5.1 測試
    - 5.1.1 單元測試
    - 5.1.2 測試替身
    - 5.1.3 整合測試
    - 5.1.4 端到端測試
  - 5.2 持續整合和持續發布（CI/CD）
    - 5.2.1 回歸測試
    - 5.2.2 持續整合
    - 5.2.3 持續交付＆持續部屬
    - 5.2.4 GitHub Action 實作 CI/CD
  - 5.3 程式碼品味、準則、風格與格式化
    - 5.3.1 程式碼品味
    - 5.3.2 程式碼寫作準則與風格
    - 5.3.3 善用工具
    - 5.3.4 整合工具到 CI/CD
  - 5.4 如何寫文件
    - 5.4.1 文件分類
    - 5.4.2 Markdown 教學
    - 5.4.3 工程師必備的繪圖工具
- 結語


## 勘誤表

雖然本書經過多次校稿，但仍難免有錯誤，若讀者發現任何錯誤，歡迎來信指正，謝謝！

以下堪誤內容針對不同版本用代號做區分是否適用：
- 第一版第一刷：E1P1
- 第一版第二刷：E1P2
- 第一版第三刷（包含電子書）：E1P3

堪誤表如下：
- pxiv，「5.2」標題更正為「持續整合和持續部署（CI/CD）」（E1P1、E1P2）
- p29，〈Xcode〉小節中，第 6 行更正為「各種iOS 裝置上的應用程式」，第 11 行更正為「開發者發布應用程式至 App Store」。（E1P1、E1P2）
- p35，圖 1-10 說明文字中應更正為「Stack Overflow 2023 問卷調查不同 IDE 的使用比例」。（E1P1）
- p48，「Go to Defination」更正為「Go to Definition」。（E1P1、E1P2）
- p95，第 7 行，「stdout（fd=0）」應更正為「stdout（fd=1）」。（E1P1、E1P2）
- p114，`route` 執行範例程式區塊，第二行「255.255.240.0」更正為「255.255.255.0」。（E1P1、E1P2、E1P3）
- p114，「eno5」全部更正為「eth5」。（E1P1、E1P2、E1P3）
- p123，第二個程式碼區塊中的「cat .ssh/authorized_keys」更正為「cat ~/.ssh/authorized_keys」。（E1P1、E1P2）
- p127，（非錯誤調整）下方程式碼區塊「[sudo] password for mujin:」 更新為 「[sudo] password for acliu:」。（E1P1、E1P2、E1P3）
- p153，「例如我們從指看根目錄的檔案和目錄」，筆誤「指」為「只」。（E1P1、E1P2、E1P3）
- p156，「在做從上而下閱讀時兩者使用方法概念相同」，「從上而下」應該為「從下而上」。（E1P1、E1P2、E1P3）
- p158，「確認函式傳遞鏈」段落中，「或者反過來看誰呼叫切呼點」的「切呼點」更正為「切入點」。（E1P1、E1P2、E1P3）
- p188，倒數第二行「行別宣告」更正為「型別宣告」。（E1P1、E1P2、E1P3）

## 補充內容

不定期更新書籍的補充內容，作為第二版的靈感來源。

- git 章節小練習中可以加入 `git commit --amend` 和 `git stash`。
