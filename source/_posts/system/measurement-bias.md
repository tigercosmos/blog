---
title: 我們與誤差的距離——論系統軟體實驗測量誤差
date: 2020-08-30 18:30:00
tags: [system software, measurement, bias, spec cpu]
des: "論文提出即便我們實驗看似都沒問題，但卻可能因為初始設定默默得到不正確的結果數據，原因在於在 Unix 系統中，環境變數的設置和 C++ 編譯器連結物件的順序改變，都可能導致程式跑出來的結果差異非常大。"
lang: zh
translation_key: measurement-bias
---

## 1. 簡介

做系統軟體的電腦科學家的研究不外乎就是開發或改良一套系統軟體，為了證明設計出來的系統是有效的，往往必須要有搭配實驗證明效能真的符合預期，大部分就是跑跑 Benchmark 或是客製化的應用程式，但是這些實驗測試出來的數據真的正確嗎？

本文主要介紹 [Todd Mytkowicz](https://scholar.google.com/citations?user=Z4y_Z3sAAAAJ&hl=en) 等人提出的論文「[Producing Wrong Data Without Doing Anything Obviously Wrong!](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf)」。這篇文章引用次數高達三百以上，不僅吸引了我的目光，內容更是震驚了我。


論文提出即便我們實驗看似都沒問題，但卻可能因為**初始設定**默默得到不正確的結果數據，原因在於在 Unix 系統中，**環境變數的設置**和 **C++ 編譯器連結物件的順序改變**，都可能導致程式跑出來的結果差異非常大。

## 2. 程式的敏感程度

底下這張圖顯示左邊程式碼在不同環境變數大小執行的結果。

![paper figure 1](https://user-images.githubusercontent.com/18013815/91653233-3ff05180-ead1-11ea-9340-743c0b04cb84.png)

可以看到隨著環境變數差異，在 95% 信心水準下，程式執行的 Cycles 可以變化 33% 左右，甚至還有暴衝 300% 差異。環境變數會事先載入記憶體中，不同的環境變數大小會導致程式之後的變數在記憶體對齊 (Alignment) 上有不同結果，進而造成效能差異。

這個簡單範例告訴我們程式其實比我們想像的還要敏感，系統環境的起始設定完全可能左右程式運行結果。

## 3. 實驗

實驗取 SPEC CPU 2006 的部分 Benchmark 來用。

![benchmark](https://user-images.githubusercontent.com/18013815/91654974-e5122680-eadf-11ea-99b7-14aba6c8cbd8.png)

實驗用的裝置選了兩種 CPU Core2、Pentium 4 和一個模擬器 m5-O3CPU，藉此驗證跟這種偏差結果跟裝置無關。

![image](https://user-images.githubusercontent.com/18013815/91654988-007d3180-eae0-11ea-8c42-c0c693b53bf8.png)

### 3.1 Link 的順序造成的偏差

編譯器在編譯程式的時候，會將好幾個 `.o` 檔案連結在一起，依據不同的連結方式，會對記憶體配置造成不同的結果。

以下圖 Perlbench 結果為例，預設的連結順序、照字母排列順序還有各種隨機排列的順序(數字 3~33)，可以看到 GCC 以 O2 和 O3 編譯執行的 Cycles 數比值卻差異很大，如果沒有偏差的話，理論上比值應該要固定才對。

![figure 2-a](https://user-images.githubusercontent.com/18013815/91655565-1a207800-eae4-11ea-9839-e8f1a3103447.png)

所以的 Benchmarks 放在一起比較：

![figure 2-b](https://user-images.githubusercontent.com/18013815/91655656-ba769c80-eae4-11ea-86d2-7b5cc359c365.png)

可以看到依照連結不同，O2/O3 的比值範圍最大可以從 0.95 到 1.10，其餘也幾乎都有 0.05 左右的差異。依據論文的實驗，這跟裝置或編譯器無關，都會有類似的結果發生。這意味著，以 perlbench 這支來說，因為範圍差異達到 0.15，某個實驗如果使用這個 Benchmark，**宣稱其系統快 10% 也有可能其實是慢 5 %**。

論文有特別去探討這個現象的原因，在 m5 模擬器上用 O3CPU 模型時，以 bzip2 Benchmark 來說，熱點的 loop 有沒有剛好在一個 Cache Line 對效能影響就會很大了。因為模擬器可以看到原始碼，所以能歸納出原因，但是現實的硬體卻因為開發商揭露細節過少，所以作者說他們無法確定 Intel Core2 是否也是因為相同的原因，只能猜測大概是這樣，以 2020 的今天來說，不知道現在的 Intel CPU 是否已經提供足夠的資訊了呢？

### 3.2 環境變數大小造成的偏差

前面提過環境變數的大小會直接對 Stack 造成影響，以 Perlbench 為例可以看到 O2/O3 的範圍也在跳動，並且這種偏差毫無規律，幾乎不能歸納或預測。

![perlbench environment size](https://user-images.githubusercontent.com/18013815/91655921-01659180-eae7-11ea-8c70-de929019bc9b.png)

全部的 Benchmark 畫成 Violin Plot 結果如下。

![all benchmark environment size](https://user-images.githubusercontent.com/18013815/91655945-3376f380-eae7-11ea-823c-c9f639052c63.png)

可以看到大部分的 Benchmark 範圍差異很小，除了 perlbench 和 lbm 之外，其餘大約都在 1% ~ 4% 之間，雖然已經遠小於 Link 順序造成的偏差，但在實驗上來說，一樣是足以造成對實驗分析產生誤解的程度。**以 lbm 來說，你可能會以為你慢了 12%，但事實上你有 9% 的進步！**

如果我們將程式起始的 Stack 位置固定，實驗證明 O2/O3 的數值基本上不會隨著環境變數大小而有變化。此外像是 perlbench 會在初始將環境變數拷貝進 Heap，會直接對程式之後的物件對齊 (Alignment) 造成極大差異，所以如果固定 Heap 的起始位置，一樣也可以確保不會產生偏差。

## 4. 偏差的未知性

你可能會想說，既然初始設定會造成實驗結果有很大差異化，那我們是否能 Tune 出一個最合適的數值？

不過很遺憾的是，如同先前的數據，這種偏差完全沒有規律可言，我們頂多選一個最佳表現的參數，但是這台機器的最佳數值會等於另一台機器的最佳數值嗎？

![different device link order](https://user-images.githubusercontent.com/18013815/91656149-f3b10b80-eae8-11ea-905c-e83010ea2aa8.png)

上圖是 Core 2 和 Pentium 4 的比較圖，x-y 表示同一個 Link 的順序在兩台機器上的 Cycle 數，我們發現這些點並無規律，此外有圈起來的是在兩台機器上表現最佳的 Link 方式，但分別是不同的 Link 順序組合，因此我們得之 Link 偏差在不同機器之間也是毫無規律的。

![different device env size](https://user-images.githubusercontent.com/18013815/91656220-a08b8880-eae9-11ea-9e9d-96f7c6d496cf.png)

同樣的我們在上圖環境變數大小的差異上也可以得到一樣結論，不同機器上校率最高的環境變數大小數值都不一樣。

## 5. 如何避免偏差

既然我們已經知道實驗可能會造成偏差，我們就該盡力去避免，實際上偏差並不是一個新議題，在不同科學領域中常常都要去處理偏差。

### 5.1 更大的 Benchmark Suite

我們也許可以增加 Benchmark 的廣度和複雜度，如此一來偏差的影響可能就會剛好被抵銷。

可是很遺憾的是，作者將 Benchmark Suite 做測試，將所有 Benchmark 用 66 種不同的配置，分別實驗做平均，畫出不同 Benchmark 平均後的分布圖。如果複雜度足夠抵銷偏差的影響，我們應該會看到一個很窄的分布圖，但實際上我們看到一個分布很散的分布圖，意味著**我們不能利用 Benchmark 的複雜性來抵銷初始設置造成的偏差。**

![Distribution of speedup due to O3 as we change the experimental setup.](https://user-images.githubusercontent.com/18013815/91656380-e09f3b00-eaea-11ea-813f-eebcdeb1bea0.png)


### 5.2 隨機實驗初始值

一種方式我們可以借助統計學來增加我們的說服力，我們可以設定很多隨機的初始值，然後藉由 T-Test 這類的統計學方法來說明我們的信心水準。

![Using setup randomization to determine the speedup of O3 for perlbench](https://user-images.githubusercontent.com/18013815/91656507-c74abe80-eaeb-11ea-9a81-84d8ac297597.png)

如上圖所示，作者使用 22 種不同的 Link 順序加上 22 種不同的環境變數大小共 484 種初始設定，跑三次做平均，得到 95% 信心水準下平均的 O2/O3 比值是 1.007 ± 0.003。這下，我們可以確信的說 O3 的確優化比較好！

### 5.3 因果推論

再來，要能確信我們對數據的結論推倒無誤，我們可以採用因果推論(causal analysis)。

以下採用維基定義：

> 一般來說，因果還可以指一系列因素（因）和一個現象（果）之間的關係。對某個結果產生影響的任何事件都是該結果的一個因素。直接因素是直接影響結果的因素，也即無需任何介入因素（介入因素有時又稱中介因素）。從這個角度來講，因果之間的關係也可以稱為因果關聯（causal nexus）。 如果有 A，B 兩個事務，如果沒有A，B 就不會發生，則 A 是 B 的因，B 是 A 的果。

有嚴謹的因果推論下，就比較不容易因為實驗偏差而歸納出錯誤的結論。

## 6. 結論

作者分析包含 ASPLOS、PACT、PLDI, 和 CGO 等的 133 篇論文，發現裡面完全沒有人提到他的論文中點出的偏差性，這其實滿令人震驚的，我們如何確信他們的實驗結果並沒有受到初始值或是其他的偏差性而影響呢？

我自己在看論文的時候也發現，大部分的論文幾乎都沒有討論實驗誤差，統計學的方式去分析甚至畫上 Error Bar 都沒有，而我們應該更注重實驗的偏差性，這篇論文給我足夠大的震撼，偏差性可以左右高達 10% 的效能，當我們對系統效能錙銖必較時，我們所宣稱的 5% 效能進步，是否被 10% 偏差性而影響了呢？
