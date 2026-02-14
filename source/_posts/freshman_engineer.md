---
title: 菜鳥程式設計師的精進之路
date: 2019-04-09 01:23:39
tags: [software engineering,the clean coder,programming,growing up,軟體工程,軟體工程師]
lang: zh
translation_key: freshman_engineer
---

### The Clean Coder 心得、摘要與補充

<img class="dz t u gn ak" src="https://miro.medium.com/max/4122/0*Pvn4y7pOrW5BB_Yc.JPG" role="presentation"><br/>

前陣子把 《<em class="hr">The Clean Coder</em>》(中譯： 無瑕的程式碼 番外篇－專業程式設計師的生存之道) 看完，剛好可以回答一個網路上最常看到的問題之一：

<span style="font-size: 16px; color: #696969; font-style:italic">菜鳥工程師與資深工程師的差別是甚麼？</span>

<!-- more --> 

很多人會覺得是技術上的差別，例如菜鳥只會懂得技術比較少，只會使用框架，會用的工具比較少等等。相反的資深工程師懂得比較多技術，能告訴你技術的差別，能針對不同情境下選用最佳的架構，然後因為踩過的地雷很多，通常都可以直接避開問題等等。

我一開始也以為是這樣，當然這的確也是兩者的差別之一，但不是最關鍵的部分—待人處事才是。《<em class="hr">The Clean Coder</em>》 這本書就是完全在針對一個專業有素養的工程師，講述如何與人溝通以及如何去把寫程式這件事做好。事實上，書中講述的概念不只適用工程師，而是適用所有行業，只不過用軟體開發舉例罷了。

## 說「Yes」與說「No」

我們都會被交付任務，對象包含主管、PM、同事或客戶，學會正確的說「Yes」和「No」。一個專業人士不該過度承諾，面對做不到的事要說「No」，並給予對方合理的承諾，在自己能力範圍內說 「Yes」。

<span style="font-size: 16px; color: #696969; font-style:italic">能就是能，不能就是不能，不要說試試看</span>

書中給了幾個例子，說明 Yes 和 No 使用情境，要親自去閱讀才能感受那個情境。我覺得要能正確的說好與不好，需要具備兩個能力，一個是對事情高度掌握，能清楚分析複雜度和執行期限，並且還能說服對方接受你的論述，過程中勢必會需要大量溝通。

## 精實程式開發

如同一般人認知，專業的程式設計師應該具有更好、更有效率的做事方法。有點老生常談，但值得自我檢視一下。

### 心情

程式設計師是一個需要創造力的工作，如同藝術家，我們會需要「靈感」和「專注力」。有些人寫程式需要有固定的儀式，有些人喜歡在半夜打 Code，有些人喜歡聽音樂邊寫，有些人需要保持自己的節奏。總之，保持好心情，調整成自己最舒服的樣子，然後開始激發創意。

### 幫助

大家是同一個團隊，互相幫助是彼此職責。不吝嗇幫助別人，自己也許獲得更多，此外也不要自負不肯找人協助，有時候我們會需要他人幫忙，像是有個很有趣的除錯方法—<a href="https://zh.wikipedia.org/wiki/%E5%B0%8F%E9%BB%84%E9%B8%AD%E8%B0%83%E8%AF%95%E6%B3%95" class="dj by jn jo jp jq" target="_blank" rel="noopener nofollow">小黃鴨除錯法</a>：

<span style="font-size: 16px; color: #696969; font-style:italic">在程式的偵錯、除錯或測試過程中，操作人耐心地向小黃鴨解釋每一行程式的作用，以此來激發靈感與發現矛盾。</span>

<img class="dz t u gn ak" src="https://miro.medium.com/max/1528/0*tFvBXruadSGl0FjN.jpg" role="presentation"><br/>

你的同事就在扮演小鴨的腳色。此外藉由 Code Review 的過程，幫忙檢查同事的程式碼時，其實大家也是在互相切磋。

### 測試的重要性

書中一直強調測試的重要性，當然我們都理解，是吧？<br>既然書中已經給了一些例子，我想這邊我分享我自己的親身經歷。

以前我還在 Cloudmosa 實習，有一天我發現，為什麼 Puffin 在網頁中有連結的地方上面點右鍵選單，加入書籤的時候，會抓不到連結的標題。

這個標題其實就是 <code class="gs kb kc kd ke b">&lt;a&gt;</code> 標籤匡著的文字內容，或是他有聲明 <code class="gs kb kc kd ke b">title</code> 是什麼。總之，加入書籤的對話框跳出來，標題欄位會是空的，所以我就要解這個 bug。研究了很久，從堆疊一層層追蹤，發現是 Mango （Puffin 的瀏覽器引擎實體，參考 <a class="dj by jn jo jp jq" target="_blank" rel="noopener" href="/coding-neutrino-blog/440c91cece8f"><em class="hr">How the puffin browser works</em></a>）傳過來的資料少了東西。所以解法也不難，在 Mango 與 Puffin 之間的 Protocol 新增傳送的資料就好。

然後我就很開心啦～Puffin 可以順利收到資料，加入書籤的對話框就有標題了。然後隔兩天，我收到 CTO Sam 的信，「Tiger 你的 patch 會讓我們的服務掛掉」理由是，Puffin 和 Mango 都有非常多版本，所以在改 protocol 的時候，要考慮向下相容性。不過 Sam 很好心，幫我直接修好，不過嚇得我一身冷汗。

這件事告訴我兩件事，一是寫測試真的很重要，如果有完善的測試，這種錯誤會即時發現，不過因為我開發的是開發分支，所以不會每個 commit 都跑完善測試。二是考慮<strong class="hf kf">向下相容性</strong>應該是 common sense，但因為我太菜了，所以忽略了這麼重要的事情。

書中很強調測試驅動開發（Test Driven Development, TDD），我也認為 TDD 是比較好的開發方式，既可以將規格、邏輯制定好，又不會陷入之後很難寫測試的困境。書中很張狂的聲明：

<span style="font-size: 16px; color: #696969; font-style:italic">TDD 的各項優點表明了一件事情，不使用 TDD 就說明你可能還不夠專業 — The Clean Coder</span>

不過你依然要視情況做出最佳選擇，沒有某個法則是鐵則。

通常測試就會被包在持續開發與持續整合（CI/CD）的流程，更廣泛來說他是維運（DevOps）的一環，其中包含單元測試、整合測試、交付測試、品質測試等等，這部分就留給大家自行探索。

### 練習

當我們越來越資深，可能就會越來越少去寫很簡單的東西，像是如何寫一個 Quick Sort，每個想要面試的人都會去刷 LeetCode 然後刷完有工作之後就忘了—如同我們應付大學考試一樣。

這種牛刀小試其實對維持手感和靈感很有幫助，老實說在寫這邊文章的同時，要我在 15 分鐘內寫出一個不會有 bug 的 Quick Sort 大概也是困難重重，概念還記得但就是生疏了。

<span style="font-size: 16px; color: #696969; font-style:italic">As the saying goes: “Practice makes you better"</span>

此外多寫一些小專案也挺不錯，資工系的學生即便修過 Compiler、OS、Compuer Architecture 都不見得自己寫過一個完整的系統，只懂得原理而沒有親手做過，其實可以等於不會，因為現實是殘酷的，實務總是比較難。

<span style="font-size: 16px; color: #696969; font-style:italic">很多人覺得自己寫程式速度已經夠快，那是因為他們沒有想要更快</span>

不知道大家有沒有看過 <a href="https://medium.com/u/5879ccb41e31?source=post_page-----5b55f279630c----------------------" class="kg az by" target="_blank" rel="noopener">Jim Huang</a> (jserv) 在演講的時候直接現場打 Code，我認為 Live Coding 真的很困難，畢竟寫程式就是在創作，需要靈感、需要醞釀。但別人做得到寫得快又寫得好，我們加強練習也做得到。

## 時間、評估與規劃

每個人時間都有限，如果工作效率不好，那就要看看自己時間分配和做事方法要不要改進。番茄工作法可能會是不錯的做法，也就是集中精神專心衝刺一段時間工作，然後休息一下，如此往復。人的精神和身體是有極限的，所以過度加班和睡眠不足都不是好策略。

開會本身是成本很高的事情，公司花了很多錢聘請每個人，如果開會本身的價值小於耗費大家的開發時間，那就完全不值得。有一些方法，例如 Agile 或 Scrum，可能都是可以參考的做法。此外自己也該只選擇自己有義務或有助益的會議參加即可。

預估就是承諾，當你跟別人說「Yes」的時候，你就有責任在期限內交付任務。當然你可以把時間抓寬一點，也許是兩個標準差，這樣可以確保你 99% 機率沒問題。

### 設計規劃

區分 Junior、Senior、Lead 的方式也包含對開發規劃與評估的能力

<ul>
<li id="2e3e" class="hd he em at hf b hg hh hi hj hk hl hm hn ho hp hq kh ki kj">Junior 通常在開發時，不知道從何規劃，做事就是想到甚麼做甚麼</li><li id="25dd" class="hd he em at hf b hg kk hi kl hk km hm kn ho ko hq kh ki kj">Senior 通常會有一個好的規劃，知道怎樣將任務拆解成步驟，甚至產生出設計文件，能區分事情重要性</li><li id="bcb8" class="hd he em at hf b hg kk hi kl hk km hm kn ho ko hq kh ki kj">Lead 包含 Senior 特質，讓專案有條有理的運作，並可以作換位思考，將整個團隊放在對整個公司都有利的立場</li>
</ul>

所以工程師的差異不僅僅只是技術懂得多不多而已。

## 團隊協作

除非你是獨立工作者，不然一定是一個團隊。團隊就會包含協作開發、權責分配、專案管理。如果沒機會接觸大型的協作開發，我建議可以去嘗試貢獻開源專案，例如 Mozilla 的 <a href="https://github.com/servo/servo" class="dj by jn jo jp jq" target="_blank" rel="noopener nofollow">Servo </a>瀏覽器引擎就是很好的練手地方，裡面共有快一千名貢獻者，你將會看到完整的大型專案開發流程。一個團隊怎樣有凝聚力，沒什麼好說的就是溝通、溝通再溝通，而溝通是一門藝術，任何專業人士都該具備。

## 輔導

最後，任何資深的工程師都應該照顧和輔導比較資淺的工程師，程式設計師就像工藝中的學徒，透過不斷學習來成長。好的導師（mentor）可以協助你快速成長、少走很多冤枉路和在你心中立下榜樣。大家都受過別人幫助，我們也應該熱心地去幫助後輩。長江後浪撲前浪，青山還比青山綠。

## 總結

本文整理了《<em class="hr">The Clean Coder</em>》一書中的重點，並且我根據自身經驗加了不少詮釋。成長過程中我在不少公司待過，看過不少工程師，見過各種文化，即便我在業界的時間不多，但看這本書也是頗有感觸，希望我的一些見解可以幫助到其他人。

那麼？我是 Junior 工程師嗎？縱橫本文的觀點，是的，我依舊還只是個菜鳥，精進自己的路還很遙遠。
