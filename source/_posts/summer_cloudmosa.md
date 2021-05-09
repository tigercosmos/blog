---
title: 這個暑假我在 CloudMosa 所見所聞
date: 2018-09-08 00:00:00
tags: [cloudmosa,intern,puffin browser,]
---

### What I saw and heard at CloudMosa in this summer.

大約在一年前，因緣際會我開始接觸瀏覽器開發，一開始是貢獻 Mozilla 的 Servo Brower Engine 專案，後來覺得還是不夠了解瀏覽器，就想說就由寫文章來強迫自己學習。ITHome 的鐵人賽剛好給了我一個行動的契機，於是就開始寫了有關於瀏覽器的系列文章。

<!-- more --> 

> 點我： [《來做個網路瀏覽器吧！》](/tags/%E4%BE%86%E5%81%9A%E5%80%8B%E7%B6%B2%E8%B7%AF%E7%80%8F%E8%A6%BD%E5%99%A8%E5%90%A7%EF%BC%81/)

寫著寫著，因為我都會到處去分享我的文章，恰好文章就被 CloudMosa 的人看到。CloudMosa 是一家臺灣的瀏覽器開發公司，產品為 Puffin Browser，關於甚麼是 Puffin 我在另外一篇文章中有詳細介紹。

> 點我： [How the Puffin Browser Works](/post/2018/09/puffin/)

國內有人對瀏覽器這麼有興趣，他們覺得非常少見，所以覺得我應該會滿有趣的，於是我分別收到 CEO Shioupyn 和 CTO Sam 的私訊。

CloudMosa 覺得光是有人喜歡瀏覽器就很難得了，也不管我實力怎樣，就給我機會，讓我進去公司體驗看看，所以這個暑假的兩個月我就在 CloudMosa 當 RD intern。

> 我朋友 Reggie Hsu 有寫了另一個英文版本的心得，大家也可以參考 <[Cloudmosa RD Internship 2020](https://reggiehsu111.medium.com/cloudmosa-rd-internship-2020-aaa184a6bf84)>

---

## 公司環境

我同事 Wens 之前有一篇推坑文，大家也可以參考看看。

> 點我：[[推坑] Puffin Browser 背後的公司 - CloudMosa 美商雲端科技](https://www.ptt.cc/bbs/Soft_Job/M.1419923169.A.41E.html)

不得不說，位於台北市樞紐中高樓的最上層(21樓)，視野真的是美不勝收，看外面風景就讓人心舒暢。

<img class="dz t u jv ak" src="https://miro.medium.com/max/4000/0*DvlFmzuUTBggQTgU" role="presentation"><br/>

我覺得有機會一定要去 101 的高樓層辦公室，或是去杜拜塔頂端上工作，那種俯視的快感一定更棒！

在來 CloudMosa 之前，我大學期間幾乎不會來北車，許多街道都令人懷念，高中念師大附中時幾乎都在這邊混，發現以前喜歡吃的店竟然倒了的時候也是滿感傷的。北車附近滿多吃的，不過後來除了跟同事一起吃，不然我幾乎都是買池上便當，而且只點里肌豬排飯，我覺得這是工程師的單調性吧。

公司鄰近台大醫院站。我當年指考在北一女考，我還記得那天從北一女走到台大醫院站，那種興奮又忐忑不安的心情，每當我上班走在台大醫院站的月台，那種奇妙的滋味就會浮上心頭。高三的暑假，一得一失，得到台大的門票，失去了女朋友。

## 辦公室環境

我很喜歡新創的工作氛圍，不管是之前在 Fusion$360 或是 CloudMosa，總是給我活力滿滿的感覺。而我在 Delta 的感覺就是一片死氣沉沉。

辦公室採用格子空間，第一眼的印象會覺得有點封閉，但是這裡人很開放，所以不會因此而缺少互動和交流。

<img class="dz t u jv ak" src="https://miro.medium.com/max/6528/1*-7wKLzfe4m7f95tXyhnTLg.jpeg" role="presentation"><br/>

上班時間很彈性，正職的同事可以隨時在家工作。有些同事住在新竹，周一和周五就是固定在家工作，也有同事長時間住美國，所以都是和台北辦公室遠端合作。

我的話通常都是約 11 點去公司，因為我都四五點睡，會起不來 :P。但我待在辦公室的時間也不會少，通常都是待到晚上 8 點才走。有時候還會必須強迫自己離開，碰到難纏的 bug 的時候，真的會走火入魔。

<img class="dz t u jv ak" src="https://miro.medium.com/max/6528/1*kZAOSgmh5Ytj4Mm4FnrO6w.jpeg" role="presentation"><br/>

因為是做瀏覽器的，公司有滿滿的測試機，iOS、Android、Windows 各類機款不計其數，公司前方還有好幾個大螢幕電視，專門用來測試 Puffin TV。

<img class="dz t u jv ak" src="https://miro.medium.com/max/4896/1*pNUj5vhin8rjeXHMkNJhKw.jpeg" role="presentation"><br/>

## 開發團隊

好的團隊真的很重要，跟 CloudMosa 的人共識之後，我才了解前老闆以前跟我說他覺得團隊的程度不如以前他在矽谷的水準是怎樣一回事。我想我在 CloudMosa 已經見識到，甚麼是世界一流的水準了。

CloudMosa 的員工絕對是一流等級，因為面試門檻真的非常高。聽說曾經有人應徵沒上，最後卻錄取 Google。並不是說我們比矽谷大廠還猛，只是從這件事就能理解能被錄取真的不容易，這也是我自嘆能力不足之處。

CEO Shioupyn 曾經跟我講他覺得 the <a href="https://zh.wikipedia.org/wiki/Big.LITTLE" class="dj by kr ks kt ku" target="_blank" rel="noopener nofollow">big.LITTLE</a> 很荒謬，我們只需要一顆超強的 CPU 不是嗎？我覺得這個觀點也用在 CloudMosa 的人事上，公司的理念就是每個進來的人都是超強的 CPU，一個要能打十個。

### 面試

公司的面試也滿有趣的，分為四輪獨立面試，由四組同事負責。每組會針對面試情況做出同意或否定，並將結果呈交給 CEO。最後 CEO 會針對四組的結果與同事做出決定。基本上就是要四組都同意才會被錄取。

每個人喜歡面試的東西都不太一樣，但久了大家也有默契，會對面試者考不一樣的東西，例如有些人就比較喜歡考技術方面，有些人比較喜歡考驗人格特質。但整體來說，CloudMosa 會希望進來的人有很強的技術背景，然後對瀏覽器、網路技術感興趣，也對 Puffin 有一定了解和想法。此外每個人必須能自己找到事情做，做的事情最好還能夠對公司有很大的助益。

### 各司其職

螞蟻、蜜蜂等是一種不可思議的動物。他們不需要別人發號司令，但他們會知道自己要做甚麼，井然有序。

CloudMosa 就是一個這樣的生態，採用扁平化管理，大家自己做好自己的事。這也是因為在找人的時候會很看重人格特質，所以才能如此放任大家自我管理並維持高產值。

我覺得同事的比喻滿有趣的，他說公司會錄取的員工大概分為三種，分別是將軍、遊俠和大兵。將軍就是有本事統領一堆人做事，遊俠是本領很強、喜歡獨來獨往、別人有需要時能拔刀相助，大兵則是能力足夠，在職務範圍能將事情做很好。以公司的情況來看，我覺得 CloudMosa 是一個遊俠為主的社會。

### 職務

CloudMosa 的職位簡單來分就是 CEO、RD 、QA、機房、業務。

每天日常就是 RD 一直解 ticket ，QA 一直生出 ticket。短短八年，不到三十人的團隊，已經累積了超過一萬條 tickets。這其實滿猛的，大家可以去看看其他大型開源專案的 issues 數量。

每天的信箱，差不多就是二三十封關於 commit 的通知，以及三四十封關於 tickets 的通知。此外還有幾百封 CI、用戶回報等等雜七雜八的信。

## 開發流程

Puffin 的開發流程跟一般大型軟體其實差不多。因為基於 Chromium 引擎，所以版本會跟著 Chromium 一起跑。

例如目前最新的 Puffin 7.7 就是基於 Chromium 66。我在寫這篇的時候，已經有 Chromium 69 了，所以其實有新的分支正在進行引擎升級。

此外因為 Puffin Client 與 Remote Browser Engine 是分開的，所以可以視為兩個產品線。而 Puffine Client 底下又有各個平台，包含 Windows, iOS, Android, Linux，則為子產品線。

大致上就是正在升級的版本為 dev branch，而已經上線的產品則進入 stable branch。這種開發模式應該滿常見的，Firefox 的版本與分支也是差不多的運作方式。

RD 開發好功能之後，QA 會先做確認，確認沒問題之後才會正式釋出。QA 之後也會追蹤產品狀況，視情況追加修正版。此外新功能通常會等版本穩定後才進行追加升級。

## 貢獻

我在 Cloudmosa 只有兩個月，時間算滿短的。這段期間我大致上做了兩件事情。

第一件是幫 Puffin Android 的 download to cloud(D2C) 功能完整化，這個功能是讓你可以在瀏覽網路想要下載檔案的時候，可以直接下載到雲端，能做到這件事情跟 Puffin 的架構有關，可以參考本文開頭附的連結，了解 Puffin 原理。

7 月幾乎都在弄這個模組，因為算是獨立的 feature，所以是一個很好的起手式，在做這個功能的時候可以順便了解整個公司的產品。

我進公司的時候，剛好公司正在開發全新的 Puffin Linux，所以第二件就是針對 Puffin Linux 的 UI 介面做貢獻。

完成 D2C 差不多一個月，8 月就在幫 Puffin 處理 UI 相關的東西，弄了顯示比例的放大鏡、顯示書籤的星星、搜尋列的右鍵選單還有一堆 UI 的邏輯修正。

在公司的這段期間我貢獻的 commits 數量大約有 100 條，我覺得生產力應該算不差了，公司也很放心讓我這個實習生貢獻。但跟同事比，我覺得我還是太弱了，每天都被滿滿的 commits log 灌爆信箱，讓我自嘆不如。

開發過程中的體驗，是我覺得公司比較像一群遊俠的原因，大部分的功能、模組都是一個人自行負責，很少會有某個功能兩個人以上一起開發。事實上 RD 也不到十個人，要開發的東西又很多，人力都不太夠了的情況下，更不太有機會多人同時在弄同件事情。

## 心得

CloudMosa 找我進去是希望我去體驗看看，CEO 也跟我講他不預期我做出甚麼，他認為找實習生是企業的一種社會責任。換句話說，他不是找我進來當生產力資源的，是來玩的。

但我自己期許能有所作為，我希望我有實質的貢獻，所以一開始我顯得很焦慮和拼命。不過欲速則不達也不是沒道理，好險有發現我的心態不正確，後來有調整過來，也感謝我的導師 <a href="https://medium.com/u/a01e1da0e781?source=post_page-----9d0e701dc166----------------------" class="lh az by" target="_blank" rel="noopener">fcamel</a> 耐心的指導。

做完 D2C 後，比較熟悉整體開發狀況，就能穩穩做貢獻了。我通常都挑優先度低的 P2、P3 bug，以及弄 Puffin Linux 的 UI 介面沒人碰的部分。

我如果靠面試的話，應該是進不了 CloudMosa，一個跟我滿好的同事就直言，我的程度還差地遠。確實是事實，也讓我一直滿自卑的，畢竟我等於走了後門進公司(覺得我挺有趣就讓我進來)。後來索性就仗著實習生的身分巴著其他工程師一直問白癡問題，像是我之前真的沒用過 GDB ，就叫同事教我，這真的是非常蠢的問題。:P

缺少了資工系的訓練，我覺得真的還是有差。原本不考慮念研究所的，現在我反而覺得需要去念資工所完成更扎實的基本訓練。我和同事的差距使我非常挫折，但這也給我一個努力的方向，想要盡快追上大家的程度。

整體來說，我滿慶幸有這個機會的，我覺得收穫最大的部分不在於學到甚麼技術，而是在這段期間的各種體驗，我滿喜歡 CloudMosa 的公司文化，和同事們相處也很開心，讓我覺得有歸屬感。

## 尾聲

<img class="dz t u jv ak" src="https://miro.medium.com/max/6528/1*OgfkPhrtdFFH8i2K2fg4mw.jpeg" role="presentation"><br/>

隨著暑假的收尾，我在 CloudMosa 的實習也到尾聲。因為我沒有時間在學期間上班，而且在這之前，我已經花太多時間在其他公司實習，剩下時間想好好在學校學習，所以就只到暑假結束而已。

如同我先前所述，我能力還不夠，我要把握在台大的最後時間把自己提升到和同事們一樣強。

在公司工作的最後一天，我一如往常地工作，剛好碰到同事下班就一樣說再見。當天我靜靜地待到最後，等所有人都下班之後，我將所有電燈、電扇都關掉。關上最後一盞燈的同時，也宣告我在 CloudMosa 的日子結束了，關上門就此離開公司。

我覺得這樣靜靜地離開滿好的，但心裡也滿感傷的，這個暑假我在 CloudMosa 有很棒的回憶！