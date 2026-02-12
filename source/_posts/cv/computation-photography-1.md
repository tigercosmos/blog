---
title: 計算攝影學 (Computational Photography) 簡介（一）
date: 2021-06-03 12:00:00
tags: [computational photography, camera, digital processing, HDR, time stacking, computer vision, image processing]
des: "本文介紹何謂計算攝影學，翻譯自 Vasily Zubarev 所寫的 Computational Photography 介紹"
lang: zh
translation_key: computation-photography-1
---

> 本文經作者同意翻譯自 Vasily Zubarev 所寫的「[Computational Photography-From Selfies to Black Holes](https://vas3k.com/blog/computational_photography/)」。大部分內容衷於呈現原意，少部分會根據譯者對原文詮釋稍作修改，內文中的「我」皆是指原作者，而譯者的註解會特別標示「譯按」。原文實在太長了，所以我將分成幾篇文章逐一翻譯。

現今智慧手機之所以可以這麼普遍流行，很大一部分需要歸功於搭載他們上面的相機。Pixel 可以在全黑的情況下有好效果，華為手機放大就像使用雙筒望遠鏡，三星手機搭載了八面鏡片，iPhone 的本身甚至就讓你在朋友中感覺優越了不少。在這些手機相機的背後，其實藏有著不可思議的創新。

相對地，單眼相機 (DSLRs) 似乎漸漸式微。儘管 Sony 每年仍推陳出新使人驚艷，但製造商的更新數度顯然持續減緩，甚至主要的營收來源只剩下那些影片創作者。

> 譯按：不過 Sony 確實還是滿厲害的，可以看他今年 (2021) 最新發表高階旗艦機 Sony A1 技術細節：[Sony A1 絕對影像王者 頂尖技術力的展現](https://www.mobile01.com/topicdetail.php?f=254&t=6311416)

我自己就有一台美金 $3000 的尼康相機，但每當我旅行時仍然用 iPhone 來拍照，這是為甚麼？

我在網路上尋找這個問題的答案，我發現很多有關「演算法」與「神經網路」的討論，但卻沒人可以清楚地解釋這些技術到底是怎樣去影像一張照片的呈現。新聞記者僅僅只是把一些產品規格的數據寫出來，部落客只是繼續產生更多開箱文，相機狂熱者只會在意相機呈現的顏色品質是否滿意。噢！網路呀，你給我們真多資訊，真愛你。

於是，我花了大半輩子去了解這背後的種種原理，我將在這篇文解釋我關於手機相機背後的所有事情，不然我大概也不久就忘光光了吧！

## 甚麼是計算攝影學

關於計算攝影學 (Computational Photography)，任何地方，包含[維基百科](https://en.wikipedia.org/wiki/Computational_photography)，你將會得到這樣的定義：「計算攝影學是採用數位計算的方式來產生數位影像或是影像處理，而非透過光學過程來達到。」基本上大部分都挺正確的，除了少數一些地方不對，像是計算攝影學甚至包含自動對焦的部分，並且提供有用的[光場相機](https://en.wikipedia.org/wiki/Light-field_camera)。看來官方定義仍然有些模糊的部分，我們仍然還是不太懂到底甚麼是計算攝影學！

> 光場相機是一種捕捉景物所形成光場資訊的相機，除了記錄不同位置下光的強度及顏色外，也記錄不同位置下光線的方向，而一般的相機只能記錄不同位置下光的強度。

史丹佛大學的 Marc Levoy 教授是計算攝影學的先驅，目前他正投入 Google Pixel 相機的開發中，在他的[文章中](https://medium.com/hd-pro/a25d34f37b11)提出了另一種解釋：「計算影像技術使我們加強和延伸數位攝影的可行性，使得我們拍出的照片看起來是如此的平常，卻幾乎不可能使用傳統相機辦到。」我更贊同這個定義，在接下來的文章，我將遵從這個定義。

因此，智慧手機是一切的根源—智慧手機別無選擇帶給人們全新的攝影技術：***計算攝影***。

智慧手機包含帶有雜訊的感光元件和比較粗糙的鏡頭。根據物理定律，他們應該只會帶給我們糟透的影像，但是直到一些開發人員發現如何去打破物理限制：更快的電子快門、強大的處理器以及更好的軟體。

![數位單眼跟智慧手機比較](https://user-images.githubusercontent.com/18013815/120586863-e9041800-c466-11eb-84a3-e9d50c3b1a41.jpg)

關於計算攝影學，大部分重要的研究多數在 2005-2015 年，不過這些都是過去的科學了。現在，映入咱們眼簾以及放在口袋的，將是前所未有嶄新的技術與知識。

計算攝影學不僅僅是 HDR 或夜間自拍模式。近期的黑洞拍攝如果沒有最新的計算攝影方法，是絕對不可能拍攝出來的。如果要用一般的望遠鏡去拍黑洞，我們將會需要整個地球這麼大的鏡片。但是，藉由放置在地球不同處的八個電波望遠鏡，並且經由一些[滿酷的 Python 程式碼](https://achael.github.io/_pages/imaging/)，我們得到了世界上第一張事件視界(event horizon)的照片。

<img src="https://i.vas3k.ru/87c.jpg" alt="event horizon" width=50%>

不過拿來拍自拍還是很好用啦，不用太擔心。

📝 [Computational Photography: Principles and Practice](http://alumni.media.mit.edu/~jaewonk/Publications/Comp_LectureNote_JaewonKim.pdf)
📝 [Marc Levoy: New Techniques in Computational photography](https://graphics.stanford.edu/talks/compphot-publictalk-may08.pdf)

> 這篇文章中將會穿插一些連結，他們將導向一些我發現很棒的文章 📝或影片 🎥，使你可以更深入去了解其中你有興趣的部分，畢竟我無法在短短的文章中解釋所有東西。

## 起源：數位處理 (Digital Processing)

回到 2010 年，Justin Bieber 發表他第一張專輯，哈里發塔 (Burj Khalif) 剛剛在杜拜啟用，當時我們還沒有能耐可以去記錄下那些壯觀的宇宙現象，因為我們的照片是充滿噪音的兩百萬像素 JEPG 圖片。

面對這樣的照片，當時我們有了第一個發自內心深處的願望，希望透過「Vintage」照片濾鏡來隱藏手機相機不堪一擊的照片成像，而這時 Instagram 也誕生了！

 ![](https://i.vas3k.ru/88i.jpg) 

# 數學與 Instagram

由於 Instagram 的誕生，任何人可以輕鬆地使用照片濾鏡。身為一個基於「研究目的」對曾對 X-Pro II、Lo-Fi、Valencia (都是濾鏡名稱) 做逆向工程的男人，我還記得這些濾鏡基本上包含三個部分：

- 顏色設定(色調、飽和度、亮度、對比、色階等)是基本的參數，如同過去攝影師在早期使用的濾鏡一般
![](https://i.vas3k.ru/85k.jpg) 


- 色調映射(Tone Mapping)包含一組向量的值，例如他告訴我們「一個色調 128 的紅色應該被改成色調 240」。他通常被使用在單一顏色的圖片中，如同[這個](https://github.com/danielgindi/Instagram-Filters/blob/master/InstaFilters/Resources_for_IF_Filters/xproMap.png)，是一個 X-Pro II 濾鏡的範例。
![](https://i.vas3k.ru/85i.jpg) 


- 疊加(Overlay)——使用半透明的圖片，上面包含灰塵、顆粒、小插圖或是任何東西，使其覆蓋在別的圖片上得到新的效果，不過不常使用。
![](https://i.vas3k.ru/85t.jpg)  

現代濾鏡不僅只有上述三個參數，但就會在數學方面變得更複雜一些。藉由手機支援硬體 Shader (著色器) 計算與 [OpenCL](https://en.wikipedia.org/wiki/OpenCL) 的支援，這些計算可以輕鬆的在 GPU 上面實現，確實是有夠酷！這些是我們在 2012 年時就能做到的。而現在，任何一個孩子都可以輕鬆[透過 CSS](https://una.im/CSSgram/)來辦到一樣的效果，不過他仍然沒機會邀請女孩去舞會就是了。

然而，在濾鏡上的進展仍然持續進行著，像是一些人在 [Dehancer](http://blog.dehancer.com/category/examples/) 網站上就實踐了一些非線性的濾鏡，不同於簡單的映射對應的修改方式，這些人用了不少華麗且複雜的轉換函數 (transformations)，這使得濾鏡得以有更多的可行性。

你可以藉由非線性的轉換函數來做到非常多的變化，但是就會使得處理變得複雜，而人們並不擅長這種複雜的工作，幸運的是我們可以藉由數值方法或是神經網路來做到，他們做到一樣的事情，但卻簡單了許多！

## 自動化調整與「一鍵完成」的夢想

當大家都習慣使用濾鏡之後，我們甚至直接把濾鏡直接整合進相機裡面了。誰是第一個想到把濾鏡放進相機的人已經不可考了，不過我們可以得知早在 iOS 5.0 發布的 2011 年，我們已經可以在裡面看到[「自動增強圖片」公開的 API]((https://developer.apple.com/library/archive/documentation/GraphicsImaging/Conceptual/CoreImaging/ci_autoadjustment/ci_autoadjustmentSAVE.html))。看來賈伯斯在公開 API 之前早就察覺到濾鏡已經被使用許久了。

自動化調整圖片在做的事情跟我們使用圖片編輯軟體做的事其實一模一樣，基本上就是修正光線和陰影，增加一些亮度，移除紅眼，修正臉部的顏色等，而使用者根本不會想到這個「神奇的加強版相機」其實背後靠的僅僅就是幾行的程式碼。

 ![ML Enhance in Pixelmator](https://i.vas3k.ru/865.jpg)
 (Pixelmator 裡的機器學習增強功能)

時至今日，「一鍵生成」的戰爭已經轉移至機器學習的領域了。已經厭倦做一堆風格轉換映射的操作的人們，開始轉向 [CNN 和 GAN 的懷抱](http://vas3k.com/blog/machine_learning/)，讓電腦自己去幫我們去調整修圖滑桿。換句話說，我們給機器一張圖片，他會自己去決定各種光學的參數，讓生成的圖片去接近我們所認知的「好的照片」。你可以上 Photoshop 或 Pixelmator Pro 之類的影片編輯軟體的官方網頁，看看他們如何用最新的機器學習特色功能來吸引你買單。你大概可以猜到機器學習不會永遠都行的通，但你永遠可以透過使用一堆資料集來訓練你自己的機器學習模型來做到更好。下面的一些資源也許對你有幫助，或是沒有 😂

📝 [Image Enhancement Papers](https://paperswithcode.com/task/image-enhancement)
📝 [DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks](http://people.ee.ethz.ch/~ihnatova/#dataset)


# 堆疊(Stacking)：智慧手機 90% 的功臣 

真正的計算攝影學來自於堆疊(Stacking)——一種將好幾張照片一張張疊在一起的技術。對於智慧手機來說，一秒內拍下幾十張照片輕而一舉。因為手機內部的機構沒有任何可以調整快門速度的原件，像是光圈都是固定的，並且是採用電子快門 (相較於傳統的機械快門)。處理器只需要各素感應器他應該要收多收毫秒的光子，然後我們就得到一張照片了。

技術上來說，手機可以像是拍影片一般的速度去照相(我們一般不會用 60fps 去照相對吧？)，它甚至也可以用照片的高畫質去錄影(一般錄影 4k (4000 萬像素) 已經很高畫質了，但相片動輒一億畫畫素)，但這樣做都會增加資料傳輸和處理器的負擔，因此軟體終究會因為硬體而有限制。

Stacking 技術其實已經發展有一段時間了。一些人會使用 Photoshop 的功能來做出一些瘋狂銳化的 HDR 照片或製作 18000x600 像素的全景圖，而且我們甚至有更多種嘗試的可能性，只待你去發掘！

人們將這種後製行為稱為「[Epsilon Photography](https://en.wikipedia.org/wiki/Epsilon_photography) (微調攝影)」，這意味著我們不斷更改相機參數（曝光、聚焦或位置）並合成出一張原本單靠單次拍攝不可能得到的照片。但在實踐中，我們稱這技術為堆疊。如今，所有行動裝置相機的創新中有 90％ 都基於此。

![](https://i.vas3k.ru/85d.jpeg) 

雖然有很多人不在乎手機相機背後的運作方式，但這對於理解整個智慧手機攝影卻至關重要：**現代的智慧型手機相機一打開就開始拍照**。這滿有道理的，畢竟它要在螢幕上顯示圖像給你看。除了不斷拍照外，它還將高解析度圖片保存在系統的循環緩衝區中，並將這些緩存存儲幾秒鐘。

> 當您點擊「拍攝照片」按鈕時，實際上已經手機早就拍攝了照片，相機其實只是使用緩衝區中的最後一張照片

如今，這就是任何手機相機的運作方式，至少高階智慧型手機是這樣。緩衝可以實現零[快門延遲](https://en.wikipedia.org/wiki/Shutter_lag) (按下快門到真的拍下照片的時間差)，這是攝影師期望已久的功能，有時甚至希望快門延遲可以是負的。透過按下按鈕，手機可以瀏覽過去，從緩衝區中去撈 5-10 張最後的照片，並開始對其進行瘋狂地分析和組合。所以我們甚至不再需要使用高動態範圍成像 (HDR) 或夜間模式，手機軟體會從緩衝區去處理好這些照片，用戶甚至都不會意識到拍的照片是被加工過的。 

事實上，這就是現在 iPhone 或 Pixel 在做的事情。

![](https://i.vas3k.ru/88j.jpg) 

## 曝光疊加 (Exposure Stacking)：高動態範圍成像 (HDR) 與光線控制

 ![](https://i.vas3k.ru/85x.jpg) 


一個存在已久的熱門話題是相機傳感器是否[可以捕捉我們眼睛可以看見的整個亮度範圍](https://www.cambridgeincolour.com/tutorials/cameras-vs-human-eye.htm)。 有人說不行，因為眼睛最多可以看到 25 個 [f-stops](https://en.wikipedia.org/wiki/F-number)，甚至頂級的全畫幅傳感器也就最大達到 14。有些人則覺得不正確，因為我們的眼睛是由大腦輔助的，大腦會自動調整您的瞳孔，並透過大腦的神經完成圖像。所以眼睛的瞬時動態範圍實際上不超過10-14 f-stops。太難了！讓我們把這些爭論留給科學家。 

但問題依舊存在—當你使用任何手機拍攝背對藍天的朋友時，如果沒有使用 HDR，你要馬得到一個清楚的天空但朋友是黑的，又或者朋友是清楚的但天空卻過曝了。

人們很久以前就找到了解決方案——使用 HDR（高動態範圍）擴大亮度範圍。當我們面對亮度範圍太廣的場景時，我們可以分三步（或更多）來完成拍攝。我們可以採用不同的曝光設定拍攝幾張照片——「正常」的、更亮的、更暗的各一張。然後我們可以用明亮的照片填充陰影的部分，並從較暗的照片中恢復過度曝光的區域。

這裡需要做的最後一件事是解決自動包圍的問題。我們需要知道如何將每張照片的曝光量分配調整，以免合成出的照片有過度曝光？然而，今天任何理工大學生都可以使用一些 Python 程式碼來做到這一點。

 ![](https://i.vas3k.ru/86t.jpg) 

當最新款 iPhone、Pixel 和 Galaxy 相機內的簡單算法檢測到您在晴天拍攝時，它們會自動開啟 HDR 模式。您甚至可以看到手機如何切換到緩沖模式 (Buffer Mode) 以保存更多的圖像——此時 FPS 下降，螢幕上的圖片變得更加生動。每次切換的瞬間在我的 iPhone X 上都清晰可見。下次仔細看看你的智能手機！

帶有包圍曝光的 HDR 的主要缺點是它在光線不足的情況下令人難以置信的毫無用武之地。 即使在家用燈的光線下，照片仍然很暗，甚至手機也無法將它們調整堆疊在一起。為了解決這個問題，谷歌早在 2013 年就在 Nexus 智能手機中宣布了一種不同的 HDR 方法。它使用時間疊加 (Time Stacking)。

📝 [What Is HDR: Concepts, Standards, and Related Aspects](https://www.videoproc.com/resource/high-dynamic-range.htm)

## 時間疊加 (Time Stacking)：長時曝光與時間流逝

 ![](https://i.vas3k.ru/85v.jpg) 

時間疊加可讓您透過一系列短時間曝光的照片獲得長曝光效果。這種方法是由喜歡拍攝夜空中星跡照片的天文愛好者們首創的。即使使用三腳架，也無法做到打開快門兩個小時來拍攝這樣的照片，因為您必須事先計算所有設置，但任何輕微的晃動都會破壞整個拍攝結果。所以他們決定將這個過程分成好多個幾分鐘的照片，然後在 Photoshop 中將圖片堆疊在一起。 

 ![These star patterns are always glued together from a series of photos. That make it easier to control exposure](https://i.vas3k.ru/86u.jpg) 
 (星軌是透過疊加一連串的照片合成出來的，這讓我們更容易控制曝光)

所以相機其實從來沒有長時間曝光拍攝過。我們透過組合幾個連續鏡頭來模擬效果長時曝光。長期以來，手機上有很多應用程式都在使用這個技巧，但現在幾乎每個製造商都將其添加到標準相機工具中。

 ![A long exposure made of iPhone's Live Photo in 3 clicks](https://i.vas3k.ru/86f.jpg)
 (在 iPhone 上透過三張 Live 照片製造出長時曝光的照片)

讓我們回到 Google 和它的夜間 HDR。事實證明，使用時間包圍可以在黑暗中創建一個不錯的 HDR。 這項技術首次出現在 Nexus 5 中，被稱為 HDR+。 該技術仍然十分受歡迎，以至於在最新的 Pixel 演示文稿中 [它甚至受到稱讚](https://www.youtube.com/watch?v=iLtWyLVjDg0&t=0)。 

HDR+ 的工作非常簡單：一旦相機檢測到您在黑暗中拍攝，它就會從緩衝區中取出最後 8-15 張 RAW 照片並將它們堆疊在一起。這樣，演算法會收集更多關於鏡頭暗區的訊息，以最大限度地減少噪點像素，避免某些原因導致相機出錯並且未能在每個特定幀上捕捉到光子。

想像一下：你不知道 [Capybara](https://en.wikipedia.org/wiki/Capybara) (一種動物) 長什麼樣，所以你決定問五個人。他們的故事大致相同，但每個人都會提到任何獨特的細節，因此與僅詢問一個人相比，您可以獲得更多資訊。照片上的像素也會發生同樣的情況。更多資訊、更清晰、噪點更少。

📝 [HDR+: Low Light and High Dynamic Range photography in the Google Camera App](https://ai.googleblog.com/2014/10/hdr-low-light-and-high-dynamic-range.html)

疊加從同一地點拍攝的照片會產生與上面星軌範例中相同的假長曝光效果。在幾十張照片的曝光彙總下，一張照片的錯誤可以在另一張照片上被修正。想像一下，要實現這一點，您需要在數位單反相機中猛按快門多少次。

 ![Pixel ad that glorifies HDR+ and Night Sight](https://i.vas3k.ru/86g.jpg) 

只剩下一件事，就是自動色彩空間映射。在黑暗中拍攝的照片通常會破壞色彩平衡（偏黃或偏綠），因此我們需要手動修復。在早期版本的 HDR+ 中，這個問題是透過簡單的自動色調修復解決的，就像 Instagram 濾鏡一樣。後來，他們使用了神經網絡來復原色彩。

[Night Sight 技術](https://www.blog.google/products/pixel/see-light-night-sight/) 就是這樣誕生的—Pixel 2、3 和更高版本中的「夜間攝影」技術。描述說「HDR+ 是建立在機器學習技術之上」。事實上，它只是神經網絡和所有 HDR+ 後處理步驟的一個花哨名稱。透過讓機器接受「之前」和「之後」照片數據集的訓練，我們能從一組黑暗和遭亂的照片中製作出一張漂亮的圖像。

 ![](https://i.vas3k.ru/88k.jpg) 

順便說一下，這個訓練數據集是公開的。也許蘋果公司的人會接受它並最終教他們「世界上最好的相機」在黑暗中拍攝？

此外，Night Sight 計算鏡頭中物體的[運動向量](https://en.wikipedia.org/wiki/Optical_flow)以使用標準化來處理模糊現象。因為長時間曝光中很容易產生模糊，所以智慧手機可以從其他鏡頭中取出銳利的部分並將它們堆疊起來以消除模糊的部分。

📝 [Night Sight: Seeing in the Dark on Pixel Phones](https://ai.googleblog.com/2018/11/night-sight-seeing-in-dark-on-pixel.html ".block-link")
📝 [Introducing the HDR+ Burst Photography Dataset](https://ai.googleblog.com/2018/02/introducing-hdr-burst-photography.html ".block-link")
