---
title: 在 PyCon APAC 2023 貢獻了第一支 CPython Pull Request
date: 2023-11-27 00:03:00
tags: [python, CPython, pycon]
des: "本文記錄在 PyCon APAC 2023 的 Sprint 活動中貢獻了第一支 CPython PR 的過程"
---

不久前參加了 PyCon APAC 2023，今年剛好辦在東京，而且會場就在我家不遠處，我甚至第一天是騎腳踏車去參加。

前兩天就是議程和到處找人聊天，這邊就不多做介紹了。

第三天有個 Sprint，參與者可以自己提出想做的專案，然後找其他人一起加入。

而我選擇去加入了 CPython 的小組，是由 [Donghee Na](https://twitter.com/dongheena92) 所主導，他是 Line 的工程師，並且是 CPython 核心貢獻者。

這張圖是大家正在聽 Donghee 解釋怎樣貢獻 CPython：
![大家聽 Donghee 解釋](https://github.com/tigercosmos/blog/assets/18013815/541ea140-170c-495d-aae4-171d402e2556)

CPython 專案編譯起來真的很簡單，編譯速度也比想像中快不少。由於沒有很複雜，這邊就不特別描述，請上 Github 看他的文件。

Sprint 是半天活動，時間有限，通常沒辦法作什麼大貢獻，所以基本上就是挑一些修改文件或是小的測試之類的問題作。

以 CPython 為例的話，大家都是找[「Easy」標籤的 Issue](https://github.com/python/CPython/issues?q=is%3Aopen+is%3Aissue+label%3Aeasy) 做，大多數 Easy 會很快被搶光，不過也有不少是前人做到一半的，也有一些 Easy 根本不 Easy。

判斷一個 Issue 的難易度很重要，最好的情況是可以在 Sprint 的幾小時內可以完成，因為這種大型專案的核心貢獻者通常很忙，趁他在場的時候當面問是最有效率的，不然回家之後你只能在 Github 上 tag 人家，溝通下率會大打折扣。

我大概花了一個多小時找目標，後來鎖定了「[gh-57879: Increase test coverage for pstats.py](https://github.com/python/cpython/pull/111447)」。這是一個增加測試覆蓋率的題目，通常這種就不會太複雜，同時之前已經有人有初步貢獻了，只是他後來中途放棄，我基本上就是把他做一半的拿來繼續做。

由於是基於前者貢獻繼續改，也必沒有花很久就完成，大約一小時就完成了 PR，不過比較可惜的是，Donghee 說他不太熟這邊的程式碼，說等其他人幫忙看，這就比較可惜了，因為有些人是現場就可以 Merge，而我則是後續來來回回跟其他貢獻者討論，約莫三週後才被 Merge。

我在搞懂 CPython 怎麼樣去察看覆蓋率上特別花時間，官方的「[Increase test coverage](https://devguide.python.org/testing/coverage/)」我覺得沒有寫得很清楚，比方說我完全搞不懂他裡面提到的 `COVERAGEDIR`，還有我一直不知道怎麼去有效率的察看覆蓋率報告。總之以我的 PR 來說，我是執行 `./python -m test test_profile test_pstats --coverage --coverdir=coverage`，然後在 `coverage` 資料夾中去直接讀 Raw 格式的報告，我感覺應該有更聰明的作法。此外我不知道為啥常常 `python -m test` 會出錯，我必須重新執行 `make` 重新編譯。

另外在貢獻這支 PR 的過程中，我也學到了一些 Python 的新知識，像是怎麼使用 `NamedTemporaryFile` API，還有使用上要怎麼搭配 `try..finally` 來確保暫存文件的正確關閉，透過簡單的貢獻 CPython 小 PR 就能學到很多。

參加這次 Sprint 貢獻 CPython 算是滿有趣的，解一下小的 Issue 也不會太困難，不過 CPython 對於真正的初學者來說可能還是有點挑戰，但是 Sprint 中也有各式各樣的題目，也有一些題目是使用 Python 接一下 API，我想對於新手來說也可以很快上手。

雖然我寫 Python 很久了，但不論是個人或公司，平常寫的 Python 程式都不是特別的厲害，如果不是透過這樣的機會，我想不太有機會得到社群的高手直接的指導，這就是開源好玩的地方吧！

跟我同桌寫程式的伙伴：
![跟我同桌的伙伴](https://github.com/tigercosmos/blog/assets/18013815/2cbb0709-f303-4b6d-a098-cfcbddc357ad)

最後感謝 Donghee Na 在 Sprint 指導大家去貢獻 CPython。

他的推文：
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Yesterday was PyCon APAC sprint day, and we submitted 6+ PRs to the CPython in a day. It was a wonderful experience to meet passionate people. Thanks to <a href="https://twitter.com/pyconapac?ref_src=twsrc%5Etfw">@pyconapac</a> and <a href="https://twitter.com/pyconjapan?ref_src=twsrc%5Etfw">@pyconjapan</a> for all the support. And also thank <a href="https://twitter.com/darjeelingt?ref_src=twsrc%5Etfw">@darjeelingt</a>, who suggested me to participate as an organizer. <a href="https://t.co/fiDjgw0tCD">pic.twitter.com/fiDjgw0tCD</a></p>&mdash; Donghee Na (@dongheena92) <a href="https://twitter.com/dongheena92/status/1718925777847374235?ref_src=twsrc%5Etfw">October 30, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

題外話，我同桌的伙伴 [c-bata](https://twitter.com/c_bata_) 很高興拿到 [Anthony Shaw](https://twitter.com/anthonypjshaw) 的簽名 😂，而且人家出過[三本書](https://www.amazon.co.jp/%E8%8A%9D%E7%94%B0-%E5%B0%86/e/B096XMSNRS?&linkCode=sl2&tag=nwpct1-twitter-profile-22&linkId=bfa9a74c037e611d23de81705cac7907&language=ja_JP&ref_=as_li_ss_tl)了！

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I participated in the <a href="https://twitter.com/hashtag/PyConAPAC?src=hash&amp;ref_src=twsrc%5Etfw">#PyConAPAC</a> sprint, made a modest contribution to CPython (thanks to <a href="https://twitter.com/dongheena92?ref_src=twsrc%5Etfw">@dongheena92</a> for your support). I received an autograph from <a href="https://twitter.com/anthonypjshaw?ref_src=twsrc%5Etfw">@anthonypjshaw</a>! As always, it was an incredible experience. Thank you so much. <a href="https://t.co/1IIhBccwkj">pic.twitter.com/1IIhBccwkj</a></p>&mdash; c-bata (@c_bata_) <a href="https://twitter.com/c_bata_/status/1718631302700945596?ref_src=twsrc%5Etfw">October 29, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
