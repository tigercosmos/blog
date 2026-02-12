---
title: "計算写真学（Computational Photography）入門（その1）"
date: 2021-06-03 12:00:00
tags: [computational photography, camera, digital processing, HDR, time stacking, computer vision, image processing]
des: "計算写真学（Computational Photography）とは何かを紹介します。Vasily Zubarev による Computational Photography の解説記事を翻訳したものです。"
lang: jp
translation_key: computation-photography-1
---

> 本記事は著者の許可を得て、Vasily Zubarev の「[Computational Photography-From Selfies to Black Holes](https://vas3k.com/blog/computational_photography/)」を翻訳したものです。大部分は原意に忠実に訳し、少数の箇所は訳者の解釈に基づいてわずかに調整しています。本文中の「私」は原著者を指し、訳者注は「訳者注」と明記します。原文が非常に長いため、数本の記事に分けて順次翻訳します。

いまスマートフォンがこれほどまでに普及した大きな理由の1つは、そこに搭載されたカメラです。Pixel は真っ暗な状況でも良い写真が撮れますし、Huawei のズームは双眼鏡のようで、Samsung は8枚構成のレンズを載せ、iPhone は友人の前でちょっと優越感までくれます。こうしたスマホカメラの裏側には、信じられないほどの革新が隠れています。

一方で、一眼レフ（DSLR）は徐々に影が薄くなっているように見えます。Sony は今でも毎年驚くような新製品を出していますが、メーカー全体としてはアップデートの頻度が明らかに落ちており、主要な収益源は映像クリエイター向けに寄ってきています。

> 訳者注：とはいえ Sony はやはりすごいです。2021年に発表されたハイエンド旗艦機 Sony A1 の技術的な詳細はこちら：[Sony A1 絕對影像王者 頂尖技術力的展現](https://www.mobile01.com/topicdetail.php?f=254&t=6311416)

私は3000ドルの Nikon を持っています。それでも旅行のときは iPhone で写真を撮ってしまう。なぜでしょう？

この答えをネットで探してみると、「アルゴリズム」や「ニューラルネットワーク」の話は山ほど出てきますが、それらが実際に写真の見え方にどう影響するのかを、明快に説明してくれる人はほとんどいません。記者は製品スペックの数字を並べるだけ。ブロガーは開封記事を量産するだけ。カメラマニアは色が気に入るかどうかだけ。ああ、インターネット。情報はたくさんくれるのに、肝心な説明はない。

そこで私は、この裏側の原理を理解するために人生のかなりの時間を費やしました。このシリーズでは、スマホカメラの背後で起きていることを全部説明します。そうしないと、たぶんすぐ忘れてしまうので。

## 計算写真学とは何か

計算写真学（Computational Photography）について、どこで調べても、例えば [Wikipedia](https://en.wikipedia.org/wiki/Computational_photography) を見ても、だいたい次のような定義が出てきます：「計算写真学とは、光学的な過程ではなくデジタル計算を用いてデジタル画像を生成したり、画像処理を行ったりすること」。大筋は合っていますが、いくつか微妙な点もあります。例えば計算写真学にはオートフォーカスも含まれますし、実用的な [ライトフィールドカメラ](https://en.wikipedia.org/wiki/Light-field_camera) も含みます。公式っぽい定義はまだ曖昧で、結局「計算写真学って何？」がよく分からないままです。

> ライトフィールドカメラは、シーンが作る光場（light field）情報を取得するカメラです。位置ごとの光の強度や色に加え、光線の方向も記録します。通常のカメラは位置ごとの光の強度しか記録できません。

スタンフォード大学の Marc Levoy 教授は計算写真学の先駆者で、現在は Google Pixel のカメラ開発に携わっています。彼は[記事](https://medium.com/hd-pro/a25d34f37b11)の中で別の説明を提示しています：「計算イメージング技術はデジタル写真の可能性を強化・拡張する。私たちが撮る写真はあまりに“普通”に見えるが、伝統的なカメラではほとんど不可能だ」。私はこの定義のほうが好きで、以降はこの定義に従います。

つまり、スマートフォンこそがすべての根源です。スマートフォンは、人々に新しい撮影技術をもたらす以外に選択肢がなかった――***計算写真学***です。

スマホにはノイズの多いセンサーと、比較的粗いレンズが載っています。物理法則から言えば、ひどい画像しか出ないはずでした。しかし開発者は物理的制約を破る方法を見つけます：高速な電子シャッター、強力なプロセッサ、そして優れたソフトウェア。

![數位單眼跟智慧手機比較](https://user-images.githubusercontent.com/18013815/120586863-e9041800-c466-11eb-84a3-e9d50c3b1a41.jpg)

計算写真学の重要な研究の多くは 2005〜2015 年に集中していました。ただ、それはもう「過去の科学」です。いま私たちの目の前にあり、ポケットに入っているのは、前例のない新しい技術と知識です。

計算写真学は HDR や夜間セルフィーモードだけではありません。最近のブラックホール撮影は、最新の計算写真学なしには絶対に実現できません。普通の望遠鏡でブラックホールを撮るなら地球サイズのレンズが必要ですが、地球上の8基の電波望遠鏡を配置し、そして[かなりイケてる Python コード](https://achael.github.io/_pages/imaging/)を動かすことで、私たちは世界初の事象の地平線（event horizon）の写真を得ました。

<img src="https://i.vas3k.ru/87c.jpg" alt="event horizon" width=50%>

とはいえ、セルフィー用途にも十分役立ちます。心配しないで。

📝 [Computational Photography: Principles and Practice](http://alumni.media.mit.edu/~jaewonk/Publications/Comp_LectureNote_JaewonKim.pdf)
📝 [Marc Levoy: New Techniques in Computational photography](https://graphics.stanford.edu/talks/compphot-publictalk-may08.pdf)

> 本シリーズでは、私が良いと思った記事 📝 や動画 🎥 へのリンクを途中に挟みます。短い記事で全部は説明できないので、興味のあるところを深掘りできるようにするためです。

## 起源：デジタル処理（Digital Processing）

2010年に戻りましょう。Justin Bieber がデビューアルバムを出し、ブルジュ・ハリファ（Burj Khalif）がドバイで開業し、私たちはまだ壮大な宇宙現象を記録できませんでした。なぜなら、写真はノイズまみれの 200 万画素 JPEG だったからです。

そんな写真に直面したとき、私たちの心からの願いはただ1つ。「Vintage」フィルタでスマホカメラの貧弱な画質をごまかしたい。そうして Instagram が誕生しました。

 ![](https://i.vas3k.ru/88i.jpg) 

# 数学と Instagram

Instagram の登場で、誰でも簡単に写真フィルタを使えるようになりました。「研究目的」で X-Pro II、Lo-Fi、Valencia（いずれもフィルタ名）をリバースエンジニアリングした男として、私はこれらのフィルタが基本的に次の3つで構成されていることを覚えています：

- 色設定（色相、彩度、明るさ、コントラスト、レベルなど）は基本パラメータで、昔の写真家が使っていたフィルタと同じようなものです。
![](https://i.vas3k.ru/85k.jpg) 


- トーンマッピング（Tone Mapping）は値のベクトルに対する写像で、例えば「赤のトーン 128 は 240 に変えるべき」といった指示をします。単一色の補正で使われることが多いです。[これ](https://github.com/danielgindi/Instagram-Filters/blob/master/InstaFilters/Resources_for_IF_Filters/xproMap.png)は X-Pro II フィルタの例です。
![](https://i.vas3k.ru/85i.jpg) 


- オーバーレイ（Overlay）は、埃・粒状感・小さなイラストなどが入った半透明画像を別の画像に重ねて新しい効果を作る方法です。頻繁には使われません。
![](https://i.vas3k.ru/85t.jpg)  

現代のフィルタはこの3つだけではありませんが、数学的にはさらに複雑になります。スマホはハードウェアシェーダ（shader）計算や [OpenCL](https://en.wikipedia.org/wiki/OpenCL) をサポートしているので、こうした計算を GPU 上で簡単に実行できます。マジでクールです。これは2012年にはもうできていました。いまやどんな子どもでも [CSS](https://una.im/CSSgram/) で同じようなことができますが、彼がプロムに誘えるかどうかは別問題です。

とはいえ、フィルタの進化は続いています。例えば [Dehancer](http://blog.dehancer.com/category/examples/) では、単純なマッピング変更とは異なる非線形フィルタが実装されています。彼らは派手で複雑な変換関数（transformations）を使い、フィルタの可能性を広げています。

非線形変換関数でできることは非常に多い一方、処理は複雑になります。そして人間は複雑な作業が得意ではありません。幸運なことに、数値計算やニューラルネットワークを使えば、同じことをずっと簡単に実現できます！

## 自動調整と「ワンクリック」の夢

みんながフィルタに慣れると、フィルタはカメラに統合されました。誰が最初にカメラにフィルタを入れたかは分かりませんが、少なくとも iOS 5.0（2011年）時点で、すでに[「自動増強画像」の公開 API]((https://developer.apple.com/library/archive/documentation/GraphicsImaging/Conceptual/CoreImaging/ci_autoadjustment/ci_autoadjustmentSAVE.html))がありました。Jobs は公開 API より前から、フィルタが長く使われていることに気づいていたのでしょう。

自動調整がやっていることは、画像編集ソフトで私たちがやる作業とほぼ同じです。ハイライトとシャドウの補正、明るさの追加、赤目除去、肌色補正など。ユーザーは、この「魔法の強化カメラ」が実は数行のコードで動いているとは思いもしません。

 ![ML Enhance in Pixelmator](https://i.vas3k.ru/865.jpg)
 (Pixelmator の機械学習エンハンス機能)

そして今日、「ワンクリック生成」の戦いは機械学習領域へ移りました。面倒なスタイルマッピングに疲れた人々は [CNN や GAN](http://vas3k.com/blog/machine_learning/) に寄りかかり、コンピュータにレタッチのスライダを勝手に動かしてもらうようになります。つまり、画像を渡すと機械が光学的パラメータを自分で決め、私たちが認知する「良い写真」に近づけた画像を生成するのです。Photoshop や Pixelmator Pro の公式ページを見れば、最新の ML 機能でいかに購買意欲を煽っているかが分かります。ML が永遠に万能ではないことは想像がつくでしょう。でも、たくさんのデータセットで自分の ML モデルを訓練すれば、より良い結果は作れます。以下のリソースが役に立つかもしれないし、立たないかもしれません 😂

📝 [Image Enhancement Papers](https://paperswithcode.com/task/image-enhancement)
📝 [DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks](http://people.ee.ethz.ch/~ihnatova/#dataset)


# スタッキング（Stacking）：スマホの90%の功労者

本当の計算写真学はスタッキングから生まれます。スタッキングとは、複数の写真を1枚ずつ重ねる技術です。スマホなら1秒で数十枚撮ることは容易です。スマホ内部にはシャッター速度を調整する機械部品がなく（絞りも固定）、電子シャッター（機械シャッターではない）を使うためです。プロセッサはセンサーに「何ミリ秒ぶん光子を集めるべきか」を指示し、それで1枚の写真が得られます。

技術的には、スマホは動画のような速度で写真を撮れます（普通、60fpsで写真は撮らないですよね？）。さらに、写真並みの画質で動画を撮ることも理屈の上では可能です（一般に4K動画は高画質でも約4000万画素ですが、写真は簡単に1億画素を超えます）。ただしそれはデータ転送と処理負荷を増やすため、結局ソフトウェアはハードウェア制約を受けます。

スタッキング技術自体は昔からあります。Photoshop を使って異常にシャープな HDR 写真を作ったり、18000×600 ピクセルのパノラマを作ったりする人もいます。まだまだ試せることはたくさんあります。あなた次第です！

こうした後処理は「[Epsilon Photography](https://en.wikipedia.org/wiki/Epsilon_photography)（微調整写真）」とも呼ばれます。露出・フォーカス・位置といったカメラパラメータを変え続け、単発撮影では不可能な画像を合成で作る、という意味です。ただ実務では、私たちはこれを単にスタッキングと呼びます。今日のモバイルカメラ革新の90%はこれに基づいています。

![](https://i.vas3k.ru/85d.jpeg) 

多くの人はスマホカメラの仕組みに興味がないかもしれませんが、スマホ写真を理解するには極めて重要な事実があります：**現代のスマホカメラは、開いた瞬間から撮影を始めている**。画面にプレビューを出す必要があるので合理的です。連続撮影するだけでなく、高解像度フレームをリングバッファ（循環バッファ）に保存し、数秒間保持しています。

> あなたが「撮影」ボタンを押したとき、実際にはスマホはずっと前に写真を撮っています。カメラは単にバッファ内の最後の1枚を使っているだけです。

これが現在のスマホカメラの動作です（少なくともハイエンド機では）。バッファリングによりゼロ[シャッターラグ](https://en.wikipedia.org/wiki/Shutter_lag)（ボタンを押してから撮影されるまでの遅延）が実現できます。写真家が長年望んできた機能で、場合によっては「シャッターラグは負であってほしい」とまで言われます。ボタンを押すとスマホは「過去」を遡り、バッファから直近の 5〜10 枚を掬い上げ、猛烈に解析して合成を始めます。つまり、HDR や夜景モードをユーザーが意識しなくても、スマホ側がバッファから適切に処理してしまい、ユーザーは「加工されている」ことに気づかないことすらあります。

実際、いまの iPhone や Pixel がやっているのはこれです。

![](https://i.vas3k.ru/88j.jpg) 

## 露出スタッキング（Exposure Stacking）：HDR と光の制御

 ![](https://i.vas3k.ru/85x.jpg) 


長年の論争として、カメラセンサーが[人間の目が見える輝度レンジ全体を捕捉できるか](https://www.cambridgeincolour.com/tutorials/cameras-vs-human-eye.htm)という話があります。否定派は「目は最大 25 [f-stops](https://en.wikipedia.org/wiki/F-number) を見られるが、最高級のフルサイズセンサーでもせいぜい 14 だ」と言います。一方で「目は脳に補助されている。脳が瞳孔を自動調整し、神経で画像を“完成”させる。だから瞬間的ダイナミックレンジは 10〜14 f-stops 以上ではない」と主張する人もいます。難しすぎる！この議論は科学者に任せましょう。

それでも問題は残ります。青空を背にした友人をスマホで撮ると、HDR がなければ「空は綺麗だが友人が黒い」か、「友人は明るいが空が白飛び」かのどちらかになりがちです。

この解決策は昔からあります――HDR（High Dynamic Range）で輝度レンジを広げることです。輝度レンジが広すぎるシーンでは、3段階（またはそれ以上）に分けて撮影できます。露出を変えて「普通」「明るい」「暗い」写真を撮り、明るい写真で影を埋め、暗い写真で白飛び領域を復元します。

最後に必要なのはオートブラケティングの解決です。各フレームの露出をどう配分・調整すれば、合成結果が過露出にならないかを知る必要があります。しかし今日では、理工系の大学生なら少しの Python コードでそれを実現できます。

 ![](https://i.vas3k.ru/86t.jpg) 

最新の iPhone・Pixel・Galaxy のカメラ内の単純なアルゴリズムが「晴天で撮影している」と検知すると、自動的に HDR を有効にします。バッファモード（Buffer Mode）に切り替わってより多くの画像を保存する様子も分かることがあります。FPS が下がり、プレビューが鮮やかになる。iPhone X ではその切り替わる瞬間がはっきり見えました。次にスマホを使うとき、ぜひ観察してみてください。

ブラケットHDRの最大の欠点は、暗所で驚くほど役に立たないことです。家庭の照明下でも写真は暗く、スマホでもうまく整列・スタックできないことがあります。これを解決するため、Google は 2013 年の Nexus スマホで別の HDR 手法を発表しました。時間方向にスタックする（Time Stacking）方法です。

📝 [What Is HDR: Concepts, Standards, and Related Aspects](https://www.videoproc.com/resource/high-dynamic-range.htm)

## タイムスタッキング（Time Stacking）：長時間露光とタイムラプス

 ![](https://i.vas3k.ru/85v.jpg) 

タイムスタッキングは、短時間露光の写真列から長時間露光の効果を得る方法です。夜空の星の軌跡（star trails）を撮る天文ファンが最初に考えたと言われています。三脚を使っても、2時間シャッターを開けっぱなしにするのは現実的ではありません。設定の事前計算が必要で、少しの揺れで結果が台無しになるからです。そこで彼らは、数分単位の写真をたくさん撮り、Photoshop で重ねることにしました。

 ![These star patterns are always glued together from a series of photos. That make it easier to control exposure](https://i.vas3k.ru/86u.jpg) 
 (星の軌跡は写真列を重ねて作られます。これにより露出制御が容易になります)

つまりカメラは実際には長時間露光をしていません。連続ショットを組み合わせて効果を“模擬”しています。このテクニックは昔からスマホアプリで使われてきましたが、いまでは多くのメーカーが標準機能として組み込んでいます。

 ![A long exposure made of iPhone's Live Photo in 3 clicks](https://i.vas3k.ru/86f.jpg)
 (iPhone の Live Photo を3クリックで長時間露光風にした例)

Google の夜間 HDR に戻りましょう。時間方向のブラケティングを使うと、暗所でもそれなりの HDR が作れることが分かりました。この技術は Nexus 5 に初めて搭載され、HDR+ と呼ばれました。いまでも人気が高く、最新の Pixel のプレゼンでも[褒められています](https://www.youtube.com/watch?v=iLtWyLVjDg0&t=0)。

HDR+ の仕組みはとてもシンプルです。カメラが暗所撮影を検知すると、バッファから直近の 8〜15 枚の RAW を取り出してスタックします。これにより暗部の情報をより多く集め、ノイズピクセルを最小化し、何らかの理由で特定フレームに光子が入らなかった場合の失敗も避けます。

例えば、あなたが [Capybara](https://en.wikipedia.org/wiki/Capybara)（動物）を見たことがないとして、5人に聞いてみるとします。話の筋は似ていますが、それぞれが固有のディテールを言い、1人に聞くより多くの情報が得られます。写真のピクセルも同じです。情報が増え、よりシャープで、ノイズが減ります。

📝 [HDR+: Low Light and High Dynamic Range photography in the Google Camera App](https://ai.googleblog.com/2014/10/hdr-low-light-and-high-dynamic-range.html)

同じ場所で撮った写真をスタックすることは、先ほどの星の軌跡の例と同じく、擬似的な長時間露光を生みます。数十枚の露出を合成すれば、ある1枚の誤差は別の1枚で補正できます。これを DSLR でやるなら、シャッターボタンを何回連打する必要があるでしょうか。

 ![Pixel ad that glorifies HDR+ and Night Sight](https://i.vas3k.ru/86g.jpg) 

残る最後の課題は、自動的な色空間マッピングです。暗所写真は色バランスが崩れやすく（黄み、緑み）、手動で修正が必要になります。初期の HDR+ では、Instagram のフィルタのような簡単な自動トーン補正で対処していましたが、後にニューラルネットワークで色を復元するようになりました。

[Night Sight](https://www.blog.google/products/pixel/see-light-night-sight/) はこうして生まれました。Pixel 2、3 以降の「夜間撮影」機能です。説明では「HDR+ は機械学習技術に基づく」とされています。実際には、ニューラルネットワークと HDR+ の後処理一式の“オシャレな呼び名”です。「Before/After」データセットで学習させることで、暗くて荒れた写真群から美しい画像を作れます。

 ![](https://i.vas3k.ru/88k.jpg) 

ちなみに、この学習データセットは公開されています。Apple の人がこれを取り込んで、「世界最高のカメラ」に暗所撮影を教える日が来るかもしれません。

さらに Night Sight は、フレーム中の物体の[モーションベクトル](https://en.wikipedia.org/wiki/Optical_flow)を計算し、手ぶれ補正（stabilization）によってブレを扱います。長時間露光はブレやすいので、スマホは別フレームからシャープな部分を取り出してスタックし、ブレた部分を消していきます。

📝 [Night Sight: Seeing in the Dark on Pixel Phones](https://ai.googleblog.com/2018/11/night-sight-seeing-in-dark-on-pixel.html ".block-link")
📝 [Introducing the HDR+ Burst Photography Dataset](https://ai.googleblog.com/2018/02/introducing-hdr-burst-photography.html ".block-link")

