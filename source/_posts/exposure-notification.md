---
title: 「台灣社交距離」App 背後技術原理：接觸通知系統與保護隱私的接觸追蹤技術
date: 2021-05-13 23:25:00
tags: [接觸通知系統, 公共衛生, covid-19]
des: "近日衛服部公佈了「台灣社交距離」App，有鑑於該 App 完全沒提及技術細節，以及網路上各大新聞媒體報導也沒介紹，於是本文介紹該 App 背後的技術細節。"
---

## 簡介

近日衛服部公佈了「台灣社交距離」App，有鑑於該 App 完全沒提及技術細節（實在是非常不應該），以及網路上各大新聞媒體報導也沒介紹（大多就是說這東西有用喔），於是將由本文介紹該 App 背後的技術細節。

## 起源

台灣政府目前公告確診診的作法差不多等於是直接公佈是誰了，以至於大家會瘋狂肉搜甚至公審，確診者似乎毫無隱私可言，我們看之前機師與情婦的例子就知道。當然政府可以說「公共利益」比「個人隱私」來的重要，但有沒有可能，我們**同時顧及公共利益和個人隱私**呢？

大約四月的時候，Google 和 Apple 兩大智慧手機陣營思考著如何讓手機可以更有效的幫助防疫，畢竟現在人人一隻手機，手機基本上就無時無刻記錄著你的位置資訊，那麼如果今天有人確診，是否可以讓確診者「主動」將他過去的位置記錄分享呢？這樣曾經出現在附近的人就會知道自己可能有被感染的風險。

但我們知道現代人最重視的就是隱私了，我們當然不能明目張膽直接公佈說，某某某的這隻 09xxxx 號碼的手機，曾經在地點 A、B、C 地點出沒，這樣就沒人願意「主動」分享了。

所以如何匿名就很重要了，只有確診者知道他可以完全匿名情況下，他才會比較願意主動分享，而因為是匿名的，其他人只會得到消息說自己曾經跟確診者接觸過，但他不會知道那是誰。

為了達到可以讓「匿名分享」成立，Apple 與 Google 提出一套技術協議，讓擁有手機的大家可以在「匿名」情況下彼此分享過去的行蹤，並且保證絕對的隱私，這套系統甚至不會記錄地理資訊，而是記錄人與人之間的「接觸」歷史。

## 接觸通知系統

Goole 與 Apple  共同研發的這套技術叫做「[保護隱私的接觸追蹤（Privacy-Preserving Contact Tracing）](https://covid19.apple.com/contacttracing)」，而基於這套技術提供了「接觸通知 API (Exposure Notification API)」。

每個國家的公共衛生主管機關可以基於這個 API 去開發自己國家的 App，稱為**接觸通知系統**。以台灣為例就是「台灣社交距離」，你可以在[這個網頁](https://www.google.com/covid19/exposurenotifications/select/)看到每個國家的 App。

使用各國主管機開發的接觸通知系統 App 後，如果用戶可能接觸過 COVID-19 患者，手機會收到回報，並透過 App 向你傳送通知。

## 接觸通知系統的簡單原理

下面這個影片簡單介紹了原理，但很可惜的是 Google 竟然沒提供中文？

<iframe width="560" height="315" src="https://www.youtube.com/embed/1Cz2Xzm6knM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

以下是我將 Apple 提供的圖例翻譯：

接觸通知系統圖示 1：
![接觸通知系統-圖示1](https://user-images.githubusercontent.com/18013815/118144059-97a3d280-b43e-11eb-9491-d289648076bd.jpg)

接觸通知系統圖示 2：
![接觸通知系統-圖示2](https://user-images.githubusercontent.com/18013815/118144066-98d4ff80-b43e-11eb-83dd-20b5376213ce.jpg)

以下是改編自 [Google 官方說明](https://www.google.com/covid19/exposurenotifications/select/)：

1. 當你下載接觸通知系統的 App，你的裝置就會得到一組隨機產生的 ID。為了確保你的身分或位置不會因為這些隨機 ID 而外洩，系統每 10 到 20 分鐘就會變更一次隨機 ID。
2. 你的手機和周遭的手機會透過藍牙在背景交換隨機 ID；由於交換的是隨機 ID，所以能保障隱私權。並且你不必開啟相關應用程式，這個程式也能正常執行。同時因為藉由藍芽的交換技術，個人的地理位置是不會被得知的。
3. 你的手機會定期比對本身的記錄清單與所有 COVID-19 陽性確診病例的隨機 ID，尋找是否有相符的項目。
4. 如果發現有符合，代表你曾經在某個地方（但不會知道是哪裡）曾經接觸過患者。該 App 就會向你傳達當地公共衛生主管機關的進一步指示，告知你該如何保護自己和旁人免於病毒威脅。

## 接觸通知系統的詳細技術原理

如上面介紹，要做到接觸通知系統，背後靠的就是藍芽，以及隨機 ID 的技術。

所以官方有兩個技術規範：
- [Bluetooth® Specification](https://covid19-static.cdn-apple.com/applications/covid19/current/static/contact-tracing/pdf/ExposureNotification-BluetoothSpecificationv1.2.pdf?1)
- [Cryptography Specification](https://covid19-static.cdn-apple.com/applications/covid19/current/static/contact-tracing/pdf/ExposureNotification-CryptographySpecificationv1.2.pdf?1)

技術規範有興趣的人就自己點進去看吧，大家都是工程師，看幾頁英文應該沒問題。

開發者可以根據 [API 規範](https://developer.apple.com/documentation/exposurenotification)來開發 App，所以「台灣社交距離」背後的程式碼不外乎就是這些 API，不過「台灣社交距離」在其程式裡面的任何地方都完全沒有提及它使用的是這套 API。

> 2021.7.22 補充：台灣社交距離原始碼已開源。[IOS 原始碼](https://github.com/ailabstw/social-distancing-ios)

Apple 官方甚至有將他們如何實現接觸通知系統的[開放原始碼釋出](https://developer.apple.com/exposure-notification/)。

節錄一下原始碼裡面 README 的 Overview 給大家看看：

> This code project contains code snippets from the iOS implementation of Exposure Notification. These snippets don't represent the entirety of the Exposure Notification implementation. Instead, it focuses on the critical paths needed to understand the functionality of the most significant parts of the codebase. This project is separated into four groups of functionality.

對如何實現有興趣的人可以去下載 Apple 的程式碼來看看。

## 結論

本文介紹了甚麼是 Google 和 Apple 共同研發的接觸通知系統，並且解釋背後的原理，而「台灣社交距離」背後其實就是基於這套系統提供的 API 所開發。
