---
title: "使用 tcpdump 分析網路封包"
date: 2025-01-03 00:30:00
tags: [tcpdump,  程式設計原來不只有寫 CODE！, 資工基本素養, 軟體工程師自我成長]
des: "本文介紹如何使用 tcpdump 來分析網路封包，來檢查程式的網路行為是否符合預期，以及是否有異常的網路行為，內容摘自《程式設計原來不只有寫 CODE！》一書的〈使用 tcpdump & Wireshark 分析網路行為〉章節。"
---

<blockquote style="border: 5px solid #ee9c6b;border-radius:30px;">

本文摘自[**《程式設計原來不只有寫 CODE！銜接學校與職場的五堂軟體開發實習課》**](/books/beyond-just-coding-book.html)一書的〈使用 tcpdump & Wireshark 分析網路行為〉。本文為精簡版，敬請[支持原書](/books/beyond-just-coding-book.html)來觀看完整內容以及精美排版。

</blockquote>

## 網路封包分析簡介

我們日常生活寫的程式往往跟網路脫不了關係，例如我們可能會寫一個程式發 HTTP 的 API 給某個服務的網站（即 HTTP 客戶端），抑或是開發物聯網裝置的應用軟體時，需要與裝置建立 UDP/TCP 連線。一般來說，我們都可以用函式庫、**開發套件**（SDK）來輕鬆寫出網路連線的程式。但是有時候不知道怎麼地，連線就是建立不起來，或是收到的網路資料不符合預期，這時候我們就需要工具幫助我們去進行網路封包分析（簡稱封包分析）。

![cover image](https://github.com/user-attachments/assets/01c18d00-0746-4c78-a97d-a7ebd4c14eee)

學習封包分析是理解和解決網路相關問題的重要一環，相信很多人聽到分析網路封包的第一直覺就是跟網路安全有關，只能說網路安全確實是一大宗應用，但是我們平常寫的任何有關網路的程式，其實都會有機會用到封包分析，常見的應用情境包含：

- **網路故障排除**：當網路出現問題時，網路封包分析能夠提供深入的洞察，幫助快速定位和解決問題。透過分析封包，可以發現連線問題、丟包情況、延遲原因等。
- **網路安全**：對於網路安全性而言，了解網路封包是一種重要的技能。透過分析封包，可以檢測不尋常的網路活動，發現潛在的攻擊或入侵行為。
- **程式效能最佳化**：透過了解數據在網路中的傳輸過程，可以最佳化應用程式和網路設置的互動，提高整體程式效能。

而許多工具都可以進行網路封包分析，例如以下兩個常見的工具：

- **tcpdump**：是一個命令列工具，我們可以直接在終端機上監測和顯示網路封包，用此工具監測特定協議（例如FTP協議），並顯示來源 IP、目標 IP、**payload**（酬載） 等。
- **Wireshark**：是一個圖形化的網路封包分析工具，提供了豐富的圖形界面和強大的分析功能。Wireshark 能夠以人類可讀的形式顯示封包內容，方便使用者直接用圖形介面進行深入的分析。

本文將著重介紹 tcpdump 如何使用，而 Wireshark 基本上概念一模一樣，只是前者是終端機介面程式而後者為圖形化介面程式。




## 安裝 tcpdump

首先我們得安裝 tcpdump，可以透過以下命令來安裝：

```shell
$ sudo apt install tcpdump       # Ubuntu/Debian
$ brew install tcpdump           # macOS
```

## tcpdump 基礎用法

一個最簡單的 tcpdump 使用方法，就是什麼參數都不下直接執行它，記得要使用 sudo 來以 root 權限執行：

```shell
$ sudo tcpdump
IP 172.21.110.24 > LAPTOP-P7QFA4QB.mshome.net: ICMP 172.21.110.24 udp port 60517 unreachable, length 80

IP LAPTOP-P7QFA4QB.mshome.net.domain > 172.21.110.24.60517: 56836 NXDomain 0/0/0 (44)

IP 172.21.110.24 > LAPTOP-P7QFA4QB.mshome.net: ICMP 172.21.110.24 udp port 60517 unreachable, length 80

ARP, Request who-has 172.21.110.24 (00:15:5d:f2:a4:d6 (oui Unknown)) tell LAPTOP-P7QFA4QB.mshome.net, length 28

ARP, Reply 172.21.110.24 is-at 00:15:5d:f2:a4:d6 (oui Unknown), length 28

IP LAPTOP-P7QFA4QB.mshome.net.domain > 172.21.110.24.42757: 45165 NXDomain 0/0/0 (90)
（會一直跑下去，以下省略）
```

這時候 tcpdump 其實就是在監控我這台電腦所有的網路介面，因為我用的是 WSL，WSL 背後就一直跟 Windows 交換資料，「172.21.110.24」是 WSL 對外的介面 IP 位置，「mshome.net.domain」則應該是 Windows 系統相關的服務。

## tcpdump 使用過濾器

把所有網路封包都印出來有點惱人，比方說我有好多個網路介面（假設電腦有五個 LAN 連接埠），每個介面可能都在做不一樣的事情，一個連印表機、一個連網際網路等等，就算是同一個介面也可能同時有好多網路程式在進行，我們只想要特定的網路封包該怎麼做呢？這時候我們可以下參數，讓 tcpdump 進行過濾。

例如我只監聽「eth0」介面上所有 443 埠的封包 ，我可以先輸入 `sudo tcpdump -i eth0 -n 'port 443'`，`-i`讓我們可以指定網路介面，`-n` 讓我們可以單純印出 IP 位置，後面的字串則是要過濾的條件。執行之後 tcpdump 會開始監聽，同時我們在另一個終端機上輸入 `curl https://tigercosmos.xyz/` 訪問一個 HTTPS 網站，就會看到 tcpdump 攔截到所有 443 埠的封包了。

```shell
$ sudo tcpdump -i eth0 -n 'port 443'
（等待輸入curl指令）
IP 172.21.110.24.33002 > 185.199.110.153.443: Flags [S], seq 1314411051, win 64240, length 0

IP 185.199.110.153.443 > 172.21.110.24.33002: Flags [S.], seq 1168621290, ack 1314411052, win 65535, length 0

IP 172.21.110.24.33002 > 185.199.110.153.443: Flags [.], ack 1, win 502, length 0

IP 172.21.110.24.33002 > 185.199.110.153.443: Flags [P.], seq 1:518, ack 1, win 502 , length 517

IP 185.199.110.153.443 > 172.21.110.24.33002: Flags [.], ack 518, win 285, length 0
（以下省略）
```

我們可以去做更複雜的過濾，例如條件為（1）在「eth0」介面、（2）TCP 協議封包、（3）來源是「192.168.3.100」、（4）3553 埠：

```shell
$ sudo tcpdump -i eth0 -n 'tcp and src host 192.168.3.100 and port 3553'
```

或是我們想要監控 ARP 協議封包、目標是「192.168.1.12」：

```shell
$ sudo tcpdump -i eth0 'arp and dst host 192.168.1.12'
```

tcpdump 可以有各式各樣的過濾條件，端看使用需求而定，更詳細的用法可以使用 `man tcpdump 來查看。

## tcpdump 案例

最後舉一個 tcpdump 應用的場景，比方說有個監控攝影機會固定發送 **UDP 廣播**（UDP broadcast），我們寫的程式不知道為什麼收不到攝影機發送的 UDP 廣播封包，這時候我們就可以用 tcpdump 來監控一下，看是不是真的有網路封包被送出。有可能是我們自己程式寫錯，比方說程式沒有**連接**（bind）正確的網路介面；或是網路介面設定不正確，例如這個攝影機只接受從「192.168.120.1/24」子網域的連線，結果我們發現網路介面設定成「192.168.4.1」，那當然收不到封包；也很有可能，我們發現原來網路線根本沒插好！總之這種時候 tcpdump 可以幫助我們去釐清到底根本原因是什麼。

<div style="display: flex;justify-content: center;align-items: center;">
<blockquote style="border: 5px solid #ee9c6b;border-radius:30px; width:80%;">

欲閱讀更多內容，請參閱[《程式設計原來不只有寫 CODE！銜接學校與職場的五堂軟體開發實習課》](/books/beyond-just-coding-book.html)介紹頁面。
<a href="/books/beyond-just-coding-book.html" style="display: flex;justify-content: center;align-items: center;"><img src="https://raw.githubusercontent.com/tigercosmos/beyond-just-coding-book/refs/heads/master/book_picture.jpg" width="100%" alt="「程式設計原來不只有寫 CODE！銜接學校與職場的五堂軟體開發實習課」書籍封面" style="max-width:350px"></a>

</blockquote>
</div>
