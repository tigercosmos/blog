---
title: Linux/Unix 用指令設定 DNS 
date: 2020-04-14 11:01:00
tags: [unix, network, dns, note, squid]
lang: zh
translation_key: dns-setting
---

事情是這樣的，從昨天開始我實驗室的主機突然不能上網了，平常我都用那台當作 proxy，因為有時候查論文需要用交大 IP。然後我一開始以為是 Squid 出問題，在那邊弄了老半天設定，才發現問題好像也不是 Squid。接著我很震驚地發現，原來我可以 SSH 進主機，卻不能從他連出去。
<!-- more -->

輸入以下指令通通沒用：

```sh
$ ping google.com
$ wget google.com
```

然後才發現原來是 DNS 的關係，我原本好像只有用 `1.1.1.1`，不知道為啥不能用了，所以要換一個。

解決辦法是修改 `/etc/resolv.conf`：

```sh
$ sudo vim /etc/resolv.conf
```

然後在裡面內容加上：

```py
# OpenDNS
nameserver 208.67.222.222
nameserver 208.67.220.220
# Google
nameserver 8.8.8.8
nameserver 8.8.4.4
# Cloudflare
nameserver 1.1.1.1
nameserver 1.0.0.1
```

`nameserver` 可以一次設多個，當一個失敗時可以換下一個，然後可以挑你想要的設定就好，不一定要全放。

接著就可以來看網路有沒有順利連通了：

```sh
$ ping google.com
$ dig google.com
$ sudo apt update
```

順利的話應該就會有東西出來了，沒有的話可能要檢查一下 `route` 有沒有設定好，或是也有可能是各種奇怪的問題 XD
