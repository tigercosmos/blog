---
title: "Linux/Unix でコマンドから DNS を設定する"
date: 2020-04-14 11:01:00
tags: [unix, network, dns, note, squid]
lang: jp
translation_key: dns-setting
---

経緯はこうです。昨日から研究室のマシンが突然インターネットに繋がらなくなりました。普段そのマシンを proxy として使っていて、論文を調べるときに交大（NCTU）の IP が必要なことがあるからです。最初は Squid が壊れたのだと思い、設定をいじってかなり時間を使ってしまいましたが、どうやら問題は Squid ではなさそうでした。さらに驚いたのは、SSH でそのマシンには入れるのに、そこから外向きに通信できないという点です。
<!-- more -->

以下のコマンドを入力しても全部ダメでした：

```sh
$ ping google.com
$ wget google.com
```

そして原因は DNS だと分かりました。もともと `1.1.1.1` だけを使っていた気がするのですが、なぜか使えなくなったので、別のものに切り替える必要がありました。

解決方法は `/etc/resolv.conf` を編集することです：

```sh
$ sudo vim /etc/resolv.conf
```

中身に以下を追加します：

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

`nameserver` は複数指定できます。1 つが失敗した場合は次のものにフォールバックします。また、好きなものだけを選べばよく、全部入れる必要はありません。

あとはネットワークが正常に疎通できるか確認します：

```sh
$ ping google.com
$ dig google.com
$ sudo apt update
```

うまくいけば何かしら出力が出るはずです。出ない場合は `route` の設定が正しいか確認したり、あるいは他にも色々と怪しい原因があるかもしれません XD

