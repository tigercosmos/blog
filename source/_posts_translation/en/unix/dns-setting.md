---
title: "Configure DNS on Linux/Unix from the Command Line"
date: 2020-04-14 11:01:00
tags: [unix, network, dns, note, squid]
lang: en
translation_key: dns-setting
---

Here’s what happened: starting yesterday, a machine in my lab suddenly couldn’t access the Internet. I usually use that machine as a proxy, because sometimes I need an NCTU IP address for looking up papers. At first I thought Squid was broken, so I spent a long time tweaking the configuration—only to realize the issue didn’t seem to be Squid. Then I was shocked to find that I could SSH into the machine, but it couldn’t connect outbound.
<!-- more -->

None of the following commands worked:

```sh
$ ping google.com
$ wget google.com
```

Eventually I found out it was a DNS issue. I think I was only using `1.1.1.1` before, and for some reason it stopped working, so I needed to switch to a different resolver.

The fix is to edit `/etc/resolv.conf`:

```sh
$ sudo vim /etc/resolv.conf
```

Then add the following content:

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

You can set multiple `nameserver` entries. If one fails, it will fall back to the next one. You can also pick only the resolvers you want—there’s no need to include all of them.

After that, you can check whether the network is working again:

```sh
$ ping google.com
$ dig google.com
$ sudo apt update
```

If everything is fine, these commands should produce output. If not, you may want to double-check whether `route` is configured correctly—or it could be one of many other weird issues XD

