---
title: SPEC CPU 2006 安裝執行
date: 2020-06-16 00:00:00
tags: [spec cpu, note]
des: "這篇介紹怎樣在 Ubuntu 16.04 上安裝執行 SPEC CPU 2006 v1.1，預期會有一些錯誤，這篇指出該怎樣去解決。"
lang: zh
translation_key: spec-cpu-note
---

## 前言

SPEC CPU 是 The Standard Performance Evaluation Corporation (SPEC) 組織建立的 CPU 效能測試指標 (Benchmark)。最近的指標有 SPEC CPU 2006 和 SPEC CPU 2017，因為具有極大公信力，許多研究都會以這套指標當作評量測試。

如果想要弄到這套指標，網路上是找不到的，是需要跟這機構購買，而且要價不斐，要用的話就是想辦法靠關係弄一套來。我最近要重現一篇論文，裡面有用到 SPEC CPU 2006/2017，實驗室剛好有 2006 年版，但在使用過程中滿多坑的，為此紀錄。

<!-- more -->

## 環境

我個人是用 Docker 開環境來測試。

這篇基於 Ubuntu 16.04，以官方 Docker Image 建構。使用 SPEC CPU v1.1 (2008)，這個年代有點復古，所以推薦使用 GCC 4/5/6 左右版本。伺服器架構為 Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz。

誠心建議情況允許的話不要隨便換環境，不然這種龐然大物隨便都一堆坑。

## 原始碼處理

SPEC CPU 2006 原本是燒在光碟中的，檔名是 `SPEC_CPU2006_v1.1.iso`，所以官方教學是掛載 `iso` 檔處理。

不過因為我用 Docker 沒法做掛載，我就直接解壓縮使用。

```shell
$ sudo apt-get install p7zip-full p7zip-rar
$ mkdir target_dir && cd target_dir
$ mv spec.iso target_dir && 7z x spec.iso
```

## 建構工具

原始的 `iso` 裡面已經有一些平台的可執行檔，但以 Ubuntu 來說就需要自己編譯。

```shell
$ cd tools/src
$ ./buildtools
```

在編譯過程中會發生一些錯誤，要依序解決。

## 預期發生的錯誤

### Conflicting types for ‘getline’

錯誤訊息

```shell
In file included from md5sum.c:38:0:
lib/getline.h:31:1: error: conflicting types for 'getline'
 getline PARAMS ((char **_lineptr, size_t *_n, FILE *_stream));
 ^
```

這是因為 `getline` 和 `getdelim` 被同時宣告在 `getline.h` 和 `stdio.h`，補上下面兩行：

```diff
+# if __GLIBC__ < 2
 int
 getline PARAMS ((char **_lineptr, size_t *_n, FILE *_stream));

 int
 getdelim PARAMS ((char **_lineptr, size_t *_n, int _delimiter, FILE *_stream));

+#endif
```

### Undefined reference to `pow`

錯誤訊息：

```shell
libperl.a(pp.o): In function `Perl_pp_pow':
pp.c:(.text+0x2a76): undefined reference to `pow'
```

編譯時讓他 link `libm`：

```shell
$ PERLFLAGS="-A libs=-lm -A libs=-ldl" ./buildtools
```


### You haven’t done a “make depend” yet!

錯誤訊息：

```
You haven't done a "make depend" yet!
make[1]: *** [hash.o] Error 1
```

這是因為 `/bin/sh` 被動過，執行以下即可，編譯後可以自己再變回來。

```shell
$ sudo rm /bin/sh
$ sudo ln -s /bin/bash /bin/sh
```

### `asm/page.h` file not found

錯誤訊息：

```
SysV.xs:7:25: fatal error: asm/page.h: No such file or directory
```

解決方法是改動 `SysV.xs` 檔案：

```diff
 #include <sys/types.h>
 #ifdef __linux__
-#   include <asm/page.h>
+#define PAGE_SIZE      4096
 #endif
```

### perl test fail

解決所有上述問題之後，執行 `$ PERLFLAGS="-A libs=-lm -A libs=-ldl" ./buildtools` 應該會有大約 9/900 所右的測試失敗，都是 perl 相關的，但因為他們是可以忽略的，我們就不管了。

跑完他會問你：

```shell
Hey!  Some of the Perl tests failed!  If you think this is okay, enter y now:
```

**這邊就輸入 `y` 即可**，要注意的是，他有時間限制，**太久沒回應他他就會自己當作 NO**。結果害我又重新跑了全部編譯流程一次，這真的超花時間的。

全部成功之後應該就會看到

```shell
Tools built successfully.  Go to the top of the tree and
source the shrc file.  Then you should be ready.
```

## 執行

### 設定檔

理論上 `config` 資料夾底下應該會有不少範例的 `.cfg`，你可以根據你的系統環境選一份。此外 `config/flags` 底下應該也會有範例 flags 設定檔，但也可能有例外，像我拿到的這份就沒有，所以你要去 [SPEC 官網](https://www.spec.org/cpu2006/flags/)上抓符合你環境的設定檔。 

### 啟用設定

```shell
$ source ./shrc
```

### 執行

`runspec` 用來跑 Benchmark，在上一步 `source` 的時候會納入環境變數中。

舉例來說要跑 `mcf` 這支測試，我的設定檔是 `Example-linux-ia64-gcc.cfg`，可以執行以下：

```shell
$  runspec --iterations 1 --size ref \
           --action onlyrun \
           --config Example-linux-ia64-gcc.cfg \
           --noreportable mcf
```

跑出來的結果：

```shell
Reading config file '/work/spec/config/Example-linux-ia64-gcc.cfg'
Benchmarks selected: 429.mcf
Compiling Binaries
  Up to date 429.mcf base ia64-gcc42 default


Setting Up Run Directories
  Setting up 429.mcf ref base ia64-gcc42 default: existing (run_base_ref_ia64-gcc42.0000)
Running Benchmarks
  Running 429.mcf ref base ia64-gcc42 default
/work/spec/bin/specinvoke -d /work/spec/benchspec/CPU2006/429.mcf/run/run_base_ref_ia64-gcc42.0000 -e speccmds.err -o speccmds.stdout -f speccmds.cmd -C

Run Complete

The log for this run is in /work/spec/result/CPU2006.018.log

runspec finished at Tue Jun 16 01:21:05 2020; 379 total seconds elapsed
```

得到結果後可以進一步做系統效能分析，不過不在此篇討論範圍內。

## 結論

這篇介紹怎樣在 Ubuntu 16.04 上安裝執行 SPEC CPU 2006 v1.1，預期會有一些錯誤，本文指出該怎樣去解決。

## 參考資料

- [Install / execute spec cpu2006 benchmark](https://sjp38.github.io/post/spec_cpu2006_install/)
