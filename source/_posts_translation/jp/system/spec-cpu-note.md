---
title: "SPEC CPU 2006 のインストールと実行"
date: 2020-06-16 00:00:00
tags: [spec cpu, note]
des: "本記事では Ubuntu 16.04 上で SPEC CPU 2006 v1.1 をインストールして実行する方法を紹介します。いくつかのエラーが発生することを前提に、その解決方法を記録します。"
lang: jp
translation_key: spec-cpu-note
---

## 前書き

SPEC CPU は The Standard Performance Evaluation Corporation（SPEC）が策定した CPU 性能評価指標（Benchmark）です。近年の指標には SPEC CPU 2006 と SPEC CPU 2017 があり、公信力が非常に高いため、多くの研究で評価指標として使われています。

この指標を入手するには、ネット上で無料で手に入るものではなく、SPEC から購入する必要があり高価です。使う必要がある場合は、研究室や共同研究者経由で何とか入手することになります。最近、SPEC CPU 2006/2017 を使った論文を再現する必要があり、研究室にたまたま 2006 版がありました。ただ、使う過程で落とし穴が多かったので記録しておきます。

<!-- more -->

## 環境

私は Docker で環境を作ってテストしました。

本記事は Ubuntu 16.04 をベースに、公式 Docker Image から構築しています。SPEC CPU は v1.1（2008）を使用します。この年代は少しレトロなので、GCC は 4/5/6 あたりのバージョンを推奨します。サーバー CPU は Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz です。

可能なら、環境をむやみに変えないことを強くおすすめします。こういう巨大なものは、環境が変わるだけで簡単にいろいろハマります。

## ソースの準備

SPEC CPU 2006 は元々ディスクの ISO として提供され、ファイル名は `SPEC_CPU2006_v1.1.iso` です。そのため公式手順では ISO をマウントして作業します。

ただし私は Docker を使っていてマウントが難しかったので、単純に解凍して使いました。

```shell
$ sudo apt-get install p7zip-full p7zip-rar
$ mkdir target_dir && cd target_dir
$ mv spec.iso target_dir && 7z x spec.iso
```

## ツールのビルド

ISO にはいくつかのプラットフォーム向けの実行ファイルが含まれていますが、Ubuntu の場合は自分でビルドする必要があります。

```shell
$ cd tools/src
$ ./buildtools
```

ビルド中にいくつかエラーが出るので、順番に解決していきます。

## 想定されるエラー

### Conflicting types for ‘getline’

エラーメッセージ：

```shell
In file included from md5sum.c:38:0:
lib/getline.h:31:1: error: conflicting types for 'getline'
 getline PARAMS ((char **_lineptr, size_t *_n, FILE *_stream));
 ^
```

`getline` と `getdelim` が `getline.h` と `stdio.h` の両方で宣言されているのが原因です。次の 2 行を追加します：

```diff
+# if __GLIBC__ < 2
 int
 getline PARAMS ((char **_lineptr, size_t *_n, FILE *_stream));

 int
 getdelim PARAMS ((char **_lineptr, size_t *_n, int _delimiter, FILE *_stream));

+#endif
```

### Undefined reference to `pow`

エラーメッセージ：

```shell
libperl.a(pp.o): In function `Perl_pp_pow':
pp.c:(.text+0x2a76): undefined reference to `pow'
```

ビルド時に `libm` をリンクします：

```shell
$ PERLFLAGS="-A libs=-lm -A libs=-ldl" ./buildtools
```

### You haven’t done a “make depend” yet!

エラーメッセージ：

```
You haven't done a "make depend" yet!
make[1]: *** [hash.o] Error 1
```

これは `/bin/sh` が変更されていることが原因です。次を実行してください。ビルド後は必要に応じて元に戻せます。

```shell
$ sudo rm /bin/sh
$ sudo ln -s /bin/bash /bin/sh
```

### `asm/page.h` file not found

エラーメッセージ：

```
SysV.xs:7:25: fatal error: asm/page.h: No such file or directory
```

`SysV.xs` を次のように変更します：

```diff
 #include <sys/types.h>
 #ifdef __linux__
-#   include <asm/page.h>
+#define PAGE_SIZE      4096
 #endif
```

### perl test fail

上記の問題をすべて解決した後でも、`$ PERLFLAGS="-A libs=-lm -A libs=-ldl" ./buildtools` を実行すると、900 個中 9 個程度のテストが失敗します（すべて perl 関連）。これは無視できるので、ここでは無視します。

ビルド後に次のように聞かれます：

```shell
Hey!  Some of the Perl tests failed!  If you think this is okay, enter y now:
```

**ここでは `y` を入力してください。** ただし時間制限があり、**反応が遅いと NO 扱いで進んでしまいます。** 私は一度これで全ビルドをやり直す羽目になりました。本当に時間がかかります。

すべて成功すると次の表示が出ます：

```shell
Tools built successfully.  Go to the top of the tree and
source the shrc file.  Then you should be ready.
```

## 実行

### 設定ファイル

理論上は `config` 配下に複数のサンプル `.cfg` があり、環境に合うものを選べます。また `config/flags` 配下にもサンプルの flags 設定ファイルがあるはずですが、例外もあります。たとえば私が入手したものには含まれていなかったので、環境に合う設定ファイルを [SPEC 公式サイト](https://www.spec.org/cpu2006/flags/) から取得する必要があります。

### 設定の有効化

```shell
$ source ./shrc
```

### 実行

ベンチマークは `runspec` で実行します。前の手順で `source` したときに、環境変数として取り込まれます。

例として `mcf` を実行する場合、私の設定ファイルは `Example-linux-ia64-gcc.cfg` で、次のように実行できます：

```shell
$  runspec --iterations 1 --size ref \
           --action onlyrun \
           --config Example-linux-ia64-gcc.cfg \
           --noreportable mcf
```

実行結果は次のようになります：

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

結果を得た後は、さらにシステムの性能分析を行えますが、それは本記事の範囲外とします。

## 結論

本記事では Ubuntu 16.04 上で SPEC CPU 2006 v1.1 をインストールして実行する方法を紹介しました。いくつかのエラーが発生することを前提に、その解決方法を記録しました。

## 参考資料

- [Install / execute spec cpu2006 benchmark](https://sjp38.github.io/post/spec_cpu2006_install/)
