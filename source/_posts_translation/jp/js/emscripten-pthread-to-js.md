---
title: "Emscripten で Pthread を JavaScript に変換し、性能を分析する"
date: 2020-07-07 18:00:00
tags: [JavaScript, web worker, nodejs, c, pthread, parallel programming, browser, browsers, 效能分析, 平行化]
des: "本記事では、Emscripten を使って C/C++ の Pthread を Web Worker と WebAssembly に変換する方法を紹介し、(1) ネイティブ C (2) Emscripten 生成の JS/WASM (3) JavaScript で直接書いた Web Worker の 3 つのケースで性能を比較します。-O3 最適化を有効にすると、Native C は Pthread→WASM より約 30% 速く、Pthread→WASM は純粋な Web Worker（JS）と概ね同程度です。"
lang: jp
translation_key: emscripten-pthread-to-js
---

## 概要

Emscripten は C/C++ を WebAssembly に変換できるツールです。裏側では LLVM を経由して変換し、Pthread の変換もサポートしています。Pthread は JavaScript の Web Worker と WebAssembly に変換されます。さらに OpenGL を WebGL に変換することもでき、ブラウザ上でネイティブに近い性能でプログラムを動かせます。

本記事の焦点は、Pthread を Web Worker + WebAssembly に変換する部分です。実際に例題プログラムを用意して変換してみます。ただ、良いテストプログラムを見つけるのは簡単ではないので、変換テスト用に π を計算する Pthread の平行プログラムを書きました。

まずは Emscripten で Pthread を JS に変換する方法を紹介します。公式ドキュメント通りに進める過程でいくつか落とし穴に遭遇したので、同じ罠に落ちないように記録しておきます。その後、(1) ネイティブ C (2) Emscripten が生成した JS/WASM (3) JavaScript で直接書いた Web Worker の 3 ケースで性能差を分析します。
<!-- more -->

## Pthread の例題プログラム

例題は、Pthread を使って π を平行に計算する小さなプログラムです。これを選んだ理由はいくつかありますが、平行プログラムで重要なのは「スレッドを作れるか」「共有メモリを使えるか」「ロックできるか」「他スレッドを待てるか」といった基本を確認することです。この π 計算の Pthread プログラムは、ちょうどそれらの要件を満たしています。

紙幅の都合上、Pthread の詳細は以前の記事「[簡易 Pthreads 平行化範例與效能分析](/post/2020/07/simple-pthread-usage/)」を参照してください。

pi.c:
```c
// pi.c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUMTHRDS 4
#define MAGNIFICATION 1e9

typedef struct
{
   int thread_id;
   int start;
   int end;
   double *pi;
} Arg;

pthread_t callThd[NUMTHRDS];
pthread_mutex_t mutexsum;

void *count_pi(void *arg)
{

   Arg *data = (Arg *)arg;
   int thread_id = data->thread_id;
   int start = data->start;
   int end = data->end;
   double *pi = data->pi;

   double x;
   double local_pi = 0;
   double step = 1 / MAGNIFICATION;
   for (int i = start; i < end; i++)
   {
      x = (i + 0.5) * step;
      local_pi += 4 / (1 + x * x);
   }

   local_pi *= step;

   pthread_mutex_lock(&mutexsum);
   *pi += local_pi;
   pthread_mutex_unlock(&mutexsum);

   printf("Thread %d did %d to %d:  local Pi=%lf global Pi=%.10lf\n", thread_id, start,
          end, local_pi, *pi);

   pthread_exit((void *)0);
}

int main(int argc, char *argv[])
{
   pthread_mutex_init(&mutexsum, NULL);

   pthread_attr_t attr;
   pthread_attr_init(&attr);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

   double *pi = malloc(sizeof(*pi));
   *pi = 0;

   int part = MAGNIFICATION / NUMTHRDS;

   Arg arg[NUMTHRDS];
   for (int i = 0; i < NUMTHRDS; i++)
   {
      arg[i].thread_id = i;
      arg[i].start = part * i;
      arg[i].end = part * (i + 1);
      arg[i].pi = pi;
      pthread_create(&callThd[i], &attr, count_pi, (void *)&arg[i]);
   }

   pthread_attr_destroy(&attr);

   void *status;
   for (int i = 0; i < NUMTHRDS; i++)
   {
      pthread_join(callThd[i], &status);
   }

   printf("Pi =  %.10lf \n", *pi);

   free(pi);

   pthread_mutex_destroy(&mutexsum);
   pthread_exit(NULL);
}
```

## Emscripten のダウンロード

Emscripten を使うには、まず GitHub のリポジトリを取得します。事前に Git をインストールしておいてください。

```shell
# emsdk repo を取得
git clone https://github.com/emscripten-core/emsdk.git

# ディレクトリに入る
cd emsdk
```

次に、手順に従って実行します：

```shell
# 最新 SDK をインストール
./emsdk install latest

# 最新 SDK を有効化
./emsdk activate latest

# 環境変数を設定
source ./emsdk_env.sh
```

一度インストール・有効化すれば、以降 Emscripten を使うたびに `emsdk` ディレクトリで `source ./emsdk_env.sh` を実行するだけで済みます。

毎回環境変数を設定したくない場合は `.bashrc` に `source ./emsdk_env.sh` を追加する方法もあります。ただし、この設定は NodeJS のパスを上書きすることがあるため、あまりおすすめしません。

## Emscripten の入門

`emcc` / `em++` は Emscripten のフロントエンドです（Clang と LLVM の関係のようなものです）。やや抽象的ですが、要するにこのコマンドで C/C++ を WebAssembly（WASM）へコンパイルできます。

簡単な例：

```c
// hello.c
#include <stdio.h>

int main() {
  printf("hello, world!\n");
  return 0;
}
```

コンパイル：

```shell
$ emcc hello.c
```

これにより `a.out.js` と `a.out.wasm` が生成されます。現在、WASM はブラウザでも NodeJS でも JavaScript による起動が必要なためです。

```shell
$ emcc hello.c -o hello.html
```

HTML を出力することもできます。この場合、ブラウザで開けるサンプルになり、ページ内の仮想ターミナルでプログラムの実行結果を確認できます。

注意点として、HTML を出力した場合はローカルサーバを立ててページを開く必要があります。これはブラウザが `file://` の XHR リクエストを許可しないためです。また Emscripten は HTTP ヘッダのファイルタイプにも要求があり、そこそこ面倒です。詳細は Emscripten 公式ドキュメントを参照してください。

より詳しい使い方は [Emscripten の Tutorial](https://emscripten.org/docs/getting_started/Tutorial.html#emscripten-tutorial) を参照してください。

## Emscripten で Pthread をコンパイルする

次に `pi.c` をコンパイルします。公式ドキュメントによると、Pthread をコンパイルするには `-s USE_PTHREADS=1` を付けます。

```shell
$ emcc pi.c -s USE_PTHREADS=1
```

この出力を実行してみましょう（NodeJS では `--experimental-wasm-threads --experimental-wasm-bulk-memory` が必要です）：

```
$ node  --experimental-wasm-threads --experimental-wasm-bulk-memory a.out.js

// 完全に無反応
```

はい、固まりました。

GitHub で質問して分かったのですが、これは Emscripten の仕組みに起因する問題です（詳細は私も完全には理解していません）。私が提出した [issue](https://github.com/emscripten-core/emscripten/issues/11543#issuecomment-654317178) を参照してください。要するに、解決策は 3 つあります：

1. `-s PROXY_TO_PTHREAD` を付けてコンパイルする
2. `-s PTHREAD_POOL_SIZE=N`（N > 0）を付けてコンパイルする
3. `main()` を `emscripten_set_main_loop()` に置き換える

ここでは次のようにコンパイルします：

```shell
$ emcc pi.c  -s USE_PTHREADS=1  -s PTHREAD_POOL_SIZE=4
```

これで正常に動きます！

```
$ node  --experimental-wasm-threads --experimental-wasm-bulk-memory a.out.js
Thread 1 did 250000000 to 500000000:  local Pi=0.874676 global Pi=0.8746757835
Thread 2 did 500000000 to 750000000:  local Pi=0.719414 global Pi=1.5940897827
Thread 0 did 0 to 250000000:  local Pi=0.979915 global Pi=2.5740044352
Thread 3 did 750000000 to 1000000000:  local Pi=0.567588 global Pi=3.1415926536
Pi =  3.1415926536」」
```

当初は Pthread の基本構文の多くが動かず、本当に焦りました。正直、ドキュメントに問題があると思いますし、開発者側からも改善が必要だとする [issue](https://github.com/emscripten-core/emscripten/issues/11554) が出ています。

## 性能分析

次は、同じ π 計算の平行ロジックについて、(1) ネイティブ pthread (2) pthread→WASM (3) JS の Web Worker の 3 ケースで性能差を測ります。

(1)、(2) は本記事の `pi.c` と Emscripten が生成した WASM を使い、(3) は以前の記事「[
Evaluation of Web Worker for Parallel Programming with Browsers, NodeJS and Deno](/post/2020/06/js/web-worker-evaluation/#NodeJS)」にある JS Web Worker のコードを使います。紙幅の都合で (3) のコードはここでは再掲しません。興味のある方は記事を参照してください。

3 つの実装は基本的に同じロジックで、総ループ回数も同一、スレッド数も同一です。実験環境は Windows 10 の WSL 1（Ubuntu 20.04）、CPU は AMD Ryzen 7 2700X 3.7 GHz（8 コアだが 4 スレッドのみ使用）です。NodeJS は v12.18、emcc は v1.39 を使用しました。gcc と emcc はどちらもデフォルトの -O2 最適化です。本記事の結果は、以前の [Evaluation of Web Worker](/post/2020/06/js/web-worker-evaluation/) とも比較できます（NodeJS の実行結果を基準にできます）。

唯一の違いは、(3) のコードが整数型で π を扱っている点です。SharedArrayBuffer を Atomics（ロック）と組み合わせる場合、整数型のバッファしか使えません。浮動小数点を扱うにはエンコードが必要ですが、本実験では型変換（エンコード）は行いません。


| Case  | Time(s)  |
|---|---|
|  pthread | 0.751  |
|  em2wasm | 1.174  |
|  js |  0.486 |

そして、JS 版がとんでもなく速いという結果になりました。信じがたいです！

後から考えると、整数演算は浮動小数点演算よりずっと速い可能性がありますし、精度も違います。そこで公平性のために、`pi.c` の `double* pi` を `unsigned* pi` に変更し、C 側もいったん拡大して π を計算し最後に割り戻すようにしました。このバージョンを `pi2.c` とします。JavaScript で浮動小数点を計算するときは `double` なので、元の C の `double` と精度は一致します。ところが、変更後の `pi2.c` の速度はほぼ変わらず、むしろ少し遅くなって 0.77s でした。

次に「mutex ロックが遅いのでは」と考えました。JS 側は Atomics を使っているので、`pi2.c` を `atomic_fetch_add_explicit` によるロックへ変更し、`pi3.c` としました。結果は 0.755s で、ほとんど差がありません。ただ、考えてみれば当然で、mutex は確かに遅いですが、差が顕著になるにはもっと倍率が必要です。

その後 `perf` で C コードを見たところ、次のようになっていました：

```shell
       │     local_pi += 4 / (1 + x * x);
       │       movsd     -0x8(%rbp),%xmm0
  0.02 │       mulsd     -0x8(%rbp),%xmm0
  0.02 │       movsd     _IO_stdin_used+0x60,%xmm1
  0.02 │       addsd     %xmm1,%xmm0
  0.13 │       movsd     _IO_stdin_used+0x68,%xmm1
       │       divsd     %xmm0,%xmm1
 13.20 │       movapd    %xmm1,%xmm0
  0.07 │       movsd     -0x28(%rbp),%xmm1
 34.67 │       addsd     %xmm1,%xmm0
 42.64 │       movsd     %xmm0,-0x28(%rbp) 
```

なるほど！以前の記事「[簡易 Pthreads 平行化範例與效能分析](/post/2020/07/simple-pthread-usage/)」でこの問題を指摘していたのを忘れていました。-O2 はなぜかメモリ処理に時間を使い過ぎる一方、-O3 ではその問題がありません。

そこで `pi3.c` を -O3 でコンパイルして試しました：

```shell
$ time gcc pi3.c -lpthread -g -O3
real    0m0.177s
$ time ./a.out
real    0m0.350s
```

実行時間だけ見ると 0.350s で、ようやく JS より速くなりました。しかし JS は JIT です。C のコンパイル時間 0.177s も含めると合計 0.527s になり、JS の 0.486s より遅いです。もう降参です。V8、強すぎる。いったいどうやってコンパイルしているんだ！

そこで、`pi2.c`（あえて pthread の mutex を測る）を使い、gcc と emcc の両方で -O3 を有効にしてもう一度計測しました：

| Case  | Time(s)  |
|---|---|
|  pthread | 0.346  |
|  em2wasm | 0.525  |
|  js |  0.504 |

ようやく期待通りの結果になりました。気持ちいい！

Pthread を Web Worker + WASM に変換した場合の性能は、JS で直接 Web Worker を書く場合よりほんの少し遅い程度です。それでも V8 は本当にすごいです。gcc と emcc を -O3 にするとコンパイル時間が長くなりますが、V8 の「コンパイル＋実行」時間は gcc の「コンパイル＋実行」より速いです。プログラムが小さいためだと推測しています。より大きなプログラムであれば、C と JS の性能差はより顕著になるはずです。

## 結論

本記事では、Emscripten を使って C/C++ の Pthread を Web Worker と WebAssembly に変換する方法を紹介し、(1) ネイティブ C (2) Emscripten 生成の JS/WASM (3) JavaScript で直接書いた Web Worker の 3 つのケースで性能を比較しました。-O3 最適化を有効にすると、Native C は Pthread→WASM より約 30% 速く、Pthread→WASM は純粋な Web Worker（JS）と概ね同程度です。

