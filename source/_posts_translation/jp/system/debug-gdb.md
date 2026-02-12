---
title: "GDB で Rust をデバッグする方法"
date: 2020-09-21 17:17:00
tags: [servo, debug, gdb, rust, english]
des: "本記事では、Rust のコードや Servo プロジェクトを開発・デバッグするために GDB を使うコツを紹介します。このデバッグ手法は C/C++ にも応用できます。"
lang: jp
translation_key: debug-gdb
---

## イントロダクション

Rust で書かれた Servo ブラウザのように、大規模なオープンソースプロジェクトは非常に巨大です。行数を数えたことがあるのですが、Servo プロジェクトには 10 万行近いコードがあります。こうした大規模プロジェクトを開発するには、正しい方法でデバッグできることがとても重要です。なぜなら、ボトルネックを素早く効率的に見つけたいからです。

本記事では、Rust のコードや Servo プロジェクトを開発・デバッグするために GDB を使うコツを紹介します。このデバッグ手法は C/C++ にも応用できます。

<!-- more --> 

## では、どうやってデバッグするのか？

ここでは、あなたがソフトウェア開発に詳しいわけではないものの、ある程度コードを書けることを前提にします。

「箱の中がどうなっているか」を知りたいとき、次のようにログを追加することがあります：

```
println!("{:?}", SOMETHING);
```

これはシンプルな方法です。十分に直感的で、コードの中で何が起きているかを理解する助けになります。

しかし、別の変数の値を知りたくなるたびに再コンパイルが必要になります。また、プログラムがクラッシュしたりメモリリークを起こしたりした場合、根本原因を追跡するのは難しくなります。

つまり、より強力なツールが必要です。Linux なら GDB、macOS なら LLDB を使えます。（Windows では Visual Studio のデバッガも非常に強力ですが、この記事では扱いません。）

ここでは GDB の使い方を説明します。LLDB は GDB と非常によく似ており、基本的にコマンドもほぼ同じなので、本記事では GDB を使って Rust と Servo をデバッグする方法を紹介します。

## GDB とは

> “GDB, the GNU Project debugger, allows you to see what is going on ‘inside’ another program while it executes — or what another program was doing at the moment it crashed.” — from gnu.org

言い換えると、GDB を使うと実行中のプログラムを制御し、コードの内部情報をより多く取得できます。

たとえば、ファイル内の特定の行でプログラムを停止できます。これを “breakpoint”（ブレークポイント）と呼びます。プログラムがブレークポイントで停止したら、そのスコープにある変数の値を表示（print）して確認できます。

また、ブレークポイントからバックトレース（backtrace）することもできます。バックトレースとは、ブレークポイントに到達するまでに呼び出された関数列を表示することです。クラッシュは必ずしも「クラッシュした行のコード」が原因とは限りません。もっと前に問題が起きていて、後から不正なパラメータが渡されてクラッシュすることもあります。

他にもいくつか使い方がありますが、以降で順に説明します。

## Rust を GDB でデバッグする

まずは Rust プロジェクトで GDB を使う方法を示すために、簡単な <a href="https://doc.rust-lang.org/book/second-edition/ch01-03-hello-cargo.html" target="_blank" >Hello World</a> を作ります。Rust と Cargo はすでにインストール済みですよね？

“the Rust book” の <a href="https://doc.rust-lang.org/book/second-edition/ch01-03-hello-cargo.html#creating-a-project-with-cargo" target="_blank" >steps</a> に従って Hello World を作成してください。コンパイルと実行ができて、Cargo が何をしているか理解できることを確認します。

プロジェクトの作成：

```
cargo new hello_cargo --bin
cd hello_cargo
```

では始めましょう！

GDB の使い方を示すため、サンプルコードを用意しました。次のコードを `./src/main.rs` にコピーしてください：

```
fn main() {
    let name = "Tiger";
    let msg = make_hello_string(&name);
    println!("{}", msg);
}

fn make_hello_string(name: &str) -> String {
    let hello_str = format!("Hi {}, how are you?", name);
    hello_str
}
```

このコードのビルドは `cargo build` を実行するだけです。

実行ファイルは `./target/debug/hello_cargo` に生成されます。デフォルトでは debug ビルドになっているため、この debug ビルドを GDB で実行できます。一方、release ビルドではデバッグ情報が失われるため、GDB でのデバッグはできません。

GDB で起動するには：

```
gdb target/debug/hello_cargo 
```

これで完了です。GDB の画面は次のようになります：

```
gdb (Ubuntu 8.1-0ubuntu3) 8.1.0.20180409-git
Copyright (C) 2018 Free Software Foundation, Inc.
...
(gdb)
```

これで GDB にコマンドを入力できるようになりました。

## C/C++ を GDB でデバッグする

C/C++ のコードを書く場合は、コンパイル時に `-g` フラグを付けてデバッグ情報を含めてビルドします。すると生成された実行ファイルを GDB で読み込めます。

```shell
$ gcc hello_world.c -g -o hello_world
$ gdb ./hello_world
```

## GDB コマンド

GDB には多くのコマンドがありますが、ここでは最も重要なものだけを紹介します。私の場合も、普段使うのはほとんどこれらです。

### break

先ほど述べた通り、ブレークポイントはプログラムを特定位置で停止させます。設定方法は 2 通りあります。

ブレークポイントを設定するコマンドは `break` または `b` です。

1 つ目は関数で止める方法です。（大規模プロジェクトでは `mod::mod::function` のようにフルパスが必要です）

```
(gdb) break make_hello_world
Breakpoint 1 at 0x55555555beca: file src/main.rs, line 8.
```

または、ファイルパスと行番号を指定して止める場所を定義できます。

```
(gdb) b src/main.rs:9
Breakpoint 2 at 0x55555555bf6a: file src/main.rs, line 9.
```

正しく設定できたか確認します。

```
(gdb) info break
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   0x000055555555beca in hello_cargo::make_hello_string at src/main.rs:8
2       breakpoint     keep y   0x000055555555bf6a in hello_cargo::make_hello_string at src/main.rs:9
```

今回はブレークポイントを 1 つだけにしたいので、`del` コマンドで 1 つ目を削除します。

```
(gdb) del 1
(gdb) info break
Num     Type           Disp Enb Address            What
2       breakpoint     keep y   0x000055555555bf6a in hello_cargo::make_hello_string at src/main.rs:9
```

### run

これでブレークポイントは 1 つだけになりました。では実行してみましょう。

開始は `run` です。

```
(gdb) run
Starting program: /home/tigercosmos/Desktop/hello_cargo/target/debug/hello_cargo 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Breakpoint 2, hello_cargo::make_hello_string (name=...) at src/main.rs:9
9	    hello_str
```

見ての通り、プログラムは狙った場所で停止しました。ここからブレークポイント上で操作できます。

### backtrace

このブレークポイントに到達するまでに、上流でどの関数が呼ばれてきたのか知りたい場合は `backtrace` または `bt` を使います。

```
(gdb) backtrace
#0  hello_cargo::make_hello_string (name=...) at src/main.rs:9
#1  0x000055555555bdd5 in hello_cargo::main () at src/main.rs:3
```

GDB は `main.rs` の 3 行目（#1）、つまり `let msg = make_hello_string(&amp;name);` が、`make_hello_string` に属する `main.rs` の 9 行目（#0）を呼び出したことを示しています。

「それは当たり前では？」と思うかもしれません。

確かにこの例ではそうです。しかし Servo のような巨大なオープンソースプロジェクトをデバッグしていて、あるモジュールのバックトレースを知りたいとしたらどうでしょうか。一般的にモジュール内では、ブレークポイントに到達するまでに 30〜40 個の関数呼び出しがあることもあります。コードを読むだけでバックトレースを把握するのは非常に難しいですが、GDB を使えば直接取得できます。

### frame

frame はバックトレースにおけるプログラム状態の 1 つです。切り替えたい frame を選び、その frame のスコープで情報を確認できます。

このコマンドは `frame` または `f` です。

前の手順で `src/main.rs:9` にブレークポイントを設定し、バックトレースには `#0` と `#1` の 2 つの frame が出ています。

ここでは frame `#1` に切り替えて、`src/main.rs:2` で宣言されている `name` の値を frame `#1` のスコープで見てみます。

```
#1  0x000055555555bdd5 in hello_cargo::main () at src/main.rs:3
3	    let msg = make_hello_string(&name);
(gdb) print name
$1 = "Tiger"
```

つまり `frame` を使うと、その frame のスコープに入れます。そこで `print` を使って変数の値を表示したり、`call` を使ってそのスコープの関数を呼び出したりできます。

では frame `#0` に切り替えて `hello_str` の値を見てみましょう。

```
(gdb) frame 0
#0  hello_cargo::make_hello_string (name=...) at src/main.rs:9
9	    hello_str
(gdb) print hello_str
$2 = alloc::string::String {vec: alloc::vec::Vec<u8> {buf: alloc::raw_vec::RawVec<u8, alloc::heap::Heap> {ptr: core::ptr::Unique<u8> {pointer: core::nonzero::NonZero<*const u8> (0x7ffff6c22060 "Hi Tiger, how are you?\000"), _marker: core::marker::PhantomData<u8>}, cap: 34, a: alloc::heap::Heap}, len: 22}}
```

### continue

必要な情報を確認したら、プログラムを続行したくなるはずです。続行するには `continue` または `c` を使います。プログラムは次のブレークポイントに到達するか、実行が終了するまで走り続けます。

```
(gdb) c
Continuing.
Hi Tiger, how are you?
```

この例ではブレークポイントが 1 つだけなので、そのまま最後まで実行されます。

### その他

ブレークポイントで停止した後、`step` で 1 行ずつ実行できます。

`frame <number>` を直接使う代わりに、`up` と `down` で frame を切り替えることもできます。

プログラムが実行中（すでに `run` を実行済み）のときに `Ctrl+C` を押すと、GDB を割り込んで即座に停止できます。プロセスはその時点の実行位置で一時停止します。割り込みによって止まった場所は「一度限りの手動ブレークポイント」のようなものです。その場で追加のコマンドを実行し、終わったら `c` で続行できます。

その後 `break`、`call`、`step` などのコマンドを使えます。

GDB には他にも多くのコマンドがあります。詳しくはドキュメントを参照してください。

## Servo をデバッグする

ここまでで Rust のデバッグ方法は分かりました。次は Servo プロジェクトをデバッグしてみましょう。ここでは Servo をコンパイルできることを前提にします。

debug モードでビルドするには：

```
$ ./mach build -d
```

ビルドが完了したら、Servo をデバッグします：

```
$ ./mach run -d https://google.com --debug
Reading symbols from /home/tigercosmos/servo/target/debug/servo...done.
(gbd)
```

これで Servo をデバッグできます。

## 結論

本記事の考え方は Rust プロジェクトだけでなく、C++ プロジェクトにも適用できます。Firefox や Chromium をデバッグしたい場合も同様の方法を使えます。GDB の複雑な設定方法などの詳細は扱わないため、より高度な内容は別の記事を探して学んでください。

GDB は開発者に必須ではありませんが、確実に「ハッカーの力」になります。Servo のコントリビューターにはフロントエンド開発者もいます。実はその中の一人は私の親友で、すでに Servo に多くのコミットをしています。フロントエンド開発者として彼はそれまで GDB を使ったことがありませんでした。しかし最近 Servo 開発で困っていて、もし GDB の使い方を知っていれば、プログラムの不可解な挙動の原因をより簡単に突き止められただろうと思います。

私は彼のため、そして Servo コミュニティに参加したばかりのすべての開発者のためにこの記事を書きました。少しでも役に立てば嬉しいです。良い hacking day を！ :)

---

### Special thanks

レビューを手伝ってくれた <a href="https://github.com/ko19951231" target="_blank" >@SouthRa</a> と <a href="https://github.com/cybai" target="_blank" >@cybai</a> に感謝します。
