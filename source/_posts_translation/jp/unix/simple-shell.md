---
title: "Shell の仕組みと簡単な実装"
date: 2020-01-18 15:01:00
tags: [unix, shell, fork, dup2, pipe]
des: "本記事では Shell の仕組みと、簡単な Shell プログラムの書き方を紹介します。"
lang: jp
translation_key: simple-shell
---

## Shell 紹介

ローカルの Unix、Windows、あるいはサーバーへ接続するときでも、私たちは端末（Terminal）を開きます。ログイン直後に表示される画面が Shell で、開発者はここにコマンドを入力してプログラムを実行します。ある意味では、「端末を使う」ということは「Shell を使う」とほぼ同義に語られることもあります（厳密には元々の意味は一致しません）。Shell の詳細は英語版 Wikipedia の「[C shell](https://en.wikipedia.org/wiki/C_shell)」も参考になります。

<!-- more --> 

<img width="500" alt="shell image" src="https://user-images.githubusercontent.com/18013815/72660990-4b927a80-3a10-11ea-9b89-d6971cf200b1.png">

Unix を普段使っていると、端末から様々なコマンドを実行します。例えば `ls` はカレントディレクトリ内のファイルを表示し、`cat` はファイル内容を表示し、`grep` は文字列の抽出に使えます。あるいは `g++ main.cc` や `node index.js` のようにコンパイルや実行をします。

Shell が強力な理由の 1 つは、日常的な作業を素早く片付けられる点です。例えば、今 Chrome がどんなプロセスを起動しているか知りたければ `ps au | grep chrome` を実行します。ここで `ps` はプロセス一覧を出し、`grep` は `chrome` に一致する行を抽出します。`|` は「pipe」で、`A | B` は A の出力を B に渡すことを意味します。B は stdin を受け取って処理を開始します。

さらに、Chrome のプロセスを全部止めたいなら `ps au | grep chrome | awk -F ' ' '{print $2}' | xargs kill` のようにできます。平たく言うと、`ps` でプロセス一覧を出し、出力を `grep` に渡して `chrome` を含む行を抽出し、出力を `awk` に渡して 2 列目（PID）だけ取り出し、最後に `xargs` を使って各 PID に対して `kill` を実行します。結果として Chrome のプロセスがすべて終了します。

このように 1 行で多くのことをしています。一般的なプログラミング言語（C++ や Python など）で同じことをやろうとすると、何十行も書かないと実現できませんが、Shell と pipe を使えば 1 行でできます。Shell を使った効率的な開発に興味があれば、「[打造高效的工作环境 – SHELL 篇](https://coolshell.cn/articles/19219.html)」も参考になります。

Shell には次のような機能があります：

- ワイルドカード（Wildcarding）：例 `rm *.cpp`
- I/O リダイレクト（Redirection）
  - `>` stdout をファイルへ出力
  - `>&` stdout と stderr をファイルへ出力
- コマンド結合（Joining）
  - `A && B`：A が成功したら B を実行
  - `A || B`：A が失敗したら B を実行
- パイプ（Piping）
  - `A | B`：A と B を同時に実行し、B が A の stdout を受け取る
  - `A |& B`：A と B を同時に実行し、B が A の stdout と stderr を受け取る
- そのほか変数、簡単な制御構造、バックグラウンド実行など。

この中でも特に重要なのは I/O と Piping です。Shell が強力なのは、多数のコマンドを連結し、input と output を 1 本のパイプライン（Pipeline）として繋げられるからです。さらに、前段の出力を次段がそのまま input として受け取れるため、前段が完全に終了するのを待たずに次段が動き始められ、効率面でも有利になります。

## Shell の仕組み

では Shell はどのようにしてプログラムを実行するのでしょうか？

Shell（`pid 0`）が `ls` のようなコマンドを受け取ると、まず `fork()` で新しいプロセス（`pid 1`）を作ります。続いて `pid 1` は `exec` 系の関数を呼び、fork された Shell を `ls` に置き換えて実行します。このとき Shell（`pid 0`）は `waitpid()` で `ls`（`pid 1`）が終了して出力を終えるのを待ち、終わったら次の入力へ進みます。

Shell:

```shell
$            # pid 0
-----------------------------------------------------
$ ls         # pid 0 fork() 出 pid 1
A B C D      # pid 1 執行 ls
-----------------------------------------------------
$            # pid 0 用 waitpid() 等 pid 1 結束才繼續
```

次に知っておくべきなのが `pipe()` という system call です。これはプロセス間通信の手段の 1 つで、Shell を実装するときには `A | B` を実現するために `pipe()` と `dup2()` を使います。先に次の 3 つの記事を読んでから、以下の例を見てください。

- [pipe() System call](https://www.geeksforgeeks.org/pipe-system-call/)
- [C program to demonstrate fork() and pipe()](https://www.geeksforgeeks.org/c-program-demonstrate-fork-and-pipe/)
- [dup() and dup2() Linux system call](https://www.geeksforgeeks.org/dup-dup2-linux-system-call/)

## Shell の例

`g++ shell.cpp && ./a.out` を実行します。

<pre><code class="c++">// shell.cpp
#include &lt;errno.h&gt;
#include &lt;fcntl.h&gt;
#include &lt;iostream&gt;
#include &lt;signal.h&gt;
#include &lt;stdio.h&gt;
#include &lt;sys/wait.h&gt;
#include &lt;unistd.h&gt;

int main(int argc, char **argv) {

  // 處理 SIGCHLD，可以避免 Child 疆屍程序
  struct sigaction sigchld_action = {.sa_handler = SIG_DFL,
                                     .sa_flags = SA_NOCLDWAIT};

  // 原本指令 ls | cat | cat | cat | cat | cat | cat | cat | cat
  // 假設 Shell 已經將指令 Parse 好

  char **cmds[9];

  char *p1_args[] = {"ls", NULL};
  cmds[0] = p1_args;

  char *p2_args[] = {"cat", NULL}; // 只是 DEMO，所以重複利用
  for (int i = 1; i &lt; 9; i++)
    cmds[i] = p2_args;

  int pipes[16]; // 需要共 8 條 pipe
  for (int i = 0; i &lt; 8; i++)
    pipe(pipes + i * 2); // 建立 i-th pipe

  pid_t pid;

  for (int i = 0; i &lt; 9; i++) {

    pid = fork();
    if (pid == 0) { // Child
      // 讀取端
      if (i != 0) {
        // 用 dup2 將 pipe 讀取端取代成 stdin
        dup2(pipes[(i - 1) * 2], STDIN_FILENO);
      }

      // 用 dup2 將 pipe 寫入端取代成 stdout
      if (i != 8) {
        dup2(pipes[i * 2 + 1], STDOUT_FILENO);
      }

      // 關掉之前一次打開的
      for (int j = 0; j &lt; 16; j++) {
        close(pipes[j]);
      }

      execvp(*cmds[i], cmds[i]);

      // execvp 正確執行的話，程式不會繼續到這裡
      fprintf(stderr, "Cannot run %s\n", *cmds[i]);

    } else { // Parent
      printf("- fork %d\n", pid);

      if (i != 0) {
        close(pipes[(i - 1) * 2]);     // 前一個的寫
        close(pipes[(i - 1) * 2 + 1]); // 當前的讀
      }
    }
  }

  waitpid(pid, NULL, 0); // 等最後一個指令結束

  std::cout &lt;&lt; "===" &lt;&lt; std::endl;
  std::cout &lt;&lt; "All done." &lt;&lt; std::endl;
}
</pre></code>

出力は次のようになります：

<pre><code class="shell">$ g++ shell.cpp && ./a.out
- fork 8244
- fork 8245
- fork 8246
- fork 8247
- fork 8248
- fork 8249
- fork 8250
- fork 8251
- fork 8252
FILE_A
FILE_B
FILE_C
===
All done.
</pre></code>

