---
title: 簡明 Shell 原理與實作
date: 2020-01-18 15:01:00
tags: [unix, shell, fork, dup2, pipe]
des: "本篇介紹 Shell 的原理，以及如何寫一個簡單的 Shell 程式。"
lang: zh
translation_key: simple-shell
---

## Shell 介紹

不管使用本地端 Unix、Windows 或是連線到伺服器，我們都會打開終端機（Terminal），而連線等入進去的起使畫面會是 Shell，然後開發者可以輸入指令來跑程式。某方面來說，當我們在講「使用終端機」就等同「使用 Shell」，即便他們原本代表意思其實不一樣。關於 Shell 的詳細介紹可以參考「[C Sehll](https://en.wikipedia.org/wiki/C_shell)」英文維基。

<!-- more --> 

<img width="500" alt="shell image" src="https://user-images.githubusercontent.com/18013815/72660990-4b927a80-3a10-11ea-9b89-d6971cf200b1.png">

試想我們平常在用 Unix 時，常常在終端機下指令，例如 `ls` 可以顯示當前目錄有哪些檔案，`cat` 可以顯示檔案內容，`grep` 可以擷取檔案內容。或是執行一些其他工作，比方說 `g++ main.cc`，或是 `node index.js` 之類的。

Shell 最強的地方是，一些平常的工作可以快速解決，像是查詢目前 chrome 開了哪些執行序，就可以下 `ps au | grep chrome`，其中 `ps` 是列出程序，`grep` 是擷取符合 `chrome` 的文字。`|` 是「pipe」，`A | B` 代表將 A 輸出的東西丟給 B，B 會收取 input 然後開始執行。 

如果想要終止所有 chrome 的執行序可以下 `ps au | grep chrome | awk -F ' ' '{print $2}' | xargs kill`，翻成白話文就是： `ps` 列出程序，輸出傳給 `grep`，`grep` 擷取有 chrome 片段，輸出傳給 `awk`，`awk` 只取第二欄為的資料（程序 pid），輸出傳給 `xargs`，`xargs` 針對每個 pid 執行 `kill`。於是所有 chrome 程序都被殺掉了。

可以看到我們做了許多事情，如果用一般的程式語言例如 C++ 或是 Python 之類的，不寫個幾十行是辦不到一樣的事情，但是用 Shell 搭配 pipe 就可以在一行內完成我們想做的事情。關於使用 Shell 做高效率開發可以參考這篇「[打造高效的工作环境 – SHELL 篇](https://coolshell.cn/articles/19219.html)」。

Shell 有很多功能，包含以下：

- 通配符（Wildcarding）：例如 `rm *.cpp`
- I/O 重新導向（Redirection）
  - `>` stdout 輸出到檔案
  - `>&` stdout 和 stderr 輸出到檔案
- 指令結合（Joining）
  - `A && B`：先執行 A，A 成功的話繼續執行 B
  - `A || B`：先執行 A，A 失敗的話繼續執行 B
- Piping
  - `A | B`：A 跟 B 同時執行，並且 B 會吃 A 的 stdout
  - `A |& B`：A 跟 B 同時執行，並且 B 會吃 A 的 stdout 和 stderr
- 剩下還包含變數、簡單程式邏輯、背景執行等等功能。

其中，就以 I/O 和 Piping 最為重要，Shell 之所以強大就是他可以串連許多指令，讓彼此的 input 和 output 建立一條流水線（Pipeline），在一行指令內就做到複雜的工作。不僅如此，因為上一個程序的 output 直接可以給下一個程序接收為 input，所以不用等上一個完全執行完，下一個程序就可以開始工作，因此效率也會比較高。

## Shell 原理

所以 Shell 是怎樣跑一個程式的呢？

Shell（`pid 0`）接到指令，比方說 `ls` 好了，他就會先 `fork()` 出一個新 process（`pid 1`），然後新的 `pid 1` 使用 `exec` 指令將 forked 的 Shell 取代成 `ls` 並執行。此時 Shell（`pid 0`）會用 `waitpid()` 等 `ls`（`pid 1`）執行完印出輸出，才繼續執行 Shell。

Shell:

```shell
$            # pid 0
-----------------------------------------------------
$ ls         # pid 0 fork() 出 pid 1
A B C D      # pid 1 執行 ls
-----------------------------------------------------
$            # pid 0 用 waitpid() 等 pid 1 結束才繼續
```

接下去，我們需要知道 `pipe()` 這個 System Call，他是一種讓程序之間可以溝通的方式之一，在實作 Shell 時，我們會需要用 `pipe()` 和 `dup2()` 來搞定 `A | B`。請大家先看「[pipe() System call](https://www.geeksforgeeks.org/pipe-system-call/)」、「[C program to demonstrate fork() and pipe()](https://www.geeksforgeeks.org/c-program-demonstrate-fork-and-pipe/)」和「[dup() and dup2() Linux system call](https://www.geeksforgeeks.org/dup-dup2-linux-system-call/)」這三篇文章，再接下去看我下面的範例。


## Shell 範例

執行 `g++ shell.cpp && ./a.out`

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

輸出會像這樣:

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
