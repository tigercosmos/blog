---
title: "An Example of Signals with `sigaction` on Unix"
date: 2019-11-24 11:01:00
tags: [unix, network programming, signal, sigaction]
lang: en
translation_key: sigaction
---

There are many ways for processes to communicate on Unix. This post introduces a simple way to use signals. You can first read [Beej’s introduction](http://beej.us/guide/bgipc/html/single/bgipc.html#signals). As the name suggests, a signal is a notification sent and received by processes. For example, when you use a shell, `Ctrl-C` can interrupt a program because the shell catches the `SIGINT` signal triggered by `Ctrl-C`, recognizes it as an interrupt signal, and terminates the program.

To send signals, you can use [`sigaction()`](http://man7.org/linux/man-pages/man2/sigaction.2.html) or [`signal()`](http://man7.org/linux/man-pages/man7/signal.7.html). I recommend `sigaction` because it’s newer. For a detailed comparison, see the Stack Overflow thread “[What is the difference between sigaction and signal?](https://stackoverflow.com/questions/231912/what-is-the-difference-between-sigaction-and-signal)”.

If you want to trigger a signal, you can use [`kill()`](http://man7.org/linux/man-pages/man2/kill.2.html) or [`sigqueue()`](http://man7.org/linux/man-pages/man3/sigqueue.3.html). The difference is that the latter is Linux-only, but it lets you attach additional information via `siginfo`. In addition, some system calls can trigger signals as well. For example, if you try to `send()` to a non-existent socket, you’ll get a `SIGPIPE`.

<!-- more -->

Below is a simple example. If you want to understand the full set of parameters, you still need to read the man pages:

<pre><code class="c++">// signal.cc
#include &lt;signal.h&gt;
#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;string.h&gt;
#include &lt;unistd.h&gt;

int main(int argc, char **argv) {
  // 宣告 sigaction 物件
  struct sigaction act;
  memset(&act, 0, sizeof act);

  // 建立 signal handler，應該盡可能簡化 handler 要做的事情
  // 並且要小心 signal 執行時，原程式已經在做了一樣的事情
  act.sa_sigaction = [](int signo, siginfo_t *info, void *context) {
    if (signo == SIGINT) { // 接收 Ctrl-C
      printf("Received SIGINT.\n");
    } else if (signo == SIGUSR1) { // 接收數字
      int int_val = info-&gt;si_value.sival_int;
      printf("recieve: %d\n", int_val);
    } else if (signo == SIGUSR2) { // 接收指標
      void *ptr = info-&gt;si_value.sival_ptr;
      printf("recieve: %p\n", ptr);
    }
  };

  // sa_sigaction 必須搭配 SA_SIGINFO，否則會呼叫 act.sa_handler
  // 但我們要 sa_sigaction 才能接收 siginfo
  //
  // 其他參數：
  //  重新啟動可被中斷的 system call 要設 SA_RESTART
  //  SIGCHLD 可以設 SA_NOCLDWAIT 來避免疆屍
  act.sa_flags = SA_SIGINFO;

  // 設定 SIGUSR1
  if (0 != sigaction(SIGUSR1, &act, NULL)) {
    perror("sigaction () failed installing SIGUSR1 handler");
    return EXIT_FAILURE;
  }
  // 設定 SIGUSR2
  if (0 != sigaction(SIGUSR2, &act, NULL)) {
    perror("sigaction() failed installing SIGUSR2 handler");
    return EXIT_FAILURE;
  }

  // 設定 SIGINT
  if (0 != sigaction(SIGINT, &act, NULL)) {
    perror("sigaction() failed installing SIGUSR2 handler");
    return EXIT_FAILURE;
  }

  // 執行 15 秒，過程中可以嘗試按 Ctrl-C
  for (int i = 1; i &lt;= 15; i++) {
    // 第二秒觸發 SIGUSR1 並送出 123
    if (i == 2) {
      union sigval value;

      // 目標 process 的 pid，這邊要送給自己，為自己的 pid
      int pid = getpid();

      // sigval 是 union，sival_int 或 sival_ptr 只能擇一
      value.sival_int = 123;

      if (sigqueue(pid, SIGUSR1, value) == 0) {
        printf("signal sent successfully!!\n");
      } else {
        perror("SIGSENT-ERROR:");
      }
    }

    // 第二秒觸發 SIGUSR2 並送出 act 的地址
    if (i == 4) {
      union sigval value;
      pid_t pid = getpid();

      value.sival_ptr = &act;
      if (sigqueue(pid, SIGUSR2, value) == 0) {
        printf("signal sent successfully!!\n");
      } else {
        perror("SIGSENT-ERROR:");
      }
    }

    // 第二秒觸發 SIGUSR1 並送出 3333
    if (i == 6) {
      union sigval value;
      int pid = getpid();

      value.sival_int = 3333;
      if (sigqueue(pid, SIGUSR1, value) == 0) {
        printf("signal sent successfully!!\n");
      } else {
        perror("SIGSENT-ERROR:");
      }
    }

    printf("Tick #%d.\n", i);

    sleep(1);
  }

  return EXIT_SUCCESS;
}

</pre></code>

The output looks like this:

<pre><code class="shell">$ g++ signal.cc
$ ./a.out
Tick #1.
recieve: 123
signal sent successfully!!
Tick #2.
Tick #3.
^CReceived SIGINT.
recieve: 0x7ffde08e7970
signal sent successfully!!
Tick #4.
Tick #5.
^CReceived SIGINT.
recieve: 3333
signal sent successfully!!
Tick #6.
^CReceived SIGINT.
Tick #7.
Tick #8.
^CReceived SIGINT.
Tick #9.
^CReceived SIGINT.
Tick #10.
Tick #11.
^CReceived SIGINT.
Tick #12.
Tick #13.
Tick #14.
Tick #15.
</pre></code>

