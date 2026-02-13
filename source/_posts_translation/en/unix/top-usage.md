---
title: "A Detailed Guide to the Unix/Linux `top` Command"
date: 2020-04-16 15:01:00
tags: [unix, linux, shell, top command]
des: "This post explains how to use the Unix/Linux `top` command. `top` is one of the most fundamental tools on Unix/Linux: it is similar to Windows Task Manager and lets you monitor the execution status of all running programs. It is also one of the simplest ways to monitor a program—use it to observe memory usage, CPU usage, and various other metrics."
lang: en
translation_key: top-usage
---

## TOP Overview

The `top` command is one of the most fundamental tools on Unix/Linux. It is similar to Windows Task Manager and lets you monitor the execution status of currently running programs. `top` is also one of the simplest ways to monitor a program: you can use it to observe how much memory and CPU a program consumes, along with many other details. There are many similar tools (such as `htop` and `gtop`) which are extended versions. If you’re interested, you can look them up, but for basic needs, `top` is more than sufficient.
<!-- more -->

Using it is straightforward:

```sh
$ top
```

![top snapshot](https://user-images.githubusercontent.com/18013815/79328399-90af4580-7f48-11ea-9926-880e9f44f84b.png)

By default, the displayed columns include:
- `PID`: Process ID
- `USER`: The user running the task
- `PR`: Task priority
- `NI`: Nice value; negative means higher priority, positive means lower priority
- `VIRT`: Total virtual memory used (kB)
- `RES`: Resident memory size (kB)
- `SHR`: Total shared memory used (kB)
- `S`: Status
    - R: running
    - D: uninterruptible sleep (typically waiting for I/O; cannot be interrupted by signals)
    - S: sleep (interruptible / can be woken)
    - T: stopped or traced; may be stopped by `SIGSTOP` / `SIGTSTP`, or traced (e.g., by a debugger via ptrace)
    - Z: zombie; usually occurs when the child has finished and is waiting for the parent to wait()/reap it
- `%CPU`: CPU usage percentage. Note that one core is 100%, so on multicore systems this can exceed 100%.
- `%MEM`: Percentage of total memory used
- `TIME+`: Total CPU time used
- `COMMAND`: Command name

## TOP Interactive Mode

One way to use `top` is to enter the interactive UI and operate it there. Below I list some of the most commonly used keys. For the rest, please refer to the links in [參考資料](#%E5%8F%83%E8%80%83%E8%B3%87%E6%96%99). The recommended way to learn is to read the article while trying the operations yourself.

### Sorting tasks

Sorting tasks is probably the most important operation, because the common use case of `top` is to see which processes are consuming the most CPU and memory.

- `M`: sort by memory usage
- `N`: sort by PID
- `P`: sort by CPU usage
- `T`: sort by running time
- `>`/`<`: move the sort column to the right/left
- `R`: reverse sort

### General operations

The key thing here is: press `q` to quit 😂

- `h`/`?`: help
- Enter/Space: refresh the screen (default: every 3 seconds)
- `=`: clear task filters (after searching)
- `B`: bold display
- `d`/`s`: change refresh interval
- `I`: toggle between showing CPU usage percentage / showing all CPU cores
- `k`: kill a specific PID
- `L`: search for a string (use with `&`)
- `q`: quit
- `r`: renice a PID (positive values lower priority, negative values raise priority; requires sudo)
- `u`/`U`: filter by user
- `W`: write current settings (they persist next time)
- `Z`: change colors

### Top panel settings

- `l`: show/hide uptime
- `m`: show/hide memory
- `t`: show/hide tasks (CPU summary)
- `1`: show/hide per-core values

### Task panel settings

Press `f` to see more system fields. Combined with `b`, `x`, and `y`, it becomes easier to see what’s going on.

- `b`: highlight the task panel
- `x`: highlight the selected column (requires `b`)
- `y`: highlight the selected row / running task (requires `b`)
- `c`: show full command line
- `f`: configure which fields to display (there are many)
    - `d`: toggle display
    - Right arrow: adjust ordering (use up/down keys)
    - Left arrow: cancel ordering
    - `esc`/`q`: exit configuration
- `i`: hide idle tasks
- `n`/`#`: set and show how many tasks
- `j`: right-align columns

### Example

Below is what it looks like after sorting by CPU (`P`) and enabling highlighting with `b`, `x`, and `y`:

![demo](https://user-images.githubusercontent.com/18013815/79379777-fffd5780-7f91-11ea-9c64-3399cc7de154.png)

Help text shown by pressing `h`:

![demo help](https://user-images.githubusercontent.com/18013815/79380863-9ed68380-7f93-11ea-9680-73c974c366f1.png)

## TOP Command-Line Options

`top` also supports command-line options:

```sh
$ top -h
top -hv | -bcHisS -d delay -n limit -u|U user | -p pid -w [cols]
```

Here I’ll highlight the most important ones; for detailed parameter descriptions, please see the [參考資料](#%E5%8F%83%E8%80%83%E8%B3%87%E6%96%99) links.

For example, this shows the top 10 tasks for the user `acliu`:

```sh
$ top -n 10 -u acliu
```

One useful trick is to use batch mode (option `-b`) together with other shell tools. For example:

```sh
$ top -b -n 3 | grep "python"
```

This output extracts the lines containing “python” from `top`. Also, because we set `-n` (in batch mode, this is the number of iterations), it will print the batch output three times (one per refresh).

## 參考資料

1. [top manual](http://manpages.ubuntu.com/manpages/precise/en/man1/top.1.html)
2. [Linux top command](https://www.computerhope.com/unix/top.htm)

