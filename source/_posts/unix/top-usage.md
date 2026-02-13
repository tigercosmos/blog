---
title: Unix/Linux TOP 指令使用詳解
date: 2020-04-16 15:01:00
tags: [unix, linux, shell, top command]
des: "本文介紹 Unix/Linux TOP 指令使用方式。top 指令 (top command) 可以說是使用 Unix/Linux 最基本的工具了，他就像 Windows 系統中的工作管理員，可以監看目前所有程式的執行狀況。top 也是最簡單監控程式的方法，用它來觀察程式運行所花的記憶體、CPU 還有各種其他資訊。"
lang: zh
translation_key: top-usage
---

## TOP 簡介

`top` 指令 (top command) 可以說是使用 Unix/Linux 最基本的工具了，他就像 Windows 系統中的工作管理員，可以監看目前所有程式的執行狀況。`top` 也是最簡單監控程式的方法，用它來觀察程式運行所花的記憶體、CPU 還有各種其他資訊。類似的程式很多，像是 `htop`、`gtop` 都是衍伸版，大家有興趣可以去查查看，但以基本需求來說 `top` 就非常夠用了。
<!-- more -->

使用方式很簡單：

```sh
$ top
```

![top snapshot](https://user-images.githubusercontent.com/18013815/79328399-90af4580-7f48-11ea-9926-880e9f44f84b.png)

其中顯示的項目預設包含：
- `PID`: 執行任務的 Process ID
- `USER`: 執行任務的使用者是誰
- `PR`: 任務的優先度 (Priority)
- `NI`: 任務的 Nice Value，負的值代表優先度高，正的值代表優先度低
- `VIRT`: 總共用到多少 kB 虛擬記憶體 (Virtual Memory)
- `RES`: 實體記憶體 (Resident Size) 大小 kB
- `SHR`: 總共用到多少 kB 的共享記憶體 (Shared Memory)
- `S`: 狀態 (Status)
    - R 代表執行中
    - D 代表不可中斷睡眠 (不可被 signal 打斷通常在等 I/O)
    - S 代表睡眠 (可被喚醒)
    - T 中斷中或停止，可能是被 `SIGSTOP` 或 `SIGTSTP` 停止，或是被 degubber 中斷 (ptrace)
    - Z 代表殭屍，通常發生在 Child 已經執行完，等待 Parent 結束或回收
- `%CPU`: 占用到多少 CPU %，注意到一個核心是 100%，所以多核心是可以超過 100% 的
- `%MEM`: 占用到多少全部記憶體多少比例
- `TIME+`: 已經執行多少時間
- `COMMAND`: 任務的指令名稱

## TOP 互動模式

一種使用 `top` 方式是進去程式裡面之後，再去操作 `top`。以下我挑出比較常用到的一些指令做介紹，剩下的大家可以參考[參考資料](#%E5%8F%83%E8%80%83%E8%B3%87%E6%96%99)中的連結。推薦的學習辦法是，邊看文章邊一起試著操作。

### 任務排序

排列任務我想是最要的指令了，所以先介紹，通常我們用 `top` 不外乎就是一看執行的程式花了多少 CPU 和記憶體。

- `M`: 照占用記憶體比例排序
- `N`: 照 PID 大小排列
- `P`: 照 CPU 使用多寡排序
- `T`: 照執行時間長短排序
- `>`/`<`: 排列欄位向左向右切換 
- `R`: 反向排列

### 一般操作

這邊比較重要的是要知道按 `q` 可以離開 😂

- `h`/`?`: 查看說明
- Enter/Space: 刷新螢幕 (預設 3 秒)
- `=`: 跳出限制任務數量 (在使用搜尋之後)
- `B`: 使用粗體
- `d`/`s`: 更改更新頻率
- `I`: 切換成顯示 CPU 使用比例/全部 CPU 數量
- `k`: 終止特定 pid
- `L`: 查詢字串，搭配 `&` 使用
- `q`: 離開程式
- `r`: 重新設定 pid 的 nice value，設定成正的會失去優先度，設成負的則會提高優先度 (需要 sudo)
- `u`/`U`: 選特定使用者
- `W`: 寫入目前設定，下次重開就會是一樣的
- `Z`: 改變顏色

### 上方面板設定

- `l`: 顯示/隱藏運行時間
- `m`: 顯示/隱藏記憶體
- `t`: 顯示/隱藏任務
- `1`: 顯示/隱藏每個單一核心的數值

### 任務面板設定

用 `f` 可以看到更多系統資料，搭配 `b`、`x`、`y` 可以清楚知道那些程式在運作。

- `b`: 開啟任務面板高亮
- `x`: 欄高亮，選取的欄位會全亮 (要先按 b)
- `y`: 行高亮，正在執行的程式會亮 (要先按 b)
- `c`: 顯示指令全名
- `f`: 設定要顯示的欄位 (非常多系統資訊)，進入之後
    - `d`: 設定顯示
    - 方向鍵右鍵: 可排序 (用上下鍵)
    - 方向鍵左鍵: 取消排序
    - `esc`/`q`: 離開設定欄位
- `i`: 隱藏閒置任務
- `n`/`#`: 設並顯示多少任務
- `j`: 靠做靠右對齊

### 範例

以下是先 `P` 做 CPU 排序，再 `b`、`x`、`y` 高亮後的樣子：

![demo](https://user-images.githubusercontent.com/18013815/79379777-fffd5780-7f91-11ea-9c64-3399cc7de154.png)

輸入 `h` 顯示的說明：


![demo help](https://user-images.githubusercontent.com/18013815/79380863-9ed68380-7f93-11ea-9680-73c974c366f1.png)


## TOP 命列列操作

`top` 也有參數可以下:

```sh
$ top -h
top -hv | -bcHisS -d delay -n limit -u|U user | -p pid -w [cols]
```

這邊我挑出最重要的介紹，詳細的參數說明請查閱[參考資料](#%E5%8F%83%E8%80%83%E8%B3%87%E6%96%99)連結。

舉例來說，這樣會顯示 `acliu` 這個使用者頭 10 個任務：

```sh
$ top -n 10 -u acliu
```

比較值得一提的是，我們可以用批次輸入 (batch mode) (參數 `-b`) 來搭配 Shell 其他工具使用，例如：

```sh
$ top -b -n 3 | grep "python"
```

這樣輸出會抓出 `top` 裡面包含 "python" 的片段，同時因為我們設定 `-n` (此時是設定批次數量)，所以會輸出三次批次狀態 (每次刷新的狀態)。

## 參考資料

1. [top manual](http://manpages.ubuntu.com/manpages/precise/en/man1/top.1.html)
2. [Linux top command](https://www.computerhope.com/unix/top.htm)
