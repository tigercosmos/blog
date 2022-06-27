---
title: Docker 使用筆記
date: 2020-03-04 01:31:00
tags: [docker, 筆記]
---

![docker image](https://user-images.githubusercontent.com/18013815/75844309-97f31780-5e10-11ea-9743-4c02a065cb5d.png)

[Docker](https://www.docker.com/) 是一個輕量的虛擬環境。

## 取得 Docker

### Windows

Windows 下裝可以直接裝 Docker，基本上就是去官網[下載安裝檔](https://hub.docker.com/editions/community/docker-ce-desktop-windows/)安裝，可以參考[官方教學](https://docs.docker.com/docker-for-windows/install/)。裝完後如果想要共享資料夾，要去 Docker 設定的地方「File Sharing」將之打算共享的硬碟勾選。

也可以用虛擬環境來操作，使用 [VMware Workstation Player](https://www.vmware.com/products/workstation-player/workstation-player-evaluation.html) 安裝 [Ubuntu 18.04](https://ubuntu.com/download/server/thank-you?country=TW&version=18.04.4)。Ubuntu 下安裝 Docker 可以參考[這篇教學](https://www.digitalocean.com/community/tutorial/show-to-install-and-use-docker-on-ubuntu-18-04)。

進入虛擬機的 Ubuntu 之後，為了讓你的 Docker 可以不用用 `sudo`，請執行以下指令：

```sh
    sudo groupadd docker
    sudo gpasswd -a $USER docker
    sudo service docker restart
```

### Linux

Linux 用戶可以參考 [Docker 文檔](https://docs.docker.com/install/linux/docker-ce/centos/)（左側選單可以選擇 Linux 發行版）。

為了讓你的 Docker 可以不用用 `sudo`，請執行以下指令：

```sh
sudo groupadd docker
sudo gpasswd -a $USER docker
sudo service docker restart
```

### macOS

請參考 [Docker 文件](https://docs.docker.com/docker-for-mac/install/) 下載安裝 Docker。

## 取得 Docker Image

Docker Image 是一個事先建立好的 Docker 開發環境。

用以下指令來取得 Ubuntu 官方 Docker Image 來做練習：

```shell
$ docker pull ubuntu:latest
```

## Docker 操作說明

接著我們要進入 Docker 的工作環境，此外我們將與 Docker 共享剛剛 pull 下來的工作目錄。

```shell
$ docker run -ti --rm --cap-add=SYS_PTRACE -v /shared_folder_local_full_path:/shared_folder_docker_full_path ubuntu
```

指令說明：

- `run`：啟動，後面可以接參數，最後要指令要哪個 Docker Image，此範例中是 `ubuntu`
- `-ti`：在 Docker 中可以用互動模式
- `--rm`：每次關閉 Docker 恢復原狀 (例如新 apt install 東西會消失，但共享資料夾不會)。
- `-v`：共享資料夾，用法為 `<本地資料夾絕對路徑>:<共享在 Docker 裡面絕對路徑>`。注意：若是 Windows 下安裝 Docker，請直接分享整個 C 碟 (直接指定資料夾路徑會無效)，Windows 會跳出訊息問你要不要分享，請點確認。

共享資料夾的幾個好處是，你可以在你本來的環境開發，最後做確認時再進 Docker 確認可以順利編譯執行就好；同時，你可以在你電腦中用你喜歡的 IDE 去撰寫程式，因為共享資料夾，你在 IDE 改動的檔案 Docker 可以直接看到，不然你在 Docker 中大概只剩 Vim 可以用了。

## 其他指令

- `docker ps`: 如果你 `docker run` 沒有設定 `--rm`，每次執行的 docker 都會被保留，可以查看之前的工作狀態
- `docker commit`: 可以將之前的工作狀態 `commit` 起來，下次就可以直接進入這個工作階段

範例如下：

```shell
$ docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS              NAMES
c3f279d17e0a        ubuntu:12.04        /bin/bash           7 days ago          Up 25 hours                            desperate_dubinsky
197387f1b436        ubuntu:12.04        /bin/bash           7 days ago          Up 25 hours                            focused_hamilton

$ docker commit c3f279d17e0a  myimage:version1
f5283438590d

$ docker run myimage:version1
```

有時候會想要同時有好幾個 Shell 操作同一個 Container，這時候可以執行：

```
$ docker exec -it <container> bash
```

更多指令可以參考 [Docker 文件](https://docs.docker.com/engine/reference/commandline/docker/)。
