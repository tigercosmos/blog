---
title: 使用 Elixir 建立簡易 HTTP Server 
date: 2020-08-24 00:09:00
tags: [http server, elixir]
des: "本文介紹如何用 Elixir 建立一個簡單的 HTTP Server，藉此來學習開發 Elixir 專案，其中包含如何用 Mix 新建一個專案，用 Hex 做套件管理，解釋一些 Elixir 語言特性，介紹如何用 Cowboy、Plug、Poison 等套件，以及如何執行專案。"
lang: zh
translation_key: elixer-simple-server
---

## 簡介

[Elixir](https://elixir-lang.org/) 是一個動態 Functional Programming 語言，專門用來打造可擴展且易維護的系統。

Elixir 基於 Erlang VM 所建，適合用在需要低延遲、分散式、錯誤容忍（fault-tolerant）的系統上，同時也可以用在網頁開發、嵌入式軟體、資料處理、多媒體處理等用途上。這篇文章「[Game of Phones: History of Erlang and Elixir](https://serokell.io/blog/history-of-erlang-and-elixir)」介紹了 Elixir 來龍去脈，值得一看。

Elixir 寫起來滿有趣的，寫了之後才意識到原來 JS 和 Rust 從 Functional Programming 借鏡了不少概念，所以在從頭學 Elixir 的過程，一些 FP 的特色，像是 Pattern Matching、Enumerable 都已經在先前接觸過。

看完[官方的語言教學](https://elixir-lang.org/getting-started/introduction.html)後，覺得需要藉由寫小專案更熟悉，所以就研究如何用 Elixir 建造一個簡單的 HTTP Server。

<!-- more -->

本文範例參考 Wes Dunn 的「[Elixir Programming: Creating a Simple HTTP Server in Elixir](https://www.jungledisk.com/blog/2018/03/19/tutorial-a-simple-http-server-in-elixir/)」，不過刪減和補充一些原文沒有的內容，讓我們藉由撰寫一個簡易 HTTP Server 練習開發 Elixir 專案。

強烈建議跟著本文一起進行操作，會對開發 Elixir 更有感覺！

## 設定 Elixir 環境

你可以參考[官方文件](https://elixir-lang.org/install.html)來安裝 Elixir。

如果你是 macOS 直接用 `brew install elixir` 即可。

如果是 Linux 用戶，`yum install elixir` 或 `apt install elixir`。不過我自己用 apt 時候失敗了，這時候可以考慮用 `asdf` 版本管理工具 （很鬧的名字）來安裝，可以參考這個 [Gist](https://gist.github.com/rubencaro/6a28138a40e629b06470)。

## 準備 Elixir專案

首先我們要建立一個 Elixir 專案

```shell
$ mkdir simple_server && cd simple_server
$ mix new . --sup --app simple_server
```

`mix` 是 Elixir 用的專案管理工具，`mix new` 後面接要建立的位置，`.` 代表當下目錄因為我們已經進入 `simple_server`。`--app simple_server` 代表我們建立的程式的名字。

`--sup` 代表這是「Supervisor」程式，意味著 Elixir 會監控這支程式和其子孫，因為這是一個 HTTP Server 的程式，我們有必要管理 Server 處理 Requests 的 Processes，可以參考下方定義。

關於 Supervisor:
> The act of supervising a process includes three distinct responsibilities. The first one is to start child processes. Once a child process is running, the supervisor may restart a child process, either because it terminated abnormally or because a certain condition was reached. For example, a supervisor may restart all children if any child dies. Finally, a supervisor is also responsible for shutting down the child processes when the system is shutting down

跑完後長這樣：

```shell
$ mix new . --sup --app simple_server
* creating README.md
* creating .formatter.exs
* creating .gitignore
* creating mix.exs
* creating lib
* creating lib/simple_server.ex
* creating lib/simple_server/application.ex
* creating test
* creating test/test_helper.exs
* creating test/simple_server_test.exs

Your Mix project was created successfully.
You can use "mix" to compile it, test it, and more:

    mix test

Run "mix help" for more commands.
```

## 用 Hex 管理套件

Elixir 用 Hex 來做套件管理，首先我們要在系統安裝 Hex：

```shell
$ mix local.hex
Are you sure you want to install "https://repo.hex.pm/installs/1.10.0/hex-0.20.5.ez"? [Yn] y
* creating /Users/tigercosmos/.mix/archives/hex-0.20.5
```

`mix local.hex` 代表將 Hex 裝在自己系統中，可以看到在 `Users/tigercosmos` 底下的 `.mix` 已經加入 `hex`。

接下來會用到幾個套件分別是：

- Cowboy: Erlang/OPT 的 HTTP Server 套件
- Plug: 用來連接 Erlang VM
- plug_cowboy: Cowboy 的接口
- Poison: 用來解析 JSON

在 Elixir 中 `mix.exs` 相當於 NodeJS 的 `package.json`，基本上就是專案設定檔。

找到 `mix.exs` 中的 `deps` 函數，在 `[]` 裡面插入：

```
{:plug_cowboy, "~> 2.3"},
{:cowboy, "~> 2.8"},
{:plug, "~> 1.10"},
{:poison, "~> 3.1"}
```

在 `mix.exs` 的 `application` 改成：

```
def application do
[
  extra_applications: [:logger, :cowboy, :plug, :poison],
  mod: {SimpleServer.Application, []}
]
end
```

改完之後執行 `mix deps.get` 取得套件：

```shell
$ mix deps.get
Resolving Hex dependencies...
Dependency resolution completed:
Unchanged:
  cowboy 2.8.0
  cowlib 2.9.1
  mime 1.4.0
  plug 1.10.4
  plug_cowboy 2.3.0
  plug_crypto 1.1.2
  poison 3.1.0
  ranch 1.7.1
  telemetry 0.4.2
...
...
```

## 設定 Server

接著我們找到 `lib/simple_server/application.ex`，找到 `start` 裡面得 `children` 改成：

```
children = [
  Plug.Adapters.Cowboy.child_spec(
    scheme: :http,
    plug: SimpleServer.Router,
    options: [port: 8085]
  )
]
```

代表我們要用 `Cowboy` 來啟動我們的 Server，設定成監聽 HTTP 8085 Port，裡面得 `plug` 是我們設定路由 (Route) 的地方。

接著我們建立路由檔案 `simple_server_router.ex`：

```shell
touch lib/simple_server/simple_server_router.ex
```

並在裡面填入

```
defmodule SimpleServer.Router do
  use Plug.Router
  use Plug.Debugger
  require Logger
plug(Plug.Logger, log: :debug)


plug(:match)

plug(:dispatch)

# >>>
# TODO: 加入路由！
# <<<

end
```

我們建立了 `SimpleServer.Router` 模組，裡面用 `Plug` 去和 Cowboy 連結。當 Cowboy 連線進來 Router 會被觸發，接著他會觸發 `:match` plug，使用 `match/2` 函數作匹配，選擇對應的路由，接著用 `:dispatch` plug 去派發工作來執行該段程式碼。

## 加入路由

在剛剛 TODO 的地方插入以下程式碼：

```
# >>>

  # 簡單的 GET
  get "/hello" do
    conn
    |> put_resp_content_type("text/plain")
    |> send_resp(200, "Hello world")
  end

  # 基本的 POST 處理 JSON
  post "/post" do
    {:ok, body, conn} = read_body(conn)

    body = Poison.decode!(body)

    IO.inspect(body) # 印出 body

    send_resp(conn, 201, "created: #{get_in(body, ["message"])}")
  end

  # 沒匹配到的預設回應
  match _ do
    send_resp(conn, 404, "not found")
  end

# <<<
```

以上程式碼都還算直觀。路由的部分是用 Elixir 的 [Pattern Matching](https://elixir-lang.org/getting-started/pattern-matching.html)（模式匹配）來判斷。

`/hello` 就是簡單的 GET 請求，`Plug` 本身會提供 [`conn` Struct](https://github.com/elixir-plug/plug#the-plugconn-struct)，代表 Request 本身的資訊，我們將他 Pipe（`|>`）給 `put_resp_content_type` 再 Pipe 給 `send_resp` 送出。[Pipe](https://elixir-lang.org/getting-started/enumerables-and-streams.html#the-pipe-operator) 是 Elixir 非常有趣的運算子，可以將資料一個傳一個給函數執行，概念跟 Shell 的 Pipe 一樣。

在 `/post` 的部分，`read_body` 是 `Plug` 提供的函數，可以將 Request 做拆解，我們取得 Body 的部分，用 `Poison.decode!` 去解析 JSON，最後再送出回應。

最後如果 `match _` 代表含括剩下所有部分，也就是預設值，在這邊我們設定沒有規範的路徑就是 `404`。

## 啟動 Server

程式碼都改好後，接著我們要啟動 Server:

```shell
$ mix run --no-halt
$ iex -S mix # 互動模式
```

`mix run` 就是啟動專案，會去找 `mix.exs` 執行，但是預設是 Callback 回應一次就會關閉 VM，我們要讓 VM 一直跑下去所以要用 `--no-halt`。另一種方式是用 `iex -S mix`，那就會開一個互動式的 Shell，`iex` 讓你可以和程式互動同時程式也正在跑。

我們可以用以下請求測試：

```shell
$ curl -v "http://localhost:8085/hello"
$ curl -v "http://localhost:8085/should-be-404"
$ curl -v -H 'Content-Type: application/json' "http://localhost:8085/post" -d '{"message": "hello world" }'
```

然後可以看到 Elixir 輸出：

```shell
23:53:47.295 [debug] GET /hello
23:53:47.295 [debug] Sent 200 in 52µs

23:53:53.553 [debug] GET /should-be-404
23:53:53.553 [debug] Sent 404 in 89µs

23:53:59.919 [debug] POST /post
%{"message" => "hello world"}
23:53:59.933 [debug] Sent 201 in 14ms
```

耶！回應都有照我們預期。

## 結論

本文介紹如何用 Elixir 建立一個簡單的 HTTP Server，藉此來學習開發 Elixir 專案，其中包含如何用 Mix 新建一個專案，用 Hex 做套件管理，解釋一些 Elixir 語言特性，介紹如何用 Cowboy、Plug、Poison 等套件，以及如何執行專案。
