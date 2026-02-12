---
title: "Elixir で簡易 HTTP サーバーを作る"
date: 2020-08-24 00:09:00
tags: [http server, elixir]
des: "本記事では Elixir でシンプルな HTTP サーバーを作りながら、Mix によるプロジェクト作成、Hex による依存管理、Elixir の言語特性の説明、Cowboy/Plug/Poison などのパッケージの利用方法、そして実行方法まで、Elixir プロジェクト開発の練習として一通り紹介します。"
lang: jp
translation_key: elixer-simple-server
---

## イントロダクション

[Elixir](https://elixir-lang.org/) は動的型付けの関数型プログラミング言語で、スケーラブルで保守性の高いシステムを構築するために設計されています。

Elixir は Erlang VM 上に構築されており、低レイテンシ・分散・フォールトトレラント（fault-tolerant）が求められるシステムに向いています。また、Web 開発、組込みソフトウェア、データ処理、マルチメディア処理など幅広い用途で利用できます。「[Game of Phones: History of Erlang and Elixir](https://serokell.io/blog/history-of-erlang-and-elixir)」という記事では Elixir の背景が紹介されており、読む価値があります。

Elixir は書いていてとても面白い言語です。実際に触ってみると、JavaScript や Rust が関数型プログラミングから多くの概念を取り入れていることに気づきます。そのため、Elixir をゼロから学ぶ過程でも、Pattern Matching や Enumerable といった FP の特徴は、以前に触れた経験があるものとして自然に理解できます。

[公式の言語ガイド](https://elixir-lang.org/getting-started/introduction.html) を読んだあと、小さなプロジェクトを書いてより慣れる必要があると感じたため、Elixir で簡単な HTTP サーバーを作る方法を調べました。

<!-- more -->

本文のサンプルは Wes Dunn の「[Elixir Programming: Creating a Simple HTTP Server in Elixir](https://www.jungledisk.com/blog/2018/03/19/tutorial-a-simple-http-server-in-elixir/)」を参考にしています。ただし、原文の一部を削減し、原文にはない内容も補足しています。簡易 HTTP サーバーの作成を通して、Elixir プロジェクト開発の練習をしていきましょう。

本記事は、ぜひ実際に手を動かしながら進めることを強くおすすめします。Elixir 開発の感覚がぐっと掴みやすくなります。

## Elixir 環境のセットアップ

Elixir のインストールは [公式ドキュメント](https://elixir-lang.org/install.html) を参照してください。

macOS の場合は `brew install elixir` でインストールできます。

Linux の場合は `yum install elixir` または `apt install elixir` を使えます。ただ、私の場合 apt では失敗しました。その際は `asdf` というバージョン管理ツール（なかなか攻めた名前）を使ってインストールする方法もあります。こちらの [Gist](https://gist.github.com/rubencaro/6a28138a40e629b06470) が参考になります。

## Elixir プロジェクトの準備

まず Elixir プロジェクトを作成します。

```shell
$ mkdir simple_server && cd simple_server
$ mix new . --sup --app simple_server
```

`mix` は Elixir のプロジェクト管理ツールです。`mix new` の後には作成先を指定します。`.` はカレントディレクトリを意味します（すでに `simple_server` に入っているため）。`--app simple_server` はアプリケーション名の指定です。

`--sup` は「Supervisor」アプリケーションであることを意味します。つまり Elixir がこのプロセスとその子孫プロセスを監視します。HTTP サーバーではリクエスト処理のプロセスを管理する必要があるため、Supervisor を使う意義があります（定義は下記参照）。

Supervisor について：
> The act of supervising a process includes three distinct responsibilities. The first one is to start child processes. Once a child process is running, the supervisor may restart a child process, either because it terminated abnormally or because a certain condition was reached. For example, a supervisor may restart all children if any child dies. Finally, a supervisor is also responsible for shutting down the child processes when the system is shutting down

実行すると次のようになります：

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

## Hex による依存管理

Elixir は Hex でパッケージ管理を行います。まずシステムに Hex をインストールします：

```shell
$ mix local.hex
Are you sure you want to install "https://repo.hex.pm/installs/1.10.0/hex-0.20.5.ez"? [Yn] y
* creating /Users/tigercosmos/.mix/archives/hex-0.20.5
```

`mix local.hex` は Hex をシステムにインストールするコマンドです。`Users/tigercosmos` 配下の `.mix` に `hex` が追加されているのが確認できます。

ここで使う主なパッケージは次の通りです：

- Cowboy: Erlang/OTP の HTTP サーバーパッケージ
- Plug: Erlang VM と接続するためのレイヤ
- plug_cowboy: Cowboy のアダプタ
- Poison: JSON 解析

Elixir の `mix.exs` は Node.js の `package.json` に相当し、基本的にはプロジェクト設定ファイルです。

`mix.exs` の `deps` 関数を探し、`[]` の中に次を追加します：

```
{:plug_cowboy, "~> 2.3"},
{:cowboy, "~> 2.8"},
{:plug, "~> 1.10"},
{:poison, "~> 3.1"}
```

次に `mix.exs` の `application` を次のように変更します：

```
def application do
[
  extra_applications: [:logger, :cowboy, :plug, :poison],
  mod: {SimpleServer.Application, []}
]
end
```

変更後、`mix deps.get` を実行して依存関係を取得します：

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

## サーバー設定

次に `lib/simple_server/application.ex` を開き、`start` 内の `children` を次のように変更します：

```
children = [
  Plug.Adapters.Cowboy.child_spec(
    scheme: :http,
    plug: SimpleServer.Router,
    options: [port: 8085]
  )
]
```

これは `Cowboy` でサーバーを起動し、HTTP の 8085 ポートを監視する設定です。`plug` はルーティング（Route）を定義する場所を指定します。

続いてルーティング用のファイル `simple_server_router.ex` を作成します：

```shell
touch lib/simple_server/simple_server_router.ex
```

そして次を記述します：

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

`SimpleServer.Router` モジュールを作り、`Plug` を使って Cowboy と接続します。Cowboy に接続が入ると Router が呼び出され、次に `:match` plug が走って `match/2` でルートを選択し、その後 `:dispatch` plug が処理をディスパッチして該当コードを実行します。

## ルートの追加

先ほどの TODO 部分に次のコードを挿入します：

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

上のコードは概ね直感的です。ルーティングは Elixir の [Pattern Matching](https://elixir-lang.org/getting-started/pattern-matching.html)（模式匹配）を使って判定します。

`/hello` は単純な GET リクエストです。`Plug` は [`conn` Struct](https://github.com/elixir-plug/plug#the-plugconn-struct) を提供しており、これはリクエスト情報を表します。これを Pipe（`|>`）で `put_resp_content_type` に渡し、さらに Pipe で `send_resp` に渡してレスポンスを返します。[Pipe](https://elixir-lang.org/getting-started/enumerables-and-streams.html#the-pipe-operator) は Elixir の面白い演算子で、データを順に関数へ渡していく点がシェルのパイプに似ています。

`/post` の部分では、`read_body` は `Plug` が提供する関数で、リクエストを分解します。Body を取り出して `Poison.decode!` で JSON を解析し、最後にレスポンスを返します。

最後の `match _` は残りすべてを包含するデフォルトで、ここでは未定義パスに対して `404` を返すようにしています。

## サーバー起動

コードを変更し終えたら、サーバーを起動します：

```shell
$ mix run --no-halt
$ iex -S mix # 互動模式
```

`mix run` はプロジェクトを起動し、`mix.exs` を実行します。ただしデフォルトではコールバックが一度返ると VM が終了してしまうため、VM を動かし続けるには `--no-halt` を付けます。別の方法として `iex -S mix` もあり、こちらは対話的なシェルを開きます。`iex` により、プログラムが動作している状態で対話しながら操作できます。

次のリクエストでテストできます：

```shell
$ curl -v "http://localhost:8085/hello"
$ curl -v "http://localhost:8085/should-be-404"
$ curl -v -H 'Content-Type: application/json' "http://localhost:8085/post" -d '{"message": "hello world" }'
```

Elixir の出力は次のようになります：

```shell
23:53:47.295 [debug] GET /hello
23:53:47.295 [debug] Sent 200 in 52µs

23:53:53.553 [debug] GET /should-be-404
23:53:53.553 [debug] Sent 404 in 89µs

23:53:59.919 [debug] POST /post
%{"message" => "hello world"}
23:53:59.933 [debug] Sent 201 in 14ms
```

やった！期待通りの応答になっています。

## 結論

本記事では Elixir でシンプルな HTTP サーバーを作りながら、Mix によるプロジェクト作成、Hex による依存管理、Elixir の言語特性の説明、Cowboy/Plug/Poison などのパッケージの利用方法、そして実行方法まで紹介しました。
