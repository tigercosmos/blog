---
title: "Building a Simple HTTP Server in Elixir"
date: 2020-08-24 00:09:00
tags: [http server, elixir]
des: "This post introduces how to build a simple HTTP server in Elixir as a way to practice developing an Elixir project, including creating a project with Mix, managing dependencies with Hex, explaining some Elixir language features, using packages such as Cowboy/Plug/Poison, and running the project."
lang: en
translation_key: elixer-simple-server
---

## Introduction

[Elixir](https://elixir-lang.org/) is a dynamic functional programming language designed for building scalable and maintainable systems.

Elixir is built on the Erlang VM, which makes it suitable for systems that require low latency, distribution, and fault tolerance. It is also used for web development, embedded software, data processing, multimedia processing, and more. The article “[Game of Phones: History of Erlang and Elixir](https://serokell.io/blog/history-of-erlang-and-elixir)” explains the background and history of Elixir and is well worth reading.

Elixir is quite fun to write. After writing some Elixir, I realized that JavaScript and Rust have borrowed many concepts from functional programming. So when learning Elixir from scratch, many FP features—such as Pattern Matching and Enumerable—felt familiar because I had encountered them before.

After reading the [official language guide](https://elixir-lang.org/getting-started/introduction.html), I felt I needed a small project to get more hands-on experience, so I looked into how to build a simple HTTP server with Elixir.

<!-- more -->

This post is based on Wes Dunn’s “[Elixir Programming: Creating a Simple HTTP Server in Elixir](https://www.jungledisk.com/blog/2018/03/19/tutorial-a-simple-http-server-in-elixir/)”. However, I removed some parts and added additional content that the original post does not cover, so we can practice Elixir project development by building a simple HTTP server.

I strongly recommend following along and actually doing the steps yourself—you will get a much better feel for developing in Elixir.

## Setting Up the Elixir Environment

You can install Elixir by following the [official documentation](https://elixir-lang.org/install.html).

On macOS, you can simply run `brew install elixir`.

On Linux, you can use `yum install elixir` or `apt install elixir`. However, my own attempt using apt failed, and in that case you can consider installing via the `asdf` version manager (a funny name). You can refer to this [Gist](https://gist.github.com/rubencaro/6a28138a40e629b06470).

## Preparing an Elixir Project

First, create an Elixir project:

```shell
$ mkdir simple_server && cd simple_server
$ mix new . --sup --app simple_server
```

`mix` is Elixir’s project management tool. `mix new` is followed by the target path. `.` means the current directory because we have already entered `simple_server`. `--app simple_server` specifies the name of the application.

`--sup` means this is a “Supervisor” application. In other words, Elixir will monitor this process and its children. Since this is an HTTP server, we need to manage the processes that handle requests—see the definition below.

About Supervisor:
> The act of supervising a process includes three distinct responsibilities. The first one is to start child processes. Once a child process is running, the supervisor may restart a child process, either because it terminated abnormally or because a certain condition was reached. For example, a supervisor may restart all children if any child dies. Finally, a supervisor is also responsible for shutting down the child processes when the system is shutting down

After running it, you should see something like:

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

## Managing Dependencies with Hex

Elixir uses Hex for package management. First, install Hex on your system:

```shell
$ mix local.hex
Are you sure you want to install "https://repo.hex.pm/installs/1.10.0/hex-0.20.5.ez"? [Yn] y
* creating /Users/tigercosmos/.mix/archives/hex-0.20.5
```

`mix local.hex` installs Hex into your system. You can see that `hex` has been added under `.mix` in `Users/tigercosmos`.

We will use the following packages:

- Cowboy: an HTTP server library for Erlang/OTP
- Plug: an interface layer that connects to the Erlang VM
- plug_cowboy: the adapter for Cowboy
- Poison: a JSON parser

In Elixir, `mix.exs` is similar to Node.js’s `package.json`; it is essentially the project configuration file.

Find the `deps` function in `mix.exs`, and insert the following into the `[]`:

```
{:plug_cowboy, "~> 2.3"},
{:cowboy, "~> 2.8"},
{:plug, "~> 1.10"},
{:poison, "~> 3.1"}
```

Then change `application` in `mix.exs` to:

```
def application do
[
  extra_applications: [:logger, :cowboy, :plug, :poison],
  mod: {SimpleServer.Application, []}
]
end
```

After that, run `mix deps.get` to fetch dependencies:

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

## Configuring the Server

Next, open `lib/simple_server/application.ex` and modify `children` inside `start` to:

```
children = [
  Plug.Adapters.Cowboy.child_spec(
    scheme: :http,
    plug: SimpleServer.Router,
    options: [port: 8085]
  )
]
```

This means we will use `Cowboy` to start our server, listen on HTTP port 8085, and use `plug` to specify the routing (Route).

Next, create the routing file `simple_server_router.ex`:

```shell
touch lib/simple_server/simple_server_router.ex
```

And fill it with:

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

We created the `SimpleServer.Router` module and used `Plug` to connect with Cowboy. When Cowboy receives a connection, it triggers the router. The router then triggers the `:match` plug to match routes via `match/2`, selects the corresponding route, and then uses the `:dispatch` plug to dispatch work and execute the corresponding code.

## Adding Routes

Insert the following code at the TODO position:

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

The code above is fairly straightforward. For routing, we use Elixir’s [Pattern Matching](https://elixir-lang.org/getting-started/pattern-matching.html) to decide which route matches.

`/hello` is a simple GET request. `Plug` provides a [`conn` Struct](https://github.com/elixir-plug/plug#the-plugconn-struct), which represents request information. We pipe (`|>`) it into `put_resp_content_type`, then pipe it into `send_resp` to send the response. [Pipe](https://elixir-lang.org/getting-started/enumerables-and-streams.html#the-pipe-operator) is a very interesting Elixir operator: it passes data from one function to the next, similar in spirit to the shell pipe.

For `/post`, `read_body` is a function provided by `Plug` that parses the request. We take the body, parse the JSON with `Poison.decode!`, and then send the response.

Finally, `match _` matches all remaining cases as the default. Here we return `404` for undefined paths.

## Starting the Server

After modifying the code, start the server:

```shell
$ mix run --no-halt
$ iex -S mix # 互動模式
```

`mix run` starts the project and executes `mix.exs`. By default, the VM will stop after handling callbacks once, so we use `--no-halt` to keep the VM running. Another option is `iex -S mix`, which starts an interactive shell; `iex` lets you interact with the program while it is running.

You can test with the following requests:

```shell
$ curl -v "http://localhost:8085/hello"
$ curl -v "http://localhost:8085/should-be-404"
$ curl -v -H 'Content-Type: application/json' "http://localhost:8085/post" -d '{"message": "hello world" }'
```

Then you should see output like:

```shell
23:53:47.295 [debug] GET /hello
23:53:47.295 [debug] Sent 200 in 52µs

23:53:53.553 [debug] GET /should-be-404
23:53:53.553 [debug] Sent 404 in 89µs

23:53:59.919 [debug] POST /post
%{"message" => "hello world"}
23:53:59.933 [debug] Sent 201 in 14ms
```

Yay! The responses match what we expected.

## Conclusion

This post introduced how to build a simple HTTP server in Elixir as a way to practice developing an Elixir project, including creating a project with Mix, managing dependencies with Hex, explaining some Elixir language features, using packages such as Cowboy/Plug/Poison, and running the project.
