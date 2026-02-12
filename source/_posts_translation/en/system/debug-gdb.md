---
title: "How to Debug Rust via GDB"
date: 2020-09-21 17:17:00
tags: [servo, debug, gdb, rust, english]
des: "I will teach you some tips for using GDB to develop and debug your Rust code and the Servo project. The same debugging approach can also be applied to C/C++."
lang: en
translation_key: debug-gdb
---

## Introduction

Big open source projects are huge, such as the Servo browser in Rust. I have counted the lines of code for you: there are almost a hundred thousand lines of code in the Servo project. To develop such a big project, knowing how to debug in the right way is very important, since you want to find the bottleneck quickly and efficiently.

In this article, I will teach you some tips for using GDB to develop and debug your Rust code and the Servo project. The same debugging approach can also be applied to C/C++.

<!-- more --> 

## So, How to Debug?

I assume you are not familiar with software development, but you have some skills to write code.

When you want to know what’s inside the box, you might add some lines, such as:

```
println!("{:?}", SOMETHING);
```

This is a simple method. Straightforward enough! It does help you figure out what’s going on in the code.

However, it requires recompiling each time you want to inspect another variable’s value in the program. Besides, when your program crashes or causes a memory leak, it is hard to trace the underlying problem.

This simple way is not powerful enough, which means you need a more capable tool. It could be GDB on Linux, or LLDB on macOS. (On Windows, the Visual Studio debugger is also very strong, but it is out of scope for this article.)

So I will talk about how to use GDB. LLDB is very similar to GDB—basically, their commands are almost the same—so I will focus on how to use GDB to debug Rust and Servo.

## Introduction to GDB

> “GDB, the GNU Project debugger, allows you to see what is going on ‘inside’ another program while it executes — or what another program was doing at the moment it crashed.” — from gnu.org

In other words, GDB allows you to control the running program and to get more information from inside the code.

For example, you can stop the program at a certain line in a file, which is called a “breakpoint”. When the program stops at the breakpoint, you can print variables to see their values in the breakpoint scope.

You can also backtrace from the breakpoint. Backtrace means printing all functions called before the breakpoint. Sometimes a crash is not caused by the code where it crashes; it might happen earlier, and an invalid parameter is passed later to cause the crash.

There are other features as well, and I will mention them in the following sections.

## GDB the Rust

First of all, I will create a simple <a href="https://doc.rust-lang.org/book/second-edition/ch01-03-hello-cargo.html" target="_blank" >Hello World</a> to demonstrate how to use GDB in a Rust project. You might have installed Rust and Cargo, haven’t you?

Please follow the <a href="https://doc.rust-lang.org/book/second-edition/ch01-03-hello-cargo.html#creating-a-project-with-cargo" target="_blank" >steps</a> in “the Rust book” to create a Hello World. Make sure you can compile, run the code, and understand what Cargo is doing.

To create a project:

```
cargo new hello_cargo --bin
cd hello_cargo
```

Then, let’s start!

In order to show how to use GDB, I have designed a sample code. Please copy the following code to your `./src/main.rs`:

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

To build this code, simply run `cargo build`.

There will be an executable file at `./target/debug/hello_cargo`. The build is in debug mode by default, and we can use the debug build with GDB. However, if it is a release build, you cannot run it with GDB, since the debug information is lost.

To run the program with GDB:

```
gdb target/debug/hello_cargo 
```

That’s it. You would see the interface in GDB like this:

```
gdb (Ubuntu 8.1-0ubuntu3) 8.1.0.20180409-git
Copyright (C) 2018 Free Software Foundation, Inc.
...
(gdb)
```

Now you can enter some commands in GDB!

## GDB the C/C++

If you write C/C++ code, add the `-g` flag while compiling, which means building with debug information, and then the executable can be loaded by GDB.

```shell
$ gcc hello_world.c -g -o hello_world
$ gdb ./hello_world
```

## GDB commands

There are many commands in GDB, but I will only introduce the most important ones. In my personal case, I usually only use these commands as well.

### break

As mentioned before, a breakpoint allows you to stop the program at a certain position. There are two ways to set breakpoints.

Use `break` or `b` to set a breakpoint.

In the first case, you can break at a function. (In a big project, you would need to enter the full path, like `mod::mod::function`.)

```
(gdb) break make_hello_world
Breakpoint 1 at 0x55555555beca: file src/main.rs, line 8.
```

Or, you can add the file path with a line number to define where to stop.

```
(gdb) b src/main.rs:9
Breakpoint 2 at 0x55555555bf6a: file src/main.rs, line 9.
```

Let’s see whether we have set it successfully.

```
(gdb) info break
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   0x000055555555beca in hello_cargo::make_hello_string at src/main.rs:8
2       breakpoint     keep y   0x000055555555bf6a in hello_cargo::make_hello_string at src/main.rs:9
```

But I just want one breakpoint, so I can delete the first one using the `del` command.

```
(gdb) del 1
(gdb) info break
Num     Type           Disp Enb Address            What
2       breakpoint     keep y   0x000055555555bf6a in hello_cargo::make_hello_string at src/main.rs:9
```

### run

Now there is just one breakpoint. Let’s run the program and see what happens!

Use `run` to start.

```
(gdb) run
Starting program: /home/tigercosmos/Desktop/hello_cargo/target/debug/hello_cargo 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Breakpoint 2, hello_cargo::make_hello_string (name=...) at src/main.rs:9
9	    hello_str
```

As you can see, the program has stopped at the position we want. Now we can do something at this breakpoint.

### backtrace

If you are wondering which upstream functions have been called before running to this breakpoint, you can use `backtrace` or `bt`.

```
(gdb) backtrace
#0  hello_cargo::make_hello_string (name=...) at src/main.rs:9
#1  0x000055555555bdd5 in hello_cargo::main () at src/main.rs:3
```

Now GDB tells you that `main.rs` line 3 (#1), which is `let msg = make_hello_string(&amp;name);`, called `main.rs` line 9 (#0), which belongs to `make_hello_string`.

You might say, that’s really obvious, doesn’t it?

Yep! However, what if you are debugging a big open source project such as Servo, and you need to figure out the backtrace for a module? Generally speaking, there are about thirty to forty function calls before the breakpoint in a module. Finding the backtrace by reading code is super hard, but with GDB we can get it directly.

### frame

A frame is one of the program states in the backtrace. We can switch to a frame we want and check information in that frame.

Use `frame` or `f` to use this command.

After the previous step, we have already set a breakpoint at `src/main.rs:9`, and there are two frames in the backtrace: `#0` and `#1`.

Now I want to check frame `#1`, and see the value of `name` (declared at `src/main.rs:2`) in the scope of frame `#1`.

```
#1  0x000055555555bdd5 in hello_cargo::main () at src/main.rs:3
3	    let msg = make_hello_string(&name);
(gdb) print name
$1 = "Tiger"
```

So, `frame` lets you enter the scope in frame `#1`, and then you can use `print` to print the value of a variable, or you can use `call` to call a function in that scope.

How about switching to frame `#0` and checking the value of `hello_str`?

```
(gdb) frame 0
#0  hello_cargo::make_hello_string (name=...) at src/main.rs:9
9	    hello_str
(gdb) print hello_str
$2 = alloc::string::String {vec: alloc::vec::Vec<u8> {buf: alloc::raw_vec::RawVec<u8, alloc::heap::Heap> {ptr: core::ptr::Unique<u8> {pointer: core::nonzero::NonZero<*const u8> (0x7ffff6c22060 "Hi Tiger, how are you?\000"), _marker: core::marker::PhantomData<u8>}, cap: 34, a: alloc::heap::Heap}, len: 22}}
```

### continue

After checking some information we want, we might want the program to continue. Use `continue` or `c` to continue running. The program will keep running until it hits another breakpoint or finishes execution.

```
(gdb) c
Continuing.
Hi Tiger, how are you?
```

Since there is just one breakpoint in this example, the program runs to the end.

### Something else

Once you stop at a breakpoint, you can use `step` to run code line by line.

Once you stop at a breakpoint, you can use `up` and `down` to switch frames instead of using `frame <number>` directly.

If the program is running (you have already issued `run`), you can press `Ctrl+C` to interrupt GDB. Then the program breaks immediately. The process pauses at where it was just running to. The stop point triggered by the interrupt works like a one-time manual breakpoint. You can run other commands there, and once you are done, you can enter `c` to continue the process.

Then, you can call commands such as `break`, `call`, `step`, etc.

GDB provides many more commands. You can check the documentation for more information.

## Debug Servo

Now we know how to debug Rust code. Let’s debug the Servo project. I assume you are able to compile Servo.

To build in debug mode:

```
$ ./mach build -d
```

Once the build is done, and we want to debug Servo:

```
$ ./mach run -d https://google.com --debug
Reading symbols from /home/tigercosmos/servo/target/debug/servo...done.
(gbd)
```

You can debug Servo now!

## Conclusion

The concepts in this article can be applied not only to Rust projects but also to C++ projects. If you want to debug Firefox or Chromium, you can use the same approach. I will not discuss details such as complex GDB configuration, so you may need to search for more advanced articles to learn further.

GDB is not a must-have for developers, but it does empower hackers. Some Servo contributors are front-end developers. One of them is my best friend, and he has already contributed many commits to Servo. As a front-end developer, he had never used GDB before. However, he recently ran into troubles while developing Servo. I think if he knows how to use GDB, it would be easier to figure out the reasons behind weird program behavior.

I’m writing this article for him and all developers who have just become part of the Servo community. I hope this article is helpful for all of you. Have a nice hacking day! :)

---

### Special thanks

Thanks <a href="https://github.com/ko19951231" target="_blank" >@SouthRa</a> and <a href="https://github.com/cybai" target="_blank" >@cybai</a> for helping me review the article.
