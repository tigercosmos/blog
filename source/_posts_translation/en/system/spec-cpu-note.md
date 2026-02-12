---
title: "Installing and Running SPEC CPU 2006"
date: 2020-06-16 00:00:00
tags: [spec cpu, note]
des: "This post describes how to install and run SPEC CPU 2006 v1.1 on Ubuntu 16.04. Some errors are expected, and this post records how to resolve them."
lang: en
translation_key: spec-cpu-note
---

## Preface

SPEC CPU is a CPU benchmark suite created by The Standard Performance Evaluation Corporation (SPEC). Recent suites include SPEC CPU 2006 and SPEC CPU 2017. Because of its strong credibility, many research projects use it as an evaluation benchmark.

If you want to obtain this suite, you cannot find it freely on the Internet—you need to purchase it from SPEC, and it is expensive. In practice, if you need it, you often have to find a copy through your lab or collaborators. Recently I needed to reproduce a paper that uses SPEC CPU 2006/2017. Our lab happened to have the 2006 version, but there are many “gotchas” in the process, so I’m recording them here.

<!-- more -->

## Environment

Personally I tested using Docker.

This post is based on Ubuntu 16.04, built from the official Docker image. The SPEC CPU version is v1.1 (2008). This is a bit retro, so I recommend GCC 4/5/6. The server CPU is Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz.

If possible, do not casually change environments. For something this large, there are pitfalls everywhere.

## Source Preparation

SPEC CPU 2006 originally comes on a disc image named `SPEC_CPU2006_v1.1.iso`, so the official guide mounts the ISO.

However, because I could not mount inside Docker, I just extracted it.

```shell
$ sudo apt-get install p7zip-full p7zip-rar
$ mkdir target_dir && cd target_dir
$ mv spec.iso target_dir && 7z x spec.iso
```

## Building Tools

The ISO contains some prebuilt binaries for certain platforms, but on Ubuntu you need to build the tools yourself.

```shell
$ cd tools/src
$ ./buildtools
```

During compilation, you will encounter some errors that need to be fixed one by one.

## Expected Errors

### Conflicting types for ‘getline’

Error message:

```shell
In file included from md5sum.c:38:0:
lib/getline.h:31:1: error: conflicting types for 'getline'
 getline PARAMS ((char **_lineptr, size_t *_n, FILE *_stream));
 ^
```

This happens because `getline` and `getdelim` are declared in both `getline.h` and `stdio.h`. Add the following two lines:

```diff
+# if __GLIBC__ < 2
 int
 getline PARAMS ((char **_lineptr, size_t *_n, FILE *_stream));

 int
 getdelim PARAMS ((char **_lineptr, size_t *_n, int _delimiter, FILE *_stream));

+#endif
```

### Undefined reference to `pow`

Error message:

```shell
libperl.a(pp.o): In function `Perl_pp_pow':
pp.c:(.text+0x2a76): undefined reference to `pow'
```

Link `libm` during the build:

```shell
$ PERLFLAGS="-A libs=-lm -A libs=-ldl" ./buildtools
```

### You haven’t done a “make depend” yet!

Error message:

```
You haven't done a "make depend" yet!
make[1]: *** [hash.o] Error 1
```

This is because `/bin/sh` has been modified. Run the following; after the build you can change it back.

```shell
$ sudo rm /bin/sh
$ sudo ln -s /bin/bash /bin/sh
```

### `asm/page.h` file not found

Error message:

```
SysV.xs:7:25: fatal error: asm/page.h: No such file or directory
```

Fix it by modifying `SysV.xs`:

```diff
 #include <sys/types.h>
 #ifdef __linux__
-#   include <asm/page.h>
+#define PAGE_SIZE      4096
 #endif
```

### perl test fail

After fixing all issues above, running `$ PERLFLAGS="-A libs=-lm -A libs=-ldl" ./buildtools` will still have around 9/900 tests failing, all related to perl. Since they are ignorable, we can ignore them.

After the build, it will prompt:

```shell
Hey!  Some of the Perl tests failed!  If you think this is okay, enter y now:
```

**Enter `y` here.** Note that there is a time limit: **if you don’t respond in time, it will treat it as NO.** I once missed it and had to redo the entire build process, which is extremely time-consuming.

When everything succeeds, you should see:

```shell
Tools built successfully.  Go to the top of the tree and
source the shrc file.  Then you should be ready.
```

## Running

### Config files

In theory, there should be several example `.cfg` files under `config`, and you can choose one according to your system environment. Also, under `config/flags`, there should be example flags files, but there can be exceptions. For example, the copy I got did not include them, so you need to download the appropriate flags files from the [SPEC website](https://www.spec.org/cpu2006/flags/).

### Enable the environment

```shell
$ source ./shrc
```

### Run

`runspec` is used to run benchmarks, and it is added to your environment variables when you `source` in the previous step.

For example, to run the `mcf` benchmark, my config file is `Example-linux-ia64-gcc.cfg`, and you can run:

```shell
$  runspec --iterations 1 --size ref \
           --action onlyrun \
           --config Example-linux-ia64-gcc.cfg \
           --noreportable mcf
```

The output looks like:

```shell
Reading config file '/work/spec/config/Example-linux-ia64-gcc.cfg'
Benchmarks selected: 429.mcf
Compiling Binaries
  Up to date 429.mcf base ia64-gcc42 default


Setting Up Run Directories
  Setting up 429.mcf ref base ia64-gcc42 default: existing (run_base_ref_ia64-gcc42.0000)
Running Benchmarks
  Running 429.mcf ref base ia64-gcc42 default
/work/spec/bin/specinvoke -d /work/spec/benchspec/CPU2006/429.mcf/run/run_base_ref_ia64-gcc42.0000 -e speccmds.err -o speccmds.stdout -f speccmds.cmd -C

Run Complete

The log for this run is in /work/spec/result/CPU2006.018.log

runspec finished at Tue Jun 16 01:21:05 2020; 379 total seconds elapsed
```

After you get the results, you can do further system performance analysis, but that is outside the scope of this post.

## Conclusion

This post described how to install and run SPEC CPU 2006 v1.1 on Ubuntu 16.04. Some errors are expected, and this post recorded how to resolve them.

## References

- [Install / execute spec cpu2006 benchmark](https://sjp38.github.io/post/spec_cpu2006_install/)
