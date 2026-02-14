---
title: "Programming for Beginners — A Learning Guide with the Right Approach"
date: 2020-05-26 20:01:00
tags: [資工, computer science, 程式, beginner, 程式設計入門, 學習指南]
lang: en
translation_key: code-beginner
---

## Preface

In an era where information is everywhere, many people want to jump in and learn some programming. For students in science and engineering, being able to code is as important as being able to speak English—when dealing with scientific data, you often need programs as tools. For business or humanities students, programming may not be strictly necessary, but having some understanding can still be very helpful: it not only lets you understand how engineers think, but it can also help you write small scripts to simplify tedious office work.

![Hello World](https://user-images.githubusercontent.com/18013815/82930700-64abc900-9fb8-11ea-9426-4f564fcda98d.png)

## How should I start learning programming?

When you want to learn a programming language, I think you can consider a few approaches:

- **Buy/borrow a book like “An Introduction to XX Programming Language”**
    This is actually the approach I recommend the most. In my experience, reading a book is the most efficient way. A good introductory book is already organized systematically into learning units. Usually, these beginner books are not that different from one another—just pick one you like.
- **Use an online learning platform and find a course like “Introduction to XX Programming Language”**
    Let me mention a few well-known options. The open course websites of NTU, NTHU, and NCTU all have excellent courses. If a course is published there, it is almost certainly a good one, so you can learn with confidence—students at these schools learn this way too. Coursera courses are mostly taught by professors from overseas universities who tend to be good teachers. They are basically in English, but many have Chinese subtitles, and you can learn for free if you don’t buy the certificate. Hahow is a Chinese commercial course platform, usually taught by industry instructors. Courses cost roughly 1,000–2,000. I once bought a course and felt the value was very high for the price.
- **Skill exchange / find someone to teach you**
    I once did a skill exchange with a designer: she taught me drawing, and I taught her programming. If you have friends like that, this can be a lot of fun. Of course, you can also hire a tutor, or ask a friend to teach you—buy them an afternoon tea for a lesson, and your engineer friends will probably be happy to help.
- **Various beginner tutorials on the internet**
    I don’t recommend this approach as much. For beginners, it’s hard to judge the quality of online resources, and online materials are often too scattered; it’s also hard for beginners to know what learning path to follow. But if you have friends who are familiar with programming, you can ask them to help plan your learning resources. Experienced engineers can tell what you should learn first and what can wait, and what you should definitely learn versus what is optional.

    So I don’t recommend that you simply Google “Python beginner” by yourself. Searching is not a bad thing, but the results may not actually help you—and might even mislead you. The best approach is to ask an experienced friend to filter and curate resources for you.

## Which programming language should I learn?

I think there are two broad categories. If you are in business or the humanities, the most important thing about programming is to understand the concepts of programming, and to be able to use code to solve small everyday problems—for example, basic data processing. Once you can write simple programs, it will also be easier to understand what engineers are talking about and communicate with them.

If you are in science or engineering, you need a deeper understanding of programming, because mastering the details has a fundamental impact on problem-solving. Whether it’s program structure or performance, you need good design—both academic research and industry products tend to have higher requirements for software quality.

Based on these two groups, here are the beginner languages I recommend:

- Business / humanities: Python
- Science / engineering: C/C++

### Why should business/humanities students learn Python?

The reason to choose Python is that it is simple: there isn’t a lot of complicated syntax, and it also hides some more advanced programming language concepts (which you typically only need when learning C++). For beginners, Python is very friendly.

Python is an interpreted language, meaning you can write and run code line by line. That’s convenient for beginners too—you don’t have to finish writing an entire program before you can run it.

For example, you can do something like this in Python:

![image](https://user-images.githubusercontent.com/18013815/82886931-d57fc080-9f79-11ea-9045-be2a8f254d5b.png)

First, open a terminal (also called the command line). `> python` means launching Python.

`>>>` indicates user input. You can see that I typed one line at a time; each line runs immediately. When I call `print()`, it immediately prints out the value.

If you don’t understand what the screenshot is showing, that’s fine—typical tutorials will explain how to start writing code. The point here is that the development interface is highly “interactive”.

When you learn Python, you can follow a book or course video and type along while observing the output, because hands-on practice is the most helpful for building understanding.

Following the structure of a good book, you can learn important concepts step by step: variable declarations, `if`/`else` conditionals, `for` loops, function definitions, importing libraries, file I/O, and so on. Along the way, try writing small programs for practice—for example, how to find all primes below 1000.

One more benefit of Python is that you can use a visual tool like [Jupyter Notebook](https://jupyter.org/) to write code. Jupyter provides a web interface for writing Python interactively, so you might not even need to know what a terminal/command line is, which can make things even easier.

The Jupyter interface looks like this:

![image](https://user-images.githubusercontent.com/18013815/82887161-30191c80-9f7a-11ea-9313-2851bc267a38.png)


After you learn to write simple Python, you can do small tasks like batch-renaming files, reading a CSV file, and computing the mean and standard deviation. Also, data science and machine learning largely use Python today, so if you are interested in those areas, it’s easy to find more resources.

### Why should science/engineering students learn C/C++?

“C/C++” refers to C and C++. C was created earlier, and C++ came later as an improved version of C. Both are widely used today.

Why do I recommend C/C++?

The most important reason is that learning C/C++ helps you understand computer fundamentals more deeply. In particular, you can manipulate memory directly—this is the concept of pointers (Pointer). Languages like Python or JavaScript hide memory operations, so you can’t do that in the same way.

When solving engineering problems, using memory resources effectively is often a key factor. Only when you can work with memory can you precisely and efficiently control it. This is also why programs written in C/C++ are essentially always faster than those written in Python.

The learning method for C/C++ is actually similar to Python: follow a book and learn important concepts step by step—variable declarations, `if`/`else` conditionals, `for` loops, function definitions, importing libraries, file I/O, and the crucial concept of ***pointers***. The main difference is that C/C++ are compiled languages: you typically write the program, compile it, and then run the compiled output.

Note that C++ adds concepts like object-oriented programming (OOP) and templates (Template). Beginners don’t need to understand these topics at the very beginning.

C is widely used in embedded systems and operating systems. If you want to get started in these areas, learning C is recommended.

Otherwise, I would recommend learning C++ directly. C++ has stood the test of time. Most large applications—browsers, PC games, video editing software, and so on—are largely written in C++. System programming is also often done in C++. With its broad applicability, C++ is an excellent introductory language for understanding programming languages and computer fundamentals.

After you finish learning C/C++, you should still learn Python, because you don’t want to use C/C++ for simple small tasks—it’s too much trouble! And since you already know C/C++, learning Python should feel natural.

### Do I have to start with Python or C++?

There are so many languages in the world. You might ask: why not Java, Rust, JavaScript, and so on?

Of course, Python and C++ are only my recommendations. You can still learn other languages—after all, the basic concepts of programming languages are shared.

But in my view, learning other languages can create additional barriers. Python has a similar language in JavaScript, but JavaScript often requires you to understand how programs run in terms of “synchronous/asynchronous” behavior, which is already too difficult for beginners—it’s an advanced concept.

Languages similar to C++ include Java and Rust. Java is a compiled language, but it doesn’t have pointers. Although it is widely used, it’s clearly not as good as C++ for working with memory. Rust does have pointers, but because it emphasizes safety, it introduces the concepts of ownership (Ownership) and lifetime (Lifetime). In fact, you need these concepts in any language eventually, but you definitely don’t need to understand them at the beginning—people usually learn them later when they get more advanced. Rust forces you to understand them early, which makes the entry barrier too high for beginners.

So I’m not very encouraging about choosing other languages as your first language.

## What comes after the basics?

I often compare learning programming to learning English: it’s a must-have skill, easy to start but hard to master. Learning programming basics is like passing the entry-level English proficiency test—most people can speak a little, but the level is still shallow, and you need lots of practice and effort to become truly good.

Think about how we learn English: through lots of reading and writing. Writing code is similar. In addition to practicing a lot, you also need to learn foundational computer science knowledge to go further.

### Practice coding

Practice is the most efficient way to learn, because it validates whether you really understand. I recommend that after learning a concept, you go to the “[High School Online Judge](https://zerojudge.tw/)” to practice. Click “分類題庫” (categorized problem sets), and you can quickly find exercises that match what you’ve learned.

![image](https://user-images.githubusercontent.com/18013815/82929743-c2d7ac80-9fb6-11ea-8d3e-0aa23ec64bc7.png)

For example, take the problem “a006: quadratic equation”: you solve a quadratic equation using the formula you learned in middle school. Along the way, you need to compute square roots using a math library, and understand how to use `if` to check $4ac-b^2 >= 0$ to determine whether real solutions exist.

For beginners, if you practice through the “a-series” problems, you can roughly consider yourself to have reached an introductory level of programming.

### Improve your programming skills

After you learn programming basics, if you want to become much stronger, I think there are three courses you should learn next: “Algorithms and Data Structures”, “Operating Systems”, and “Computer Architecture”. I discussed them in depth in “[Three Must-Take Courses for Software Engineers](/post/2018/12/engineer_class/)”.

To summarize briefly: data structures and algorithms help your code run efficiently, but they don’t solve system-level problems. System-level problems require understanding operating system principles—how Windows or Linux works under the hood—so that your program can run faster on that system. Problems that OS knowledge can’t solve require understanding computer architecture, which involves assembly, memory systems, CPU design, and other hardware-oriented knowledge. Understanding hardware is what enables software to run faster.

These courses are available on the open course platforms of NTU, NTHU, and NCTU as well. I highly recommend Professor **Chih-Hsing Chang** at NTU’s “[Data Structures and Algorithms](http://mirlab.org/jang/courses/dsa/schedule.asp)”, which includes recordings for each lecture. I also recommend Professor **Yun-Nung Chen** at NTU’s “[Algorithm Design and Analysis](https://www.youtube.com/watch?v=gg7U1ZOPzSA&list=PLOAQYZPRn2V5C4Cx5tSLW8z0saWUm8LD-)”, which is an advanced version of algorithms.

If you plan to go into computer science in the future, or become a software engineer, I think you will need to build core CS literacy, which involves debugging skills, good coding habits and style, the ability to read code, collaboration and processes in software development, and general foundational knowledge about computers. I discussed these in depth in “[Core Competencies for Computer Science Students]((/post/2020/04/thought-about-cs-student/))”.

### Build applications

Beyond small scripts, you might also want to build applications. Once you understand the basic concepts of programming—functions, loops, file handling, and so on—you can consider learning to build websites, apps, chatbots, etc. That often requires learning new languages: web development involves HTML, JavaScript, and CSS. But since you already know how to program, getting started won’t be too difficult. Android development may require Java or Kotlin, but the fundamental logic is the same, so it should be approachable as well.

If you are interested in data science, after learning Python you can learn what NumPy is—a library that makes mathematical computation convenient in Python. With it, you can start implementing simple mathematical models to solve machine learning, deep learning, NLP, and related problems. I think O’REILLY books in data science are generally quite good. Usually, you only need basic programming concepts, and the rest is about understanding the math. In other words, as long as you can translate math into code, programming won’t be your bottleneck when learning data science.

## Conclusion

In this post, I proposed concrete ways to learn programming languages. Using the right approach matters, and I hope this helps reduce the number of “trial and error” mistakes people make. Please don’t believe in books or articles like “Understand Python in three hours” or “Python quick start”—in most cases, you still won’t understand after reading them. The key is to learn programming concepts systematically and practice applying what you learn by writing programs.

You can follow the methods suggested here to learn systematically—whether by reading books, watching videos, or getting help from others. For the two types of beginners, I recommend Python and C/C++ respectively. After you understand basic programming concepts, you can start building applications, or keep improving your programming skills.
