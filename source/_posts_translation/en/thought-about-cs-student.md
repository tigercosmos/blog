---
title: "Core Competencies for Computer Science Students"
date: 2020-04-24 00:00:00
tags: [大學,研究所,資工,資訊工程,想法,基本素養, computer science]
lang: en
translation_key: thought-about-cs-student
---

What skills should a CS undergraduate have upon graduation? What skills should a CS graduate student have after finishing a master’s program?
Beyond what you learn in specialized courses, I want to seriously discuss the “core competencies” that CS students should develop.
<!-- more --> 

## Preface

I did not major in CS as an undergraduate; I only studied CS in graduate school. Before grad school, my path of learning programming was non-typical (see [〈How I Became Interested in Computer Science〉](/post/2019/12/cs-to-me/)). In my freshman year, I started getting interested in programming thanks to advice from an older student. Then I explored most beginner knowledge through self-study. I also had opportunities to intern at multiple software companies and contribute to open-source communities. It wasn’t until the second half of my junior year and the first half of my senior year that I actually started taking CS department courses. Only in graduate school did I truly become a “CS student”.

Because of this special learning path, I’ve been able to observe that many classmates lack what I consider important core competencies. I think it’s a pity if CS students graduate without developing these competencies. To avoid sounding like I’m praising myself, I’ll admit that I’m also missing some of them—so this is also a reminder for myself to build them properly.

The core competencies I’m talking about include five points:

- Programming development and debugging skills
- Good coding habits and style
- The ability to read code
- Collaboration and process in software development
- Fundamental knowledge about “everything computer-related”

> In addition to this post, you can also refer to my teaching video series: “[拯救資工系學生的基本素養](https://www.youtube.com/playlist?list=PLCOCSTovXmudP_dZi1T9lNHLOtqpK9e2P)”

## Programming development and debugging skills

There’s not much to say about “programming development”: everyone can write code, but writing it well is difficult. Since you are a CS student, you should train your coding ability seriously. Try to become familiar with the characteristics of languages like C++, Python, and JavaScipt. If you have extra time, refer to advanced books such as *C++ Primer*, *Effective C++*, and *Effective Python*. If you search online, you’ll find many experts sharing their personal reading lists. And it’s not just that: writing good code also requires combining various knowledge, including making good use of the standard library (STD), data structures and algorithms, and design patterns (Design Pattern).

Among your classmates, there are surely C++ masters who feel impossibly strong—people who deeply understand language features, and can even serve as someone else’s “spec”. Becoming that kind of “god-level” person is too hard, and you don’t necessarily need to. But as far as I know, better companies tend to have colleagues with strong language mastery. After all, this is a tool you’ll use to make a living for a lifetime—so you should also set the expectation to become fluent.

When you write code, you will inevitably encounter bugs. You can’t escape them. Sometimes you debug for days and feel like you want to die. So how do you debug? Are you still using `print` everywhere? Learn to use debugging tools. Using debuggers skillfully can make development much smoother. You don’t have to learn GDB specifically—modern IDEs have great debugging tools. The concepts are the same; as long as you achieve the goal, it’s fine.

Using `print` is convenient and simple, but if you can set breakpoints with a debugger, you can obtain the data you want more easily. For example, you can inspect the backtrace and go through stack frames one by one to observe how pointers and values change as they are passed around. Doesn’t that sound more efficient? Being able to quickly identify where the bug truly comes from is real skill.

Finally, make good use of development tools: a good text editor (I use VSCode) or an integrated development environment (IDE), plus helpful plugins like static analysis, auto-formatting, syntax suggestions, and auto-completion. You should also learn the tools that work best on different platforms, because you may develop on Windows, macOS, and Linux. It’s also highly recommended to learn Vim (no Emacs flame wars 🤣), because even if you use an IDE in daily work, there will be times when you need to edit code directly on a server—and using Nano can be inconvenient.

## Good coding habits and style

There’s a saying: “If you haven’t written 100,000 lines of code, you’re not a qualified CS student.” I’m actually skeptical of that—given typical school workloads, it’s hard to reach. But from another angle, there’s also the saying that 10,000 hours makes an expert, and that time is roughly equivalent to 100,000 lines of code. Still, it can’t be chaotic code written only to “make it run”. Only well-written code is meaningful.

Good habits and style are also called clean code. The code you write won’t be read only by you; it should also be clear to others, and it should still make sense to you years later instead of leaving you confused. Beautiful code can be self-explanatory without comments.

How you write code also affects how easy it is to write tests. Unplanned code is very hard to test, which is why the concept of test-driven development (Test Driven Development, TDD) exists. You don’t have to do TDD, but you should always think: will this code be easy to test in the future? Will it be easy to debug? If the answer is no, then your habits and style still need improvement.

To build good coding habits, you can keep reading excellent engineers’ code and reflecting on how it differs from yours. You can also read books like *Clean Code* and learn directly from how bad examples are fixed.

## The ability to read code

For typical homework or personal small projects, your code is probably at most a few hundred lines—maybe a thousand. But most well-known open-source projects are large: hundreds of thousands or millions of lines. Mature software developed by software companies is also often on the scale of millions of lines. What do you do then?

If you want to understand how Linux pthread works, do system-level development, or modify a bit of Linux source code, you have to first understand what’s going on inside Linux—and Linux itself has millions of lines of code. Chromium is a well-known open-source browser engine. Many browsers and applications adopt its core, so if you work at a company built on Chromium, you need to be able to read Chromium’s source code. These examples show that it’s very likely you will deal with large projects. So how do you quickly find the code relevant to your goal, understand the execution flow, and understand the context? That is what you need to practice.

To illustrate how strong engineers can be in this area: when I contributed to the Servo project (around 100,000 lines of code), a new contributor modified an entire module in their first contribution. When I made my first contribution, I only changed a few lines; but they already understood this not-small project, knew how to modify it, and contributed around a thousand lines of code.

As for how to develop this ability, you can start practicing with smaller projects and gradually scale up, experiencing how architecture and organization change as projects grow. This article—[〈在 Linux 理解大型 C/C++ 專案的輔助工具〉](https://medium.com/fcamels-notes/%E5%9C%A8-linux-%E7%90%86%E8%A7%A3%E5%A4%A7%E5%9E%8B-c-c-%E5%B0%88%E6%A1%88%E7%9A%84%E8%BC%94%E5%8A%A9%E5%B7%A5%E5%85%B7-f794c3aa43f7)—is also worth reading; it mainly discusses which tools can help you. But the most important thing is still: practice more, and observe more.

## Collaboration and process in software development

Before entering CS grad school, I had already interned at several companies and participated in many open-source projects, so I’m very familiar with software development processes and collaborative development. In software companies, a Git + GitHub-like version control workflow is fundamental. On top of that, you also need testing, quality control, deployment, and so on. CS students will eventually work in software companies, and you can’t expect companies to teach or train you from scratch—so it’s better to learn these things in advance.

There are many version control systems, but the mainstream is already Git + GitHub. Why learn Git? Your own projects can be version-controlled. If you mess something up, you can go back to a previous version. When developing with others, you can use Git to merge code—please don’t share zipped source code packages with each other for collaboration anymore.

How to use Git for `git clone`, `git commit`, `git push`, `git branch`, `git checkout`, and `git merge` are the basics of the basics. And you should at least know how to fork a project on GitHub and open a Pull Request. Too many students (whether at NTU or NCTU) don’t know this workflow, yet they often have group projects or need to help TAs teach. So I plan to write a proper tutorial later.

I think writing tests is a good habit. How do you know the code you changed still works every time? Verifying each change with tests increases your confidence in your code. But that doesn’t guarantee your tests cover everything—so tests must be continuously updated too. There’s a lot of knowledge around testing: unit tests, integration tests, E2E tests, QA tests, and so on. Interested readers can search for more information.

Also, continuous integration / continuous delivery (CI/CD) is a classic software development buzzword; if you’re interested, go learn about it too.

## Fundamental knowledge about computer systems

Computers and programming languages are like a knife and its blade: your code must be good (a sharp blade), and you must also understand computers (a tough knife). Intro to computer science, computer organization, computer networks, operating systems, algorithms, and data structures—CS students all learn these. They are the most important tools a CS student should have. When solving problems, you often have to return to the most fundamental principles or the underlying system architecture.

If you understand other CS areas, even better: natural language processing, machine learning, computer graphics, computer vision, compiler principles, heterogeneous computing, cybersecurity, and much more. I think you may have opportunities to use these pieces of knowledge. Even if you don’t, learning their scientific principles and broadening your horizons is still interesting.

## Conclusion

In this post I mentioned several core capabilities: programming development and debugging skills, good coding habits and style, the ability to read code, collaboration and process in software development, and fundamental knowledge about computer systems. I deeply admire some well-known computer scientists, software engineers, and CS professors, and I hope I can be as strong as them. I think the first step is to master these core competencies.

I don’t want this post to be misunderstood as criticizing the status quo. These core competencies come from real pain in my own experience. When contributing to open-source projects, I spent a huge amount of time getting corrected and having my work rejected because I didn’t know how to use Git; at Company C, I bugged coworkers for a long time before learning GDB, and later I realized how foolish it was not to know it; at Company S, I wasted a lot of coworkers’ valuable time being taught Vim, and later I realized there were many opportunities to use Vim—writing commit logs too; after interning at many companies, I realized that because I didn’t come from a CS background, my understanding of CS knowledge was too shallow. There were too many things I didn’t know, which made me feel inferior. At the same time, my mastery of programming languages was too weak, and what I wrote was far worse than others.

There are so many moments of “if only I had known earlier”. I hate the feeling of being looked down on, being thought useless and weak. I believe you don’t want that either. So why not stop saying “if only I had known earlier”?

This post is also meant to encourage CS students at all levels. And it’s not limited to CS students—anyone who wants to step into the CS field can refer to the capabilities I listed here. I hope we can all improve together.

