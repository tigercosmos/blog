---
title: "Why You Shouldn’t Choose Computer Science as Your University Major?"
date: 2020-07-23 07:00:00
tags: [資工, computer science, 想法, 校系選擇]
des: "This post introduces what CS majors typically require and what different CS subfields study. It also discusses what university life looks like and what future career prospects may be."
lang: en
translation_key: why-dont-choose-cs
---
## Introduction

Recently, it’s the season of university placement results, and many students are busy collecting information and deciding which department to study. Choosing a school and a major is a major life decision. Don’t count on transferring majors—that’s wishful thinking. You need to make the right decision from the start. So I want to share my perspective: why I don’t recommend that applicants follow the crowd and choose a CS major, so that they won’t end up with long nights and endless nightmares later.

First, a bit of background about me. I graduated from NTU’s Bioenvironmental Systems Engineering. During college I took courses and did projects in the CS department, and I am currently studying in an NCTU CS graduate program. In the past, I interned at four software companies, and I’m currently a part-time student worker at the National Center for High-performance Computing. I also have extensive open-source development experience, I like writing CS technical articles, and I’ve given several public talks. If you’re interested, you can read “[How I Became Interested in Computer Science](https://tigercosmos.xyz/post/2019/12/cs-to-me/)”, which describes how I discovered my passion for CS. My views come from my understanding of the CS industry and academia.

## What do CS majors basically learn?

Let’s first look at the required courses for NTU CS.

![台大的必修](https://user-images.githubusercontent.com/18013815/88225236-bca33b00-cc9c-11ea-9769-62e47043c9d6.png)

Calculus and the usual freshman-year science courses are similar across engineering and science majors, so don’t worry too much about them.
In the first semester of freshman year, you learn basic **programming**. In the second semester, you take **data structures and algorithms**, learning how to write programs “efficiently”.

In sophomore year, I think three courses are relatively important: **algorithm design**, **system programming**, and **operating systems**. The remaining **math** is a toolkit you might use if you happen to need it, and **digital systems** teaches the basics of logic circuit design.

**Algorithm design** is an extension of data structures and algorithms from the second semester of freshman year. It uses mathematics to analyze how different computation methods affect runtime and memory usage.

The software we write day-to-day is called an application—usually websites, apps, and general-purpose software. **System programming** teaches you how to write lower-level programs on an operating system—programs that need to communicate directly with the OS. For example, applications often use existing libraries for networking, but those libraries themselves have to handle system-level tasks like networking, thread management, and communication management. In short: applications are written for general users, while system programming is more like writing programs for programmers.

Today, the three major operating systems on the market are Windows, macOS, and Linux. Without an operating system, your computer is basically a piece of scrap metal. The purpose of an **operating system** is to communicate directly with hardware and provide users with a convenient interface. The OS integrates hardware devices like the CPU, memory, disk, and networking into an interface, so when you open Windows in daily life, you don’t need to manage hardware devices manually; the computer still runs in an orderly way, and you can run all kinds of applications on it. The operating systems course is about the principles behind all of that.

In junior year, I think the most important courses are **computer architecture** and **computer networks**. The remaining **automata and formal languages** is, simply put, about understanding what “computation” really is—a very mathematical course.

**Computer networks** introduces why the internet works: what mechanisms build the networks we see everywhere today.

**Computer architecture** mainly introduces how the CPU works under the hood. Roughly speaking, it explains why Intel and AMD CPUs can run, what hardware pipelines happen when you give a task to the CPU, and how understanding those principles can help when you write programs—like reading the manual when you use a tool.

Some students might think: “Programming is in demand, so CS must be a good choice.” That’s true to some extent. But if that’s your only reason, you don’t necessarily need to major in CS. As you can see, CS is not about teaching you how to build websites or apps—if you come to CS expecting that, you definitely won’t learn it. And honestly, many professors probably don’t write much application code either. If you’re not interested in the science and engineering principles behind computers, majoring in CS can be very boring.

In fact, many departments offer basic programming courses. Universities’ computer centers also often provide general education courses that teach how to build websites, apps, or do numerical computation with programs. If you want to learn application development, online course platforms or programming bootcamps can actually be the fastest approach.

If you worry that not knowing computer science will disadvantage you in the future, you can consider taking the three most important CS courses. I summarize them as **data structures and algorithms**, **operating systems**, and **computer organization**. Taking these three is like getting a glimpse into the mysteries of computer science. For more detailed introductions, you can refer to my previous post: “[Three Must-Take Courses for Software Engineers](https://tigercosmos.xyz/post/2018/12/engineer_class/)”. Students from other departments can take or audit these courses, so you can major in something else and treat computer science as a supporting skill.

Finally, the total required credits for CS are only around 50-something. I honestly think that’s pretty few. By the second semester of junior year, you may have no required courses left. In my previous major, we had more than 70 required credits—and even in the first semester of senior year, we still had required courses. So if you’re the type who won’t do extracurricular learning or explore other fields on your own, I really don’t recommend choosing CS. After the second semester of junior year, you’ll have a lot of free time—if you spend all of it gaming or wasting time, that would be seriously problematic.

## What else do CS majors learn?

Beyond the basic required courses, the rest are electives. Undergraduate electives are basically the same as graduate-level courses, meaning CS undergrads and CS grad students take the same set of electives together. But don’t worry too much: I think CS graduate-level courses are among the easiest in science and engineering. As long as you can code, everything is negotiable.

There are many CS subfields. Here are a few examples:
- **High-performance computing**: using every possible method to make computers compute faster—for example, using more CPUs, or having GPUs help with computation. But as you add more devices, management becomes harder and communication efficiency between devices decreases.
- **Operating systems, distributed systems**: making systems run more efficiently and faster. One approach is to treat many systems as one system—this is called a distributed system.
- **Programming languages**: studying the design of programming languages—for example, the differences between Python and C++.
- **Compilers**: how can a compiler make the compiled program run faster? For example, without changing the original source code, turning a program that takes 1 minute into one that finishes in 10 seconds. You can also make a compiler compile faster, but compilation time and runtime performance often require trade-offs.
- **Computer vision**: how to give computers human-like vision—for example, how to stitch panoramic photos together, or how to reconstruct a 3D model from multiple photos.
- **Image processing**: how to process photos—for example, how PhotoShop processes images under the hood.
- **Networking**: wireless networks, broadband communication, IoT networks—studying how to make network communication faster.
- **Computer architecture**, **embedded systems**, **system-on-chip**: studying how to make hardware run faster. For example, Google developed its own TPU for deep learning services, which can compute faster than GPUs.
- **Artificial intelligence**: including machine learning and deep learning. The idea is simple. For example, we can learn to tell the difference between cats and dogs, and we can also have machines learn—perhaps by recognizing information such as color, shape, and size, and building a mathematical model so computers can judge cats vs. dogs too.
- **Natural language processing**: enabling computers to understand human language, or generate human language. In the past, this was done through probabilistic statistics. For example, after “我喜歡” (“I like”), there’s a high probability the next word is “你” (“you”). Recent research typically uses machine learning and deep learning methods.
- **Information security**: we often hear about “security vulnerabilities” in software. This field also includes computer viruses, trojans, eavesdropping, spyware, and so on—how to discover and prevent them is what this discipline is about.
- **Human-computer interaction**: computers, phones, smart glasses, smart watches—these are all “human-computer interfaces”. UX and user design are big topics. For example, confirm buttons are usually placed on the right, cancel buttons are often red, and when your mouse accidentally leaves a menu you may have a few seconds to “regret” and return—these are all part of interaction design.

There are many more areas, such as **bioinformatics**, **database systems**, **big data**, **computer animation**, **wearable devices**, and **cloud computing**—I won’t introduce them one by one. The basic requirement is always: you must be able to code. Also, CS required courses are the foundation of computer science. You must learn them well, otherwise you will never be able to learn these applied fields well.

## Life in a CS major

Some people share that during their CS major they joined student associations, traveled around, joined clubs, and lived an easy and joyful life. I think those people are one-in-ten-thousand talents. For most people, it’s absolutely impossible to be that relaxed.

In reality, CS comes with a huge amount of coding. There’s a saying: if you haven’t written 100,000 lines of code during college, you’re not a passing CS student. What does 100,000 lines mean? A strong engineer might write around a hundred lines of code per day at work. If you do the math, it’s roughly achievable if you seriously code throughout all four years of college. Writing from morning to night, or even from midnight until the next noon, is commonplace. There is no such thing as a CS student who can’t code—you can code badly, but you can’t not know how to code.

From my observation of strong classmates around me: during the school year or summer, they will definitely intern in industry. Accumulating industry experience early helps growth and future employment. They also do research—if you want to go to graduate school or become an expert in some domain, you need research to understand that domain more deeply. People still join clubs or travel, but most of their remaining time is spent on learning, work, and research.

If you like exploring the world, you can also visit other domains. Many fields need to collaborate with CS. For example, biomedical science plus CS extends into bioinformatics. And physics, chemistry, atmospheric science, and economics all need computers for scientific computing. Even civil engineering has computer-aided groups. Computer science is more like a tool that applies to all industries.

## Future career prospects in CS

CS students’ salary prospects have a very strong positive correlation with both their ability to code and their understanding of specialized domains—just like a great pianist charges much more than a third-rate pianist. A typical new graduate’s salary can range from monthly 30–40k to annual 1–2 million.

Common career paths include software engineer, hardware engineer, firmware engineer, data scientist, data engineer, web designer, and system engineer. Job requirements vary flexibly depending on company needs. For example, software engineers might need to understand natural language processing or image processing; data scientists may need biology background knowledge in addition to machine learning.

If you work at a regular company doing app or web development, it might be around 30–60k per month. This is the career path for the vast majority of CS graduates. The next step is better companies such as MediaTek, TSMC, Microsoft, and so on—doing system programming, complex software, or hardware architecture—where you can get an annual salary of around 1 million. If you’re strong enough to join a top-tier domestic software company as a core developer, you might negotiate 1.5 million. If you’re truly strong, I’ve also heard of people getting 2 million right after graduation. But those who can get high salaries are still a minority. I think only the top 10% of all CS students have a chance of earning over a million.

So if you have unrealistic fantasies about CS—thinking that because CS is “hot” you’ll definitely get a high salary—you’re wrong. It’s true that because AI is popular and “artificial intelligence” is hyped everywhere, the IT industry is very hot and there are indeed more openings. But most openings are relatively low-tech, and the pay is correspondingly not great. The market has large demand for web, Android, and iOS software engineers, but most of those roles also have relatively low technical requirements. It’s still relatively rare to need truly top experts for building websites—unless you’re talking about companies like Google, Facebook, or Twitter, which need very strong people. Other higher-paying jobs are usually in more advanced specialized domains. On one hand, demand is smaller; on the other hand, reaching the required capability is not easy.

So studying CS won’t automatically make you a “winner in life”. You still need to work hard to become truly strong before you can have better prospects. If you’re choosing CS just because the IT industry is popular, you should reconsider. I also keep thinking that the IT industry may be in a bubble. My main reason is that I think AI is a hyped topic. It’s almost bizarre how so many graduate programs at NTU are doing deep learning. It doesn’t feel like we especially need so many people researching deep learning right now, and in the future we may need even fewer people doing it.

## Conclusion

This post introduced what CS majors typically require and what different CS subfields study. It also discussed what university life looks like and what future career prospects may be.

In fact, majoring in CS is mostly about learning computer science, and you also need to practice coding a lot on your own. You can ask yourself: are you interested in how computers work? Why can code run faster? Why can browsers render webpages? Why can Google find answers among billions of data points in just a few seconds? How does Alpha Go play Go? How do chatbots understand your questions? How does Pixar compute animation? How does NASA control rockets with computers? Why did Apple want to abandon Intel and switch to ARM?

There are so many interesting questions. These are what CS explores. But if you are not interested in any of the questions above, then maybe don’t study CS. Of course, you can separate interest from career—you can major in CS for a living, and I think you can definitely live a pretty decent life. But if your career is also your interest, wouldn’t you live more happily? You may have better choices.
