---
title: "A Junior Programmer’s Path to Growth"
date: 2019-04-09 01:23:39
tags: [software engineering,the clean coder,programming,growing up,軟體工程,軟體工程師]
lang: en
translation_key: freshman_engineer
---

### Notes, Summary, and Commentary on *The Clean Coder*

<img class="dz t u gn ak" src="https://miro.medium.com/max/4122/0*Pvn4y7pOrW5BB_Yc.JPG" role="presentation"><br/>

Recently, I finished reading *<em class="hr">The Clean Coder</em>* (Chinese title: “無瑕的程式碼 番外篇－專業程式設計師的生存之道”). It happens to answer one of the most common questions I see online:

<span style="font-size: 16px; color: #696969; font-style:italic">What is the difference between a junior engineer and a senior engineer?</span>

<!-- more --> 

Many people think the difference is purely technical: juniors know fewer technologies, rely more on frameworks, and have fewer tools at their disposal. Seniors, on the other hand, know more technologies, can explain trade-offs, choose the best architecture for different scenarios, and—because they’ve stepped on many landmines—can often avoid problems directly.

At first, I thought so too. And yes, that is indeed one of the differences—but it is not the most critical part. The most critical part is how you handle people and situations. *<em class="hr">The Clean Coder</em>* is entirely about what it means to be a professional, disciplined engineer: how to communicate with people, and how to do the work of programming well. In fact, the concepts in the book apply not only to engineers but to all professions; it just uses software development as the running example.

## Saying “Yes” and Saying “No”

We are all assigned tasks by managers, PMs, coworkers, or clients. You need to learn how to say “Yes” and “No” properly. A professional should not overcommit. When faced with something you cannot do, say “No”, and then give the other party a reasonable commitment—say “Yes” within your capabilities.

<span style="font-size: 16px; color: #696969; font-style:italic">If you can, you can; if you can’t, you can’t—don’t say “I’ll try”.</span>

The book gives several examples to illustrate when to use Yes and No. You need to read them yourself to feel the context. I think saying “yes” or “no” correctly requires two abilities. One is having a strong grasp of the situation—being able to analyze complexity and deadlines clearly. The other is being able to persuade the other party to accept your reasoning, which inevitably requires a lot of communication.

## Lean Software Development

As people generally understand it, a professional programmer should have better and more efficient ways of working. This may sound like old advice, but it’s worth self-checking.

### Mood

Programming is a creative job. Like an artist, you need “inspiration” and “focus”. Some people have rituals before coding; some like coding late at night; some like coding with music; some need to maintain their own rhythm. In short: keep yourself in a good mood, adjust yourself to the most comfortable state, and then start generating ideas.

### Help

We are on the same team, and helping each other is part of the job. Don’t be stingy about helping others—you may gain more in return. Also, don’t be arrogant and refuse to ask for help. Sometimes we need others’ assistance. For example, there is an interesting debugging method—the <a href="https://zh.wikipedia.org/wiki/%E5%B0%8F%E9%BB%84%E9%B8%AD%E8%B0%83%E8%AF%95%E6%B3%95" class="dj by jn jo jp jq" target="_blank" rel="noopener nofollow">rubber duck debugging</a> technique:

<span style="font-size: 16px; color: #696969; font-style:italic">During debugging, troubleshooting, or testing, the operator patiently explains what each line of code does to a rubber duck, which helps trigger insights and uncover contradictions.</span>

<img class="dz t u gn ak" src="https://miro.medium.com/max/1528/0*tFvBXruadSGl0FjN.jpg" role="presentation"><br/>

Your coworkers are basically playing the duck. Also, through Code Review, when you help check your coworker’s code, everyone is learning from one another.

### The importance of testing

The book repeatedly emphasizes the importance of testing. Of course we all understand that, right?<br>Since the book already provides some examples, I want to share one of my own experiences here.

Back when I was interning at Cloudmosa, one day I noticed a strange bug: in Puffin, when you right-clicked a link on a webpage and chose “Add Bookmark”, Puffin couldn’t capture the link’s title.

That title is either the text content wrapped by the <code class="gs kb kc kd ke b">&lt;a&gt;</code> tag, or whatever is declared in <code class="gs kb kc kd ke b">title</code>. In short, when the “Add Bookmark” dialog popped up, the title field was empty, so I needed to fix the bug. After investigating for a long time—tracing through the stack layer by layer—I found that the data passed from Mango (Puffin’s browser engine implementation; see <a class="dj by jn jo jp jq" target="_blank" rel="noopener" href="/coding-neutrino-blog/440c91cece8f"><em class="hr">How the puffin browser works</em></a>) was missing something. The fix wasn’t hard: just add the missing data to the protocol between Mango and Puffin.

I was happy—Puffin could receive the data, and the bookmark dialog now had a title. Two days later, I received an email from CTO Sam: “Tiger, your patch will take down our service.” The reason was that Puffin and Mango had many versions deployed, so when you change the protocol you must consider backward compatibility. Sam was kind enough to fix it directly, but it still scared me.

This taught me two things. First, testing is really important: with solid tests, this kind of issue would be caught immediately. But because I was working on the development branch, we didn’t run a full test suite on every commit. Second, considering <strong class="hf kf">backward compatibility</strong> should be common sense—but I was too inexperienced and overlooked something so important.

The book strongly advocates Test Driven Development (Test Driven Development, TDD). I also think TDD is a good development approach: it helps you define requirements and logic, and it prevents you from getting stuck later when tests are hard to write. The book even makes a bold statement:

<span style="font-size: 16px; color: #696969; font-style:italic">The benefits of TDD imply one thing: if you are not using TDD, it may mean you are not professional enough — The Clean Coder</span>

But you still need to make the best choice depending on the context; no single rule is absolute.

Testing is usually part of the continuous integration / continuous delivery (CI/CD) pipeline. More broadly, it’s part of operations (DevOps), including unit tests, integration tests, acceptance tests, quality tests, and so on. I’ll leave that for everyone to explore on their own.

### Practice

As we get more senior, we may write fewer “simple” things—for example, how to implement Quick Sort. Everyone who wants to interview will grind LeetCode, and then after getting a job they forget it—just like cramming for college exams.

This kind of “quick drill” is actually helpful for maintaining your touch and inspiration. Honestly, while writing this post, if you asked me to write a bug-free Quick Sort within 15 minutes, I’d probably struggle. I still remember the concepts, but I’m rusty.

<span style="font-size: 16px; color: #696969; font-style:italic">As the saying goes: “Practice makes you better"</span>

In addition, writing more small projects is also a good idea. Even CS students who have taken Compiler, OS, and Computer Architecture may not have built a complete system themselves; if you only understand the principles but have never done it hands-on, it can be equivalent to “not knowing” it—reality is harsh, and practice is always harder.

<span style="font-size: 16px; color: #696969; font-style:italic">Many people think they already code fast enough, but that’s because they don’t want to be faster.</span>

I don’t know if you have seen <a href="https://medium.com/u/5879ccb41e31?source=post_page-----5b55f279630c----------------------" class="kg az by" target="_blank" rel="noopener">Jim Huang</a> (jserv) live-code on stage. I think live coding is truly difficult: programming is creation—it needs inspiration, and it needs incubation. But if others can code fast *and* well, we can improve through practice too.

## Time, Estimation, and Planning

Everyone has limited time. If your efficiency is poor, you should examine whether your time allocation and working methods need improvement. The Pomodoro Technique can be a good approach: focus intensely for a period, then rest, and repeat. Human energy and the body have limits, so excessive overtime and sleep deprivation are not good strategies.

Meetings themselves are expensive. The company pays a lot to hire each person; if the value of a meeting is less than the development time it consumes, it is not worth it at all. There are methods like Agile or Scrum that can be helpful. Also, you should only attend meetings you are obligated to attend or that bring you real value.

Estimation is commitment. When you say “Yes” to someone, you are responsible for delivering within the deadline. You can, of course, add buffer—maybe two standard deviations—so you can be confident you have a 99% chance of being fine.

### Design and planning

One way to distinguish Junior, Senior, and Lead is also their ability to plan and estimate development.

<ul>
<li id="2e3e" class="hd he em at hf b hg hh hi hj hk hl hm hn ho hp hq kh ki kj">Juniors often don’t know how to plan; they work by doing whatever comes to mind.</li><li id="25dd" class="hd he em at hf b hg kk hi kl hk km hm kn ho ko hq kh ki kj">Seniors usually have a good plan: they know how to break tasks down into steps, may even produce design documents, and can distinguish priorities.</li><li id="bcb8" class="hd he em at hf b hg kk hi kl hk km hm kn ho ko hq kh ki kj">Leads have senior qualities, can keep projects running in an orderly way, and can think from others’ perspectives—positioning the whole team in a way that benefits the company.</li>
</ul>

So the differences between engineers are not merely about “how much tech you know”.

## Team Collaboration

Unless you are a solo freelancer, you will be part of a team. Teams involve collaborative development, responsibility division, and project management. If you don’t have opportunities to work on large-scale collaborative development, I recommend trying to contribute to open-source projects. For example, Mozilla’s <a href="https://github.com/servo/servo" class="dj by jn jo jp jq" target="_blank" rel="noopener nofollow">Servo </a> browser engine is a great place to practice—it has nearly a thousand contributors, and you will see the complete workflow of a large project. How does a team build cohesion? There’s not much to say: communication, communication, and more communication. Communication is an art that any professional should possess.

## Mentorship

Finally, any senior engineer should take care of and mentor more junior engineers. Programmers are like apprentices in a craft, growing through continuous learning. A good mentor can help you grow quickly, avoid many detours, and set an example in your mind. We have all benefited from others’ help, and we should enthusiastically help the next generation too.

## Summary

This post organized key points from *<em class="hr">The Clean Coder</em>* and added quite a bit of my own interpretation based on personal experience. During my growth, I have worked at several companies, seen many engineers, and experienced different cultures. Even though my time in the industry is not long, reading this book still resonated with me. I hope some of my thoughts can help others.

So—am I a junior engineer? Judging by the perspective throughout this post: yes. I’m still just a rookie, and the road to improving myself is still long.

