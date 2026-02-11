---
title: "New Software Engineer in Training (3): Breakthrough Growth and Becoming an Open Source Maintainer"
date: 2024-09-29 04:00:00
tags: [Japan, work, software engineer, New Software Engineer in Training, open source, open source projects]
des: "My second year working in Japan: I grew by leaps and bounds, and I even became a maintainer of the PcapPlusPlus project!"
lang: en
translation_key: software-engineer-grow-up-3
---

## Foreword

I said I would post an update every half year, but I dragged it out and a whole year passed. In the blink of an eye, I had already been at Mujin for two years. I’ve gradually adapted to how to survive here—you could say I’ve become an old hand. Starting from this past half year, I felt my skills grow extremely fast. On one hand, that’s from being sharpened by two years at the company. On the other hand, it relies even more on the skills I learned from the open source community. At this point, I’m familiar with most of the company code. Development is no longer a problem for me. Even though our codebase is still expanding, it doesn’t slow me down. Overall, things feel increasingly natural and under control.

This series (ongoing):
- [New Software Engineer in Training (1): My First On-the-Job Experience at Mujin in Japan](/post/2023/02/software-engineer/software-engineer-grow-up-1/)
- [New Software Engineer in Training (2): Workplace Socialization and Survival](/post/2023/11/software-engineer/software-engineer-grow-up-2/)
- [New Software Engineer in Training (3): Breakthrough Growth and Becoming an Open Source Maintainer](/post/2024/09/software-engineer/software-engineer-grow-up-3/) (this post)
- [New Software Engineer in Training (4): Promoted to Mid-Level Engineer, Quitting to Start a Company (Finale)](/post/2025/06/software-engineer/software-engineer-grow-up-4/)

<img src="https://github.com/tigercosmos/blog/assets/18013815/a96e7331-f056-4408-82e7-f58f99050014" alt="cover image"  width="500px"></img>

## The PcapPlusPlus open source project

In the previous post, I mentioned working on a networking-related project at the company. That project required using the [PcapPlusPlus](https://github.com/seladb/PcapPlusPlus) library. During development, I noticed many parts of PcapPlusPlus that could be improved, as well as bugs. So I kept filing issues and submitting pull requests. The author, seladb, is very patient. It’s common to have one or two hundred comments under a single PR discussion. To get a PR merged, sometimes it can take one or two months. After contributing steadily for about half a year, perhaps because seladb saw my enthusiasm for programming, he invited me to become one of the maintainers of PcapPlusPlus—even though I was still a newbie engineer.

<img src="https://github.com/user-attachments/assets/bbb9a506-c630-4370-ad4d-4316ef76584b" alt="github maintainer invitation"  width="500px"></img>

I can’t even describe how happy I was to receive that invitation. PcapPlusPlus is a project with close to 3K stars. On GitHub, projects with over 1K stars are already among the better ones. More importantly, I felt recognized. That sense of achievement is something a company simply cannot give me.

In open source, everyone is equal. Even though I only had a bit over one year of full-time experience, I could still discuss development as an equal with engineers who have ten years of experience. Even if my knowledge isn’t broad enough, occasionally I can still offer an idea that makes people go “oh, interesting.” In contrast, in a company you’re often constrained, and managers aren’t necessarily happy to hear your opinions. At first I tried to share my thoughts, but gradually I gave up. The survival rule at work is: do less, make fewer mistakes. That doesn’t mean I’m not thinking. On the contrary, I remember all the things that are wrong—I just don’t say them. Protecting yourself is the priority.

To improve as an engineer, you need a lot of learning. Honestly, a company doesn’t teach you that much. Contributing to open source perfectly fills that gap. By consistently contributing to open source projects, I keep learning new knowledge. The open source world also has more—and stronger—engineers you can interact with, and they’re often more willing to teach than senior coworkers at a company. Over this year, my ability to write code improved noticeably.

The open source community can also be warm. For example, not long ago Taiwan had a major earthquake. Hardly anyone at the office checked in on me, but seladb immediately DM’d me and told me to reach out if I needed any help. I was genuinely touched. seladb and I are just internet friends in the open source world, yet he felt warmer than many people in the “real world.” I think that’s also one reason I love contributing to open source software so much.

Sharing the PcapPlusPlus project at COSCUP 2024 in the sciwork track:

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Thanks to everyone&#39;s enthusiastic participation over the past two days at <a href="https://twitter.com/coscup?ref_src=twsrc%5Etfw">@coscup</a> . Whether participating in the career roundtable, project sprint, technical speeches, or communicating with us, everyone&#39;s enthusiastic participation made every part so exciting and successful! <a href="https://t.co/fQ5dRtIAVB">pic.twitter.com/fQ5dRtIAVB</a></p>&mdash; sciwork (@sciwork) <a href="https://twitter.com/sciwork/status/1820083713197449291?ref_src=twsrc%5Etfw">August 4, 2024</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

## A workplace mentor

Before I joined the company, I asked a foreign coworker what he thought of the company. He said he felt lucky because his manager was very willing to teach him. Even though he didn’t do a master’s degree, his manager was like a mentor to him. He joined at the right time and met a manager who was willing to teach, so I believe he must have learned a lot.

When I joined, however, the company had already grown too large. There was no longer a system of hand-holding newcomers. Managers were already busy just keeping up, and the newly promoted team leads hadn’t truly grown into their roles yet.

So after the expansion, Mujin is basically a “let the sheep graze” model. Learning is on you, and there is naturally no such thing as a “mentor.” That’s why I’m very grateful that [yyc](https://github.com/yungyuc) continues to teach me. If you’re not familiar with my story, you’ve probably seen me mention him many times in earlier posts. Ever since I was his TA when he taught part-time at NCTU during my master’s, he has continuously given me guidance and advice. That’s exactly what I lack in the workplace—he’s truly my workplace mentor.

Being able to meet someone willing to help you in life is incredibly fortunate. From graduation to two years into working, I’ve kept asking yyc about workplace issues and life direction. He always gives guidance generously. And it’s not just me. Any student he taught at NCTU, or any friend he met in the community, he’s willing to help. He’s a senior I genuinely respect.

yyc has a teaching project: [modmesh](https://github.com/solvcon/modmesh). Over the past half year I also contributed quite a bit of code, mainly implementing a NumPy-like low-level array. This topic was later used for a [PyCon JP 2024 talk](https://www.slideshare.net/slideshow/crafting-your-own-numpy-do-more-in-c-and-make-it-python-pycon-jp-2024/272070611). I had been contributing small bits since I graduated from my master’s, but my skills weren’t good enough back then. Now that I’ve improved a lot, I feel I can understand the project from a much broader perspective. PcapPlusPlus and modmesh have been the two biggest contributors to my growth in programming.

## The workplace

After more than two years of work, by Google’s level definitions, I think I’ve grown from roughly L3 to L4. My mastery of the company codebase, my independent development ability, my collaboration skills, and my raw coding ability—all improved significantly. Mujin has a similar leveling system. After two years, I expected to be promoted, but disappointingly my level stayed the same. It forced me to reflect on what I’m still lacking.

In the past two years, I built several relatively large features. I also have two sizable independent projects under my belt. And yet I was still assessed as having insufficient impact. That is extremely frustrating, because in the first two years, what tasks you get to take on is not really up to you. But now that I’ve been here for two years, my voice should increase. I can only try to “fight for” projects that are considered high-value and high-impact. There are many articles online about leveling and promotions. I’ve read a lot, but actually executing and producing results still takes time to explore and learn.

In terms of pure coding skill, I think I’ve already hit the bar. I review open source code almost every day. For the parts of the company codebase I’m familiar with, I can also do helpful reviews. But I can still improve further. Compared to truly senior engineers, I’m still too green. So I’ll keep writing more and reading more, pushing toward the next level. Three years is often considered a time threshold for “meeting the senior engineer bar,” so I hope to fill my missing parts over the next year.

## Writing a book

This year I also wrote a book, from January all the way to November. Back in my master’s days, I made a series of videos called “The basic essentials for CS students,” and I always wanted to turn that topic into a book. So I talked to 博碩出版社, and it went pretty smoothly. First I proposed a writing plan, designed the outline and content plan, and after signing the contract, I started writing.

Because I can’t write very fast, I planned for eight months. In total I had to write over 300 pages in A4 format, which would translate into around 400 pages in a normal book size. My daily progress was usually only one or two pages. The main reason is that you have to spend a lot of time thinking through the content, and then after you know what to say, you still need time to research, look up materials, draw diagrams, write sample code, and so on. Every step is time-consuming. The book was planned to be published by the end of 2024. I’m looking forward to it, and I’ll write another post after it’s officially published. I truly admire people who can deliver a manuscript in three or four months. For me, that feels unbelievable. You only realize how hard it is to finish a book once you actually try to write one.

## Conclusion

<img src="https://github.com/user-attachments/assets/105454bf-873e-4949-a8da-3a77fbb26f81" alt="wife picture"  width="700px"></img>
(Just casually showing off my beautiful wife.)

This year also had a lot of messy, random things: my Japanese girlfriend became my Japanese wife; I got much better at skiing; I traveled to many places; I spoke at COSCUP and PyCon JP; I met a ton of people; and so on. Too much happened, and it’s impossible to share everything. As I write, I even get lazy. I originally planned to post every half year, but it feels like many things require a longer time horizon to truly feel their impact. So going forward, posting once a year is probably enough. This post should have covered the “personal growth” part of this year.

Without realizing it, I’m about to turn 28. It feels unbelievable. I feel like I’m always in a learning state. After you improve a bit, you realize there’s even more you don’t understand. Life is learning forever—it doesn’t stop. It’s tiring, but also very interesting.

