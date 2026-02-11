---
title: "My 2025 Fall Mid-Level SWE Job Hunt in Japan and Taiwan"
date: 2025-10-21 01:00:00
tags: [Japan, Taiwan, job, software engineer, T2]
des: "This post documents my 2025 fall job hunt for mid-level software engineering roles in Japan and Taiwan, including the companies I applied to, a detailed record of my interviews with T2, and my takeaways."
lang: en
translation_key: find-swe-job-in-japan-2025
---

## Overview

This year (2025), I started a part-time startup in February. At the end of May, I left Mujin to work on the startup full-time. In mid-September, the startup officially failed, and I started my 2025 fall job-hunting journey.

At Mujin, I worked in backend (in the broadest possible sense of "backend"). My strength is C++. And because I spent the eight months of my startup working on AI, the roles I targeted this time were AI application engineer, backend engineer, and C++ engineer. My total full-time experience was exactly three years. My final level at Mujin was E2, roughly equivalent to Google L4 (mid-level engineer), so I also searched for mid-level and senior roles.

One advantage I have in Japan is that I hold permanent residency. It might help a bit, but since Japan's visas are comparatively easy to obtain (unlike the U.S.), this advantage does not change things much. What really expands opportunities is Japanese proficiency. But roles that require Japanese often do not pay well (typical Japanese companies). That said, companies like Google and Amazon sometimes have positions that require Japanese, and if you cannot speak Japanese, you miss out on those high-paying opportunities.

Because I cannot do business Japanese, I could only apply to English-speaking roles, which means the pool is relatively small. I spent about one day applying to essentially all relevant openings I could find on LinkedIn in Japan. It felt like Japan might not be that easy, and with the language disadvantage, I decided to try Taiwan as well. I figured that, in the worst case, I would have to go back to Taiwan. So I spent another day applying to all relevant openings in Taiwan.

In Japan, I used LinkedIn and Japan Dev. I also had xAI run a deep search to look for companies that matched my background, especially those with careers pages on their websites but without postings on LinkedIn. It did find a few, and I applied manually. In Taiwan, I used LinkedIn, Yourator, and Cake. I also casually applied to some roles in Europe and the U.S., but heard nothing. I worked with recruiters on both the Japan and Taiwan sides as well. My daily routine was basically refreshing LinkedIn and constantly talking to new recruiters. In the end, recruiters did not help that much, though I think there is a lot of luck involved.

I started looking in mid-September. By late October, I received an offer from T2, Inc. in Japan. I found a job in about a month. I was very lucky, and I genuinely feel blessed.

## Application Log

A brief record of the companies I applied to:

- **Ghosted & resume rejected**: Woven by Toyota (I applied to many roles and got zero response; maybe I was blacklisted from previous applications), Cohera, Tier IV, OpenAI, Anthropic, ispace, Mapbox, Pairs, Amazon TW, Amazon JP, Applied Intuition, Rakuten, LegalOn, CookPad, Zeals, DeepX, MODE, CADDi, LexxPluss (surprisingly rejected despite my AGV background), Rapyuta Robotics, TSMC, Japan AI, Sony, etc. You can see that most are still centered around robotics and AI, with a primary focus on Japan, plus some random applications.
- **Responded but I declined**: Appier JP (a friend referred me, but I only heard back a month later; it was my original top choice because I heard the Ad Cloud Bidding team had great conditions and did not do LeetCode-style interviews, but I declined after getting the T2 offer), 詹姆斯科技 (declined after getting an offer), 江夏株式会社 (a China-backed startup; not interested in the domain), Alpaca (not interested in crypto), Speechify (sent an online assessment without even saying hello), Citadel Securities (same: assessment first, no greeting). I declined two because they reached out too late, and I declined several because I really dislike companies that send assessments without even a basic HR introduction. I also care a lot about transparent funding and industries with a strong outlook.
- **Rejected after the first round**: PicCollage (had take-home work first, then the first round was HR + Tech Lead together. I found it very odd to interview with HR and an engineer in the same session. For a startup, it felt very old-school. Overall, it was a bit boring.)
- **Rejected after the second round**: Citatel AI (a Japanese startup; first round was a chat with the CTO, then take-home work, then a second round to continue discussing the take-home work; my answers were not ideal, and later they said they chose a more senior candidate), Vibranium Labs (a Silicon Valley startup; first round with the U.S. CTO was okay, but I did not click with the Taiwan lead in the second round and got rejected).
- **Got an offer**: T2 (HR reached out proactively; everything went smoothly and I was hired).

I applied to basically all roles that were visible online. In the end, I only got four interviews (not counting those that arrived after I had already received an offer). That is not a lot. Most of the time, job hunting felt strangely empty. This is why I think "spraying and praying" is still important. During the job-search phase, you want to feel busy so you are less anxious: always having new companies to apply to, talking to new recruiters, and having a few companies invite you to interviews. These activities may not directly help you find the right landing spot, but psychologically they provide stability, and they also force you to keep practicing how to talk and interview.

## Interview Experience at T2 (Kabushiki Kaisha)

T2 is a Japanese startup working on autonomous trucks. In just three years, they scaled to 200 employees, which shows how strong their funding is. It seems the initial trigger was that a T2 technical manager saw my job-hunting post on LinkedIn and asked HR to contact me. I also had a former colleague from Mujin working at T2, which helped as well. It was basically an interview invitation that fell from the sky.

Before the formal interviews, I had an HR call. It was mostly an introduction to what kind of company T2 is. HR's English was not great (T2 is still primarily a Japanese-speaking company), but I had already asked my friend about it, so I basically knew what they were doing. The funny part is that it was not even really an interview. It was just a light chat with HR, and they immediately started asking how much money I would need to be willing to join T2. That was a bit shocking at first. It made the company feel like it was buying talent, but in reality T2's funding is transparent: it is backed by multiple major corporations. So it is probably just because they are extremely short on people and have to compete aggressively. After the HR call, I did not even know which role I was interviewing for. They said the role would be decided after the interviews.

Then they scheduled the first interview: a technical interview directly with the Head of Engineering. It lasted about an hour and was fairly standard. We talked about my background, then they asked follow-up questions based on it. They also tested some C++ knowledge; if you have read *Effective C++* carefully, you should be able to answer. At the end, they asked about my interests. I said skiing, and I also mentioned that I wrote a book. He seemed interested in that. I also brought up that my long-term goal is to move into management, and I am not sure if that helped. I still did not know what role I was interviewing for, but it seemed T2 had already decided to put me into simulation. T2 had just completed Level 2 testing for autonomous trucks and planned to jump directly to Level 4. They needed someone to build a simulation environment for the Level 4 system, and I think they valued my ability to implement things.

Next was the second round, onsite at T2 headquarters. Unfortunately, my knee was injured at the time, but I did not want to drag the process out. The faster I finished interviews, the sooner I could start, so I went in anyway. The interviewer was the motion planning team lead, because the new simulation team would initially report to him. The interview had two stages. First was a coding exercise: C++ file handling and math computations. The problem was not hard, but I kept getting tripped up by weird little details, and I got nervous for a while. The second stage was a technical interview plus behavioral questions. They asked about system architecture, test design, organizational communication, refactoring, and so on. The scope was broad, and I surprised the interviewer with my answers to some advanced C++ questions. The second round was originally scheduled for three hours, but it ended up taking almost four. I felt that was a very good sign: it usually means the interviewer is interested in the candidate.

After every step (HR call, first round, second round), they asked the same question: what is my current compensation, and what is my expected compensation. I kept avoiding the question because I did not want to be anchored too early at a low number, and I also worried that throwing out a number that was too high too early might scare them away. I always said we could discuss it after an offer. Up to that point, everything had been positive, and I felt very confident.

T2 basically only has two rounds. After I passed the second round, I initially thought I would get an offer the next day, but I ended up waiting almost a full week for a response. Even though everything felt smooth, I still worried something might go wrong, so I was quite anxious in the middle. Luckily, nothing unexpected happened.

### Salary Negotiation with T2

To be honest, this was my first time negotiating compensation. Before this, my previous (and first) full-time job was at Mujin. I took whatever they offered. First, the number was reasonable. Second, at the time I cared more about getting to Japan quickly to reunite with my (now) Japanese wife. I had no leverage and could only accept unilaterally. This time I was an experienced hire, and I did not have urgent pressure to find a job, so I was not at a disadvantage. I could wait for a better opportunity.

After I heard back from T2, I spent a lot of time researching and doing homework to prepare for negotiation. I watched many instructional videos. First, I set a target. I prepared three ranges: a high number above expectations, a mid number that reflects market reality, and a low number that I would accept in the worst case. I did market research through various channels so I could adapt during the negotiation.

I also practiced negotiation through repeated simulations with AI so that when I went into the real conversation, I could communicate clearly. In the process, I analyzed which conditions benefited me, which positions I should insist on, and how to attack or defend depending on whether I got positive or negative responses. I think these AI practice sessions helped a lot in the final negotiation with HR.

Here are some of my negotiation premises and arguments:

Prerequisites:
- I learned that T2 actually pays very high numbers in practice.
- T2 is extremely short on headcount and is under schedule pressure.
- The new simulation team needs someone with relevant industry background and strong implementation skills.
- I received very positive feedback throughout the interviews.

Arguments:
- It is a brand-new simulation team, which is harder work and comes with more responsibility.
- I built a deep technical foundation through Mujin and open-source work, and my industry experience can be applied directly.
- The new simulation team will hire more people later (I heard at least five), and I am ready to take on management responsibility at any time.
- I can join quickly. With permanent residency, I can start within two weeks (a visa process typically takes two to three months or more).

The negotiation went smoothly. My preparation paid off, and I could confidently explain why I deserved the number I asked for. HR said they would try to get approval from the CEO. The waiting period felt abnormally long. Every day I was uneasy, both excited and afraid of disappointment. I even dreamed about the result.

A few days later, I finally got the outcome, and it was beyond my expectations. T2 accepted the number I asked for directly, without cutting even a single yen. This also confirmed a point emphasized in many negotiation guides: if you prepare thoroughly and advocate proactively without fear, you improve your odds. Of course, luck played a major role as well: my abilities happened to match what the hiring manager was looking for. When the result came, there was no hesitation. I told HR I would definitely join T2 and that I would not let them down!

## Overall Takeaways from Job Hunting in Japan and Taiwan (2025)

Overall, I do not think the 2025 fall market was particularly good. As a mid-level engineer, I expected it to be easier than for entry-level candidates, but in practice, almost all of my applications went nowhere. If AI-related roles did not respond because my startup experience was only eight months and therefore too short, fine. But even for C++ roles and roles directly related to my previous robotics-industry background, I still got rejected at the resume stage.

In this job search, I basically did not grind LeetCode. I only did a few problems from the "Top 75" list to get my C++ fingers back, because during the past few months of working on the startup I wrote almost no C++ and felt rusty. Among the companies I actually interviewed with, none of them asked LeetCode-style questions, which suggests that grinding is not always necessary. That said, part of the reason is that I skipped companies that explicitly required LeetCode, so if you only target large tech companies, you probably cannot avoid it.

If you want high compensation in Taiwan, you basically have to go to semiconductor companies or foreign companies. But I did not want to grind LeetCode at all, I was not very interested in the ecosystem at big companies, and I wanted some degree of remote flexibility so I could return to Japan at any time. So I basically skipped big companies. After going through a round of searching, I also realized that Taiwan's pure-software market is not very competitive. "Senior" software engineers are mostly below TWD 2M per year (sometimes only TWD 1.5M). Silicon Valley startups hiring in Taiwan can reach TWD 2M+, but honestly that is not that high either. No wonder almost everyone goes to TSMC, NVIDIA, MediaTek, and similar companies. Friends with similar years of experience at big companies are generally at TWD 2.5M+.

Japan is not that great either. Pure-software compensation is generally better than Taiwan, but Taiwan's semiconductor salaries are so high and taxes are so low that Japan ends up looking unimpressive. For many mid-to-senior software roles in Japan, the upper bound is around JPY 10M to 15M. If you want higher numbers, there are only a few companies: Woven by Toyota, Amazon, Google, Indeed, JPMorgan, Bloomberg, and so on. Those can reach JPY 20M+, but they are very hard to get into. For example, Google basically has no openings; if there are openings, they are often for internal transfers. Overall, if you can work at a major company in Taiwan (large local or foreign), the value-for-money is likely far better than even top-tier companies in Japan.

> For a Taiwan vs. Japan comparison, this article is quite good: "[日本軟體工程師的薪水如何？到底值不值得去？](https://life.huli.tw/2024/02/12/japan-software-engineer-salary/)"

We can also refer directly to levels.fyi to get a rough sense of compensation for software engineers in Tokyo:
![2025 日本東京軟工薪水情報](/img/2025-Tokyo-SWE-Salary.png)

You can see that the median is JPY 8M. This is the starting compensation at a slightly-better-than-average company. A "high" benchmark is JPY 12M, which is a common number for experienced engineers in Tokyo. The 90th percentile is JPY 15M, which is a typical level for many engineering managers. JPY 20M+ is an exception for companies like Google and Indeed. But looking at the aggregate chart is not very precise, because years of experience and company effects are huge. The good part is that levels.fyi lets you expand and inspect individual entries, so you can estimate the range for people with similar industry background and years of experience. When I did research, I expanded and checked hundreds of individual salary entries one by one.

Take my former employer Mujin as an example. On Japan Dev, you can see that the range for the "Senior Embedded Software Engineer for Functional Safety" role is JPY 7M to 11M. That upper bound matches our understanding of the market for senior engineers. For "Senior Computer Vision Engineer", the budget is JPY 9M to 14M. Mujin hires for traditional CV, so a higher budget makes sense, but overall the senior range is still within the normal market. Mujin is already one of the companies in Japan that can pay relatively well (roughly upper-middle of the market), so most other companies will offer less.

My new company, T2, posts engineering budgets directly at JPY 10M to 20M. Aside from Google and Indeed, there are currently not many companies in Japan that can pay at that level. I joined right when T2 was hiring aggressively, so I was extremely lucky. And if you are interested, feel free to reach out for a referral.

## Conclusion

Overall, I am very grateful that T2 gave me this opportunity. If I had not happened to run into this opportunity at the right time, and if I had not happened to be recognized by the hiring manager, it is entirely possible that I would have spent several months or even half a year job hunting. I want to thank every friend who helped with referrals, everyone who cheered me on, and my mentors who always gave me wise advice. Most of all, I want to thank my wife, my parents, and my parents-in-law for their support and encouragement, which made this journey far less lonely.

I strongly feel that life has its own plan. I studied bioengineering in college and did terribly in mechanical fundamentals at the time. I thought I would be a software engineer for the rest of my life and never have anything to do with mechanical engineering again. But then I met my Japanese wife, started working in Japan, entered the robotics industry (which I never thought I would do), and later joined T2 to work on autonomous driving. The knowledge I gained from robotics can be applied directly. Without realizing it, all the dots connected, as if the universe was telling me that my life value will shine in robotics and autonomous driving. Next up is to join the new company and work hard.

