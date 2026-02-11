---
title: "NCTU Observations and Reflections (6): Third Year — Graduation"
date: 2022-05-20 00:00:00
tags: [graduate school, master's, thoughts, NYCU ICSE, NCTU observations and reflections]
des: "This post records my reflections from my final year at the Institute of Computer Science and Engineering, NCTU/NYCU."
lang: en
translation_key: nctu-6
---

![A photo of myself](https://user-images.githubusercontent.com/18013815/169416614-08fb3993-846c-49af-9170-2b597494803d.jpg)

I never expected that I would pursue a master's degree—let alone spend three years on it. During those three years, the university's name even changed from NCTU to NYCU. Looking back, these three years of graduate school were both sweet and bitter. They were full and precious to me. At NCTU/NYCU, I learned a lot and had a great time across coursework, teaching as a TA, studying abroad, extracurricular activities, and academic research.

## Exchange Student

In the fall semester of my third year, I went to Italy as an exchange student. I think the exchange experience completely reshaped my values: only after seeing the world do you realize how small you are—and how narrow your view can be.

For my experiences in Italy, you can read these posts:

- 2021 Politecnico di Milano exchange guide
    - [Before departure preparation](/post/2021/10/italy/2021-exchange-polimi/)
    - [Arriving in Milan and survival](/post/2021/10/italy/2021-exchange-polimi-arrive-milan-survive/)
    - [Overview and reflections on Politecnico di Milano](/post/2021/12/italy/2021-exchange-polimi-school/)
- [An unbelievable love story that started in Milan with a Japanese girl](/post/2022/04/story/japanese-girlfriend/)

I am not an essayist, and I cannot write travel experiences as vividly as some writers do. Still, there were many emotions and realizations that I simply cannot fully convey in words. If possible, I think everyone should spend some time abroad at least once. In short, I spent the first half of my third year overseas. And while I traveled, I also spent some time doing research and writing my thesis.

## Master's Research

After arriving in Milan, I still talked with my advisor regularly. Because of the time difference, I usually met with him either early in the morning or late at night. Luckily, with modern networks—and with computer science, where you can work anywhere as long as you have a laptop—this was manageable. One fun aspect of doing research in Europe was that I set up an experimental environment on AWS in Europe. I originally thought I could use the results from Europe to submit a paper, but later I found issues with the experimental data, so I gave up on that idea. In the end, the thesis data was still collected back in Taiwan. Interestingly, Milan has an AWS region, so you can basically treat Milan AWS as an edge server. In Taiwan, however, the nearest AWS region is Hong Kong, which at best serves as a normal cloud server. Sometimes I find it strange that Taiwan's IT industry is so developed, and yet there is still no AWS region in Taiwan. While abroad, I also mostly finished writing the thesis, and my advisor helped revise it. I found it especially interesting to interact with my advisor while abroad—it felt like cross-border collaboration, even though we are both Taiwanese.

When I returned to Taiwan from Europe in March, I was extremely anxious. Graduation was full of uncertainties: the key performance issue in my thesis had not been solved yet; I also had to pass a programming certification exam to graduate; and I needed to find a job in Japan as quickly as possible. On top of that, I had to stay in a quarantine hotel for 16 days after returning. It was the darkest time of my year.

In Milan, I had thought the research was going smoothly—I had even finished writing the thesis, and I assumed I could just go back to Taiwan and wrap up graduation. But during the final check, my advisor discovered a fatal issue in the experiments. My mood instantly crashed. Even the conference paper I was preparing to submit had to be postponed. The issue was found in late January, when Taiwan was about to enter Lunar New Year. I had also planned to enjoy the final month, so I decided to leave it for later and continue after returning to Taiwan. Being stuck with this research problem made me anxious: I urgently wanted to graduate and did not want to be blocked by this issue. Although it took only a bit more than two weeks to resolve (maybe being locked in a hotel really helped), at the time everything felt uncertain—I had no idea how to solve it, or how long it would take.

The performance issue was the key problem in my research. The cause was unclear. I had been searching for the underlying reason for half a year since going abroad. In Milan, I thought I had found the answer, but right before paper submission, when my advisor checked the data, we discovered that the previous "answer" was actually wrong—so I had to start over. The process of finding the real cause felt like looking for a needle in the ocean: there are too many possible causes of performance problems. The only thing you can do is repeatedly form hypotheses, validate them, and eliminate them. Fortunately, I did find the answer in the end—and I found it before the quarantine ended. The cause was related to shared memory. I had suspected shared memory before, but in my research, shared memory overhead only becomes significant under certain memory-size conditions, which was a phenomenon I had never considered when testing. So for a long time, I could not reproduce that shared-memory overhead as the cause of the anomaly. Looking back, I actually feel this bottleneck became a highlight of my master's journey. Compared with completing research smoothly from start to finish, encountering setbacks and learning how to break through them is what gave me the most valuable gains.

## Programming Certification

To graduate from CS at NCTU/NYCU, you also have to pass a programming certification exam. The exam is very similar to the problems used in informatics competitions: you are given a problem, you handle input/output, and you solve it using data structures and algorithms. More bluntly, it is checking whether students have practiced LeetCode-style problems before graduation. Honestly, I think this exam is meaningless. The existence of this requirement is basically just proof that our "great university" (NYCU) is a diploma mill—look at world-class universities; which ones require a programming certification to graduate? At the very least, I am sure NTU does not. I also believe the University of Tokyo, Peking University, MIT, and CMU do not either.

In March, I still spent some time practicing. I was job hunting anyway, so I used LeetCode to keep my hands warm. If I failed the certification, there was a good chance I would not graduate on time, because the exam is held only once or twice a month—and each attempt comes with a lot of pressure. On the exam day, I even got stuck on a basic problem at first, and I almost wanted to give up. But I calmed down, checked carefully, and eventually got all the baseline points. That time, I ranked 8th out of more than a hundred people. Probably fewer than one-fifth of test takers passed. From the pass rate, the university really does "gatekeep" whether students have seriously practiced problems 🙃. I failed twice when I took it cold before going abroad, but I finally passed on the third attempt. Passing was a huge relief—it meant I was not far from graduation anymore.

## Master's Thesis Defense

After passing the programming certification—effectively resolving the only variable left—the rest of my time was basically spent revising my thesis, preparing for the defense, and job hunting. The thesis and defense themselves were not stressful, because in general, once your advisor agrees to let you defend, you have basically passed (?). Typically, the advisor invites the defense committee, usually professors or scholars they know well. But I like to do things my own way. I told Professor 游逸平 that I had specific committee members in mind, and after getting his approval, I emailed invitations myself.

I invited Professor 洪士灝 from NTU CS—my undergraduate project advisor—because I really wanted him to see how I had grown. I also invited Professor 葉宗泰, a new faculty member at the university; I enjoyed his AI hardware design course and wanted him to see what my research was about. The other two committee members were Professor 楊武, who has a good relationship with our lab, and my advisor.

Although I prepared my slides for a long time, on the defense day it still felt like I did not present well. The professors seemed unclear about many parts. What made me sad was realizing that the committee members had not actually read my master's thesis. The professors are busy, so the defense is basically just listening to the student's presentation—but I truly hoped they would read the thesis carefully and give feedback. Another disappointment was that I had hoped Professor 洪士灝 would give some affirmation, but on the defense day he only pointed out the weaknesses and lack of practicality in the research. I was really discouraged in that moment. I had thought the work was quite complete, perhaps even above the expected master's level, but the professors' strict standards forced me to reflect: there was still a lot that could be improved.

## Graduation

I heard that at some universities or departments, after the defense you still have to spend a month revising the thesis. But in CS at NCTU/NYCU, once you pass the defense, you are basically done. Usually this is because the student already has a job, and the advisor wants them to leave quickly. In my case, it was because I had already written the thesis in a fairly complete form—and before the defense, I even submitted a conference paper. About a week after passing the defense, I completed the graduation procedures and officially graduated. At first it did not feel real. Later, when I used my EasyCard to take the bus and saw the fare change to the regular adult fare, I suddenly felt it: I was no longer a student.

After graduation, it was time to focus on job hunting. I started around mid-March. At first, it came with graduation pressure; later, when I got my first offer, I finally felt calmer. I will write the detailed process as a separate post. As of now, I have received several offers. I am still deciding which one to choose, but at least I am certain that I will be working in Japan.

## Acknowledgements

I especially like and appreciate Professor 游逸平's guidance. Interacting with him was easy and comfortable, like talking to a friend. Over these three years, I explored topics I genuinely liked, discussed with him continuously, and kept iterating on the research. Even while I was in Milan, he stayed connected with me to discuss progress and revise the thesis. That is how I was ultimately able to produce a complete piece of work.

Thank you to Professor 洪士灝, Professor 楊武, and Professor 葉宗泰 for serving as committee members, providing diverse suggestions, and ultimately affirming the results. Their strict standards were frustrating at times, but also reminded me that I still have room to grow.

Thank you to CloudMosa. Thank you to CEO 沈修平 for generously lending Puffin servers for my research. Thank you to engineers TJ and Patrick for taking time after work to help deploy my research code to the servers. With CloudMosa's support, my master's research became much more complete.

Thank you to my family, my girlfriend, and my friends for your support, companionship, and help. Thank you to my parents for always respecting my choices and giving me the motivation to charge forward without hesitation. Thank you to 悦悦 for being with me during the lowest point of my research, giving me comfort while I was under both graduation and job-hunting pressure. Thank you to my friends for encouraging each other and discussing things together throughout the process. Thank you to 永昱 for always speaking frankly to me, helping me see my shortcomings and room for improvement. Thank you to 弘昇 for offering many useful suggestions in both research and coursework, and for constantly pushing each other. Everyone in the lab was friendly. Special thanks to 奕安 for recommending a paper to me, which gave me the initial spark for this research. Thanks as well to 意喬, 博聖, 冠程, 益揚, and 宇勝 for feedback and suggestions. It was a pleasure working with everyone in the lab.

I also received help from many more people on and off campus, and I am grateful to everyone who helped me along the way. I am happy to have completed my master's degree, and I hope to contribute to society with what I have learned.

## Closing Thoughts

This master's degree has a special meaning to me—not because the degree itself is inherently special, but because what I did while pursuing it deeply shaped me. People often worry about whether they should do a master's degree. My answer is: if you can make the process meaningful, then do it. "Meaningful" can include things like: needing the credential to switch tracks; filling in foundational knowledge; being interested in a domain and even wanting to go academic; or completing things you missed during undergrad (for example, studying abroad).

But if the only reason is "a master's degree pays more," or you simply do not know what to do and you are just studying for the sake of studying, then the degree is basically meaningless—master's degrees are not worth much these days. I have seen many juniors go straight into graduate school without really wanting to learn anything; they just want to "farm" another degree. That is pretty meaningless. In computer science, companies like FAANG do not require a master's degree to apply, and having one is not necessarily useful. CS is still about ability. A bachelor's degree is enough.

This series, <[NCTU Observations and Reflections](/tags/交大觀察與心得/)>, has finally reached its last chapter. I hope it provides some inspiration for students who want to know what master's life is like, or what CS at NCTU/NYCU is actually like. If you have any questions, feel free to contact me. Next, my story finally moves on to the chapter after entering the workforce!
