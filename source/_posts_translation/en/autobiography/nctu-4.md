---
title: "NCTU Observations and Reflections (4): Second Year (Fall)"
date: 2021-01-27 14:00:00
tags: [graduate school, master's, thoughts, NYCU ICSE, NCTU observations and reflections, fine arts club, NCHC, teaching assistant, parallel programming]
des: "This post records a few stories from the fall semester of my second year at the Institute of Computer Science and Engineering, NCTU/NYCU."
lang: en
translation_key: nctu-4
---

## Preface

In the blink of an eye, the fall semester of my second year is over. Time really does feel faster as you get older. I still remember how, in elementary school, a semester felt unbelievably long—now it feels like it is suddenly vacation again.

As usual, I am writing one post per semester. So many things happened that I should record them—otherwise it would be a shame to not remember the stories later.

A lot happened this semester. I took two courses, TA'd for two courses, joined the Ironman Challenge in October and recorded videos for a month, worked part-time at NCHC, continued participating in the Fine Arts Club, and picked up a new hobby: photography.

![Basketball court at NCTU in the evening](https://user-images.githubusercontent.com/18013815/105941169-c8324d00-6097-11eb-858b-e284a93ab198.png)

(Basketball court at NCTU in the evening)

## Courses

I took Accelerator Architectures for Machine Learning (AAML) and Operating Systems (OS). With that, I completed all eight courses required for graduation.

AAML was a pleasant surprise for me. Professor 葉宗泰 is a new faculty member, and this was his first course after joining NCTU CS. Why was it a surprise? Because I have always hated the term "AI." I have always felt it is just hype. Also, I have always had a mental block with AI—my "domain power" is relatively low. I had never seriously understood what deep learning really is. Of course I learned the basic knowledge, but I never had the feeling of truly understanding it. Yet the deep-learning introduction at the beginning of this course suddenly made things click. Either I finally had an epiphany after procrastinating for so long, or it is because Professor 葉's perspective is closer to systems software/hardware design, which is closer to my background, so I can absorb what he says more easily.

I categorize "AI" into four layers: application scenarios, AI models (or algorithms), parameter tuning (fine-grained customization), and system software/hardware. A successful AI service needs all four. First, you have a problem to solve—for example, a device needs to recognize what animal it is seeing. Then you need an appropriate model. For example, ResNet is a model designed for image recognition, so you evaluate whether it can be used. But having an architecture is not enough. You also need to tune parameters to best fit the target problem. Finally, AI always runs on systems. How you optimize the system and hardware has a huge effect on performance. This course mainly focused on the last step, which I also think is the most important step: no matter how good the earlier steps are, if you run AI on hardware without fully utilizing the hardware's performance, it is all wasted. AAML is a huge topic, and I cannot do it justice in a few sentences. If I have the chance, I will write another post to introduce it. In short: I strongly recommend this course.

The other course was OS. I took OS in undergrad, but the graduate-level version goes deeper. Also, it did not seem too heavy, so I thought I would learn it again. Professor 張立平 teaches in English without much problem—he speaks smoothly, and the content is quite clear. The biggest issue is that half of the course consists of student presentations on classic OS papers. But I think CS grad-student presentations are generally bad. It is fine if you cannot speak English smoothly, but at least you should know how to present—and the content was a mess too. Each group had only 15 minutes to present a paper. For reference, in a lab paper-reading session, it often takes 40 minutes or more to clearly explain one paper. So when time is short, the right approach is to pick a few interesting points and explain them deeply or extend them. But many people try to rush through the entire paper in 15 minutes like a train schedule, which leads to slides packed with everything and no structure; combined with poor English, the slide layout becomes chaotic, and the audience cannot understand anything. The intention of having students present papers is good, but the student level is too low. Unless you need OS as a makeup requirement because you never took it in undergrad, I would not recommend this course.

When I worked part-time at Skymizer in Taiwan, I never understood what "Compiler for AI" was for. After taking AAML, I finally understood!

As a side note: I think if you enter the AI field from systems software, tight hardware–software co-design is extremely important. The unfortunate part about Skymizer is that they do not have their own hardware—they only provide software services. But many AI models require highly customized hardware, and customers typically do not have the capability to design AI-optimized hardware themselves. So customers can probably only use off-the-shelf hardware like NVDLA. Yet AI models can be accelerated much more through hardware design. That is not a problem you can solve by compilers alone. This kind of hardware–software integration is exactly where companies like 耐能 (Kneron) and Cerebras shine.

## Teaching Assistant Work

This semester I TA'd for two courses: [Parallel Programming](https://nycu-sslab.github.io/PP-f20/) and Numerical Software Development. Parallel Programming is my advisor Professor 游逸平's course. In previous years, it was usually handled by M2s. It also happens to be a field I like, so I became the head TA and basically handled everything. Numerical Software Development is a TA job I have done continuously since M1. When I first joined NCTU, Professor 陳永昱 happened to be looking for someone, and since then I have kept working with him.

I am determined to be a good TA, so I replaced all of the Parallel Programming assignments (you can take a look at the [course website](https://nycu-sslab.github.io/PP-f20/); the course is all about assignments). I thought the previous assignments were too easy and not at the graduate level. Now people cannot even think about copying assignments XD. Also, whenever students had questions, I responded immediately. In my experience, many TAs for other courses just ignore emails. There was one assignment every two weeks. I usually spent one week designing the assignment, and another week grading it, meaning I basically had no breaks between assignments. I also graded presentations. In other courses, TAs often give scores based on vibes; for my scoring, I used very detailed rubrics.


![Parallel Programming final student presentations](https://user-images.githubusercontent.com/18013815/105940784-0418e280-6097-11eb-964c-205d5ffaafff.png)
(Parallel Programming final student presentations)

Honestly, I could have chosen to do only as much work as the pay justifies. But I chose to do a lot. The TA hourly rate is about 200. If I work more than 5 hours per week for one course, I am basically losing money. In reality, I spent more like 10–20 hours per week on it. I have to rant a bit about NCTU: I TA'd for two courses, and I only got NT$8,000 per month total (NTU professors were shocked; it is even lower than what NTU pays students). At NTU, one course at least pays around NT$11,000. No wonder many TAs slack off—who wants to work for that little? The "correct" strategy is to do less and treat it as scholarship money. I just happen to have a lot of passion for teaching. I think students could feel how serious I was. I wonder whether I can get an outstanding TA award; it would be nice to have an honor like that.

## Ironman Challenge

Participating in the Ironman Challenge has basically become routine. I join it every year. From Angular to DBMS to Browsers, I have written a lot of posts. Writing is actually exhausting. My high-quality posts often take several days. For Ironman, if you do not stockpile drafts in advance, a post produced in one day is basically garbage. This year I did not want to exhaust myself, and I did not want to write low-quality posts. So I decided to join the video track, because I have long wanted to produce a video series: "[Basic Literacy to Save CS Students](https://www.youtube.com/playlist?list=PLCOCSTovXmudP_dZi1T9lNHLOtqpK9e2P)."

<iframe width="560" height="315" src="https://www.youtube.com/embed/IaHcesCpuA4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Before making this video series, I wrote a post: "[Basic Literacy for CS Students](post/2020/04/thought-about-cs-student/)." In that post, I explained the skills that I think CS students need to know but that schools often do not teach—you have to learn them yourself. You can view that post as the reading guide for the series. In the videos, I went one step further and actually taught what I believe CS students should know.

For beginners, this series should be quite practical. I hope you can recommend it to your juniors. I also hope people in education will make more use of my teaching.

For example, once a sophomore CS junior asked me how to learn touch typing and Vim, and I knew I should make an episode explaining learning methods. Or, when a teammate in my graduate course project told me they did not know how to call an API, I knew I needed to explain what an API and a server are. For students who have never interned at a company or participated in a large open-source project, they probably are not familiar enough with Git operations either. Also, more and more people are learning to code—not only CS students. In fact, all developers should have these basic literacies.

Remember how, when preparing for college entrance exams, everyone had to read "搶救國文大作戰"? Now we also need to "save" our programming abilities. My goal is to help CS students stand up to scrutiny, whether at school or during job hunting. My own ability is only so-so, but among people who are better than me, who is willing to spend hours every day making videos? This series took me four to five hours of work each day. After finishing it, I felt proud that I had done something meaningful. The only regret is that I do not know why it did not even win an honorable mention.

## Fine Arts Club

I joined the Fine Arts Club in M1, when I was full of drive for art creation. I feel like that drive has cooled down a bit now, but drawing is still fun. Also, I met many great friends in the club. I feel that friendships in arts-and-culture clubs are easier to maintain. The clubs I joined at NTU were mostly work-oriented. After the event was over, there was no bond anymore, and people stopped contacting each other. But in the Fine Arts Club, everyone is interested in art. The connection built on shared interests may be more solid.

<iframe src="https://www.facebook.com/plugins/post.php?href=https%3A%2F%2Fwww.facebook.com%2Fnctu.finart%2Fphotos%2Fa.3503794043022848%2F3503787503023502%2F&width=500&show_text=true&appId=577288832614270&height=375" width="500" height="375" style="border:none;overflow:hidden" scrolling="no" frameborder="0" allowfullscreen="true" allow="autoplay; clipboard-write; encrypted-media; picture-in-picture; web-share"></iframe>

(Nude figure sketch)

This year I also took an officer role. It is not common for graduate students to take officer roles, but I was just a small social-media editor. My job was to take lots of photos and share them on the club's Facebook page. I actually enjoyed doing it. I found that I am quite interested in photography. In regular club sessions, rather than practicing drawing, I often preferred photographing people while they were drawing and capturing their works. Later, I only spent half the time drawing, and used the other half to take photos everywhere. If you are interested, you can check out the [Fine Arts Club Facebook page](https://www.facebook.com/nctu.finart/photos) for typical activity photos.

<iframe src="https://www.facebook.com/plugins/post.php?href=https%3A%2F%2Fwww.facebook.com%2Fnctu.finart%2Fphotos%2Fa.3460872767314976%2F3460863130649273%2F&width=500&show_text=true&appId=577288832614270&height=375" width="500" height="375" style="border:none;overflow:hidden" scrolling="no" frameborder="0" allowfullscreen="true" allow="autoplay; clipboard-write; encrypted-media; picture-in-picture; web-share"></iframe>

(Instructor demonstrating an oil portrait)

## National Center for High-performance Computing (NCHC)

I already talked about going to NCHC in the previous post. The main reason I went was that I wanted to work with SmartNICs—network interface cards with onboard processors, commonly used in data centers to accelerate networking and data processing. But the vendor kept delaying. I originally thought we could start using them in June of last year, but the NICs did not arrive until October. After installing them in servers, the whole device still had a bunch of problems. So by the end of this semester, I still had not actually done SmartNIC research. During summer I read many related papers, but reading alone does not give much intuition—you need to use the hardware to really know what is going on.

Later I was reassigned to other projects to do full-stack work. I am not interested in full stack. Mainly because I already did a lot of it when I worked part-time at a software company in undergrad, and in my view it is more like grunt work. But there was no choice: the equipment I was supposed to use was not ready, so at NCHC I could only help other projects with development. Recently, the SmartNIC equipment should finally be ready. I hope I can start doing related research next semester.

The most interesting part of working at NCHC is probably understanding what they do inside. I even had the chance to visit the supercomputer machine room in the basement. To enter, you first have to register your face, and every entry and exit requires a scan. Inside looks like the supercomputer machine rooms you see in photos, but it is still nice to experience it in person (even though it is really loud). There are many racks of machines—one rack can cost several million. This is not the kind of equipment you can play with at NCTU. Once, I even helped remove GPUs inside: you have to pull out the machine trays. A rack can have multiple layers, and one layer can weigh close to 100 kg. It takes several people working together to remove one layer. In short, it was a valuable experience.

## People Who Influenced Me Deeply

I want to specifically talk about the people around me this semester who had a big impact on me.

The first is the instructor of Numerical Software Development, [陳永昱](https://www.linkedin.com/in/yungyuc/). At first, I was just a TA, and we worked together smoothly. Later, I mentioned that I was actually very interested in high-performance computing (HPC). In fact, he already knew I worked part-time at NCHC (in his generation, they called it 國高), but he thought it was just regular part-time work (and, to be fair, this semester it did become that 😅). I went to NCHC because I wanted to get more exposure to HPC research and equipment. He then told me that he used to do HPC-related work—for example, when he was in the U.S., he worked on NASA research projects; after returning to Taiwan, he worked on missile research at the National Chung-Shan Institute of Science and Technology. These are topics that require massive computation. Now at Synopsys, in some sense his work is still related to HPC.

Previously, my understanding of HPC was very abstract. What I learned from 洪士灝 was viewing it from an infrastructure perspective. But 陳永昱 comes from a mechanical engineering background, and he understands it from the application perspective. That was a brand-new perspective for me. I still think there is a clear difference between academia and industry. When discussing technology, 陳永昱 often asks me whether it can "make money." Of course, I am not saying there is anything wrong with exploring the mysteries of the world. But people do need to earn a living (?). Talking with him is always a big shock to me. I cannot say everything he says is correct, but he often offers viewpoints I never thought about. Most of the time, I agree with him.

Another person is my undergrad DBMS instructor, [駱明凌](https://www.linkedin.com/in/ming-ling-lo-94631b3/). He returned to NTU to teach mainly because he had already retired; he taught DBMS with the mindset of giving back to society. After I graduated, I stayed in touch with him. There was a time when I was very interested in DBMS and even wanted him to advise me. This semester I talked with him once, and we talked a lot. Mainly, I had never had the chance to properly hear his story, so I used this opportunity to hear the whole story—from school to retirement. I love listening to stories like this. I feel you can learn a lot from them. I also learned that he recently started a company. At some point, I suddenly lost interest in entrepreneurship, but after that conversation, I again felt that I must seize the opportunity to start something in the future. I even already decided on a company name: "微中子科技股份有限公司." I hope it becomes real one day.

In fact, I think the people I meet in daily life all inspire or influence me in some way: labmates, club friends, classmates, and even friends I met through the E.Sun Scholarship. This is also why I like spending time with people.

## Research

During summer, my research progress was fairly fast. The program architecture was basically built in summer. This semester, progress slowed down. Mainly because I spent half my time doing coursework assignments, and the other half on TA work—so there was no time left. Also, I took a bit of a detour in the research. Recently, after discussing with my advisor, I narrowed the topic, which should make it easier to maintain steady progress. My expectation is to defend next semester and finish writing the thesis. Since I plan to go to Italy as an exchange student next academic year, I need to ensure I meet all graduation requirements before going abroad, so after returning I can graduate directly. Or, if I am unlucky and cannot go abroad because of the pandemic, I can still graduate directly.

## Photography

I have always been interested in taking photos. Previously, I thought shooting with the iPhone 11 should be acceptable, but later I really felt the limitations were too strong. For example, shooting distant subjects or getting good night scenes is just too different. So I bought a camera. After researching for a long time, I decided on the Fujifilm X-S10. It is a new model released at the end of last year. Compared with mid-range mirrorless cameras from other brands, I think it has the best cost–performance ratio, so I bought it. Photography is a new hobby I have been cultivating after fine arts. I am still practicing seriously. I hope I can keep improving. If you are interested, feel free to follow my [ig: @liu_an_chi](https://www.instagram.com/liu_an_chi/), where I post my photography work.

## Conclusion

It feels weird to not have a conclusion. In short: I did quite a lot this semester. NCTU is honestly pretty good, and life has been quite exciting.
