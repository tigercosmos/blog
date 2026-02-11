---
title: "NCTU Observations and Reflections (2): First-Year Master's (Spring) — Coursework"
date: 2020-07-13 12:00:00
tags: [graduate school, master's, computer science, thoughts, NYCU ICSE, NCTU observations and reflections, courses, XR project, computer vision, NLP, computer animation]
des: "This post shares my reflections on coursework in the spring semester of my first year, covering an XR Project, Computer Animation & Visual Effects, Computer Vision, and Natural Language Processing."
lang: en
translation_key: nctu-2
---

## Preface

This semester felt like the happiest semester I have had in the past two or three years. It was fulfilling: I took four courses, still participated in the Fine Arts Club, wrote quite a few posts, and found a research direction and topic. Studying CS really is a joyful thing. Broadly speaking, my M1 spring can be divided into three parts: coursework, the Fine Arts Club, and research. This post focuses on coursework.
<!-- more -->

## Courses

All four courses this semester were of moderate difficulty, but taken together they still kept me extremely busy. The main reason is that I took a very heavy course last semester, so I only dared to take two courses then. But graduation requires eight courses in total. In order to focus on research in M2, I needed to accelerate my course progress this semester.

Due to COVID-19, only one course was in person; the rest moved online. I actually like online classes, because I can watch at my preferred time and play at 2× speed. The downside is that you do not get to know the instructors or classmates, which feels pretty bad. It also makes finding teammates difficult. In the past you could just walk up and ask someone, but now you cannot even see people. In the end, I just emailed around to find teammates. If I could choose, I would want courses to be online but with offline, in-person discussions. That is my favorite format.

This semester I took Computer Animation, Computer Vision, an XR Project, and Natural Language Processing. Even though my research is in systems software, my coursework had basically nothing to do with systems software. Part of the reason is that NCTU did not offer the courses I wanted. I wanted courses on heterogeneous computing, high-performance computing, systems software analysis, and distributed systems, but unfortunately NCTU did not offer them (and, in fact, there were none next semester either). Another reason is that if I cannot take what I want, then I might as well broaden my horizons. I deliberately chose very "visual" courses. I think these topics are closely related to daily life and are interesting. And if I want to go into high-performance computing, HPC inevitably needs application domains. These topics might help me in the future—for example, accelerating computer-vision computation or computer-animation computation.

## XR Project

I highly recommend the course "XR Project." The scale was impressive: all four instructors attended every class—莊榮宏, 謝啟民, 張宏宇, and 王銓彰. They came from the Institute of Computer Science and Engineering, the Institute of Communication Studies, the Institute of Applied Arts, and industry. This is on a completely different level from the usual co-taught courses where instructors just rotate. I liked this course a lot, but it was also a very demanding course that required spending a huge amount of time discussing and building projects with teammates.

Each instructor had a different perspective. For example, 王銓彰 analyzed feasibility from the perspective of taking client projects and development. 張宏宇 had deep insights into user experience and strongly emphasized a product's "story." 謝啟民 loved innovative ideas—the more unique, artistic, and unconventional, the better. 莊榮宏 generally cared more about the technical aspects, and he was also the organizer of the course, which deserves huge credit. All four instructors were great, and personally I liked and admired 王銓彰 the most. He often had unique insights on XR development. After class I frequently went to discuss things with him, and each conversation felt rewarding.

Photo with Professor 王銓彰:
<div align=center>
<img src="https://user-images.githubusercontent.com/18013815/87268897-3f3e3480-c4fe-11ea-8a64-06c486f64745.png" alt="Photo with Professor 王銓彰" width=400>
</div>

The semester goal was to complete three projects. Each project meant coming up with an AR/VR/MR (collectively XR) idea and building it. For the first two projects, the instructors provided prompts—for example, the first was "tell a story," and the second was "break the fourth wall." The third project was free-form. The standard team setup was two developers, one music designer, and one visual designer. There were many opportunities to collaborate with students from other departments, such as music and applied arts. This team composition felt very similar to building a project at a startup.

The course also provided a very rich set of VR/AR devices: more than ten sets of equipment including Vive, HoloLens, Oculus, and more. A single HoloLens costs around NT$120,000—without this course, there is no chance you can casually get two of them to play with at the same time. Among the devices, Vive is common and I had tried it many times. But I have to say: Microsoft's HoloLens is truly incredible. In principle you can call it an AR headset, but glasses like Conan's or Google Glasses can only present a 2D overlay on the lens. HoloLens can present 3D content in front of your eyes, making it feel as if a real 3D object exists in the real world through the headset. This enables very strong integration between the physical and virtual worlds. No wonder Microsoft calls it Mixed Reality.

The essence of this course was learning how to play your role well in a small team and build a product. In a "real" team, you typically also need someone for planning or a PM. If you do not have that, someone has to take on that responsibility, or you have to manage horizontally. During development you have to discuss intensely with teammates. Almost every week, we spent three or four days holding meetings for one or two hours each—and this is not even counting the time spent doing actual work.

During each project, we had weekly presentations. All four instructors gave feedback and guidance based on our progress and results. They had extensive experience in XR development. As described above, whether it was technical aspects, product aspects, story, or user experience design, you could learn a lot from their sharp feedback.

The second prompt was "break the fourth wall," which is abstract and difficult. Usually the "fourth wall" refers to the interaction between performing arts and the audience. So what is the fourth wall for XR devices? Most teams interpreted it as enabling interaction through two different media at once—for example, playing a game that uses an AR device together with a VR device, or combining a computer with a VR device. Our team's interpretation was: all VR users watch the same script at the same time, but the direction of the story is decided by real-time competitive gameplay among all VR users. Even if users are in different physical locations, the real-time, online interactive mechanism breaks the fourth wall between them while they experience the story.

For engineering students, it is rare to spend time with classmates from music, art, or communications backgrounds, so I think this course is a great chance to do so. The course also resembles a startup's rapid development workflow, and the team composition is fairly standard. For students without internship experience, it is also a good way to practice an industry-style development mode. Even though I had already interned at several companies and often interacted with people of different backgrounds, I still gained many new insights from this course.

## Computer Animation and Visual Effects

Before I introduce the course Computer Animation and Visual Effects, you can first watch this video to review the evolution of Pixar animation:

<iframe src="https://www.facebook.com/plugins/video.php?href=https%3A%2F%2Fwww.facebook.com%2Fthisisinsiderentertainment%2Fvideos%2F2406396736089585%2F&show_text=0&width=560" width="560" height="315" style="border:none;overflow:hidden" scrolling="no" frameborder="0" allowTransparency="true" allowFullScreen="true"></iframe>

I was genuinely moved after watching this documentary. Technological evolution is always astonishing, and that is the main reason why I wanted to take this course.

Computer animation is conceptually very similar to physical simulation. Indeed, to make animation look realistic, we need to follow physical principles. But because it is animation, many details do not have to strictly follow the physical model. This course, however, still mainly introduced physical models for animation, and spent less time on visual effects.

I found the assignments quite interesting. Since this is computer animation, the assignment results are visible, and visible outputs tend to be satisfying. If you are interested, you can check out the [computer-animation posts](/tags/%E9%9B%BB%E8%85%A6%E5%8B%95%E7%95%AB/) I wrote before.

For example, the first assignment was a 2D physical model:

![ca image1](https://user-images.githubusercontent.com/18013815/80283089-d8537f80-8747-11ea-9a81-09a995909d10.gif)

That said, I do have complaints about the assignments. Every assignment came as a huge Visual Studio project. The provided code built a lot of abstractions and physics-oriented frameworks. Before you even started writing anything, the template already had one to two thousand lines of code. If the goal is just implementing physical models, wouldn't a simple Python program plus a simple rendering engine be better?

The final project was also interesting. Since the whole semester was online, the last class was finally in-person, and most projects were animated short films, which was very entertaining to watch. However, in my opinion, since this course teaches the principles of computer animation and visual effects, the final project should involve implementing some physical model or VFX technique, or presenting some research papers. Instead, most people used Maya to make animations. It felt more suited to a course like "Applications of Computer Animation" or "Computer Animation Practice."

Staying true to the scientific spirit, our final project implemented Jos Stam's classic paper "Real-Time Fluid Dynamics for Games." We built a fluid game engine, and this is the flame effect we rendered:

![Flame](https://i.imgur.com/oDxEGw6.gif)

Overall, this course was pretty good. It is suitable for people like me who are deeply moved by computer animation.

## Computer Vision

Computer Vision is also a great course. There were five assignments: camera calibration, Fourier processing, image stitching, stereoscopic reconstruction, and image classification. Each assignment was very solid, and after completing all five, you basically get a glimpse into the core of the field. The workload is absolutely on par with top universities abroad.

For the assignments, see the posts I wrote under "[Computer Vision](/tags/%E9%9B%BB%E8%85%A6%E8%A6%96%E8%A6%BA/)." Each post corresponds to one assignment.

The lectures covered most of the content in the standard *Computer Vision* textbook, though the course spent less time on pure image processing. In the last few lectures, since the professor loves machine learning, he also talked quite a bit about ML.

I have to say: Professor 邱維辰 is the easiest English lecturer for me to listen to so far, even though he studied in Germany 😂. His catchphrases are "ok" and "right"—almost every sentence includes one—but you get used to it, and at least his English is fluent. NCTU requires at least one English-taught course to graduate. If someone only wants to take one English-taught course, I recommend taking his course.


## Natural Language Processing

This course taught NLP using classical probabilistic models. It pairs well with NTU Professor 陳縕儂's playlist, "[Applications of Deep Learning](https://www.youtube.com/playlist?list=PLOAQYZPRn2V7ZDNiCrrGAr1JVO3DssOkX)," which approaches NLP from a deep-learning perspective.

I think the early part of the course was good. The TF-IDF and bi-gram probabilistic models were quite interesting. As a classical probabilistic model, TF-IDF still performs very well in NLP and is often used as a baseline. But later in the course, the detailed discussion of NLP parsers became a bit boring. It would have been better to at least introduce the relationship between deep learning and NLP.

There were five programming labs, all simple small programs. If you can code, they do not take too much time, and they feel rewarding to implement: (1) implementing a TF-IDF model from scratch, (2) using a library to do part-of-speech processing on a corpus, (3) implementing a bi-gram model, (4) using AWS's survey platform, and (5) using a simple machine-learning model for an NLP task.

I think courses with hands-on implementation are always good, because I am a hands-on person. If I do not actually write code, it is hard to understand the principles. Overall, I found the assignments interesting, but the fourth one—using AWS's survey platform—was really boring. It was basically just wiring up APIs. I understand why it exists, because in research you often need to design survey systems for participants. But as an assignment, it is not very interesting. The main drawback of the assignments overall is that they did not give me the feeling of seeing the whole field; it felt like I only touched a small part of NLP. Perhaps the assignment scope could have been made more comprehensive.

The final project was also rather odd. It was an NLP classification competition: given a bunch of tweets, classify what emotion they express. The funny part is that despite this being a course focused on classical probabilistic NLP, everyone used deep learning to solve the task. The instructor never taught deep learning in class, and I am a complete deep-learning idiot, so in the end I had to rely on my teammates to carry me.

## Conclusion

Each course this semester was roughly moderate in workload, but together they almost killed me. Every course took turns having deadlines, and I had no spare time to do research properly. Still, because of that, I also felt that I learned a lot and grew quite a bit. It also helped me make up for some of the regret of not studying CS in undergrad. Overall, NCTU's courses are not that different from NTU's. Based on the reflections of seniors who studied in the U.S., even top American universities do not necessarily have better course quality than NCTU. That made me feel more at ease—after finishing coursework at NCTU, it seems I will be quite competitive.
