---
title: "NTU Observations and Reflections (3): Freshman Summer"
date: 2017-01-25 10:26:00
tags: [NTU, university, projects, programming, NTU observations and reflections]
lang: en
translation_key: freshman-summer
---

This post mainly describes my mindset and path from the spring semester of my freshman year through the summer.

Around the spring semester of my freshman year, I was still in the Department of Atmospheric Sciences. During winter break, I worked as a volunteer and helped Professor 林博雄 with observations. During that time, I met 石恩. He studied physics, and we got along well. Later, Professor 林博雄 needed to find students to help with software work. 石恩 was working with the professor then, so he introduced me to the professor. I had always had some exposure to programming, but at that time my level was really just a bit better than average. Still, whatever—only by taking on challenges can you level up. At the time I vaguely felt that writing code would not be a problem. But to explain that, I need to go further back.

<!-- more --> 

In high school, I wrote some C++. Back then, I kind of enjoyed the fun of informatics competitions, but later I felt I did not have the talent and gave up. After the college entrance exam, I did a science-fair project and used Python for simulation, purely for fun. Looking back now, my programming ability in that period was honestly not great.

Around winter break of freshman year, the public health professor who advised the science-fair project I did in senior year of high school was looking for students to build software. My senior, 辜鉅璋, asked whether I was interested. If I was, he would teach me and help me get started. Of course that would be great—having someone willing to teach you is the best. So in the spring semester of freshman year, I trained my programming skills again with my senior, starting from web technologies, mainly focusing on JavaScript.

Later, the public health side fizzled out. So in the summer, I joined Professor 林博雄's COOK team in the Department of Atmospheric Sciences. During the entire freshman summer, aside from spending two weeks at 集思論壇, I spent the rest of the time coding in the lab.

At the beginning, the professor assigned me the task of building visualization software for radiosonde balloons. Because I had been practicing web development, I chose to build it as a web application. A web frontend can only present data. If you want to process files, you need a backend. So you have to set up a local server. At the time I heard Node.js was popular—and it is also JavaScript—so I chose Node.js!

After I actually started, I realized I only knew the surface and could not build a project at all. At the time, all I had learned was a little object-oriented concept and how to use D3.js. So I began reading a lot of documentation and learning how to structure a frontend and backend. The frontend was easier: using Bootstrap makes it easy to get a layout, and most logic is just JavaScript operations. What I was stuck on for the longest time was how to build the backend. After setting up a Node.js server, I used Express, which was the most widely used framework. Soon I also learned how to call APIs. What I was really blocked on was file upload, because the radiosonde data files needed to be uploaded to the backend for processing and then sent back to the frontend. Writing code that could upload files took a lot of time, and later I finally found a workable template.

In addition, the professor wanted the balloon visualization to have a 3D presentation. Later I found Cesium.js, an open-source 3D globe simulation library. Even better, it can run on the frontend. In short, I spent a month completing the 3D path plot, 2D path plot, azimuth plot, Skew-T diagram, and all kinds of X–Y parameter plots. This was my first real project with actual engineering practice. I implemented all the required features. The professor did not originally ask me to implement the Skew-T diagram, but in atmospheric science, the Skew-T is too important. If it was missing, the project would have been only half complete, so I spent additional time to implement it. The Skew-T portion used another scientist's open-source code, but I could not understand the data format it expected—and on the internet, that was the only open-source implementation I could find. Later I checked the commit history and found that, long ago, the project used an older data format that I could understand. So I took the old version, and finally got the Skew-T part working!

After finishing the first project, I understood frontend and backend better, but I was still a newbie. While I was doing the first project, there was another short-term foreign student in the lab who was responsible for building an app. This was a micro-meteorological observation project. The professor bought a bunch of small weather sensors that can be plugged into a phone. The professor distributed the sensors to volunteers. Volunteers could then use the phone as a mini weather station to measure atmospheric data, and the process required an app as support. That student was developing the app.

When I finished the first project, that foreign student happened to be returning home, so the app project ended up in my hands. The app used a Hybrid App approach: an iOS/Android shell framework with web development technology on top. The advantage is that the web skills I practiced in the spring semester could be used directly to build the app, without having to learn how to build apps with Java or Swift.

After I took over the project, I rewrote almost 80% of the original code. With all the new features I added later, not much of the original code remained. In a way, I feel sorry to that student—he spent a month building it, and later I changed almost everything. But honestly, it would have been better if I had built it from the beginning. The reason this happened is that we did not have a standard software development process. Everyone just did their own thing. When you take over someone else's project, it becomes difficult to work with and hard to maintain, so the successor ends up deleting the parts that are hard to deal with and rebuilding from scratch. In commercial software companies, development follows certain standards to ensure maintainability and extensibility. But at that time, we had no such concept—and I also did not know how the outside world does things.

Building the app was similar to building a website, because of the Hybrid App approach, but there were still some features only phones can do, such as the phone's compass and Bluetooth. The app also needed to call APIs. This time, I used the lab's server, and along the way I learned how to work with MySQL and PHP. Later, I also used PHP to build a crawler for weather information, and I implemented user accounts using a database.

By around October, the app finally went online. After that, I continuously added new features. The difficulty was not high; it was mostly complexity and annoyance. For example, I did not implement an internationalization system at first. Later, adding language switching became truly painful. The second project was an app. Through it, I became more familiar with web development patterns and got a better understanding of backend operations. I also learned how to build a mobile app. And finally, publishing the app to Google Play required a developer account, and you even have to pay a fee to register the account. At last, the app project reached a milestone. In fact, the app has kept being updated, because the professor always has more features he wants to add—so even now, this project has not truly "ended."

My entire freshman summer was spent coding. My classmates were going around the island instead! But this was a very important period. My initial software development experience was built in that summer, and it helped me a lot later. Of course, there were downsides too—my myopia seems to have worsened, QQ!

[Weather Balloon (Radiosonde) Tracker — GitHub](https://github.com/tigercosmos/Weather-Balloon-Radiosonde-Tracker)

[Firefly Micro-Meteorology Web — GitHub](https://github.com/seanstone/cook-wn2nac2)

[Firefly Micro-Meteorology Web — App](https://play.google.com/store/apps/details?id=tw.edu.ntu.as.cook)

Feel free to fork ~
