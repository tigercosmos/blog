---
title: Revision Control
date: 2019-01-18 11:01:00
tags: [how things work, english,git, software engineering, revision control]
---

It's a late night, and the man is still at the office, being in deep water over the crashed program. Everything goes wrong. He doesn't even know why the code cannot run as usual. Unfortunately, he has forgotten what he had changed, regardless of the changes, which must be of paramount importance to the incident.
<!-- more --> 

It is a nightmare for all developers that the software gets out of control, so they create a tool for doing revision control, called version-control software, such as [GIT](https://en.wikipedia.org/wiki/Git) or [SVN](https://en.wikipedia.org/wiki/Apache_Subversion). When programmers develop programs, there would be many versions, such as "the big bang," "add new feature A," "fix a big bug of B." Engineers keep these versions functionality-completed, so each version should be able to run well independently, which means you can switch to any version without any risk of the crash. With the benefit of version control, programmers can modify codes at their disposal now. Once they make crashes on the program, they can easily reset the code to the last version.

Revision control also enables the collaboration of developing programs by many engineers. There are some features and bugs, and there will be many versions stand-alone. In this case, a version calls a *commit* in software engineering terminology. Each programmer heads on one of the tasks and creates a commit. Then the version-control tool like GIT can merge those commits into a new version.

Revision control is everywhere in software development. No matter projects in companies or open source communities, almost all use version-control tool. You might hear of [Github](https://github.com/). It's the biggest platform of open source projects based on GIT. Almost none of the programmers don't have a Github account--just like everyone has a Facebook account. The man works off early today because he has used the revision-control tool.
