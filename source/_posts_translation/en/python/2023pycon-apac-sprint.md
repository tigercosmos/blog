---
title: "My First CPython Pull Request at the PyCon APAC 2023 Sprint"
date: 2023-11-27 00:03:00
tags: [python, CPython, pycon]
des: "This post records how I contributed my first CPython pull request during the PyCon APAC 2023 sprint."
lang: en
translation_key: 2023pycon-apac-sprint
---

Not long ago, I attended PyCon APAC 2023. This year it was held in Tokyo, and the venue was not far from my place—I even rode my bike there on the first day.

The first two days were the talks and a lot of chatting with people, so I won’t go into those here.

On the third day, there was a sprint. Participants can propose projects they want to work on and then recruit others to join.

I chose to join the CPython group, led by [Donghee Na](https://twitter.com/dongheena92). He is an engineer at Line and also a CPython core contributor.

Here’s a photo of everyone listening to Donghee explain how to contribute to CPython:
![大家聽 Donghee 解釋](https://github.com/tigercosmos/blog/assets/18013815/541ea140-170c-495d-aae4-171d402e2556)

CPython is actually very easy to build, and the build time was faster than I expected. Since it’s not particularly complicated, I won’t describe it in detail here—please refer to the documentation on GitHub.

The sprint is a half-day event with limited time, so you typically can’t make a huge contribution. In practice, you pick small tasks like documentation fixes or minor test-related issues.

For CPython, people usually look for issues labeled [“Easy”](https://github.com/python/CPython/issues?q=is%3Aopen+is%3Aissue+label%3Aeasy). Most “Easy” issues get snatched up quickly, but quite a few are half-finished by someone else, and some “Easy” issues are not easy at all.

Judging the difficulty of an issue is very important. The best case is something you can complete within a few hours during the sprint. Core contributors on large projects are usually very busy, so asking questions in person while they’re there is by far the most efficient. Otherwise, once you’re back home, you’ll mostly be tagging them on GitHub, and communication efficiency drops a lot.

I spent a bit over an hour looking for a target and eventually settled on “[gh-57879: Increase test coverage for pstats.py](https://github.com/python/cpython/pull/111447)”. This was about increasing test coverage, which is usually not too complex. Also, someone had already made some initial progress but later gave up partway through—so I essentially continued from where they left off.

Because it was based on someone else’s partial work, I finished it without spending too long. I completed the PR in about an hour on that day (though follow-up work later probably took another five or six hours). Unfortunately, Donghee said he wasn’t very familiar with that part of the codebase and suggested waiting for other people to review it. That was a bit of a pity, because some people can merge things on the spot. In my case, I went back and forth with other contributors afterward, and it was merged about three weeks later.

I spent a lot of time figuring out how to check coverage in CPython. I felt the official guide, “[Increase test coverage](https://devguide.python.org/testing/coverage/)”, wasn’t very clear. For example, I couldn’t really understand `COVERAGEDIR`, and I didn’t know an efficient way to view the coverage report. For my PR, I ran `./python -m test test_profile test_pstats --coverage --coverdir=coverage`, and then read the raw-format reports directly under the `coverage` directory. I feel there should be a smarter approach. Also, I’m not sure why, but `python -m test` often failed for me, and I had to rerun `make` to rebuild.

During the process of contributing this PR, I also learned some new Python things—for example, how to use the `NamedTemporaryFile` API, and how to combine it with `try..finally` to ensure the temporary file is properly closed. Even with a small CPython PR, you can learn a lot.

Overall, it was pretty fun to contribute to CPython during this sprint. Solving a small issue isn’t too difficult, but CPython can still be a bit challenging for true beginners. That said, the sprint also had all kinds of tasks; some were just calling a Python API or two, so I think beginners can also get started quickly.

I’ve written Python for a long time, but whether personally or at work, the Python code I usually write isn’t particularly impressive. Without an opportunity like this, I probably wouldn’t have a chance to get direct guidance from experts in the community. That’s part of what makes open source so fun.

My teammate at the same table:
![跟我同桌的伙伴](https://github.com/tigercosmos/blog/assets/18013815/2cbb0709-f303-4b6d-a098-cfcbddc357ad)

Finally, thanks to Donghee Na for guiding everyone on contributing to CPython during the sprint.

His tweet:
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Yesterday was PyCon APAC sprint day, and we submitted 6+ PRs to the CPython in a day. It was a wonderful experience to meet passionate people. Thanks to <a href="https://twitter.com/pyconapac?ref_src=twsrc%5Etfw">@pyconapac</a> and <a href="https://twitter.com/pyconjapan?ref_src=twsrc%5Etfw">@pyconjapan</a> for all the support. And also thank <a href="https://twitter.com/darjeelingt?ref_src=twsrc%5Etfw">@darjeelingt</a>, who suggested me to participate as an organizer. <a href="https://t.co/fiDjgw0tCD">pic.twitter.com/fiDjgw0tCD</a></p>&mdash; Donghee Na (@dongheena92) <a href="https://twitter.com/dongheena92/status/1718925777847374235?ref_src=twsrc%5Etfw">October 30, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

As an aside, my tablemate [c-bata](https://twitter.com/c_bata_) was really happy to get [Anthony Shaw](https://twitter.com/anthonypjshaw)’s autograph 😂, but the other person has already published [three books](https://www.amazon.co.jp/%E8%8A%9D%E7%94%B0-%E5%B0%86/e/B096XMSNRS?&linkCode=sl2&tag=nwpct1-twitter-profile-22&linkId=bfa9a74c037e611d23de81705cac7907&language=ja_JP&ref_=as_li_ss_tl)!

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I participated in the <a href="https://twitter.com/hashtag/PyConAPAC?src=hash&amp;ref_src=twsrc%5Etfw">#PyConAPAC</a> sprint, made a modest contribution to CPython (thanks to <a href="https://twitter.com/dongheena92?ref_src=twsrc%5Etfw">@dongheena92</a> for your support). I received an autograph from <a href="https://twitter.com/anthonypjshaw?ref_src=twsrc%5Etfw">@anthonypjshaw</a>! As always, it was an incredible experience. Thank you so much. <a href="https://t.co/1IIhBccwkj">pic.twitter.com/1IIhBccwkj</a></p>&mdash; c-bata (@c-bata_) <a href="https://twitter.com/c_bata_/status/1718631302700945596?ref_src=twsrc%5Etfw">October 29, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

