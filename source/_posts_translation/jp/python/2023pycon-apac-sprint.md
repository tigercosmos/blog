---
title: "PyCon APAC 2023 の Sprint で初めて CPython に Pull Request を出した話"
date: 2023-11-27 00:03:00
tags: [python, CPython, pycon]
des: "PyCon APAC 2023 の Sprint で、初めて CPython に Pull Request を出した過程を記録します。"
lang: jp
translation_key: 2023pycon-apac-sprint
---

少し前に PyCon APAC 2023 に参加しました。今年は東京開催で、会場が自宅からそれほど遠くなかったので、初日は自転車で行きました。

最初の 2 日間はトークを聴いたり、いろいろな人と話したりしていたので、ここではあまり触れません。

3 日目には Sprint がありました。参加者は自分がやりたいプロジェクトを提案し、そこに他の人が合流する形で進みます。

私は [Donghee Na](https://twitter.com/dongheena92) が主導する CPython のグループに参加しました。Donghee は Line のエンジニアで、CPython のコアコントリビューターでもあります。

Donghee が「どうやって CPython に貢献するか」を説明しているのを、みんなで聞いている写真です：
![大家聽 Donghee 解釋](https://github.com/tigercosmos/blog/assets/18013815/541ea140-170c-495d-aae4-171d402e2556)

CPython のビルドは本当に簡単で、ビルド時間も想像よりずっと速かったです。ここでは細かく説明せず、GitHub のドキュメントを参照してください。

Sprint は半日のイベントで時間が限られているため、大きな貢献はなかなか難しいです。実際には、ドキュメント修正や小さなテスト追加など、比較的小さなタスクを選ぶことが多いです。

CPython の場合、みんな [“Easy” ラベルの Issue](https://github.com/python/CPython/issues?q=is%3Aopen+is%3Aissue+label%3Aeasy) を探して取り組みます。“Easy” はすぐに埋まりがちですが、誰かが途中までやって放置されているものもありますし、“Easy” と言いながら全然 Easy ではないものも結構あります。

Issue の難易度を見極めるのはとても重要です。理想は Sprint の数時間で完了できる規模のものです。大規模プロジェクトのコアコントリビューターは基本的に忙しいので、その場で直接質問できるのが最も効率的です。家に帰ってからだと、GitHub 上でメンションしてやり取りすることになり、コミュニケーション効率がかなり落ちます。

私は 1 時間以上かけて対象を探し、最終的に「[gh-57879: Increase test coverage for pstats.py](https://github.com/python/cpython/pull/111447)」に取り組むことにしました。テストカバレッジを増やす系は、だいたい複雑すぎないことが多いですし、すでに誰かが途中まで貢献していたものの、途中で離脱してしまった状態だったので、私はその続きから進める形でした。

他の人の途中作業をベースにできたので、それほど時間をかけずに仕上げられました。当日は 1 時間くらいで PR を出せたのですが（ただし、その後のフォローアップに 5〜6 時間はかかったと思います）、残念ながら Donghee はこの周辺のコードにあまり詳しくないとのことで、他の人のレビューを待つことになりました。現場でそのまま Merge できるケースもあるので、そこは少し惜しかったです。私は後日、他のコントリビューターとやり取りを重ね、最終的に 3 週間ほどで Merge されました。

CPython のカバレッジ確認のやり方を理解するのに、特に時間がかかりました。公式の「[Increase test coverage](https://devguide.python.org/testing/coverage/)」は、個人的にはあまり分かりやすくないと感じました。たとえば `COVERAGEDIR` の意味がいまいち掴めなかったり、カバレッジレポートを効率よく見る方法が分からなかったりしました。私の PR では `./python -m test test_profile test_pstats --coverage --coverdir=coverage` を実行し、`coverage` ディレクトリに出力される Raw 形式のレポートを直接読んでいましたが、もっとスマートなやり方がある気がします。あと、理由は分からないのですが、`python -m test` がよく失敗して、`make` をやり直して再ビルドする必要がありました。

この PR に貢献する過程で、Python の新しい知識もいくつか学べました。たとえば `NamedTemporaryFile` API の使い方や、`try..finally` と組み合わせて一時ファイルを確実にクローズする方法などです。CPython の小さな PR でも、学べることは多いです。

今回の Sprint で CPython に貢献できたのは、全体としてとても面白い経験でした。小さな Issue を解くのはそこまで難しくありませんが、CPython は本当の初心者にとっては少しハードルが高いかもしれません。それでも Sprint にはさまざまな題材があり、Python の API を少し叩くだけのものもあるので、初心者でも比較的すぐ入りやすいと思います。

私は Python を長く書いてきましたが、個人でも会社でも、普段書いている Python コードが特別すごいわけではありません。こういう機会がなければ、コミュニティの強い人たちに直接指導してもらう機会はなかなか得られないと思います。これが、オープンソースの面白さの一つですね。

同じテーブルで一緒に作業していた仲間：
![跟我同桌的伙伴](https://github.com/tigercosmos/blog/assets/18013815/2cbb0709-f303-4b6d-a098-cfcbddc357ad)

最後に、Sprint で CPython への貢献を指導してくれた Donghee Na に感謝します。

彼のツイート：
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Yesterday was PyCon APAC sprint day, and we submitted 6+ PRs to the CPython in a day. It was a wonderful experience to meet passionate people. Thanks to <a href="https://twitter.com/pyconapac?ref_src=twsrc%5Etfw">@pyconapac</a> and <a href="https://twitter.com/pyconjapan?ref_src=twsrc%5Etfw">@pyconjapan</a> for all the support. And also thank <a href="https://twitter.com/darjeelingt?ref_src=twsrc%5Etfw">@darjeelingt</a>, who suggested me to participate as an organizer. <a href="https://t.co/fiDjgw0tCD">pic.twitter.com/fiDjgw0tCD</a></p>&mdash; Donghee Na (@dongheena92) <a href="https://twitter.com/dongheena92/status/1718925777847374235?ref_src=twsrc%5Etfw">October 30, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

題外話ですが、同じテーブルの仲間 [c-bata](https://twitter.com/c_bata_) は [Anthony Shaw](https://twitter.com/anthonypjshaw) のサインをもらえてすごく喜んでいました 😂。ただ、相手はすでに [3 冊も本を出している](https://www.amazon.co.jp/%E8%8A%9D%E7%94%B0-%E5%B0%86/e/B096XMSNRS?&linkCode=sl2&tag=nwpct1-twitter-profile-22&linkId=bfa9a74c037e611d23de81705cac7907&language=ja_JP&ref_=as_li_ss_tl)んですよね！

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I participated in the <a href="https://twitter.com/hashtag/PyConAPAC?src=hash&amp;ref_src=twsrc%5Etfw">#PyConAPAC</a> sprint, made a modest contribution to CPython (thanks to <a href="https://twitter.com/dongheena92?ref_src=twsrc%5Etfw">@dongheena92</a> for your support). I received an autograph from <a href="https://twitter.com/anthonypjshaw?ref_src=twsrc%5Etfw">@anthonypjshaw</a>! As always, it was an incredible experience. Thank you so much. <a href="https://t.co/1IIhBccwkj">pic.twitter.com/1IIhBccwkj</a></p>&mdash; c-bata (@c-bata_) <a href="https://twitter.com/c_bata_/status/1718631302700945596?ref_src=twsrc%5Etfw">October 29, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
