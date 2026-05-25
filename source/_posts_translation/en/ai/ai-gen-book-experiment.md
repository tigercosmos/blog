---
title: "Generating a Book with AI"
date: 2026-05-26 04:00:00
tags: [AI, writing]
des: "This post experiments with having an AI reference the table of contents of *Binary Hacks Rebooted* to generate a similar book, and shares some thoughts on the result."
lang: en
translation_key: ai-gen-book-experiment
---

I happened to see online that *Binary Hacks Rebooted* had been translated into Chinese. The book was a hit in Japan a while back, so it's not too surprising that the buzz has reached Taiwan too. Several chapters genuinely caught my interest, which gave me a sudden idea: could I have an AI reference the original book's table of contents and generate similar content?

So I had an AI reference the table of contents of *Binary Hacks Rebooted* and try to generate a similar book. The concept isn't complicated, and the process was pretty quick.

Experiment result: https://github.com/tigercosmos/ai-gen-book-exp

> Disclaimer: This project did not feed any of the original *Binary Hacks Rebooted* material to the AI. If any content happens to resemble the original work, it is purely a coincidence stemming from overlap between the AI's knowledge base and the original.

![cover](https://github.com/user-attachments/assets/99211e54-d89a-4ba5-a046-e1b540adc67a)
(Plaza de España, Seville)

## AI Book Generation Workflow

### Build the Outline

Create a new directory and copy the table of contents of *Binary Hacks Rebooted* into `outline.txt`.

### Start Generating Content

Open Claude Code, pick Opus 4.7, and use xhigh effort (you could also use Codex or another AI API). Use the following prompt:

> Based on `outline.txt`, write out a book. The book must be written in Markdown, with Traditional Chinese (Taiwan) as the language. First define a standardized format, then use multiple sub-agents in parallel to generate every chapter. Each main chapter is one directory, and each sub-chapter is one Markdown file. Just handle chapters 1 through 3 for now.

The initial step didn't take much time — all sub-chapters across three chapters were generated in roughly ten minutes.

### Check the Content

Because each sub-chapter is generated independently by the AI, we have to make sure the content is coherent and correct, so we let the AI run a review:

> Review everything across chapters 1 to 3 and check whether the content is coherent and accurate. If anything is wrong, fix it, and record what you learned in `review.md`.

After the adjustments, you get smoother content along with a `review.md`. We later have the AI consult this experience to self-correct. This step also took only about ten minutes.

### Iterate

Once the first three chapters are produced, you need to check whether the template style needs adjusting, and whether the chapters' wording, coherence, level of detail, and so on all match your expectations. Various aspects may need more polishing, and you may want to feed additional prompts to the AI.

Once everything is set up, you can let the AI iterate on its own and crank out the whole book. I've preserved every step along the way in the git history — feel free to take a look if you're curious.

### Further Optimization

A book also contains code, concept diagrams, exercises, and more, and each of those parts needs to be checked by the AI separately. For example, the code parts need an additional skill that lets the AI try running the code as written in the book; if it doesn't work, it has to come back and revise the code, and the code's output also needs to be generated for real to be correct. Concept diagrams, exercises, and all the other details may each require their own skill.

## The Result

Feel free to head over to the [project](https://github.com/tigercosmos/ai-gen-book-exp) and check out the result yourself. If you happen to have read the original, you can compare them too.

Although I haven't read the original, from the perspective of "learning the knowledge," the generated book is probably good enough. In essence, this is similar to what I do every day on ChatGPT, chatting with the AI to learn a new technology. Suppose today I want to learn what ELF is — I'd just have ChatGPT generate some teaching content right there, and then strengthen my understanding through back-and-forth with the AI. Having an AI generate a book is just the same idea with "a book" as the medium.

The result is usable, but the content is generic. So if you actually want the book to become a great book, the hard work is unavoidable. As an author who once wrote a bestseller, I know full well what it takes to pour real effort into writing a good book — there are so many details and clever touches in a single book, and what an AI casually generates reads like an elementary-schooler's essay.

## Conclusion

Since this project was purely for fun, I didn't seriously iterate on or optimize the book's content. But overall I think generating a book with AI is entirely feasible — it hardly takes any time, and it works perfectly fine for self-study.

Stepping back, though, AI is just a tool, like a text editor. We can easily use AI to crank out ordinary content, but what makes a book endure depends on the author's creativity and depth of knowledge.

In an era where everyone can generate a book with AI, distinctive authors can still produce one-of-a-kind, excellent works with AI's help. At the same time, I hope unscrupulous publishers won't ship cheap AI-generated content to market — that would be an insult to the publishing industry.
