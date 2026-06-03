# AEO / AI-friendly SEO guide — Neutrino's Blog

How this blog is optimized for AI answer engines (ChatGPT, Perplexity, Claude,
Google AI Overviews) and classic search, and how to keep new posts optimized.

> **Key fact that shapes everything:** AI answer crawlers (GPTBot, ClaudeBot,
> PerplexityBot, OAI-SearchBot, …) **do not run JavaScript** — they read only
> the raw server-rendered HTML, with 1–5 s timeouts and no second render pass
> (Vercel, >1.3 B fetches). Hexo's static output is ideal. Every signal below is
> emitted server-side at build time; never rely on client-side JS for metadata.

---

## 1. What's already automated (no per-post work)

Driven by `themes/autumn/scripts/seo.js` + `themes/autumn/layout/_partial/head.ejs`:

- **JSON-LD `@graph`** on every page — `WebSite` + `Organization` + `Person`,
  plus `WebPage` + `ImageObject` + `BreadcrumbList` + `BlogPosting` on posts,
  wired by stable `@id`s. The **`Person` node** (with `sameAs` → GitHub, LinkedIn,
  Scholar, …) is the highest-leverage signal: it lets engines resolve "is this
  the same An-Chi Liu" → E-E-A-T + citation eligibility. Author data lives in
  `themes/autumn/_config.yml` under `author:`.
- **Open Graph + Twitter cards**, **canonical** (clean URLs), **`meta robots`**
  with `max-image-preview:large` / `max-snippet:-1`, **hreflang** alternates,
  **Atom feed** autodiscovery, accurate **title** order (`Post Title | Site`).
- **Correct heading hierarchy** — each post has exactly one `<h1>` (the post
  title). Authored sections must start at `##`. **Never use `#` in a post body.**
- **`robots.txt`** (`source/robots.txt`) welcomes all crawlers + references the
  sitemap. **`sitemap.xml`** carries `lastmod` + hreflang alternates.
- **`llms.txt`** auto-generated (`scripts/generate-llms-txt.js`).

---

## 2. Per-post authoring playbook (this is where the real AEO wins are)

The Princeton **GEO** study (KDD 2024, 10 000 queries) measured what actually
makes AI engines cite a page. The biggest, best-verified levers are content
habits — and **lower-ranked / independent sites benefit most** (+115 % for a
rank-5 source). Do these on every post:

### 2.1 Answer-first (front-load the conclusion)
- Open with a **2–3 sentence TL;DR** (`> **TL;DR**：…`) that directly answers the
  post's core question, *before* background. The scaffold (`scaffolds/post.md`)
  now includes this. AI engines weight content near the top.
- Don't bury the answer under a narrative `## 前言` / preface.

### 2.2 Fact density (the strongest cited-content signal)
- Add **statistics, version numbers, dates, benchmarks** with attribution
  (GEO: statistics +41 %, quotations +32 %, citing sources +30 %).
- One clean **definition sentence** per concept ("X 是 …").
- **Link out to primary sources** (official docs, RFCs, papers, repos).
- Keyword stuffing does **nothing** — skip it.

### 2.3 Self-contained, extractable chunks
- Make each `##` section stand alone: restate the subject noun, avoid
  cross-section pronouns ("如前所述…"), keep sections ~150–300 words.
- Use **lists and tables** for enumerations / comparisons — engines lift these.
- Phrase key `##` headings as **natural questions** in the post's language, with
  the answer immediately under the heading. (Anchor permalinks already make every
  section individually addressable.)

### 2.4 Freshness + E-E-A-T
- On a **genuine content edit**, set `updated: YYYY-MM-DD` in front-matter. This
  bumps `dateModified` + the visible "Updated" line + sitemap `lastmod`.
  **Do not** set it for typo fixes — `lastmod` must stay trustworthy.
- State first-hand experience where relevant ("身為在 T2 的工程師…") — Experience
  is the hardest-to-fake E-E-A-T signal and you have genuine engineering content.

### 2.5 Front-matter hygiene
- **`des:`** — write a real ~150-char summary. If omitted, the system auto-derives
  one from the body (fine, but a hand-written `des` is better).
- **First image** becomes the `og:image` / social preview. Put a good one early,
  or it falls back to `/img/cover.png`.

### Per-post checklist
- [ ] TL;DR / answer-first lede
- [ ] `des:` filled in
- [ ] A statistic / version / dated fact with a source link
- [ ] `##` sections (never `#`), key ones phrased as questions
- [ ] Lists/tables where it helps
- [ ] A representative first image
- [ ] `updated:` if this is a real revision of an old post

---

## 3. Deliberately NOT done (researched and rejected as hype)

Don't "fix" these later without re-checking — the 2025-2026 research was explicit:

- **`llms.txt` as a traffic strategy** — Google says it ignores it; logs show
  ~0.1 % real AI-bot consumption. We ship it as a zero-cost experiment only.
- **`SearchAction` / sitelinks searchbox** — Google retired it Nov 2024.
- **Auto-generated `FAQPage`** — FAQ rich results removed from Search May 2026.
  (Only add `FAQPage` JSON-LD manually if a post has a *real* visible Q&A section.)
- **Citation-multiplier claims** ("2.5× more AI answers from schema") — vendor
  benchmarks, directional only. JSON-LD's confirmed value is entity/E-E-A-T
  signaling for search-index-backed surfaces, not a guaranteed citation lever.

---

## 4. Config knobs

| Where | Setting |
|---|---|
| `themes/autumn/_config.yml` → `author:` | name, url, image, jobTitle, knowsAbout, alumniOf, **sameAs** (Person/E-E-A-T) |
| `themes/autumn/_config.yml` → `og_image*`, `logo` | default share image (replace `/img/cover.png` with a 1200×630 asset for best previews) |
| `_config.yml` → `updated_option: 'date'` | keeps `lastmod` stable until you set `updated:` |
| `_config.yml` → `feed.content: true` | full-text Atom feed for AI ingestion |
| `source/robots.txt` | crawler policy (currently: allow everyone, incl. training bots) |

**After deploy, validate:** Google [Rich Results Test](https://search.google.com/test/rich-results)
and [Schema.org validator](https://validator.schema.org/) on a post URL; confirm
`https://tigercosmos.xyz/robots.txt`, `/sitemap.xml`, `/llms.txt`, `/atom.xml` resolve.
