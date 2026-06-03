'use strict';

// SEO / AEO helpers for the autumn theme.
//
// All metadata for AI answer engines (GPTBot, ClaudeBot, PerplexityBot,
// OAI-SearchBot, ...) must be emitted server-side in the static HTML, because
// those crawlers do NOT execute JavaScript (Vercel, >1.3B fetches). Hexo's
// pre-rendered output is ideal for this; these helpers build the head metadata
// and a connected JSON-LD @graph at build time.

const { stripHTML } = require('hexo-util');

function toLanguageArray(languageConfig) {
  if (Array.isArray(languageConfig)) return languageConfig;
  if (typeof languageConfig === 'string') {
    return languageConfig.split(',').map((s) => s.trim()).filter(Boolean);
  }
  return [];
}

function getDefaultLang(config) {
  const langs = toLanguageArray(config && config.language);
  return langs[0] || 'en';
}

// Open Graph locale form (underscore): zh -> zh_TW, en -> en_US, ja/jp -> ja_JP
function ogLocale(lang) {
  switch (lang) {
    case 'zh': return 'zh_TW';
    case 'en': return 'en_US';
    case 'ja':
    case 'jp': return 'ja_JP';
    default: return 'en_US';
  }
}

// BCP-47 form (hyphen) for schema.org inLanguage / html lang.
function bcp47(lang) {
  switch (lang) {
    case 'zh': return 'zh-TW';
    case 'jp': return 'ja';
    default: return lang || 'en';
  }
}

function toArray(x) {
  if (!x) return [];
  if (Array.isArray(x)) return x;
  if (typeof x.toArray === 'function') return x.toArray();
  if (typeof x.forEach === 'function') {
    const out = [];
    x.forEach((item) => out.push(item));
    return out;
  }
  return [x];
}

function firstContentImage(content) {
  if (!content) return null;
  const m = /<img[^>]*\bsrc=["']([^"']+)["']/i.exec(String(content));
  return m && m[1] ? m[1] : null;
}

function base(config) {
  return String((config && config.url) || '').replace(/\/$/, '');
}

function absUrl(config, u) {
  if (!u) return u;
  if (/^https?:\/\//i.test(u)) return u;
  return base(config) + '/' + String(u).replace(/^\/+/, '');
}

function getTheme(ctx) {
  return ctx.theme || (typeof hexo !== 'undefined' && hexo.theme && hexo.theme.config) || {};
}

// Strip HTML to plain text, first removing heading permalink anchors so the
// "¶ " marker (markdown.anchors.permalinkSymbol) never leaks into summaries.
function cleanText(html) {
  if (!html) return '';
  let s = String(html).replace(/<a\b[^>]*class="[^"]*header-anchor[^"]*"[^>]*>[\s\S]*?<\/a>/gi, '');
  s = stripHTML(s).replace(/¶/g, ' ');
  return s.replace(/\s+/g, ' ').trim();
}

// Build a concise description: front-matter `des`/`description` -> excerpt ->
// post body -> site description. Never falls back to the title (a duplicate
// signal that meta/OG/JSON-LD should not share).
function getDescription(page, config, maxLen) {
  page = page || {};
  let d = (page.des || page.description) ? String(page.des || page.description).replace(/\s+/g, ' ').trim() : '';
  if (!d && page.excerpt) d = cleanText(page.excerpt);
  if (!d && page.content) d = cleanText(page.content);
  if (!d) d = (config && (config.description || config.subtitle)) || '';
  const cap = maxLen || 160;
  if (d.length > cap) d = d.slice(0, cap - 1).replace(/\s+\S*$/, '') + '…';
  return d;
}

function pageUrl(ctx, page) {
  const config = ctx.config;
  let u;
  if (page && page.permalink) {
    u = page.permalink;
  } else {
    const p = (page && (page.path || page.current_url)) || '';
    const urlFor = (typeof ctx.url_for === 'function') ? ctx.url_for.bind(ctx) : (x) => x;
    u = base(config) + urlFor('/' + String(p).replace(/^\/+/, ''));
  }
  // Prefer the clean directory URL over an explicit index.html.
  return String(u).replace(/index\.html$/, '');
}

function ogImage(ctx, page) {
  const config = ctx.config;
  const theme = getTheme(ctx);
  const fromContent = firstContentImage(page && page.content);
  const fromFm = (page && (page.cover || (toArray(page.photos)[0]))) || null;
  const fallback = theme.og_image || '/img/cover.png';
  return absUrl(config, fromContent || fromFm || fallback);
}

// Dimensions are only known for the configured default image (remote post
// images have unknown size, so we omit width/height there).
function ogImageDims(ctx, page) {
  const theme = getTheme(ctx);
  const usingDefault = !firstContentImage(page && page.content) &&
    !(page && (page.cover || toArray(page.photos)[0]));
  if (usingDefault && theme.og_image_width && theme.og_image_height) {
    return { width: Number(theme.og_image_width), height: Number(theme.og_image_height) };
  }
  return null;
}

function isoDate(d) {
  return (d && typeof d.toISOString === 'function') ? d.toISOString() : undefined;
}

// One connected JSON-LD @graph: WebSite + Organization + Person on every page,
// plus WebPage + ImageObject + BreadcrumbList + BlogPosting on posts. Nodes are
// linked by stable @id fragments so AI/Google entity-resolution can follow
// Person -> Organization -> Article (the E-E-A-T / authorship graph).
function buildGraph(ctx, page) {
  const config = ctx.config;
  const theme = getTheme(ctx);
  const b = base(config);
  const author = theme.author || {};
  const siteName = config.title;
  const lang = bcp47((page && page.lang) || getDefaultLang(config));
  const urlFor = (typeof ctx.url_for === 'function') ? ctx.url_for.bind(ctx) : (x) => x;

  const websiteId = b + '/#website';
  const orgId = b + '/#organization';
  const personUrl = author.url ? String(author.url).replace(/\/$/, '') : (b + '/about');
  const personId = personUrl + '/#person';
  const logoUrl = absUrl(config, theme.logo || theme.og_image || '/img/cover.png');

  const website = {
    '@type': 'WebSite',
    '@id': websiteId,
    url: b + '/',
    name: siteName,
    description: (config.description || config.subtitle || ''),
    inLanguage: lang,
    publisher: { '@id': orgId }
  };

  const organization = {
    '@type': 'Organization',
    '@id': orgId,
    name: siteName,
    url: b + '/',
    logo: { '@type': 'ImageObject', url: logoUrl, contentUrl: logoUrl },
    founder: { '@id': personId }
  };
  if (Array.isArray(author.sameAs) && author.sameAs.length) organization.sameAs = author.sameAs;

  const person = {
    '@type': 'Person',
    '@id': personId,
    name: author.name || config.author,
    url: author.url || (b + '/about/')
  };
  if (author.image) {
    const img = absUrl(config, author.image);
    person.image = { '@type': 'ImageObject', url: img, contentUrl: img };
  }
  if (author.jobTitle) person.jobTitle = author.jobTitle;
  if (author.description) person.description = author.description;
  if (Array.isArray(author.knowsAbout) && author.knowsAbout.length) person.knowsAbout = author.knowsAbout;
  if (Array.isArray(author.alumniOf) && author.alumniOf.length) person.alumniOf = author.alumniOf;
  if (Array.isArray(author.sameAs) && author.sameAs.length) person.sameAs = author.sameAs;

  const graph = [website, organization, person];

  if (page && page.layout === 'post') {
    const url = pageUrl(ctx, page);
    const webpageId = url + '#webpage';
    const articleId = url + '#article';
    const breadcrumbId = url + '#breadcrumb';
    const primaryImageId = url + '#primaryimage';

    const datePublished = isoDate(page.date);
    const dateModified = isoDate(page.updated) || datePublished;
    const imgUrl = ogImage(ctx, page);
    const dims = ogImageDims(ctx, page);

    const primaryImage = {
      '@type': 'ImageObject',
      '@id': primaryImageId,
      url: imgUrl,
      contentUrl: imgUrl
    };
    if (dims) { primaryImage.width = dims.width; primaryImage.height = dims.height; }

    const crumbs = [{ '@type': 'ListItem', position: 1, name: 'Home', item: b + '/' }];
    let pos = 2;
    toArray(page.categories).forEach((cat) => {
      crumbs.push({ '@type': 'ListItem', position: pos++, name: cat.name || String(cat), item: absUrl(config, urlFor(cat.path || '')) });
    });
    crumbs.push({ '@type': 'ListItem', position: pos, name: page.title });
    const breadcrumb = { '@type': 'BreadcrumbList', '@id': breadcrumbId, itemListElement: crumbs };

    const webpage = {
      '@type': 'WebPage',
      '@id': webpageId,
      url: url,
      name: page.title,
      isPartOf: { '@id': websiteId },
      primaryImageOfPage: { '@id': primaryImageId },
      breadcrumb: { '@id': breadcrumbId },
      inLanguage: lang,
      datePublished: datePublished,
      dateModified: dateModified
    };

    const article = {
      '@type': 'BlogPosting',
      '@id': articleId,
      isPartOf: { '@id': webpageId },
      mainEntityOfPage: { '@id': webpageId },
      headline: page.title,
      name: page.title,
      description: getDescription(page, config),
      datePublished: datePublished,
      dateModified: dateModified,
      inLanguage: lang,
      author: { '@id': personId },
      publisher: { '@id': orgId },
      image: { '@id': primaryImageId }
    };
    const tags = toArray(page.tags).map((t) => (t && t.name) ? t.name : t).filter(Boolean);
    if (tags.length) article.keywords = tags.join(', ');
    const sections = toArray(page.categories).map((c) => (c && c.name) ? c.name : c).filter(Boolean);
    if (sections.length) article.articleSection = sections;

    graph.push(primaryImage, webpage, breadcrumb, article);
  }

  return { '@context': 'https://schema.org', '@graph': graph };
}

hexo.extend.helper.register('seo_description', function seoDescriptionHelper(page) {
  return getDescription(page || this.page, this.config);
});

hexo.extend.helper.register('seo_canonical', function seoCanonicalHelper(page) {
  return pageUrl(this, page || this.page);
});

hexo.extend.helper.register('seo_og_image', function seoOgImageHelper(page) {
  return ogImage(this, page || this.page);
});

hexo.extend.helper.register('seo_og_image_dims', function seoOgImageDimsHelper(page) {
  return ogImageDims(this, page || this.page);
});

hexo.extend.helper.register('seo_og_locale', function seoOgLocaleHelper(lang) {
  return ogLocale(lang || (this.page && this.page.lang) || getDefaultLang(this.config));
});

// Returns the JSON-LD object as a string safe to drop inside a <script> tag
// (escapes "<" so a "</script>" inside any field can't break out).
hexo.extend.helper.register('seo_jsonld', function seoJsonldHelper(page) {
  try {
    const obj = buildGraph(this, page || this.page);
    return JSON.stringify(obj).replace(/</g, '\\u003c');
  } catch (e) {
    return '';
  }
});
