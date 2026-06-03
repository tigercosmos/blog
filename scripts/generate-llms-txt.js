'use strict';

// Generates /llms.txt (https://llmstxt.org/) — a curated Markdown index of the
// site for LLMs.
//
// Reality check (2026): Google explicitly does NOT use llms.txt, and server
// logs across studies show near-zero real AI-bot consumption. This is shipped
// as a cheap, future-facing experiment with NO traffic expectation. The
// load-bearing AEO signals remain: clean static HTML, JSON-LD, sitemap, feed.

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

function cleanText(html) {
  if (!html) return '';
  let s = String(html).replace(/<a\b[^>]*class="[^"]*header-anchor[^"]*"[^>]*>[\s\S]*?<\/a>/gi, '');
  s = stripHTML(s).replace(/¶/g, ' ');
  return s.replace(/\s+/g, ' ').trim();
}

function summarize(post, n) {
  let d = (post.des || post.description) ? String(post.des || post.description).replace(/\s+/g, ' ').trim() : '';
  if (!d && post.excerpt) d = cleanText(post.excerpt);
  if (!d && post.content) d = cleanText(post.content);
  if (d.length > n) d = d.slice(0, n - 1).replace(/\s+\S*$/, '') + '…';
  return d;
}

hexo.extend.generator.register('llms-txt', function llmsTxtGenerator(locals) {
  const config = this.config;
  const defaultLang = getDefaultLang(config);
  const base = String(config.url || '').replace(/\/$/, '');

  const posts = locals.posts.sort('-date').toArray()
    .filter((p) => (p.lang || defaultLang) === defaultLang);

  const lines = [];
  lines.push('# ' + config.title);
  lines.push('');
  lines.push('> ' + (config.description || config.subtitle || ''));
  lines.push('');
  lines.push(
    (config.author ? 'Author: ' + config.author + '. ' : '') +
    'Topics: software engineering, systems & high-performance programming, C++, Rust, ' +
    'compilers and open source. Original posts are in Traditional Chinese (zh-TW); ' +
    'selected posts are also translated into English and Japanese.'
  );
  lines.push('');
  lines.push('## Posts');
  lines.push('');
  posts.forEach((p) => {
    const url = p.permalink || (base + '/' + String(p.path || '').replace(/^\/+/, ''));
    const sum = summarize(p, 160);
    lines.push('- [' + p.title + '](' + url + ')' + (sum ? ': ' + sum : ''));
  });
  lines.push('');
  lines.push('## Optional');
  lines.push('');
  lines.push('- [About the author](' + base + '/about/)');
  lines.push('- [Books](' + base + '/books/)');
  lines.push('- [Atom feed](' + base + '/atom.xml)');
  lines.push('');

  return { path: 'llms.txt', data: lines.join('\n') };
});
