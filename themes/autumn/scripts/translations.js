'use strict';

function getDefaultLang(config) {
  const lang = config.language;
  if (Array.isArray(lang)) return lang[0];
  if (typeof lang === 'string') return lang.split(',')[0].trim();
  return 'en';
}

function getLanguages(config) {
  const lang = config.language;
  const raw = Array.isArray(lang)
    ? lang
    : (typeof lang === 'string' ? lang.split(',') : []);
  const seen = new Set();
  const result = [];
  raw.forEach((item) => {
    const normalized = String(item || '').trim();
    if (!normalized || normalized === 'default' || seen.has(normalized)) return;
    seen.add(normalized);
    result.push(normalized);
  });
  return result.length ? result : [getDefaultLang(config)];
}

function getTranslationPosts(hexo, page) {
  if (!page || !page.translation_key) return [];
  const posts = hexo.locals.get('posts');
  if (!posts) return [];
  return posts
    .filter((post) => post.translation_key === page.translation_key)
    .sort((a, b) => {
      const langA = a.lang || getDefaultLang(hexo.config);
      const langB = b.lang || getDefaultLang(hexo.config);
      return langA.localeCompare(langB);
    });
}

function getPostSlugPath(post, lang) {
  if (!post || !post.source) return post.slug;
  const source = post.source.replace(/\\/g, '/');
  const translationDir = hexo.config.translation_post_dir || '_posts_translation';
  const translationPrefix = `${translationDir.replace(/\/$/, '')}/${lang}/`;
  const legacyTranslationPrefix = (lang === 'ja')
    ? `${translationDir.replace(/\/$/, '')}/jp/`
    : null;
  const legacyPrefix = `_posts/${lang}/`;
  const legacyJpPrefix = lang === 'ja' ? '_posts/jp/' : null;
  let prefix = null;
  if (source.startsWith(translationPrefix)) prefix = translationPrefix;
  if (!prefix && legacyTranslationPrefix && source.startsWith(legacyTranslationPrefix)) prefix = legacyTranslationPrefix;
  if (!prefix && source.startsWith(legacyPrefix)) prefix = legacyPrefix;
  if (!prefix && legacyJpPrefix && source.startsWith(legacyJpPrefix)) prefix = legacyJpPrefix;
  if (!prefix) return post.slug;
  const stripped = source.slice(prefix.length).replace(/\.md$/i, '');
  return stripped;
}

function routeExists(urlPath) {
  if (typeof hexo === 'undefined' || !hexo.route || typeof hexo.route.get !== 'function') {
    return true; // fail open if the router is unavailable
  }
  let key = String(urlPath).replace(/^\/+/, '');
  if (key === '' || key.endsWith('/')) key += 'index.html';
  const candidates = [key];
  try {
    const decoded = decodeURI(key);
    if (decoded !== key) candidates.push(decoded);
  } catch (e) { /* malformed URI, ignore */ }
  try {
    const encoded = encodeURI(key);
    if (encoded !== key) candidates.push(encoded);
  } catch (e) { /* ignore */ }
  return candidates.some((candidate) => Boolean(hexo.route.get(candidate)));
}

function getListTranslationEntries(ctx, page) {
  if (!page || !(page.__index || page.archive || page.tag)) return [];

  const defaultLang = getDefaultLang(ctx.config);
  const languages = getLanguages(ctx.config);
  if (languages.length < 2) return [];

  const pageLang = page.lang || defaultLang;
  const currentUrlRaw = (typeof page.current_url === 'string') ? page.current_url : (page.path || '');
  const currentUrl = String(currentUrlRaw).replace(/^\/+/, '');
  let relativePath = currentUrl;

  if (pageLang !== defaultLang && relativePath.startsWith(`${pageLang}/`)) {
    relativePath = relativePath.slice(pageLang.length + 1);
  }

  const base = (ctx.config.url || '').replace(/\/$/, '');
  return languages
    .map((lang) => {
      const withPrefix = lang === defaultLang ? `/${relativePath}` : `/${lang}/${relativePath}`;
      const path = withPrefix.replace(/\/{2,}/g, '/');
      return {
        lang,
        path,
        abs: `${base}${path}`,
        title: page.title || '',
        isCurrent: lang === pageLang,
        isDefault: lang === defaultLang,
      };
    })
    // Only keep languages whose list page actually exists — the i18n
    // generators skip languages that have no matching posts, so fabricating
    // a URL for every configured language would emit hreflang/switcher links
    // that 404. Always keep the current page even if a route lookup misses.
    .filter((entry) => entry.isCurrent || routeExists(entry.path));
}

hexo.extend.helper.register('translation_entries', function (page) {
  const entries = getTranslationPosts(hexo, page);
  if (!entries.length) return getListTranslationEntries(this, page);

  const base = (this.config.url || '').replace(/\/$/, '');
  const defaultLang = getDefaultLang(hexo.config);

  return entries.map((post) => {
    const lang = post.lang || defaultLang;
    // Use the post's actual generated path. Hexo computes post.path from the
    // same permalink/timezone rules it uses to write the file (via the
    // post_permalink filter below), so it always matches the file on disk.
    // Rebuilding the path from post.date instead drifts whenever a timezone
    // rollover pushes the display month past the month baked into the path
    // (e.g. 2026-02-28 23:42 +08:00 -> file at /2026/03/ but format('MM') -> 02).
    const path = this.url_for(post.path);
    return {
      lang,
      path,
      abs: `${base}${path}`,
      title: post.title,
      isCurrent: page._id === post._id,
      isDefault: lang === defaultLang,
    };
  });
});

hexo.extend.filter.register('post_permalink', function (data) {
  const defaultLang = getDefaultLang(hexo.config);
  if (!data || data.layout !== 'post') return data;
  if (!data.lang || data.lang === defaultLang) return data;
  if (!data.slug && !data.source) return data;

  const year = data.date && data.date.format ? data.date.format('YYYY') : '';
  const month = data.date && data.date.format ? data.date.format('MM') : '';
  const slugPath = getPostSlugPath(data, data.lang) || data.slug;
  data.__permalink = `${data.lang}/post/${year}/${month}/${slugPath}/`;

  return data;
}, 1);
