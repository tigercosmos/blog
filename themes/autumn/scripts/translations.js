'use strict';

function getDefaultLang(config) {
  const lang = config.language;
  if (Array.isArray(lang)) return lang[0];
  if (typeof lang === 'string') return lang.split(',')[0].trim();
  return 'en';
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
  const legacyPrefix = `_posts/${lang}/`;
  let prefix = null;
  if (source.startsWith(translationPrefix)) prefix = translationPrefix;
  if (!prefix && source.startsWith(legacyPrefix)) prefix = legacyPrefix;
  if (!prefix) return post.slug;
  const stripped = source.slice(prefix.length).replace(/\.md$/i, '');
  return stripped;
}

hexo.extend.helper.register('translation_entries', function (page) {
  const entries = getTranslationPosts(hexo, page);
  if (!entries.length) return [];

  const base = (this.config.url || '').replace(/\/$/, '');
  const defaultLang = getDefaultLang(hexo.config);

  return entries.map((post) => {
    const lang = post.lang || defaultLang;
    let path;
    if (lang !== defaultLang && post.slug && post.date && post.date.format) {
      const year = post.date.format('YYYY');
      const month = post.date.format('MM');
      const slugPath = getPostSlugPath(post, lang) || post.slug;
      path = `/${lang}/post/${year}/${month}/${slugPath}/`;
    } else {
      path = this.url_for(post.path);
    }
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
