'use strict';

const { url_for } = require('hexo-util');
const { toMomentLocale } = require('hexo/lib/plugins/helper/date');

function toLanguageArray(languageConfig) {
  if (Array.isArray(languageConfig)) return languageConfig;
  if (typeof languageConfig === 'string') {
    return languageConfig
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
  }
  return [];
}

function getDefaultLang(config) {
  const langs = toLanguageArray(config.language);
  return langs[0] || 'en';
}

function getPostLang(post, defaultLang) {
  return (post && post.lang) ? post.lang : defaultLang;
}

function getPageLang(ctx, explicitLang) {
  const defaultLang = getDefaultLang(ctx.config);
  return explicitLang || (ctx.page && ctx.page.lang) || defaultLang;
}

function withLangPrefix(path, lang, defaultLang) {
  let p = String(path || '');
  if (!p || p === '/') return lang === defaultLang ? '' : `${lang}/`;
  p = p.replace(/^\/+/, '');
  if (!p) return lang === defaultLang ? '' : `${lang}/`;
  if (lang === defaultLang) return p;
  if (p === lang || p.startsWith(`${lang}/`)) return p;
  return `${lang}/${p}`;
}

hexo.extend.helper.register('url_for_lang', function urlForLangHelper(path, lang) {
  const defaultLang = getDefaultLang(this.config);
  const pageLang = getPageLang(this, lang);
  const prefixed = withLangPrefix(path, pageLang, defaultLang);
  return url_for.call(this, prefixed);
});

hexo.extend.helper.register('list_tags_i18n', function listTagsI18nHelper(options = {}) {
  const defaultLang = getDefaultLang(this.config);
  const pageLang = getPageLang(this, options.lang);

  const {
    style = 'list',
    transform,
    separator = ', ',
    suffix = '',
    orderby = 'name',
    order = 1
  } = options;

  const showCount = Object.prototype.hasOwnProperty.call(options, 'show_count') ? options.show_count : true;

  const classStyle = typeof style === 'string' ? `-${style}` : '';
  const className = (typeof options.class === 'string') ? options.class : 'tag';
  const ulClass = (options.class && options.class.ul) || `${className}${classStyle}`;
  const liClass = (options.class && options.class.li) || `${className}${classStyle}-item`;
  const aClass = (options.class && options.class.a) || `${className}${classStyle}-link`;
  const countClass = (options.class && options.class.count) || `${className}${classStyle}-count`;

  const tags = [];
  this.site.tags.forEach((tag) => {
    if (!tag.length) return;
    const count = tag.posts.filter((post) => getPostLang(post, defaultLang) === pageLang).length;
    if (!count) return;

    tags.push({
      name: tag.name,
      path: withLangPrefix(tag.path, pageLang, defaultLang),
      count
    });
  });

  if (!tags.length) return '';

  tags.sort((a, b) => {
    if (orderby === 'length') return (a.count - b.count) * order;
    return a.name.localeCompare(b.name) * order;
  });

  if (options.amount) tags.splice(options.amount);

  let result = '';

  if (style === 'list') {
    result += `<ul class="${ulClass}" itemprop="keywords">`;
    tags.forEach((tag) => {
      result += `<li class="${liClass}">`;
      result += `<a class="${aClass}" href="${url_for.call(this, tag.path)}${suffix}" rel="tag">`;
      result += transform ? transform(tag.name) : tag.name;
      result += '</a>';
      if (showCount) result += `<span class="${countClass}">${tag.count}</span>`;
      result += '</li>';
    });
    result += '</ul>';
  } else {
    tags.forEach((tag, i) => {
      if (i) result += separator;
      result += `<a class="${aClass}" href="${url_for.call(this, tag.path)}${suffix}" rel="tag">`;
      result += transform ? transform(tag.name) : tag.name;
      if (showCount) result += `<span class="${countClass}">${tag.count}</span>`;
      result += '</a>';
    });
  }

  return result;
});

hexo.extend.helper.register('list_archives_i18n', function listArchivesI18nHelper(options = {}) {
  const { config } = this;
  const defaultLang = getDefaultLang(config);
  const pageLang = getPageLang(this, options.lang);

  const archiveDir = String(config.archive_dir || 'archives').replace(/^\/+/, '').replace(/\/?$/, '/');
  const langArchiveDir = withLangPrefix(archiveDir, pageLang, defaultLang).replace(/\/?$/, '/');

  const timezone = config.timezone;
  const langLocale = toMomentLocale(pageLang || config.language);
  let { format } = options;

  const type = options.type || 'monthly';
  const { style = 'list', transform, separator = ', ' } = options;
  const showCount = Object.prototype.hasOwnProperty.call(options, 'show_count') ? options.show_count : true;
  const className = options.class || 'archive';
  const order = options.order || -1;

  const compareFunc = type === 'monthly'
    ? (yearA, monthA, yearB, monthB) => yearA === yearB && monthA === monthB
    : (yearA, monthA, yearB, monthB) => yearA === yearB;

  if (!format) format = type === 'monthly' ? 'MMMM YYYY' : 'YYYY';

  const posts = this.site.posts
    .filter((post) => getPostLang(post, defaultLang) === pageLang)
    .sort('date', order);

  if (!posts.length) return '';

  const data = [];

  posts.forEach((post) => {
    let date = post.date.clone();
    if (timezone) date = date.tz(timezone);

    const year = date.year();
    const month = date.month() + 1;
    const lastData = data[data.length - 1];

    if (!lastData || !compareFunc(lastData.year, lastData.month, year, month)) {
      if (langLocale) date = date.locale(langLocale);
      const name = date.format(format);
      data.push({ name, year, month, count: 1 });
    } else {
      lastData.count++;
    }
  });

  const link = (item) => {
    let url = `${langArchiveDir}${item.year}/`;
    if (type === 'monthly') {
      url += item.month < 10 ? `0${item.month}/` : `${item.month}/`;
    }
    return url_for.call(this, url);
  };

  let result = '';
  if (style === 'list') {
    result += `<ul class="${className}-list">`;
    data.forEach((item) => {
      result += `<li class="${className}-list-item">`;
      result += `<a class="${className}-list-link" href="${link(item)}">`;
      result += transform ? transform(item.name) : item.name;
      result += '</a>';
      if (showCount) result += `<span class="${className}-list-count">${item.count}</span>`;
      result += '</li>';
    });
    result += '</ul>';
  } else {
    data.forEach((item, i) => {
      if (i) result += separator;
      result += `<a class="${className}-link" href="${link(item)}">`;
      result += transform ? transform(item.name) : item.name;
      if (showCount) result += `<span class="${className}-count">${item.count}</span>`;
      result += '</a>';
    });
  }

  return result;
});

