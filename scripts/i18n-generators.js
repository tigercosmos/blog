'use strict';

const pagination = require('hexo-pagination');

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

function getLanguages(config) {
  const langs = toLanguageArray(config.language);
  const seen = new Set();
  const result = [];
  for (const lang of langs) {
    if (!lang || lang === 'default') continue;
    if (seen.has(lang)) continue;
    seen.add(lang);
    result.push(lang);
  }
  return result.length ? result : [getDefaultLang(config)];
}

function normalizeDir(input) {
  if (!input) return '';
  let dir = String(input);
  if (dir === '/') return '';
  if (dir.startsWith('/')) dir = dir.slice(1);
  if (dir && !dir.endsWith('/')) dir += '/';
  return dir;
}

function getPostLang(post, defaultLang) {
  return (post && post.lang) ? post.lang : defaultLang;
}

function filterPostsByLang(posts, lang, defaultLang) {
  return posts.filter((post) => getPostLang(post, defaultLang) === lang);
}

const fmtNum = (num) => (num < 10 ? `0${num}` : String(num));

hexo.extend.generator.register('index', function i18nIndexGenerator(locals) {
  const config = this.config;
  const defaultLang = getDefaultLang(config);
  const languages = getLanguages(config);

  const indexConfig = config.index_generator || {};
  const perPage = indexConfig.per_page;
  const orderBy = indexConfig.order_by || '-date';
  const paginationDir = config.pagination_dir || 'page';

  const basePath = normalizeDir(indexConfig.path || '');
  const allPosts = locals.posts.sort(orderBy);

  const pages = [];

  for (const lang of languages) {
    const langPrefix = lang === defaultLang ? '' : `${lang}/`;
    const langBase = `${langPrefix}${basePath}`;
    const langPosts = filterPostsByLang(allPosts, lang, defaultLang);
    if (!langPosts.length) continue;

    pages.push(...pagination(langBase, langPosts, {
      perPage,
      layout: ['index', 'archive'],
      format: `${paginationDir}/%d/`,
      data: {
        __index: true,
        lang
      }
    }));
  }

  return pages;
});

hexo.extend.generator.register('tag', function i18nTagGenerator(locals) {
  const config = this.config;
  const defaultLang = getDefaultLang(config);
  const languages = getLanguages(config);

  const perPage = config.tag_generator.per_page;
  const paginationDir = config.pagination_dir || 'page';
  const orderBy = config.tag_generator.order_by || '-date';
  const tags = locals.tags;

  const pages = [];

  for (const lang of languages) {
    const langPrefix = lang === defaultLang ? '' : `${lang}/`;
    tags.forEach((tag) => {
      if (!tag.length) return;

      const posts = filterPostsByLang(tag.posts.sort(orderBy), lang, defaultLang);
      if (!posts.length) return;

      const path = `${langPrefix}${tag.path}`;
      pages.push(...pagination(path, posts, {
        perPage,
        layout: ['tag', 'archive', 'index'],
        format: `${paginationDir}/%d/`,
        data: {
          tag: tag.name,
          lang
        }
      }));
    });
  }

  return pages;
});

hexo.extend.generator.register('archive', function i18nArchiveGenerator(locals) {
  const config = this.config;
  const defaultLang = getDefaultLang(config);
  const languages = getLanguages(config);

  const paginationDir = config.pagination_dir || 'page';
  const orderBy = config.archive_generator.order_by || '-date';
  const perPage = config.archive_generator.per_page;
  const baseArchiveDir = normalizeDir(config.archive_dir);
  const allPosts = locals.posts.sort(orderBy);

  const Query = this.model('Post').Query;
  const pages = [];

  for (const lang of languages) {
    const langPosts = filterPostsByLang(allPosts, lang, defaultLang);
    if (!langPosts.length) continue;

    const langPrefix = lang === defaultLang ? '' : `${lang}/`;
    const archiveDir = `${langPrefix}${baseArchiveDir}`;

    function generate(path, posts, options) {
      const data = Object.assign({ archive: true, lang }, options);
      pages.push(...pagination(path, posts, {
        perPage,
        layout: ['archive', 'index'],
        format: `${paginationDir}/%d/`,
        data
      }));
    }

    generate(archiveDir, langPosts);

    if (!config.archive_generator.yearly) continue;

    const postsByYear = {};

    langPosts.forEach((post) => {
      const date = post.date;
      const year = date.year();
      const month = date.month() + 1;

      if (!Object.prototype.hasOwnProperty.call(postsByYear, year)) {
        postsByYear[year] = Array.from({ length: 13 }, () => []);
      }

      postsByYear[year][0].push(post);
      postsByYear[year][month].push(post);

      if (config.archive_generator.daily) {
        const day = date.date();
        if (!Object.prototype.hasOwnProperty.call(postsByYear[year][month], 'day')) {
          postsByYear[year][month].day = {};
        }
        (postsByYear[year][month].day[day] || (postsByYear[year][month].day[day] = [])).push(post);
      }
    });

    const years = Object.keys(postsByYear);

    for (let i = 0, len = years.length; i < len; i++) {
      const year = +years[i];
      const yearData = postsByYear[year];
      const yearUrl = `${archiveDir}${year}/`;
      if (!yearData[0].length) continue;

      generate(yearUrl, new Query(yearData[0]), { year });

      if (!config.archive_generator.monthly && !config.archive_generator.daily) continue;

      for (let month = 1; month <= 12; month++) {
        const monthData = yearData[month];
        if (!monthData.length) continue;

        if (config.archive_generator.monthly) {
          generate(`${yearUrl}${fmtNum(month)}/`, new Query(monthData), { year, month });
        }

        if (!config.archive_generator.daily) continue;

        for (let day = 1; day <= 31; day++) {
          const dayData = monthData.day && monthData.day[day];
          if (!dayData || !dayData.length) continue;
          generate(`${yearUrl}${fmtNum(month)}/${fmtNum(day)}/`, new Query(dayData), { year, month, day });
        }
      }
    }
  }

  return pages;
});
