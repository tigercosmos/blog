'use strict';

const Promise = require('bluebird');
const { parse: yfm } = require('hexo-front-matter');
const { extname } = require('path');
const { stat } = require('hexo-fs');
const { slugize, Pattern, Permalink } = require('hexo-util');
const { toDate, timezone, isExcludedFile, isTmpFile, isHiddenFile, isMatch } = require('hexo/lib/plugins/processor/common');

function getPostDir(config) {
  const dir = config.translation_post_dir || '_posts_translation';
  return dir.endsWith('/') ? dir : `${dir}/`;
}

function normalizeLang(lang) {
  return lang === 'jp' ? 'ja' : lang;
}
let permalink;

const preservedKeys = {
  title: true,
  year: true,
  month: true,
  day: true,
  i_month: true,
  i_day: true,
  hash: true
};

function parseFilename(config, path) {
    config = config.substring(0, config.length - extname(config).length);
    path = path.substring(0, path.length - extname(path).length);

    if (!permalink || permalink.rule !== config) {
      permalink = new Permalink(config, {
        segments: {
          year: /(\d{4})/,
          month: /(\d{2})/,
          day: /(\d{2})/,
          i_month: /(\d{1,2})/,
          i_day: /(\d{1,2})/,
          hash: /([0-9a-f]{12})/
        }
      });
    }

    const data = permalink.parse(path);

    if (data) return data;

    return {
      title: slugize(path)
    };
}

function processPost(file) {
    const Post = hexo.model('Post');
    const { path } = file.params;
    const doc = Post.findOne({ source: file.path });
    const { config } = hexo;
    const { timezone: timezoneCfg } = config;
    const updated_option = config.use_date_for_updated === true ? 'date' : config.updated_option;
    let categories, tags;

    if (file.type === 'skip' && doc) return;

    if (file.type === 'delete') {
      if (doc) return doc.remove();
      return;
    }

    return Promise.all([
      file.stat(),
      file.read()
    ]).spread((stats, content) => {
      const data = yfm(content);
      const info = parseFilename(config.new_post_name, path);
      const keys = Object.keys(info);

      data.source = file.path;
      data.raw = content;
      data.slug = path.substring(0, path.length - extname(path).length);
      data.lang = normalizeLang(file.params.lang);

      if (file.params.published) {
        if (!Object.prototype.hasOwnProperty.call(data, 'published')) data.published = true;
      } else {
        data.published = false;
      }

      for (let i = 0, len = keys.length; i < len; i++) {
        const key = keys[i];
        if (!preservedKeys[key]) data[key] = info[key];
      }

      if (data.date) {
        data.date = toDate(data.date);
      } else if (info && info.year && (info.month || info.i_month) && (info.day || info.i_day)) {
        data.date = new Date(
          info.year,
          parseInt(info.month || info.i_month, 10) - 1,
          parseInt(info.day || info.i_day, 10)
        );
      }

      if (data.date) {
        if (timezoneCfg) data.date = timezone(data.date, timezoneCfg);
      } else {
        data.date = stats.birthtime;
      }

      data.updated = toDate(data.updated);

      if (data.updated) {
        if (timezoneCfg) data.updated = timezone(data.updated, timezoneCfg);
      } else if (updated_option === 'date') {
        data.updated = data.date;
      } else if (updated_option === 'empty') {
        delete data.updated;
      } else {
        data.updated = stats.mtime;
      }

      if (data.category && !data.categories) {
        data.categories = data.category;
        delete data.category;
      }

      if (data.tag && !data.tags) {
        data.tags = data.tag;
        delete data.tag;
      }

      categories = data.categories || [];
      tags = data.tags || [];

      if (!Array.isArray(categories)) categories = [categories];
      if (!Array.isArray(tags)) tags = [tags];

      if (data.photo && !data.photos) {
        data.photos = data.photo;
        delete data.photo;
      }

      if (data.photos && !Array.isArray(data.photos)) {
        data.photos = [data.photos];
      }

      if (data.link && !data.title) {
        data.title = data.link.replace(/^https?:\/\//, '').replace(/\/$/, '');
      }

      if (data.permalink) {
        data.__permalink = data.permalink;
        delete data.permalink;
      }

      const doc = Post.findOne({ source: file.path });

      if (doc) return doc.replace(data);

      return Post.insert(data);
    }).then(doc => Promise.all([
      doc.setCategories(categories),
      doc.setTags(tags)
    ]));
}

hexo.extend.processor.register(new Pattern(path => {
  const postDir = getPostDir(hexo.config);
  if (isTmpFile(path)) return;

  let result;

  if (path.startsWith(postDir)) {
    const rest = path.substring(postDir.length);
    const slashIndex = rest.indexOf('/');
    if (slashIndex === -1) return;

    const lang = normalizeLang(rest.substring(0, slashIndex));
    const postPath = rest.substring(slashIndex + 1);
    if (!lang || !postPath || isHiddenFile(postPath)) return;

    result = {
      published: true,
      lang,
      path: postPath
    };
  }

  if (!result) return;

  result.renderable = hexo.render.isRenderable(path) && !isMatch(path, hexo.config.skip_render);
  if (isExcludedFile(result.path, hexo.config)) return;

  return result;
}), function translationPostProcessor(file) {
  if (file.params.renderable) {
    return processPost(file);
  }
});
