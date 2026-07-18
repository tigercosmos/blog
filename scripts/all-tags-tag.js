'use strict';

// Fills the "所有標籤" cloud on the navigation page from the live tag database.
//
// Why a filter and not a {% tag %}: tag plugins render during Hexo's *process*
// phase, and the navigation page is often processed before most posts' tags are
// registered — so the tag DB is only partially populated at that moment. The
// `after_render:html` filter runs during *generation*, when every tag exists.
//
// The navigation markdown contains a placeholder element:
//     <div class="all-tags-cloud"></div>
// which this filter replaces with a "·"-separated cloud of every tag, sorted by
// post count (desc, then name). Each link uses the tag's own `.path`, so slug
// quirks (e.g. the `c++` tag living at /tags/c/) resolve with no special-casing.

const PLACEHOLDER = '<div class="all-tags-cloud"></div>';

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

hexo.extend.filter.register('after_render:html', function (str, data) {
  if (str.indexOf(PLACEHOLDER) === -1) return str;

  const tags = hexo.locals.get('tags');
  if (!tags || !tags.length) return str;

  const root = (hexo.config.root || '/').replace(/\/+$/, '');
  const cloud = tags
    .toArray()
    .sort((a, b) => (b.length - a.length) || a.name.localeCompare(b.name))
    .map((tag) => '<a href="' + root + '/' + tag.path + '">' + escapeHtml(tag.name) + '</a>')
    .join(' · ');

  return str.replace(PLACEHOLDER, '<div class="all-tags-cloud">' + cloud + '</div>');
});
