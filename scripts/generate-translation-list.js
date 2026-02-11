'use strict';

const fs = require('fs');
const path = require('path');
const {
  parse: parseFrontMatter
} = require('hexo-front-matter');

const ROOT = process.cwd();
const POSTS_DIR = path.join(ROOT, 'source', '_posts');
const TRANSLATIONS_DIR = path.join(ROOT, 'source', '_posts_translation');
const OUT_FILE = path.join(ROOT, 'list.md');

function listMarkdownFiles(dir) {
  const out = [];
  const stack = [dir];
  while (stack.length) {
    const cur = stack.pop();
    const entries = fs.readdirSync(cur, {
      withFileTypes: true
    });
    for (const ent of entries) {
      const full = path.join(cur, ent.name);
      if (ent.isDirectory()) {
        stack.push(full);
      } else if (ent.isFile() && ent.name.toLowerCase().endsWith('.md')) {
        out.push(full);
      }
    }
  }
  out.sort((a, b) => a.localeCompare(b));
  return out;
}

function readFrontMatter(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8');
  const normalized = raw.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const fm = parseFrontMatter(normalized);
  return {
    title: typeof fm.title === 'string' ? fm.title.trim() : '',
    date: fm.date ? String(fm.date) : '',
    lang: typeof fm.lang === 'string' ? fm.lang.trim() : '',
    translationKey: typeof fm.translation_key === 'string' ? fm.translation_key.trim() : '',
  };
}

function exists(p) {
  try {
    fs.accessSync(p, fs.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

function relFromPostsDir(filePath) {
  return path.relative(POSTS_DIR, filePath).replace(/\\/g, '/');
}

function translationPath(lang, relPostPath) {
  return path.join(TRANSLATIONS_DIR, lang, relPostPath);
}

function fmtCheck(x) {
  return x ? '✅' : '⬜';
}

function fmtDone(enOk, jpOk) {
  return enOk && jpOk ? '[x]' : '[ ]';
}

const SKIP_DIR = [
  '_posts/let-build-dbms/',
  '_posts/angular/'
];

function main() {
  if (!exists(POSTS_DIR)) {
    console.error(`Missing posts dir: ${POSTS_DIR}`);
    process.exit(1);
  }

  const postFiles = listMarkdownFiles(POSTS_DIR);

  const filteredFiles = postFiles.filter((absPath) => {
    for (const dirName of SKIP_DIR) {
      if (absPath.includes(dirName)) {
        return false;
      }
    }
    return true;
  });

  const rows = filteredFiles.map((absPath) => {
    console.log(`Processing: ${absPath}`);
    const rel = relFromPostsDir(absPath);
    const meta = readFrontMatter(absPath);
    const enFile = translationPath('en', rel);
    const jpFile = translationPath('jp', rel);
    const enOk = exists(enFile);
    const jpOk = exists(jpFile);
    return {
      rel,
      title: meta.title,
      date: meta.date,
      sourceLang: meta.lang || '(default)',
      hasKey: Boolean(meta.translationKey),
      enOk,
      jpOk,
    };
  });

  const total = rows.length;
  const doneBoth = rows.filter((r) => r.enOk && r.jpOk).length;
  const missingEn = rows.filter((r) => !r.enOk).length;
  const missingJp = rows.filter((r) => !r.jpOk).length;
  const missingKey = rows.filter((r) => !r.hasKey).length;

  const lines = [];
  lines.push('# Translation Checklist');
  lines.push('');
  lines.push('This file is generated. Re-run `node scripts/generate-translation-list.js` after adding translations.');
  lines.push('');
  lines.push(`- Total posts: ${total}`);
  lines.push(`- Posts with both EN+JP translations: ${doneBoth}`);
  lines.push(`- Missing EN translation files: ${missingEn}`);
  lines.push(`- Missing JP translation files: ${missingJp}`);
  lines.push(`- Source posts missing \`translation_key\`: ${missingKey}`);
  lines.push('');
  lines.push('## Posts');
  lines.push('');

  for (const r of rows) {
    const suffixParts = [];
    suffixParts.push(`EN ${fmtCheck(r.enOk)}`);
    suffixParts.push(`JP ${fmtCheck(r.jpOk)}`);
    suffixParts.push(`key ${r.hasKey ? '✅' : '⬜'}`);
    if (r.sourceLang && r.sourceLang !== '(default)') suffixParts.push(`src:${r.sourceLang}`);
    const title = r.title ? ` — ${r.title}` : '';
    const date = r.date ? ` (${r.date})` : '';
    lines.push(`- ${fmtDone(r.enOk, r.jpOk)} \`${r.rel}\`${title}${date} — ${suffixParts.join(', ')}`);
  }

  fs.writeFileSync(OUT_FILE, `${lines.join('\n')}\n`, 'utf8');
  console.log(`Wrote ${OUT_FILE}`);
}

if (require.main === module) {
  main();
}