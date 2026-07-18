# NOTE: `npm run deploy` (hexo deploy) is intentionally a no-op here —
# _config.yml sets `deploy.type: ''`, so this only regenerates the site into public/.
# Actual publishing to GitHub Pages is handled by peaceiris/actions-gh-pages in CI.
npm install
npm run clean
npm run deploy
cp CNAME public
cp ads.txt public