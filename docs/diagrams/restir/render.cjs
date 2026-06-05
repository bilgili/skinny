// LaTeX → standalone SVG for the skinny docs (KaTeX renders unreliably on the
// repo's GitLab, so display equations ship as committed SVGs instead).
//
// Usage: node render.cjs <equations.json> <out-dir> [pxPerEx]
//   equations.json: [{ "name": "<file-stem>", "tex": "<LaTeX math>" }, ...]
// Writes <out-dir>/<name>.svg (one display equation each). MathJax 3 / full
// LaTeX math support; glyph colour is baked to #0f172a (the diagram palette) so
// the SVG renders the same loaded as an <img> in light or dark doc themes.

const fs = require('fs');
const path = require('path');
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');

const [, , jsonPath, outDir, pxPerExArg] = process.argv;
if (!jsonPath || !outDir) {
  console.error('usage: node render.cjs <equations.json> <out-dir> [pxPerEx]');
  process.exit(1);
}
const PX_PER_EX = Number(pxPerExArg) || 9;
const COLOR = '#0f172a';

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages });
const svg = new SVG({ fontCache: 'none' }); // inline glyph paths → fully standalone
const doc = mathjax.document('', { InputJax: tex, OutputJax: svg });

function render(latex) {
  const node = doc.convert(latex, { display: true });
  let s = adaptor.innerHTML(node); // the <svg>…</svg> element
  s = s.replace(/currentColor/g, COLOR);
  // ex units → px so each file has an intrinsic, theme-independent size
  s = s.replace(/(width|height)="([\d.]+)ex"/g,
    (_, dim, v) => `${dim}="${(Number(v) * PX_PER_EX).toFixed(1)}"`);
  if (!/xmlns=/.test(s)) s = s.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ');
  return s.trim() + '\n';
}

fs.mkdirSync(outDir, { recursive: true });
const eqs = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
for (const { name, tex: latex } of eqs) {
  const out = path.join(outDir, `${name}.svg`);
  fs.writeFileSync(out, render(latex));
  console.log(`wrote ${name}.svg`);
}
console.log(`done: ${eqs.length} equations → ${outDir}`);
