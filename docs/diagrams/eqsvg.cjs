// Shared offline equation → SVG layout engine for the docs.
//
// The repo's GitLab does not render KaTeX/`$$` math reliably, so display
// equations ship as committed SVG images. The publication-quality path is
// MathJax (docs/diagrams/restir/render.cjs over a topic's equations.json); this
// dependency-free fallback renders monospace math with REAL raised/lowered
// sub/superscripts and 2-D fraction bars on a white card, so each SVG renders
// the same as an <img> in any viewer and in GitLab's light or dark theme.
//
// Markup (LaTeX-like): `_{...}`/`^{...}` (or single-char `_x`/`^x`) for sub- and
// superscripts; everything else is literal (Unicode ok: σ λ μ θ π · → −). p̂ is
// "p" + combining circumflex (U+0302). Topic generators require() this module
// and call writeAll(); see docs/diagrams/restir|skin/gen_svg_equations.cjs.

const fs = require('fs');
const path = require('path');

const F = 18;                 // base font size (px)
const SUB = 0.72;             // sub/superscript scale
const CW = F * 0.6;           // monospace advance (base)
const SUBCW = CW * SUB;       // monospace advance (sub/sup)
const ASC = F * 0.80, DESC = F * 0.22;
const SUBDY = F * 0.24;       // subscript baseline drop
const SUPDY = -F * 0.42;      // superscript baseline rise
const SUPROOM = -SUPDY, SUBROOM = SUBDY;
const PAD = 10, BARPAD = 10;
const FONT = 'ui-monospace, SFMono-Regular, Menlo, Consolas, \'Liberation Mono\', monospace';
const INK = '#0f172a', BG = '#ffffff', EDGE = '#e2e8f0';

// Visible glyph cells: code points minus combining marks (U+0300–036F).
const cells = (s) => [...s].filter((c) => {
  const u = c.codePointAt(0);
  return u < 0x300 || u > 0x36f;
}).length;
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// Parse markup into runs of {text, kind: base|sub|sup}.
function parse(s) {
  const runs = [];
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if ((c === '_' || c === '^') && i + 1 < s.length) {
      const kind = c === '_' ? 'sub' : 'sup';
      i++;
      let t;
      if (s[i] === '{') { const j = s.indexOf('}', i); t = s.slice(i + 1, j); i = j + 1; }
      else { t = s[i]; i++; }
      runs.push({ text: t, kind });
    } else {
      let j = i;
      while (j < s.length && s[j] !== '_' && s[j] !== '^') j++;
      runs.push({ text: s.slice(i, j), kind: 'base' });
      i = j;
    }
  }
  return runs;
}
const runW = (r) => cells(r.text) * (r.kind === 'base' ? CW : SUBCW);
const measure = (s) => parse(s).reduce((a, r) => a + runW(r), 0);

// Lay out a marked-up string with its baseline at (x, y). Returns SVG <text>
// elements (one per run, absolutely positioned) and the advance width.
function layout(s, x, y) {
  let cur = x;
  const out = [];
  for (const r of parse(s)) {
    if (r.text.length) {
      const ry = y + (r.kind === 'sub' ? SUBDY : r.kind === 'sup' ? SUPDY : 0);
      const fs = r.kind === 'base' ? F : F * SUB;
      out.push(`<text x="${cur.toFixed(1)}" y="${ry.toFixed(1)}" font-family="${FONT}" ` +
        `font-size="${fs.toFixed(1)}" fill="${INK}">${esc(r.text)}</text>`);
    }
    cur += runW(r);
  }
  return { svg: out.join('\n  '), width: cur - x };
}

function frame(width, height, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width.toFixed(1)}" ` +
    `height="${height.toFixed(1)}" viewBox="0 0 ${width.toFixed(1)} ${height.toFixed(1)}">\n` +
    `  <rect x="0.5" y="0.5" width="${(width - 1).toFixed(1)}" height="${(height - 1).toFixed(1)}" ` +
    `rx="7" fill="${BG}" stroke="${EDGE}"/>\n  ${body}\n</svg>\n`;
}

function row(s) {
  const baseY = PAD + SUPROOM + ASC;
  const width = measure(s) + 2 * PAD;
  const height = baseY + DESC + SUBROOM + PAD;
  return frame(width, height, layout(s, PAD, baseY).svg);
}

function frac(lhs, num, den) {
  const numBase = PAD + SUPROOM + ASC;
  const barY = numBase + DESC + SUBROOM + 4;
  const denBase = barY + 4 + SUPROOM + ASC;
  const height = denBase + DESC + SUBROOM + PAD;
  const lhsW = measure(lhs), numW = measure(num), denW = measure(den);
  const barW = Math.max(numW, denW) + 2 * BARPAD;
  const fracX = PAD + lhsW;
  const cx = fracX + barW / 2;
  const lhsBase = barY + (ASC - DESC) / 2;
  const width = fracX + barW + PAD;
  const body = [
    lhs ? layout(lhs, PAD, lhsBase).svg : '',
    layout(num, cx - numW / 2, numBase).svg,
    `<line x1="${fracX.toFixed(1)}" y1="${barY.toFixed(1)}" x2="${(fracX + barW).toFixed(1)}" y2="${barY.toFixed(1)}" stroke="${INK}" stroke-width="1.3"/>`,
    layout(den, cx - denW / 2, denBase).svg,
  ].filter(Boolean).join('\n  ');
  return frame(width, height, body);
}

// eqs: [[name, svgString], ...]; writes <dir>/<name>.svg.
function writeAll(eqs, dir) {
  for (const [name, svg] of eqs) {
    fs.writeFileSync(path.join(dir, `${name}.svg`), svg);
    console.log(`wrote ${name}.svg`);
  }
  console.log(`done: ${eqs.length} equations → ${dir}`);
}

module.exports = { row, frac, writeAll, F };
