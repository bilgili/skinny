// Offline equation → SVG renderer for docs/ReSTIR.md.
//
// The repo's GitLab does not render KaTeX/`$$` math reliably, so display
// equations ship as committed SVG images. The publication-quality path is
// `render.cjs` (MathJax 3, true LaTeX → SVG) using `equations.json`; this script
// is the dependency-free fallback used when no Node/Python math toolchain or
// network is available. It lays out monospace math with real 2-D fraction bars
// on a white card, so each SVG renders identically as an <img> in any Markdown
// viewer and in GitLab's light or dark theme.
//
// Usage: node gen_svg_equations.cjs            (writes *.svg next to this file)

const fs = require('fs');
const path = require('path');

const F = 18;                 // font size (px)
const CW = F * 0.6;           // monospace advance (Menlo/SF Mono ≈ 0.6em)
const ASC = F * 0.80, DESC = F * 0.22;
const PAD = 10, BARPAD = 10;
const FONT = 'ui-monospace, SFMono-Regular, Menlo, Consolas, \'Liberation Mono\', monospace';
const INK = '#0f172a', BG = '#ffffff', EDGE = '#e2e8f0';

// Visible width in glyph cells: code points minus combining marks (U+0300–036F).
const cells = (s) => [...s].filter((c) => {
  const u = c.codePointAt(0);
  return u < 0x300 || u > 0x36f;
}).length;
const w = (s) => cells(s) * CW;
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const txt = (s, x, y, anchor = 'start') =>
  `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}" ` +
  `font-family="${FONT}" font-size="${F}" fill="${INK}">${esc(s)}</text>`;

function frame(width, height, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width.toFixed(1)}" ` +
    `height="${height.toFixed(1)}" viewBox="0 0 ${width.toFixed(1)} ${height.toFixed(1)}">\n` +
    `  <rect x="0.5" y="0.5" width="${(width - 1).toFixed(1)}" height="${(height - 1).toFixed(1)}" ` +
    `rx="7" fill="${BG}" stroke="${EDGE}"/>\n  ${body}\n</svg>\n`;
}

function row(s) {
  const width = w(s) + 2 * PAD;
  const height = F + 2 * PAD;
  return frame(width, height, txt(s, PAD, PAD + ASC));
}

function frac(lhs, num, den) {
  const lhsW = w(lhs);
  const numW = w(num), denW = w(den);
  const barW = Math.max(numW, denW) + 2 * BARPAD;
  const fracX = PAD + lhsW;
  const cx = fracX + barW / 2;
  const numBase = PAD + ASC;                 // numerator sits in the top band
  const barY = PAD + F + 5;                   // fraction bar
  const denBase = barY + 5 + ASC;             // denominator below the bar
  const lhsBase = barY + (ASC - DESC) / 2;    // lhs vertically centred on the bar
  const width = fracX + barW + PAD;
  const height = denBase + DESC + PAD;
  const body = [
    lhs ? txt(lhs, PAD, lhsBase) : '',
    txt(num, cx, numBase, 'middle'),
    `<line x1="${fracX.toFixed(1)}" y1="${barY}" x2="${(fracX + barW).toFixed(1)}" y2="${barY}" stroke="${INK}" stroke-width="1.3"/>`,
    txt(den, cx, denBase, 'middle'),
  ].filter(Boolean).join('\n  ');
  return frame(width, height, body);
}

// Display strings mirror equations.json (LaTeX). Subscripts use underscore
// notation; p̂ is p + combining circumflex (U+0302).
const P = 'p̂';        // p-hat
const W_ = 'ω';
const EQ = [
  ['ris-weight', frac(`w_i = `, `${P}(x_i)`, `p_src(x_i)`)],
  ['ris-W', frac(`W = `, `Σ_i w_i`, `M · ${P}(y)`)],
  ['target', row(`${P} = lum(f · L_e)`)],
  ['pmix', frac(`p_mix(${W_}_i) = `,
    `M_light·p_light(${W_}_i) + M_bsdf·p_bsdf(${W_}_i)·[${W_}_i→sphere|env]`,
    `M_light + M_bsdf`)],
  ['merge', row(`w = ${P}_dst(src.y) · src.W · src.M`)],
  ['gris-ms', frac(`m_s = `, `M_s · ${P}_s(z_s)`, `Σ_j M_j · ${P}_j(z_s)`)],
  ['gris-ws', row(`w_s = m_s · ${P}_q(z_s) · W_s`)],
  ['gris-Wout', frac(`W_out = `, `Σ_s w_s`, `${P}_q(Y)`)],
  ['biased', frac(`W = `, `Σ_s ${P}_q(z_s)·W_s·M_s`, `(Σ_s M_s) · ${P}(Y)`)],
  ['resolve', row(`direct = f(y) · L_e(y) · V(y) · W`)],
  ['resolve-dir', row(`direct_dir = Σ_d f(${W_}_i,d) · L_e,d · V_d`)],
];

const outDir = __dirname;
for (const [name, svg] of EQ) {
  fs.writeFileSync(path.join(outDir, `${name}.svg`), svg);
  console.log(`wrote ${name}.svg`);
}
console.log(`done: ${EQ.length} equations → ${outDir}`);
