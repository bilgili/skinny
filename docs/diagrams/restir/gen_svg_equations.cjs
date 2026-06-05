// Offline equation → SVG renderer for docs/ReSTIR.md (ReSTIR DI).
//
// Uses the shared layout engine in ../eqsvg.cjs (real sub/superscripts + 2-D
// fraction bars). LaTeX sources for the publication-quality MathJax path are in
// equations.json (render with render.cjs). Markup: `_{}`/`^{}` for sub/sup; p̂ is
// p + combining circumflex (U+0302). Specs mirror equations.json.
//
// Usage: node gen_svg_equations.cjs            (writes *.svg next to this file)

const { row, frac, writeAll } = require('../eqsvg.cjs');

const P = 'p̂';
const EQ = [
  ['ris-weight', frac('w_{i} = ', 'p̂(x_{i})', 'p_{src}(x_{i})')],
  ['ris-W', frac('W = ', 'Σ_{i} w_{i}', 'M · p̂(y)')],
  ['target', row('p̂ = lum(f · L_{e})')],
  ['pmix', frac('p_{mix}(ω_{i}) = ',
    'M_{light}·p_{light}(ω_{i}) + M_{bsdf}·p_{bsdf}(ω_{i})·[ω_{i}→sphere|env]',
    'M_{light} + M_{bsdf}')],
  ['merge', row('w = p̂_{dst}(src.y) · src.W · src.M')],
  ['gris-ms', frac('m_{s} = ', 'M_{s} · p̂_{s}(z_{s})', 'Σ_{j} M_{j} · p̂_{j}(z_{s})')],
  ['gris-ws', row('w_{s} = m_{s} · p̂_{q}(z_{s}) · W_{s}')],
  ['gris-Wout', frac('W_{out} = ', 'Σ_{s} w_{s}', 'p̂_{q}(Y)')],
  ['biased', frac('W = ', 'Σ_{s} p̂_{q}(z_{s})·W_{s}·M_{s}', '(Σ_{s} M_{s}) · p̂(Y)')],
  ['resolve', row('direct = f(y) · L_{e}(y) · V(y) · W')],
  ['resolve-dir', row('direct_{dir} = Σ_{d} f(ω_{i,d}) · L_{e,d} · V_{d}')],
];

writeAll(EQ, __dirname);
