// Offline equation → SVG renderer for docs/SkinRendering.md (skin optics).
//
// Uses the shared layout engine in ../eqsvg.cjs (real sub/superscripts + 2-D
// fraction bars). LaTeX sources for the publication-quality MathJax path are in
// equations.json (render with ../restir/render.cjs). Markup: `_{}`/`^{}` for
// sub/sup. Equations transcribed from the shaders: melanin/hemoglobin from
// skin_bssrdf.slang::{melaninAbsorption, hemoglobinAbsorption}; Burley from
// burleyDiffusionProfile; Schlick from fresnelSchlick; HG/extinction/free-flight
// from volume_render.slang.
//
// Usage: node gen_svg_equations.cjs            (writes *.svg next to this file)

const { row, frac, writeAll } = require('../eqsvg.cjs');

const EQ = [
  ['melanin', row('σ_{a}(λ) = c_{mel} · μ_{eu}(λ)')],
  ['hemoglobin', row('σ_{a} = c_{hb} · (sat · μ_{HbO₂} + (1 − sat) · μ_{Hb})')],
  ['extinction', row('σ_{t} = σ_{a} + σ_{s}')],
  ['transmittance', row('T(d) = exp(−σ_{t} · d)')],
  ['burley', frac('R(r) = ', 'exp(−r/d) + exp(−r/(3d))', '8π · d · r')],
  ['schlick', row('F(θ) = F_{0} + (1 − F_{0})(1 − cosθ)^{5}')],
  ['fresnel0', row('F_{0} = ((1 − n) / (1 + n))^{2}')],
  ['hg-phase', frac('p(cosθ) = ', '1 − g^{2}', '4π · (1 + g^{2} − 2g·cosθ)^{3/2}')],
  ['free-flight', row('Δt = −ln(u) / σ_{maj}')],
];

writeAll(EQ, __dirname);
