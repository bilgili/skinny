// Offline equation → SVG renderer for docs/PhotonMapping.md (GPU SPPM, PM-1).
//
// Uses the shared layout engine in ../eqsvg.cjs (real sub/superscripts + 2-D
// fraction bars). LaTeX sources for the publication-quality MathJax path are in
// equations.json (render with ../restir/render.cjs over this dir's
// equations.json). Markup: `_{}`/`^{}` for sub/sup; Unicode for Greek (β τ Φ γ π).
// Specs mirror equations.json.
//
// Usage: node gen_svg_equations.cjs            (writes *.svg next to this file)

const { row, frac, writeAll } = require('../eqsvg.cjs');

const EQ = [
  // Photon flux carried from a diffuse area emitter (cosine emission cancels the
  // cosine pdf): β = Le·π / p_sel.
  ['photon-beta', frac('β = ', 'L_{e} · π', 'p_{sel}')],
  // Bare BSDF density estimate deposited per photon (no cosine — the photon-map
  // estimator): φ_j += β · f_r, f_r = response / max(ω_{i,z}, 1e-4).
  ['deposit', frac('φ_{j} += β · f_{r},   f_{r} = ', 'response(ω_{o}, ω_{i})', 'max(ω_{i,z}, 1e-4)')],
  // SPPM progressive reduction (Hachisuka & Jensen 2009), γ = 2/3.
  ['update-N', row('N\' = N + γ · M')],
  ['update-r', frac('r\' = r · √', 'N\'', 'N + M')],
  ['update-tau', frac('τ\' = (τ + Φ) · (r\'/r)² = (τ + Φ) · ', 'N\'', 'N + M')],
  // Per-pass indirect radiance estimate from the accumulated flux.
  ['radiance', frac('L_{indirect} = ', 'τ', 'N_{emitted} · π · r²')],
  // The composited per-pixel sample = direct (NEE + specular-chain) + indirect.
  ['sample', row('L = L_{d} + L_{indirect}')],
  // Spatial-hash cell count = next power of two ≥ 2·W·H.
  ['numcells', row('n_{cells} = 2^{⌈log₂(2·W·H)⌉}')],
];

writeAll(EQ, __dirname);
