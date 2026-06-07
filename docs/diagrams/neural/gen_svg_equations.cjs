// Offline equation → SVG renderer for docs/NeuralGuiding.md (SplineFlow).
//
// Uses the shared layout engine in ../eqsvg.cjs (real sub/superscripts + 2-D
// fraction bars). LaTeX sources for the publication-quality MathJax path are in
// equations.json (render with ../restir/render.cjs over this topic's JSON).
// Markup: `_{}`/`^{}` for sub/sup; Unicode ok (ω θ α β π Σ ℝ □ · → −).
//
// Usage: node gen_svg_equations.cjs            (writes *.svg next to this file)

const { row, frac, writeAll } = require('../eqsvg.cjs');

const EQ = [
  // condition encoding c = (normalised pos, normal, outgoing dir) ∈ ℝ⁹
  ['cond', row('c = ( 2(p − b_{min})/e − 1,  N,  ω_o ) ∈ ℝ^{9}')],
  // forward flow draw
  ['flow-fwd', row('z = T_θ(u ; c),    u ~ U([0,1]^{2})')],
  // one coupling layer: transform one coord, pass the other
  ['coupling', row("z'_t = RQS( z_t ; θ_L(z_c, c) ),    z'_c = z_c")],
  // monotone rational-quadratic spline, forward
  ['rqs', frac('y = y_k + ',
    'h_k [ s θ^{2} + d_k θ(1−θ) ]',
    's + (d_k + d_{k+1} − 2s) θ(1−θ)')],
  ['rqs-where', row('θ = (x − x_k) / w_k,    s = h_k / w_k')],
  // log-det Jacobian accumulates over coupling layers
  ['logdet', row('log|det ∂z/∂u| = Σ_L log| ∂y_L/∂x_L |')],
  // unit-square density from the forward log-det (base sample uniform)
  ['pdf-square', row('q_□(z) = exp( − log|det ∂z/∂u| )')],
  // square → solid-angle Jacobian
  ['pdf-omega', frac('q_ω(ω) = ', 'q_□(z)', '2π')],
  // inverse density of an arbitrary direction (for the mixture pdf)
  ['pdf-inv', row('q_ω(ω) = (1/2π) exp( log|det ∂u/∂z| ),   z = M^{-1}(ω)')],
  // one-sample MIS mixture pdf over the active proposals
  ['mix-pdf', row('p_mix(ω) = α_b p_bsdf(ω) + α_e p_env(ω) + α_n q_ω(ω)')],
  // unbiased throughput update at the bounce
  ['estimator', row('β ← β · f(ω) cosθ / p_mix(ω)')],
  // per-vertex training weight (backward attribution)
  ['contrib', frac('w_k = ', 'L_{final} − L_k', 'β_{in,k}')],
  // forward-KL / weighted-NLL training objective
  ['loss', row('L(θ) = − E_{(c,ω,w)~D}[ w · log q_θ(ω | c) ]')],
];

writeAll(EQ, __dirname);
