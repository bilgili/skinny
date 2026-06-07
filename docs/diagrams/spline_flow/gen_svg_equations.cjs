// Offline equation → SVG renderer for docs/SplineFlows.md (spline-flow theory).
//
// Uses the shared layout engine in ../eqsvg.cjs (real sub/superscripts + 2-D
// fraction bars). LaTeX sources for the publication-quality MathJax path are in
// equations.json (render with ../restir/render.cjs over this topic's JSON).
// Markup: `_{}`/`^{}` for sub/sup; Unicode ok (ω θ ξ δ α π Σ Π ∘ ∝ ⇔ ‖ → −).
//
// Usage: node gen_svg_equations.cjs            (writes *.svg next to this file)

const { row, frac, writeAll } = require('../eqsvg.cjs');

const EQ = [
  // a flow draws by pushing a base sample through the forward map
  ['forward', row('x = T_θ(u),    u ~ p_u(u)')],
  // change of variables: density of a transformed variable
  ['change-of-vars', row('p_x(x) = p_u( f(x) ) · |det ∂f/∂x|,    f = T_θ^{-1}')],
  // its log form — the trainable likelihood
  ['log-likelihood', row('log p_x(x) = log p_u( f(x) ) + log|det ∂f/∂x|')],
  // a flow is a composition of bijections; log-dets add
  ['composition', row('T_θ = T_K ∘ ··· ∘ T_1,    log|det J_T| = Σ_{k=1}^{K} log|det J_{T_k}|')],
  // one coupling layer: pass half, transform the other half conditioned on it
  ['coupling', row('y_{1:d} = x_{1:d},    y_{d+1:D} = τ( x_{d+1:D} ; Θ(x_{1:d}) )')],
  // triangular Jacobian → determinant is the product of element transforms
  ['coupling-jac', row('det ∂y/∂x = Π_{i>d} τ\'( x_i ; Θ_i )')],
  // monotone rational-quadratic spline inside bin k
  ['rqs', frac('g(ξ) = y_k + ',
    'h_k [ s_k ξ^{2} + δ_k ξ(1−ξ) ]',
    's_k + (δ_{k+1} + δ_k − 2 s_k) ξ(1−ξ)')],
  ['rqs-where', row('ξ = (x − x_k) / w_k,    s_k = h_k / w_k,    h_k = y_{k+1} − y_k')],
  // spline derivative → the per-layer log-Jacobian term
  ['rqs-deriv', frac("g'(ξ) = ",
    's_k^{2} [ δ_{k+1} ξ^{2} + 2 s_k ξ(1−ξ) + δ_k (1−ξ)^{2} ]',
    '[ s_k + (δ_{k+1} + δ_k − 2 s_k) ξ(1−ξ) ]^{2}')],
  // knot parameterisation guaranteeing a valid monotone spline
  ['params', row('w = softmax(θ^w)·2B,   h = softmax(θ^h)·2B,   δ = softplus(θ^δ) > 0')],
  // maximum-likelihood training = forward-KL minimisation
  ['mle', row('θ* = argmax_θ Σ_{n=1}^{N} log p_x(x_n ; θ) = argmin_θ KL( p* ‖ p_θ )')],
  // conditional flow log-likelihood
  ['cond', row('log p_x(x | c) = log p_u( f(x ; c) ) + log|det ∂f/∂x|')],
  // Monte-Carlo estimate of the reflected-radiance integral
  ['mc', frac('⟨L_o⟩ = (1/N) Σ_{i=1}^{N} ', 'f(ω_i) L_i(ω_i) cosθ_i', 'p(ω_i)')],
  // zero-variance importance sampler matches the integrand
  ['optimal', row('Var⟨L_o⟩ = 0   ⇔   p(ω) ∝ f(ω) L_i(ω) cosθ')],
  // one-sample MIS mixture over proposals
  ['mis', row('p_mix(ω) = Σ_i α_i p_i(ω),    Σ_i α_i = 1')],
  // contribution-weighted negative log-likelihood for path guiding
  ['weighted-nll', row('L(θ) = − Σ_n w_n log q_θ(ω_n | c_n),    w_n ∝ f L_i cosθ')],
];

writeAll(EQ, __dirname);
