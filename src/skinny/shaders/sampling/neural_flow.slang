// Conditional rational-quadratic neural spline flow — GPU inference.
//
// Faithful port of the verified PyTorch prototype (spline_flow/train.py:
// ConditionalSplineFlow2D + SplineCoupling + rqs_forward/rqs_inverse). The flow is
// an invertible map T_theta : [0,1]^2 -> [0,1]^2 conditioned on `cond`; it provides
//   sampleNeural(cond, u)  : forward  u -> wi  + exact solid-angle pdf
//   pdfNeural(cond, wi)    : inverse  wi -> pdf (for the MIS mixture)
// Weights are flat buffers (host bakes them, see the spline_flow exporter); the
// architecture is fixed at compile time and must match the bake.
//
// Conventions match the prototype: hemisphere is y-up; base sample u in [0,1]^2;
// square->solid-angle Jacobian 2*pi, so q_omega = q_square / (2*pi).

// ---- architecture (MUST match the baked weights) ----
// NF_LAYERS / NF_BINS / NF_HIDDEN are build-time configurable (study change
// neural-precision-size-study): `#ifndef`-guarded so `slangc -D NF_HIDDEN=48`
// (etc.) selects an off-default size. The host threads the same `-D` into every
// module that imports this file (the wavefront pre-pass, the inline inverse, the
// record entry) and folds the dims into the pipeline cache key; the NFW1 loader
// asserts the baked `(layers,bins,hidden,cond)` matches the built dims. With no
// override the defaults below reproduce the shipped net byte-for-byte.
#ifndef NF_LAYERS
#define NF_LAYERS 6     // coupling layers
#endif
#ifndef NF_BINS
#define NF_BINS 24      // rational-quadratic spline bins (K)
#endif
#ifndef NF_HIDDEN
#define NF_HIDDEN 96    // MLP hidden width
#endif

// ---- inference precision (mixed fp16; study change neural-precision-size-study) ----
// NF_WT = weight *storage* type (the `StructuredBuffer` element bound at 33/34);
// NF_CT = the MLP GEMM *accumulate*/activation type. Three modes via `-D`:
//   fp32          (default)      : NF_WT=float NF_CT=float — byte-identical to ship
//   fp16-storage  -D NF_WT=half               : half weights, float GEMM (½ bytes)
//   fp16-compute  -D NF_WT=half -D NF_CT=half : half weights + half GEMM
// The RQ-spline math (softmax/cumsum/exp/log + the analytic inverse-quadratic
// solve) and the returned solid-angle pdf stay `float` in EVERY mode — fp16 there
// is catastrophic-cancellation prone. Conversions at the GEMM boundary are
// implicit, so the fp32 token stream is unchanged (no redundant casts).
#ifndef NF_WT
#define NF_WT float
#endif
#ifndef NF_CT
#define NF_CT float
#endif

// ---- fp8 weight storage (e4m3; study change neural-trainer-backends) ----
// A fourth mode, selected by `-D NF_FP8=1 -D NF_WT=uint`: weights are 8-bit
// e4m3 packed 4-per-`uint`, decoded to `float` in the scalar GEMM. The decode
// is plain integer math + exp2, so it needs NO device feature (portable across
// Vulkan / Metal / MoltenVK) — fp8 *compute* would need a cooperative-matrix
// rewrite and is out of scope. The buffers stay typed `StructuredBuffer<NF_WT>`
// (= `StructuredBuffer<uint>` here), so no signature changes; only `nf_fetch`
// below branches. NF_CT stays `float`.
#ifdef NF_FP8
// e4m3 (OCP E4M3FN): 1 sign, 4 exp (bias 7), 3 mantissa; no inf, max |x| = 448.
// Bit-for-bit identical to the host encoder (neural_weights.f32_to_e4m3).
float nf_decode_e4m3(uint bits)
{
    uint s = (bits >> 7) & 0x1u;
    uint e = (bits >> 3) & 0xFu;
    uint m = bits & 0x7u;
    float val = (e == 0u) ? (float(m) * (1.0f / 512.0f))            // 2^-9 * m
                          : ((1.0f + float(m) * 0.125f) * exp2(float(int(e) - 7)));
    return (s != 0u) ? -val : val;
}
float nf_fetch(StructuredBuffer<NF_WT> buf, uint idx)
{
    uint word = buf[idx >> 2];
    uint by   = (word >> ((idx & 3u) * 8u)) & 0xFFu;
    return nf_decode_e4m3(by);
}
#else
// fp32 / fp16 storage: a plain element fetch. In fp32 (NF_WT=float) this is
// `buf[idx]`; in the fp16 modes it returns the half element (the compute type
// conversion at the multiply is unchanged), so the emitted SPIR-V matches the
// shipped kernels exactly.
NF_WT nf_fetch(StructuredBuffer<NF_WT> buf, uint idx) { return buf[idx]; }
#endif

static const int   NF_DIM    = 2;     // flow dimensionality (a direction = 2 dof)
static const int   NF_COND   = 9;     // condition dim (renderer: pos3 + N3 + wo3)
static const float NF_PI     = 3.14159265358979323846f;
static const float NF_LOG2PI = 1.83787706640934548356f;   // log(2*pi)
static const float NF_EPS    = 1e-8f;
static const float NF_MINBIN = 1e-4f;
static const float NF_ZEPS   = 1e-5f;     // chart-inverse boundary margin (== spline_flow _Z_EPS)

// ---- coupling warp (axis; change neural-nis-baseline) ----
// NF_COUPLING selects the per-coupling 1-D warp: the shipped rational-quadratic
// spline (0, default) or NIS's piecewise-quadratic warp (1, Müller et al. 2019).
// Only the warp + its parameter count change; the conditioner MLP, the chart, the
// constant-2π Jacobian and the coupling topology are identical. NF_COUPLING=0 keeps
// the original code paths verbatim (#if-guarded) → byte-identical to the shipped net.
#ifndef NF_COUPLING
#define NF_COUPLING 0   // 0 = rqs (rational-quadratic spline), 1 = nis-pq (piecewise-quadratic)
#endif

#if NF_COUPLING == 1
static const int NF_PARAMS = 2 * NF_BINS + 1;   // widths(K) + vertices(K+1)  [NIS-PQ]
#else
static const int NF_PARAMS = 3 * NF_BINS + 1;   // widths(K)+heights(K)+derivs(K+1)  [RQS]
#endif

// ---- conditioner positional encoding (axis 2; change renderer-conditioner-encoding) ----
// NF_ENCODING selects the NeRF-γ feature map applied to the condition BEFORE the
// conditioner MLP. It is a side input to the conditioner only — NEVER the u→z
// measure transform — so |J|, NF_LOG2PI and the whole pdf path are unchanged
// (axis 2 is Jacobian-free by construction). Build dim like NF_CHART / NF_HIDDEN:
//   E0 (0, default) raw passthrough — byte-identical to the pre-encoding net
//   E1 (1)          per-scalar γ(s, NF_L_POS); path regime ⇒ every condition
//                   scalar is banded (matches spline_flow make_cond_encoding "path")
//   E3 (3)          E1 + the full raw condition appended (the "encoded + raw
//                   identity" variant, include_raw=True)
// Byte-for-byte with spline_flow fourier_gamma / make_cond_encoding(regime="path"):
// per scalar γ(s) = (sin(2⁰πs),cos(2⁰πs),…,sin(2^{L-1}πs),cos(2^{L-1}πs)), groups
// emitted in condition order, E3's raw tail last. NF_L_POS == spline_flow L_pos (10).
#ifndef NF_ENCODING
#define NF_ENCODING 0
#endif
#ifndef NF_L_POS
#define NF_L_POS 10     // position-group Fourier bands (spline_flow L_pos default)
#endif

#if NF_ENCODING == 0
static const int NF_MLP_ENC = NF_COND;                          // raw passthrough
#elif NF_ENCODING == 1
static const int NF_MLP_ENC = NF_COND * (2 * NF_L_POS);         // γ bands, every scalar
#elif NF_ENCODING == 3
static const int NF_MLP_ENC = NF_COND * (2 * NF_L_POS) + NF_COND;  // E1 + raw identity tail
#else
#error "NF_ENCODING must be 0 (E0), 1 (E1) or 3 (E3)"
#endif

static const int NF_MLP_IN = 1 + NF_MLP_ENC;    // pass-through coord + encoded condition

// Conditioner positional encoding: raw cond[NF_COND] → encoded[NF_MLP_ENC]. A
// transliteration of spline_flow/train.py fourier_gamma + FourierEncoding for the
// "path" regime — byte-for-byte with the trainer (the encoded condition the net
// was fit on). E0 is the identity (compiles to a plain copy, unused at the call
// site). NEVER feeds the u→z transform, so it cannot change the Jacobian.
void nf_encode(float cond[NF_COND], out float enc[NF_MLP_ENC])
{
#if NF_ENCODING == 0
    for (int i = 0; i < NF_COND; ++i) enc[i] = cond[i];
#else
    int o = 0;
    for (int i = 0; i < NF_COND; ++i)
    {
        // γ(s, L) = (sin(2⁰πs),cos(2⁰πs),…,sin(2^{L-1}πs),cos(2^{L-1}πs)) ∈ ℝ^{2L}
        for (int l = 0; l < NF_L_POS; ++l)
        {
            float ang = cond[i] * (exp2((float)l) * NF_PI);   // 2^l · π · s
            enc[o++] = sin(ang);
            enc[o++] = cos(ang);
        }
    }
  #if NF_ENCODING == 3
    for (int i = 0; i < NF_COND; ++i) enc[o++] = cond[i];      // raw identity tail
  #endif
#endif
}

// One Linear layer's location in the flat weight/bias buffers. Row-major weights
// laid out [outDim, inDim]. There are 3*NF_LAYERS of these (3 Linear per coupling).
struct NfLayerHeader
{
    uint weightOffset;
    uint biasOffset;
    uint inDim;
    uint outDim;
};

float nf_silu(float x) { return x / (1.0f + exp(-x)); }
float nf_softplus(float x) { return log(1.0f + exp(x)); }

// Evaluate one Linear layer: out[o] = bias[o] + sum_i W[o,i]*in[i], optional SiLU.
// `n_in`/`n_out` are the runtime dims from the header (<= the static maxima).
// Weights/biases are NF_WT (fp16 storage in the fp16 modes); the accumulate +
// activations are NF_CT (fp16 in fp16-compute). The half↔float conversions at
// the GEMM boundary are implicit; in fp32 (NF_WT=NF_CT=float) every type below
// is `float`, so the emitted SPIR-V is identical to the shipped kernel. The SiLU
// nonlinearity is evaluated in `float` (cheap, fp16-exp is lossy) then stored as
// NF_CT — the *compute* this mode trades to fp16 is the O(NF_HIDDEN²) GEMM.
void nf_linear(
    StructuredBuffer<NF_WT> W, StructuredBuffer<NF_WT> B, NfLayerHeader h,
    NF_CT inp[NF_HIDDEN], int n_in, bool act, out NF_CT outp[NF_HIDDEN])
{
    for (int o = 0; o < (int)h.outDim; ++o)
    {
        NF_CT acc = nf_fetch(B, h.biasOffset + (uint)o);
        uint row = h.weightOffset + (uint)o * h.inDim;
        for (int i = 0; i < n_in; ++i)
            acc += nf_fetch(W, row + (uint)i) * inp[i];
        outp[o] = act ? nf_silu(acc) : acc;
    }
}

// First Linear of the conditioner: input is NF_MLP_IN wide — the encoded
// condition (E1/E3) can exceed NF_HIDDEN, so the input array cannot share
// nf_linear's NF_HIDDEN-sized signature. Identical math to nf_linear; only the
// input array extent differs. For E0, NF_MLP_IN == 1 + NF_COND and this is
// numerically identical to the pre-encoding first layer.
void nf_linear_in(
    StructuredBuffer<NF_WT> W, StructuredBuffer<NF_WT> B, NfLayerHeader h,
    NF_CT inp[NF_MLP_IN], int n_in, bool act, out NF_CT outp[NF_HIDDEN])
{
    for (int o = 0; o < (int)h.outDim; ++o)
    {
        NF_CT acc = nf_fetch(B, h.biasOffset + (uint)o);
        uint row = h.weightOffset + (uint)o * h.inDim;
        for (int i = 0; i < n_in; ++i)
            acc += nf_fetch(W, row + (uint)i) * inp[i];
        outp[o] = act ? nf_silu(acc) : acc;
    }
}

// 3-layer conditioner MLP for one coupling layer: (pass-through coord, encoded
// cond) -> NF_PARAMS spline parameters. `base3` is the index of this coupling's
// first header. `cond` is the ALREADY-ENCODED condition (nf_encode applied at the
// proposal call site, once per flow eval) of width NF_MLP_ENC — for E0 this is
// the raw condition (NF_MLP_ENC == NF_COND), so the math is unchanged.
void nf_mlp(
    StructuredBuffer<NF_WT> W, StructuredBuffer<NF_WT> B, StructuredBuffer<NfLayerHeader> H,
    int base3, float xcond, float cond[NF_MLP_ENC], out float params[NF_PARAMS])
{
    NF_CT a[NF_MLP_IN];
    a[0] = xcond;
    for (int i = 0; i < NF_MLP_ENC; ++i) a[1 + i] = cond[i];

    NF_CT b[NF_HIDDEN];
    nf_linear_in(W, B, H[base3 + 0], a, NF_MLP_IN, true, b);   // in -> hidden, SiLU
    NF_CT h2[NF_HIDDEN];
    nf_linear(W, B, H[base3 + 1], b, NF_HIDDEN, true, h2);     // hidden -> hidden, SiLU

    // hidden -> NF_PARAMS, linear. params stays `float` — the spline decode/eval
    // below is full-precision.
    NF_CT wide[NF_HIDDEN];
    nf_linear(W, B, H[base3 + 2], h2, NF_HIDDEN, false, wide);
    for (int p = 0; p < NF_PARAMS; ++p) params[p] = wide[p];
}

// Decode raw MLP params -> normalized spline knots (matches SplineCoupling._params).
void nf_decode(
    float params[NF_PARAMS],
    out float widths[NF_BINS], out float heights[NF_BINS], out float derivs[NF_BINS + 1])
{
    // softmax(widths), softmax(heights); + min_bin floor; renormalize.
    float mw = params[0]; float mh = params[NF_BINS];
    for (int k = 1; k < NF_BINS; ++k)
    {
        mw = max(mw, params[k]);
        mh = max(mh, params[NF_BINS + k]);
    }
    float sw = 0.0f; float sh = 0.0f;
    for (int k = 0; k < NF_BINS; ++k)
    {
        widths[k]  = exp(params[k] - mw);            sw += widths[k];
        heights[k] = exp(params[NF_BINS + k] - mh);  sh += heights[k];
    }
    float tw = 0.0f; float th = 0.0f;
    for (int k = 0; k < NF_BINS; ++k)
    {
        widths[k]  = NF_MINBIN + widths[k] / sw;   tw += widths[k];
        heights[k] = NF_MINBIN + heights[k] / sh;  th += heights[k];
    }
    for (int k = 0; k < NF_BINS; ++k) { widths[k] /= tw; heights[k] /= th; }
    for (int k = 0; k < NF_BINS + 1; ++k) derivs[k] = nf_softplus(params[2 * NF_BINS + k]) + 1e-3f;
}

// Monotone rational-quadratic spline, forward. Returns y; outputs log|dy/dx|.
// DOC:rqs#sig start
float nf_rqs_fwd(float x, float widths[NF_BINS], float heights[NF_BINS],
                 float derivs[NF_BINS + 1], out float logdet)
// DOC:rqs#sig end
{
    x = clamp(x, 0.0f, 1.0f);
    float xk = 0.0f; int idx = 0;
    // locate bin: largest knot <= x
    float cum = 0.0f;
    for (int k = 0; k < NF_BINS; ++k) { if (x >= cum) idx = k; cum += widths[k]; }
    float x0 = 0.0f; for (int k = 0; k < idx; ++k) x0 += widths[k];
    float y0 = 0.0f; for (int k = 0; k < idx; ++k) y0 += heights[k];
    float w = max(widths[idx], NF_EPS);
    float hgt = max(heights[idx], NF_EPS);
    float d0 = derivs[idx]; float d1 = derivs[idx + 1];
    float delta = hgt / w;
    float theta = (x - x0) / w;
    float t1 = theta * (1.0f - theta);
    // DOC:rqs#core start
    float num = hgt * (delta * theta * theta + d0 * t1);
    float den = delta + (d0 + d1 - 2.0f * delta) * t1;
    den = max(den, NF_EPS);
    float y = y0 + num / den;
    // DOC:rqs#core end
    // DOC:logdet#deriv start
    float dn = delta * delta * (d1 * theta * theta + 2.0f * delta * t1 + d0 * (1.0f - theta) * (1.0f - theta));
    float dydx = dn / (den * den);
    logdet = log(max(dydx, NF_EPS));
    // DOC:logdet#deriv end
    return clamp(y, 0.0f, 1.0f);
}

// Monotone rational-quadratic spline, inverse. Returns x; outputs log|dx/dy| = -log|dy/dx|.
float nf_rqs_inv(float y, float widths[NF_BINS], float heights[NF_BINS],
                 float derivs[NF_BINS + 1], out float logdet)
{
    y = clamp(y, 0.0f, 1.0f);
    int idx = 0; float cum = 0.0f;
    for (int k = 0; k < NF_BINS; ++k) { if (y >= cum) idx = k; cum += heights[k]; }
    float x0 = 0.0f; for (int k = 0; k < idx; ++k) x0 += widths[k];
    float y0 = 0.0f; for (int k = 0; k < idx; ++k) y0 += heights[k];
    float w = max(widths[idx], NF_EPS);
    float hgt = max(heights[idx], NF_EPS);
    float d0 = derivs[idx]; float d1 = derivs[idx + 1];
    float delta = hgt / w;
    float z = (y - y0) / hgt;
    float a = (d0 + d1 - 2.0f * delta);
    float A = z * a + delta - d0;
    float B = d0 - z * a;
    float C = -delta * z;
    float disc = max(B * B - 4.0f * A * C, 0.0f);
    float denom = min(-B - sqrt(disc), -NF_EPS);
    float theta = clamp((2.0f * C) / denom, 0.0f, 1.0f);
    float x = x0 + theta * w;
    float fwdlog;
    nf_rqs_fwd(x, widths, heights, derivs, fwdlog);
    logdet = -fwdlog;
    return clamp(x, 0.0f, 1.0f);
}

// ---- NIS piecewise-quadratic coupling (Müller et al. 2019) ----
// Faithful port of spline_flow nis-pq: _decode_pq + pq_forward + pq_inverse.
// Compiled only under -D NF_COUPLING=1. The pdf is piecewise-LINEAR over K bins of
// learned width (vertex pdf values at the K+1 boundaries) → the cdf is quadratic
// per bin; the inverse is a per-bin quadratic solve.
#if NF_COUPLING == 1
// Decode raw MLP params[2K+1] -> K bin widths (softmax + floor + renormalize, same
// as nf_decode's widths) and K+1 vertex pdf values (softplus, normalized so the
// trapezoidal integral Σ W_k (V_k+V_{k+1})/2 = 1). Matches SplineCoupling._decode_pq.
void nf_decode_pq(
    float params[NF_PARAMS],
    out float widths[NF_BINS], out float vertices[NF_BINS + 1])
{
    float mw = params[0];
    for (int k = 1; k < NF_BINS; ++k) mw = max(mw, params[k]);
    float sw = 0.0f;
    for (int k = 0; k < NF_BINS; ++k) { widths[k] = exp(params[k] - mw); sw += widths[k]; }
    float tw = 0.0f;
    for (int k = 0; k < NF_BINS; ++k) { widths[k] = NF_MINBIN + widths[k] / sw; tw += widths[k]; }
    for (int k = 0; k < NF_BINS; ++k) widths[k] /= tw;

    float v[NF_BINS + 1];
    for (int i = 0; i < NF_BINS + 1; ++i) v[i] = nf_softplus(params[NF_BINS + i]) + 1e-3f;
    float area = 0.0f;
    for (int k = 0; k < NF_BINS; ++k) area += widths[k] * 0.5f * (v[k] + v[k + 1]);
    area = max(area, 1e-8f);
    for (int i = 0; i < NF_BINS + 1; ++i) vertices[i] = v[i] / area;
}

// Piecewise-quadratic forward x -> y; outputs log|dy/dx|. (port of pq_forward)
float nf_pq_fwd(float x, float widths[NF_BINS], float vertices[NF_BINS + 1], out float logdet)
{
    x = clamp(x, 0.0f, 1.0f);
    float xk[NF_BINS + 1]; xk[0] = 0.0f;
    float yk[NF_BINS + 1]; yk[0] = 0.0f;
    for (int k = 0; k < NF_BINS; ++k)
    {
        xk[k + 1] = xk[k] + widths[k];
        yk[k + 1] = yk[k] + widths[k] * 0.5f * (vertices[k] + vertices[k + 1]);
    }
    int idx = 0;
    for (int k = 0; k <= NF_BINS; ++k) if (x >= xk[k]) idx = k;
    idx = clamp(idx, 0, NF_BINS - 1);
    float x0 = xk[idx];
    float w  = max(widths[idx], NF_EPS);
    float v0 = vertices[idx]; float v1 = vertices[idx + 1];
    float t  = clamp((x - x0) / w, 0.0f, 1.0f);
    float y  = yk[idx] + w * (v0 * t + 0.5f * (v1 - v0) * t * t);
    float pdf = max(v0 + t * (v1 - v0), NF_EPS);
    logdet = log(pdf);
    return clamp(y, 0.0f, 1.0f);
}

// Piecewise-quadratic inverse y -> x; outputs log|dx/dy| = -log|dy/dx|. (port of pq_inverse)
float nf_pq_inv(float y, float widths[NF_BINS], float vertices[NF_BINS + 1], out float logdet)
{
    y = clamp(y, 0.0f, 1.0f);
    float xk[NF_BINS + 1]; xk[0] = 0.0f;
    float yk[NF_BINS + 1]; yk[0] = 0.0f;
    for (int k = 0; k < NF_BINS; ++k)
    {
        xk[k + 1] = xk[k] + widths[k];
        yk[k + 1] = yk[k] + widths[k] * 0.5f * (vertices[k] + vertices[k + 1]);
    }
    int idx = 0;
    for (int k = 0; k <= NF_BINS; ++k) if (y >= yk[k]) idx = k;
    idx = clamp(idx, 0, NF_BINS - 1);
    float x0 = xk[idx];
    float w  = max(widths[idx], NF_EPS);
    float v0 = vertices[idx]; float v1 = vertices[idx + 1];
    float a = w * 0.5f * (v1 - v0);
    float b = w * v0;
    float c = -(y - yk[idx]);
    float disc = max(b * b - 4.0f * a * c, 0.0f);
    float t_quad = (2.0f * c) / min(-b - sqrt(disc), -NF_EPS);
    float t_lin  = -c / max(b, NF_EPS);
    float t = clamp((abs(a) < NF_EPS) ? t_lin : t_quad, 0.0f, 1.0f);
    float pdf = max(v0 + t * (v1 - v0), NF_EPS);
    logdet = -log(pdf);
    return clamp(x0 + t * w, 0.0f, 1.0f);
}
#endif

// ---- hemisphere chart (axis; change renderer-chart-selection) ----
// NF_CHART selects the square<->direction map sandwiched after the couplings:
//   V1 (1, default) Lambert azimuthal equal-area — byte-identical to the shipped
//                   net: the default build emits no -D NF_CHART and the V1 path +
//                   sampleNeural/pdfNeural signatures below are unchanged.
//   V0 (0)          cylindrical equal-area (φ=2πu, cosθ=v); |J| = 2π.
//   V5 (5)          equirectangular (φ=2πu, θ=(π/2)v); NON-equal-area, |J| = π²·sinθ
//                   (per-sample → nf_chart_logjac, not the constant NF_LOG2PI).
// V2 (2) is reserved but NOT implemented: it needs the outgoing dir in flow-local
// space (the .nrec schema lacks the tangent basis), so it cannot be trained to
// match. V0/V1/V5 are z-only maps → no `wo` is threaded and the V1 build is
// byte-identical. Transliterations of spline_flow/train.py _ChartV0/_ChartV1/_ChartV5.
#define NF_CHART_V0 0
#define NF_CHART_V1 1
#define NF_CHART_V2 2
#define NF_CHART_V5 5
#ifndef NF_CHART
#define NF_CHART NF_CHART_V1
#endif

// Square <-> hemisphere (y-up): Lambert azimuthal equal-area chart (V1), a
// transliteration of spline_flow/train.py `_ChartV1` + `_square_to_disk` /
// `_disk_to_square`. Shirley concentric square->disk (seam-free, area-preserving)
// followed by the Lambert lift `cosθ = 1 − r²` (pole at the disk centre). The chart
// is equal-area, so the square->solid-angle Jacobian is the SAME constant 2*pi as the
// retired cylindrical map — `NF_LOG2PI` and the pdf path are unchanged.

// Shirley concentric square [0,1]^2 -> unit disk (positive-radius, 4-wedge form).
float2 nf_concentric_square_to_disk(float2 z)
{
    float u = clamp(z.x, 0.0f, 1.0f);
    float v = clamp(z.y, 0.0f, 1.0f);
    float a = 2.0f * u - 1.0f;
    float b = 2.0f * v - 1.0f;
    bool a_dom = abs(a) >= abs(b);
    float r = a_dom ? abs(a) : abs(b);
    float a_s = (a == 0.0f) ? 1.0f : a;            // safe divisors (centre: r=0, phi unused)
    float b_s = (b == 0.0f) ? 1.0f : b;
    float q = NF_PI / 4.0f;
    bool right = a_dom && (a >= 0.0f);
    bool left  = a_dom && (a <  0.0f);
    bool top   = (!a_dom) && (b >= 0.0f);
    float phi = 1.5f * NF_PI + q * (a / (-b_s));   // bottom wedge (default)
    if (right)     phi = q * (b / a_s);
    else if (left) phi = NF_PI - q * (b / (-a_s));
    else if (top)  phi = 0.5f * NF_PI - q * (a / b_s);
    return float2(r * cos(phi), r * sin(phi));
}

// Inverse of nf_concentric_square_to_disk. Unit disk -> z in [0,1]^2, clamped off the
// exact boundary by NF_ZEPS so the RQ spline is evaluated on its open interior.
float2 nf_disk_to_square(float2 c)
{
    float r = sqrt(c.x * c.x + c.y * c.y);
    float phi = atan2(c.y, c.x);                   // (-pi, pi]
    if (phi < -NF_PI / 4.0f) phi += 2.0f * NF_PI;  // [-pi/4, 7pi/4)
    float fourpi = 4.0f / NF_PI;
    bool w0 = phi < NF_PI / 4.0f;
    bool w1 = (phi >= NF_PI / 4.0f) && (phi < 3.0f * NF_PI / 4.0f);
    bool w2 = (phi >= 3.0f * NF_PI / 4.0f) && (phi < 5.0f * NF_PI / 4.0f);
    float a; float b;
    if (w0)      { a = r;                                 b = phi * r * fourpi; }
    else if (w1) { a = (0.5f * NF_PI - phi) * r * fourpi; b = r; }
    else if (w2) { a = -r;                                b = (NF_PI - phi) * r * fourpi; }
    else         { a = (phi - 1.5f * NF_PI) * r * fourpi; b = -r; }
    float u = (a + 1.0f) * 0.5f;
    float v = (b + 1.0f) * 0.5f;
    return float2(clamp(u, NF_ZEPS, 1.0f - NF_ZEPS),
                  clamp(v, NF_ZEPS, 1.0f - NF_ZEPS));
}

// DOC:pdf-omega#map start
float3 nf_square_to_hemi(float2 z)
{
    float2 c = nf_concentric_square_to_disk(z);    // unit disk, radius r=|c|
    float r2 = clamp(c.x * c.x + c.y * c.y, 0.0f, 1.0f);
    float cos_t = 1.0f - r2;                        // Lambert lift: cosθ = 1 − r²
    float sin_t = sqrt(max(1.0f - cos_t * cos_t, 0.0f));
    float r = sqrt(r2);
    float cphi = (r > NF_EPS) ? c.x / max(r, NF_EPS) : 1.0f;   // pole at r→0: ω = (0,1,0)
    float sphi = (r > NF_EPS) ? c.y / max(r, NF_EPS) : 0.0f;
    return float3(sin_t * cphi, cos_t, sin_t * sphi);
}
// DOC:pdf-omega#map end

float2 nf_hemi_to_square(float3 wld)
{
    float3 w = normalize(wld);
    float cos_t = clamp(w.y, -1.0f, 1.0f);
    float r = sqrt(max(1.0f - cos_t, 0.0f));        // r² = 1 − cosθ
    float sin_t = sqrt(max(1.0f - cos_t * cos_t, 0.0f));
    float cphi = (sin_t > NF_EPS) ? w.x / max(sin_t, NF_EPS) : 1.0f;
    float sphi = (sin_t > NF_EPS) ? w.z / max(sin_t, NF_EPS) : 0.0f;
    return nf_disk_to_square(float2(r * cphi, r * sphi));
}

// ---- chart dispatch (renderer-chart-selection) ----
// nf_chart_square_to_dir / nf_chart_dir_to_square / nf_chart_logjac select the
// active chart by NF_CHART. V1 (default) keeps using nf_square_to_hemi /
// nf_hemi_to_square + the constant NF_LOG2PI directly in sampleNeural/pdfNeural
// (no wrapper, no SPIR-V change), so these symbols are only defined for V0/V5.
#if NF_CHART == NF_CHART_V0
// V0 cylindrical equal-area: φ=2πu, cosθ=v. Transliterates spline_flow
// square_to_hemisphere / hemisphere_to_square. The inverse does NOT clamp off the
// [0,1] boundary (matches spline_flow hemisphere_to_square exactly). |J| = 2π.
float3 nf_chart_square_to_dir(float2 z)
{
    float u = clamp(z.x, 0.0f, 1.0f);
    float v = clamp(z.y, 0.0f, 1.0f);
    float phi = 2.0f * NF_PI * u;
    float cos_t = v;
    float sin_t = sqrt(max(1.0f - cos_t * cos_t, 0.0f));
    return float3(sin_t * cos(phi), cos_t, sin_t * sin(phi));
}
float2 nf_chart_dir_to_square(float3 wld)
{
    float3 w = normalize(wld);
    float phi = atan2(w.z, w.x);
    if (phi < 0.0f) phi += 2.0f * NF_PI;
    return float2(phi / (2.0f * NF_PI), clamp(w.y, 0.0f, 1.0f));
}
float nf_chart_logjac(float2 z) { return NF_LOG2PI; }

#elif NF_CHART == NF_CHART_V5
// V5 spherical equirectangular: φ=2πu, θ=(π/2)v. NON-equal-area: |J| = π²·sinθ, so
// q_ω = q_□ / (π²·sinθ). Transliterates spline_flow _ChartV5.
static const float NF_LOGPI2 = 2.28946497f;        // log(π²)
float3 nf_chart_square_to_dir(float2 z)
{
    float u = clamp(z.x, 0.0f, 1.0f);
    float v = clamp(z.y, 0.0f, 1.0f);
    float phi = 2.0f * NF_PI * u;
    float theta = 0.5f * NF_PI * v;
    float sin_t = sin(theta);
    float cos_t = cos(theta);
    return float3(sin_t * cos(phi), cos_t, sin_t * sin(phi));
}
float2 nf_chart_dir_to_square(float3 wld)
{
    float3 w = normalize(wld);
    float theta = acos(clamp(w.y, -1.0f, 1.0f));
    float phi = atan2(w.z, w.x);
    if (phi < 0.0f) phi += 2.0f * NF_PI;
    float u = phi / (2.0f * NF_PI);
    float v = theta / (0.5f * NF_PI);
    return float2(clamp(u, NF_ZEPS, 1.0f - NF_ZEPS),
                  clamp(v, NF_ZEPS, 1.0f - NF_ZEPS));
}
float nf_chart_logjac(float2 z)
{
    float v = clamp(z.y, 0.0f, 1.0f);
    float sin_t = max(sin(0.5f * NF_PI * v), NF_EPS);    // clamp the θ=0 pole
    return NF_LOGPI2 + log(sin_t);
}
#endif

// Forward flow u -> z accumulating log|det dz/du| (coupling layers alternate the mask).
// DOC:coupling#sig start
float2 nf_flow_forward(
    StructuredBuffer<NF_WT> W, StructuredBuffer<NF_WT> B, StructuredBuffer<NfLayerHeader> H,
    float2 u, float cond[NF_MLP_ENC], out float logdet)
// DOC:coupling#sig end
{
    // DOC:coupling#core start
    float2 z = u; logdet = 0.0f;
    for (int L = 0; L < NF_LAYERS; ++L)
    {
        bool even = (L % 2 == 0);                 // even: dim0 conditions, dim1 transforms
        float xcond = even ? z.x : z.y;
        float xtr   = even ? z.y : z.x;
        float params[NF_PARAMS];
        nf_mlp(W, B, H, L * 3, xcond, cond, params);
#if NF_COUPLING == 1
        float widths[NF_BINS]; float vertices[NF_BINS + 1];
        nf_decode_pq(params, widths, vertices);
        float ld; float ytr = nf_pq_fwd(xtr, widths, vertices, ld);
#else
        float widths[NF_BINS]; float heights[NF_BINS]; float derivs[NF_BINS + 1];
        nf_decode(params, widths, heights, derivs);
        float ld; float ytr = nf_rqs_fwd(xtr, widths, heights, derivs, ld);
#endif
        logdet += ld;
        if (even) z.y = ytr; else z.x = ytr;
    }
    return z;
    // DOC:coupling#core end
}

// Inverse flow z -> u accumulating log|det du/dz|.
float2 nf_flow_inverse(
    StructuredBuffer<NF_WT> W, StructuredBuffer<NF_WT> B, StructuredBuffer<NfLayerHeader> H,
    float2 zin, float cond[NF_MLP_ENC], out float logdet)
{
    float2 z = zin; logdet = 0.0f;
    for (int L = NF_LAYERS - 1; L >= 0; --L)
    {
        bool even = (L % 2 == 0);
        float ycond = even ? z.x : z.y;
        float ytr   = even ? z.y : z.x;
        float params[NF_PARAMS];
        nf_mlp(W, B, H, L * 3, ycond, cond, params);
#if NF_COUPLING == 1
        float widths[NF_BINS]; float vertices[NF_BINS + 1];
        nf_decode_pq(params, widths, vertices);
        float ld; float xtr = nf_pq_inv(ytr, widths, vertices, ld);
#else
        float widths[NF_BINS]; float heights[NF_BINS]; float derivs[NF_BINS + 1];
        nf_decode(params, widths, heights, derivs);
        float ld; float xtr = nf_rqs_inv(ytr, widths, heights, derivs, ld);
#endif
        logdet += ld;
        if (even) z.y = xtr; else z.x = xtr;
    }
    return z;
}

// PUBLIC: sample a hemisphere direction from the flow. base sample u in [0,1]^2.
float3 sampleNeural(
    StructuredBuffer<NF_WT> W, StructuredBuffer<NF_WT> B, StructuredBuffer<NfLayerHeader> H,
    float cond[NF_MLP_ENC], float2 u, out float pdfOmega)
{
    // DOC:pdf-square#core start
    float logdet;
    float2 z = nf_flow_forward(W, B, H, u, cond, logdet);
    float log_q_square = -logdet;                       // base pdf on unit square = 1
    // DOC:pdf-square#core end
    // DOC:pdf-omega#pdf start
#if NF_CHART == NF_CHART_V1
    pdfOmega = exp(log_q_square - NF_LOG2PI);            // V1 (default): constant 2π
    return nf_square_to_hemi(z);
#else
    pdfOmega = exp(log_q_square - nf_chart_logjac(z));   // V0 (2π) / V5 (π²·sinθ)
    return nf_chart_square_to_dir(z);
#endif
    // DOC:pdf-omega#pdf end
}

// PUBLIC: exact solid-angle pdf the flow assigns to direction `wi` (for the MIS mix).
// DOC:pdf-inv#sig start
float pdfNeural(
    StructuredBuffer<NF_WT> W, StructuredBuffer<NF_WT> B, StructuredBuffer<NfLayerHeader> H,
    float cond[NF_MLP_ENC], float3 wi)
// DOC:pdf-inv#sig end
{
    // DOC:pdf-inv#core start
    if (wi.y <= 0.0f) return 0.0f;
#if NF_CHART == NF_CHART_V1
    float2 z = nf_hemi_to_square(wi);
    float logdet;
    nf_flow_inverse(W, B, H, z, cond, logdet);          // log|det du/dz| = log q_square
    return exp(logdet - NF_LOG2PI);                      // V1 (default): constant 2π
#else
    float2 z = nf_chart_dir_to_square(wi);
    float logdet;
    nf_flow_inverse(W, B, H, z, cond, logdet);
    return exp(logdet - nf_chart_logjac(z));             // V0 (2π) / V5 (π²·sinθ)
#endif
    // DOC:pdf-inv#core end
}
