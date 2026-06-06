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
static const int   NF_DIM    = 2;     // flow dimensionality (a direction = 2 dof)
static const int   NF_LAYERS = 6;     // coupling layers
static const int   NF_BINS   = 24;    // rational-quadratic spline bins (K)
static const int   NF_HIDDEN = 96;    // MLP hidden width
static const int   NF_COND   = 9;     // condition dim (renderer: pos3 + N3 + wo3)
static const float NF_PI     = 3.14159265358979323846f;
static const float NF_LOG2PI = 1.83787706640934548356f;   // log(2*pi)
static const float NF_EPS    = 1e-8f;
static const float NF_MINBIN = 1e-4f;

static const int NF_PARAMS = 3 * NF_BINS + 1;   // widths(K)+heights(K)+derivs(K+1)
static const int NF_MLP_IN = 1 + NF_COND;       // one pass-through coord + condition

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
void nf_linear(
    StructuredBuffer<float> W, StructuredBuffer<float> B, NfLayerHeader h,
    float inp[NF_HIDDEN], int n_in, bool act, out float outp[NF_HIDDEN])
{
    for (int o = 0; o < (int)h.outDim; ++o)
    {
        float acc = B[h.biasOffset + o];
        uint row = h.weightOffset + (uint)o * h.inDim;
        for (int i = 0; i < n_in; ++i)
            acc += W[row + (uint)i] * inp[i];
        outp[o] = act ? nf_silu(acc) : acc;
    }
}

// 3-layer conditioner MLP for one coupling layer: (pass-through coord, cond) ->
// NF_PARAMS spline parameters. `base3` is the index of this coupling's first header.
void nf_mlp(
    StructuredBuffer<float> W, StructuredBuffer<float> B, StructuredBuffer<NfLayerHeader> H,
    int base3, float xcond, float cond[NF_COND], out float params[NF_PARAMS])
{
    float a[NF_HIDDEN];
    a[0] = xcond;
    for (int i = 0; i < NF_COND; ++i) a[1 + i] = cond[i];

    float b[NF_HIDDEN];
    nf_linear(W, B, H[base3 + 0], a, NF_MLP_IN, true, b);   // in -> hidden, SiLU
    nf_linear(W, B, H[base3 + 1], b, NF_HIDDEN, true, a);   // hidden -> hidden, SiLU

    // hidden -> NF_PARAMS, linear. Reuse the wide local buffer then copy out.
    float wide[NF_HIDDEN];
    nf_linear(W, B, H[base3 + 2], a, NF_HIDDEN, false, wide);
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
float nf_rqs_fwd(float x, float widths[NF_BINS], float heights[NF_BINS],
                 float derivs[NF_BINS + 1], out float logdet)
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
    float num = hgt * (delta * theta * theta + d0 * t1);
    float den = delta + (d0 + d1 - 2.0f * delta) * t1;
    den = max(den, NF_EPS);
    float y = y0 + num / den;
    float dn = delta * delta * (d1 * theta * theta + 2.0f * delta * t1 + d0 * (1.0f - theta) * (1.0f - theta));
    float dydx = dn / (den * den);
    logdet = log(max(dydx, NF_EPS));
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

// Square <-> hemisphere (y-up), matching the prototype.
float3 nf_square_to_hemi(float2 z)
{
    float u = clamp(z.x, 0.0f, 1.0f);
    float v = clamp(z.y, 0.0f, 1.0f);
    float phi = 2.0f * NF_PI * u;
    float st = sqrt(max(1.0f - v * v, 0.0f));
    return float3(st * cos(phi), v, st * sin(phi));
}

float2 nf_hemi_to_square(float3 wld)
{
    float phi = atan2(wld.z, wld.x);
    if (phi < 0.0f) phi += 2.0f * NF_PI;
    return float2(phi / (2.0f * NF_PI), clamp(wld.y, 0.0f, 1.0f));
}

// Forward flow u -> z accumulating log|det dz/du| (coupling layers alternate the mask).
float2 nf_flow_forward(
    StructuredBuffer<float> W, StructuredBuffer<float> B, StructuredBuffer<NfLayerHeader> H,
    float2 u, float cond[NF_COND], out float logdet)
{
    float2 z = u; logdet = 0.0f;
    for (int L = 0; L < NF_LAYERS; ++L)
    {
        bool even = (L % 2 == 0);                 // even: dim0 conditions, dim1 transforms
        float xcond = even ? z.x : z.y;
        float xtr   = even ? z.y : z.x;
        float params[NF_PARAMS];
        nf_mlp(W, B, H, L * 3, xcond, cond, params);
        float widths[NF_BINS]; float heights[NF_BINS]; float derivs[NF_BINS + 1];
        nf_decode(params, widths, heights, derivs);
        float ld; float ytr = nf_rqs_fwd(xtr, widths, heights, derivs, ld);
        logdet += ld;
        if (even) z.y = ytr; else z.x = ytr;
    }
    return z;
}

// Inverse flow z -> u accumulating log|det du/dz|.
float2 nf_flow_inverse(
    StructuredBuffer<float> W, StructuredBuffer<float> B, StructuredBuffer<NfLayerHeader> H,
    float2 zin, float cond[NF_COND], out float logdet)
{
    float2 z = zin; logdet = 0.0f;
    for (int L = NF_LAYERS - 1; L >= 0; --L)
    {
        bool even = (L % 2 == 0);
        float ycond = even ? z.x : z.y;
        float ytr   = even ? z.y : z.x;
        float params[NF_PARAMS];
        nf_mlp(W, B, H, L * 3, ycond, cond, params);
        float widths[NF_BINS]; float heights[NF_BINS]; float derivs[NF_BINS + 1];
        nf_decode(params, widths, heights, derivs);
        float ld; float xtr = nf_rqs_inv(ytr, widths, heights, derivs, ld);
        logdet += ld;
        if (even) z.y = xtr; else z.x = xtr;
    }
    return z;
}

// PUBLIC: sample a hemisphere direction from the flow. base sample u in [0,1]^2.
float3 sampleNeural(
    StructuredBuffer<float> W, StructuredBuffer<float> B, StructuredBuffer<NfLayerHeader> H,
    float cond[NF_COND], float2 u, out float pdfOmega)
{
    float logdet;
    float2 z = nf_flow_forward(W, B, H, u, cond, logdet);
    float log_q_square = -logdet;                       // base pdf on unit square = 1
    pdfOmega = exp(log_q_square - NF_LOG2PI);
    return nf_square_to_hemi(z);
}

// PUBLIC: exact solid-angle pdf the flow assigns to direction `wi` (for the MIS mix).
float pdfNeural(
    StructuredBuffer<float> W, StructuredBuffer<float> B, StructuredBuffer<NfLayerHeader> H,
    float cond[NF_COND], float3 wi)
{
    if (wi.y <= 0.0f) return 0.0f;
    float2 z = nf_hemi_to_square(wi);
    float logdet;
    nf_flow_inverse(W, B, H, z, cond, logdet);          // log|det du/dz| = log q_square
    return exp(logdet - NF_LOG2PI);
}
