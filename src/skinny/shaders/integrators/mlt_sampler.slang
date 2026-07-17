// PSSMLT primary-sample-space sampler (change mlt-integrator).
//
// GPU port of pbrt-v4's MLTSampler (samplers.{h,cpp}): a Markov chain owns a
// vector of primary samples X[i] in [0,1) with lazy Kelemen mutations — per
// iteration either a "large step" (fresh uniforms, probability
// largeStepProbability) or a "small step" (per-dimension Gaussian perturbation
// with effective sigma sigma*sqrt(nSmall) aggregated over the iterations since
// the dimension was last touched, wrapped to [0,1)). Each dimension carries a
// one-deep backup so a rejected proposal restores the pre-mutation state
// exactly (pbrt Accept/Reject semantics).
//
// pbrt's X is a dynamically-resized vector (EnsureReady LOG_FATALs on GPU);
// here the dimension budget is FIXED at MLT_MAX_DIMS per chain. The budget is
// sized so the worst-case index for the flat-material envelope at the
// configured maxDepth is provably below it (~72 at maxDepth 5, see design D2)
// — an out-of-budget index is a broken invariant, not a supported path: the
// fallback below is memory-safety only (a fresh hash value every evaluation is
// not restored on reject and would break detailed balance).
//
// Samples are served through pbrt's three-stream interleaving (camera / light
// / connection): index = streamIndex + MLT_NUM_STREAMS * sampleIndex.
//
// The numpy mirror lives in src/skinny/sampling/mlt_sampler.py — keep the two
// in lockstep (tests/test_mlt_sampler.py asserts the semantics hostlessly).

import common;

static const uint MLT_NUM_STREAMS = 3u;

// Fixed per-chain primary-sample budget. Must match mlt_buffer_sizes on the
// host (wavefront_layout.py) and MLT_MAX_DIMS in sampling/mlt_sampler.py.
static const uint MLT_MAX_DIMS = 192u;

// One primary-sample dimension — 16 B, matches the host stride.
struct MltPrimarySample
{
    float value;
    float valueBackup;
    uint lastMod;    // iteration this dimension was last modified
    uint modBackup;  // lastMod backup for reject-restore
};

// Per-chain mutation bookkeeping (lives in the chain-metadata buffer).
// Initial state matches pbrt's freshly-constructed MLTSampler exactly:
// currentIteration = 0, largeStep = true, lastLargeStepIteration = 0 — the
// initial L evaluation runs at iteration 0 with NO preceding startIteration,
// so every dimension is fresh-filled with lastMod = 0 (design D3).
struct MltSamplerState
{
    uint rngState;               // mutation/accept RNG (common.slang RNG.state)
    uint currentIteration;
    uint lastLargeStepIteration;
    uint largeStep;              // bool
    uint streamIndex;
    uint sampleIndex;
};

MltSamplerState mltCreateSampler(uint seedIndex, uint seed)
{
    MltSamplerState s;
    s.rngState = pcgHash(seedIndex + seed * 16777259u);
    s.currentIteration = 0u;
    s.lastLargeStepIteration = 0u;
    s.largeStep = 1u;
    s.streamIndex = 0u;
    s.sampleIndex = 0u;
    return s;
}

// Inverse error function (pbrt-v4 util/math.h ErfInv — Giles' polynomial,
// https://stackoverflow.com/a/49743348). Needed for the 1-uniform-per-dimension
// inverse-CDF normal draw: Box-Muller would consume two uniforms and break the
// primary-sample-space mapping.
float mltErfInv(float a)
{
    float t = log(max(fma(a, -a, 1.0f), 1.175494e-38f));
    float p;
    if (abs(t) > 6.125f)
    {
        p = 3.03697567e-10f;
        p = fma(p, t, 2.93243101e-8f);
        p = fma(p, t, 1.22150334e-6f);
        p = fma(p, t, 2.84108955e-5f);
        p = fma(p, t, 3.93552968e-4f);
        p = fma(p, t, 3.02698812e-3f);
        p = fma(p, t, 4.83185798e-3f);
        p = fma(p, t, -2.64646143e-1f);
        p = fma(p, t, 8.40016484e-1f);
    }
    else
    {
        p = 5.43877832e-9f;
        p = fma(p, t, 1.43286059e-7f);
        p = fma(p, t, 1.22775396e-6f);
        p = fma(p, t, 1.12962631e-7f);
        p = fma(p, t, -5.61531961e-5f);
        p = fma(p, t, -1.47697705e-4f);
        p = fma(p, t, 2.31468701e-3f);
        p = fma(p, t, 1.15392562e-2f);
        p = fma(p, t, -2.32015476e-1f);
        p = fma(p, t, 8.86226892e-1f);
    }
    return p * a;
}

// N(mu, sigma) by inverse CDF from one uniform.
float mltSampleNormal(float u, float mu, float sigma)
{
    return mu + 1.41421356237f * sigma * mltErfInv(clamp(2.0f * u - 1.0f, -0.99999994f, 0.99999994f));
}

// pbrt MLTSampler::StartIteration.
void mltStartIteration(inout MltSamplerState s, float largeStepProbability)
{
    RNG rng = { s.rngState };
    s.currentIteration += 1u;
    s.largeStep = rng.next() < largeStepProbability ? 1u : 0u;
    s.rngState = rng.state;
}

// pbrt MLTSampler::StartStream.
void mltStartStream(inout MltSamplerState s, uint index)
{
    s.streamIndex = index;
    s.sampleIndex = 0u;
}

// pbrt MLTSampler::GetNextIndex.
uint mltNextIndex(inout MltSamplerState s)
{
    uint idx = s.streamIndex + MLT_NUM_STREAMS * s.sampleIndex;
    s.sampleIndex += 1u;
    return idx;
}

// pbrt MLTSampler::EnsureReady over the chain's fixed X slice.
// `xBase` is the chain's first element in the shared primary-sample buffer.
float mltEnsureReady(
    inout MltSamplerState s,
    RWStructuredBuffer<MltPrimarySample> X,
    uint xBase,
    uint index,
    float sigma)
{
    if (index >= MLT_MAX_DIMS)
    {
        // Broken invariant (design D2): unreachable in the flat envelope.
        // Memory-safety fallback only — decorrelated, NOT reversible.
        RNG rng = { s.rngState };
        float v = rng.next();
        s.rngState = rng.state;
        return v;
    }
    MltPrimarySample Xi = X[xBase + index];
    RNG rng = { s.rngState };

    // Reset X_i if a large step took place in the meantime.
    if (Xi.lastMod < s.lastLargeStepIteration)
    {
        Xi.value = rng.next();
        Xi.lastMod = s.lastLargeStepIteration;
    }

    // Apply the remaining sequence of mutations.
    Xi.valueBackup = Xi.value;
    Xi.modBackup = Xi.lastMod;
    if (s.largeStep != 0u)
    {
        Xi.value = rng.next();
    }
    else
    {
        uint nSmall = s.currentIteration - Xi.lastMod;
        float effSigma = sigma * sqrt(float(nSmall));
        Xi.value += mltSampleNormal(rng.next(), 0.0f, effSigma);
        Xi.value -= floor(Xi.value);
    }
    Xi.lastMod = s.currentIteration;

    X[xBase + index] = Xi;
    s.rngState = rng.state;
    return Xi.value;
}

// pbrt MLTSampler::Get1D / Get2D.
float mltGet1D(
    inout MltSamplerState s,
    RWStructuredBuffer<MltPrimarySample> X,
    uint xBase,
    float sigma)
{
    return mltEnsureReady(s, X, xBase, mltNextIndex(s), sigma);
}

float2 mltGet2D(
    inout MltSamplerState s,
    RWStructuredBuffer<MltPrimarySample> X,
    uint xBase,
    float sigma)
{
    float a = mltGet1D(s, X, xBase, sigma);
    float b = mltGet1D(s, X, xBase, sigma);
    return float2(a, b);
}

// pbrt MLTSampler::Accept.
void mltAccept(inout MltSamplerState s)
{
    if (s.largeStep != 0u)
        s.lastLargeStepIteration = s.currentIteration;
}

// pbrt MLTSampler::Reject — restore every dimension touched this iteration
// and rewind the iteration counter.
void mltReject(
    inout MltSamplerState s,
    RWStructuredBuffer<MltPrimarySample> X,
    uint xBase)
{
    for (uint i = 0u; i < MLT_MAX_DIMS; ++i)
    {
        MltPrimarySample Xi = X[xBase + i];
        if (Xi.lastMod == s.currentIteration)
        {
            Xi.value = Xi.valueBackup;
            Xi.lastMod = Xi.modBackup;
            X[xBase + i] = Xi;
        }
    }
    s.currentIteration -= 1u;
}
