// Wavefront MLT — Kelemen full-sample PSSMLT over the BDPT integrator
// (change mlt-integrator). Compiled ONLY with -DSKINNY_MLT: common.slang's
// RNG is then the primary-sample-space sampler (chain X vectors, binding 52),
// so BDPTIntegrator.estimateRadiance below is a deterministic function of the
// chain state — the PSSMLT target. Light-tracer splats are captured per chain
// (bdpt.slang atomicSplatRadiance override) and written post-acceptance.
//
// Per-frame staged sequence (record_mlt_loop):
//   [reset] wfMltBootstrap (breadth-tiled) → host CDF/b/seed resample →
//   wfMltInit → per frame: wfMltMutate × iterations (breadth-tiled) →
//   wfMltResolve (b-normalized film average into accumBuffer).
//
// Estimator: pbrt MLTIntegrator::Render's dual splat — proposal × a/c_p +
// current × (1−a)/c_c per mutation, resolve scale b / mpp_actual, film-
// averaged progressively (design D4). c is the luminance of the COMPLETE
// sample (eye contribution + captured splats) — full-sample chains, no depth
// strata (design D1 amendment: skinny's env transport is not
// strategy-partitioned, so per-depth chains would drop env light).

import common;
import bindings;
import scene_trace;
import cameras.pinhole;
import integrators.bdpt;
#if defined(SKINNY_SPECTRAL)
// Spectral MLT (change spectral-mlt): the target function becomes the megakernel
// spectral BDPT estimator (change spectral-bdpt-megakernel). Its estimateRadiance
// has the identical `float3 (Ray, HitInfo, inout RNG)` signature and already
// resolves Spectrum → clamped linear sRGB internally (spectrumResolveToLinearSRGB),
// so c, the splat capture, and the resolve pass all stay on the RGB code paths.
// The hero-wavelength draw `sampleWavelengths(rng.next())` at the top of
// estimateRadiance is served by the -DSKINNY_MLT PSS override → the wavelength is
// a primary-sample dimension with no estimator or mutation-logic change (design D2).
import integrators.bdpt_spectral;   // SpectralBDPTIntegrator
#endif
import wavefront.wf_display;   // wfWriteDisplay (exposure + tonemap + present)

// Per-dispatch tile window (same push-constant layout as wavefront_bdpt's
// WfTilePC so the host tile-push path is shared).
struct MltTilePC { uint streamBase; uint shadeSlot; uint streamSize; };
[[vk::push_constant]] ConstantBuffer<MltTilePC> wfTile;

// ── chain state ─────────────────────────────────────────────────

// One captured film contribution: the eye value at the chain's PSS raster
// position (slot 0) or a light-tracer splat (slots 1..BDPT_MAX_VERTS).
struct MltRecord
{
    // Scalar fields (not float3) so the Vulkan-scalar and Metal MSL strides
    // are both exactly 16 B (wavefront_layout.mlt_buffer_sizes).
    uint pixel;      // lane = y * width + x
    float r;
    float g;
    float b;
};

static const uint MLT_RECORD_SLOTS = BDPT_MAX_VERTS + 1u;   // eye + ≤6 splats

// Per-chain persistent bookkeeping (32 B).
struct MltChainMeta
{
    uint rngState;                 // mutation PCG stream (per-chain, NOT the replay seed)
    uint currentIteration;
    uint lastLargeStepIteration;
    uint seedIndex;                // bootstrap index this chain was seeded from
    float cCurrent;                // scalar contribution of the current state
    uint nRecords;                 // live records in mltCurrentRecords
    uint pad0;
    uint pad1;
};

[[vk::binding(53)]] RWStructuredBuffer<MltChainMeta> mltChainMeta;      // nChains
[[vk::binding(54)]] RWStructuredBuffer<MltRecord> mltCurrentRecords;    // nChains × MLT_RECORD_SLOTS
[[vk::binding(55)]] RWStructuredBuffer<float> mltBootstrapWeights;      // nBootstrap
[[vk::binding(56)]] RWStructuredBuffer<uint> mltChainSeeds;             // nChains
// Proposal-record scratch (change spectral-mlt, Metal live-state fix):
// mltEvaluate writes its captured records STRAIGHT here instead of returning a
// thread-local MltRecord[7] — on Metal, that array (on top of the spectral
// estimator's live subpath state) overflowed the per-thread budget and hung
// wfMltMutate forever; bootstrap/init sat just under the limit. Same values,
// same order, so Vulkan behavior is unchanged. Indexed chain*MLT_RECORD_SLOTS
// (bootstrap: slot*MLT_RECORD_SLOTS — in-flight slots ≤ nChains by the same
// tiling contract that lets the PSS buffer double as bootstrap scratch).
[[vk::binding(57)]] RWStructuredBuffer<MltRecord> mltProposalRecords;   // nChains × MLT_RECORD_SLOTS


// MLT film splat scale — Q24.8 (design D4): headroom 2^24 luminance units per
// frame against the per-frame splat-count bound; NO upper clamp is permitted
// (clamping biases exactly the caustics MLT targets). Coarser than
// BDPT_SPLAT_SCALE (Q22.10) by design: MLT trades 2 precision bits for 4×
// count headroom, and the resolve averages the quantization across frames.
static const float MLT_SPLAT_SCALE = 256.0;

float mltLuminance(float3 c)
{
    return dot(max(c, float3(0.0)), float3(0.2126, 0.7152, 0.0722));
}

void mltFilmSplat(uint pixelLane, float3 radiance)
{
    if (any(radiance != radiance)) return;
    if (any(isinf(radiance))) return;
    radiance = max(radiance, float3(0.0));
    uint base = pixelLane * 3u;
    uint orig;
    InterlockedAdd(lightSplatBuffer[base + 0u], uint(radiance.x * MLT_SPLAT_SCALE), orig);
    InterlockedAdd(lightSplatBuffer[base + 1u], uint(radiance.y * MLT_SPLAT_SCALE), orig);
    InterlockedAdd(lightSplatBuffer[base + 2u], uint(radiance.z * MLT_SPLAT_SCALE), orig);
}

// ── target-function evaluation ──────────────────────────────────

// One complete BDPT sample driven by the chain's primary samples. The captured
// contributions land in mltProposalRecords[recordBase .. recordBase+nRecords)
// (slot 0 = eye, then light splats) — device memory, never a thread-local
// array (see the binding-57 comment). Returns the record count and the scalar
// contribution c.
struct MltSample
{
    uint nRecords;
    float c;
};

MltSample mltEvaluate(inout RNG rng, uint recordBase)
{
    gMltSplatCount = 0u;
    rng.startStream(0u);   // pbrt cameraStreamIndex

    // Raster position from primary sample space — the chain explores image
    // space through its first two camera-stream dimensions (pbrt GetPixel2D).
    float2 u = rng.next2();
    uint2 pixel = min(uint2(u * float2(float(fc.width), float(fc.height))),
                      uint2(fc.width - 1u, fc.height - 1u));

    PinholeCamera cam;
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);

    HitInfo firstHit = traceScene(fc, ray);

#if defined(SKINNY_SPECTRAL)
    SpectralBDPTIntegrator<PinholeCamera> integ;
#else
    BDPTIntegrator<PinholeCamera> integ;
#endif
    integ.cam = cam;
    float3 eyeL = integ.estimateRadiance(ray, firstHit, rng) * lensWeight;
    if (any(eyeL != eyeL) || any(isinf(eyeL)))
        eyeL = float3(0.0);
    eyeL = max(eyeL, float3(0.0));

    MltSample s;
    MltRecord r0;
    r0.pixel = pixel.y * fc.width + pixel.x;
    r0.r = eyeL.x; r0.g = eyeL.y; r0.b = eyeL.z;
    mltProposalRecords[recordBase] = r0;
    s.nRecords = 1u;
    float c = mltLuminance(eyeL);
    [loop]
    for (uint i = 0u; i < gMltSplatCount && s.nRecords < MLT_RECORD_SLOTS; i++)
    {
        int2 sp = gMltSplatPixel[i];
        if (sp.x < 0 || sp.y < 0 || uint(sp.x) >= fc.width || uint(sp.y) >= fc.height)
            continue;
        MltRecord rr;
        rr.pixel = uint(sp.y) * fc.width + uint(sp.x);
        rr.r = gMltSplatRgb[i].x;
        rr.g = gMltSplatRgb[i].y;
        rr.b = gMltSplatRgb[i].z;
        mltProposalRecords[recordBase + s.nRecords] = rr;
        c += mltLuminance(gMltSplatRgb[i]);
        s.nRecords += 1u;
    }
    s.c = c;
    return s;
}

// ── kernels ─────────────────────────────────────────────────────

// Phase B (bootstrap): one thread = one bootstrap sample. Breadth-tiled by
// the host (wfTile.streamBase, batch ≤ nChains so each in-flight slot owns a
// distinct X slice). Writes the scalar contribution per bootstrap index.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfMltBootstrap(uint3 tid: SV_DispatchThreadID)
{
    uint slot = tid.x;
    if (slot >= wfTile.streamSize) return;
    uint index = wfTile.streamBase + slot;

    RNG rng = mltCreateSampler(index, fc.mltSeed, slot * MLT_MAX_DIMS, fc.mltSigma);
    // Records land in the slot's proposal-scratch slice and are ignored —
    // bootstrap only needs the scalar contribution.
    MltSample s = mltEvaluate(rng, slot * MLT_RECORD_SLOTS);
    mltBootstrapWeights[index] = s.c;
}

// Phase I (init): one thread = one chain. Reconstructs the chain's current
// state by replaying its resampled bootstrap seed (pinned initial
// bookkeeping: iteration 0, largeStep true, lastLargeStep 0, no
// startIteration — design D3), then re-seeds the MUTATION stream per chain
// (bootstrap resampling draws WITH replacement, so two chains may share a
// seedIndex; their mutation streams must still diverge).
[shader("compute")]
[numthreads(64, 1, 1)]
void wfMltInit(uint3 tid: SV_DispatchThreadID)
{
    uint chain = wfTile.streamBase + tid.x;
    if (tid.x >= wfTile.streamSize || chain >= fc.mltNumChains) return;

    uint seedIndex = mltChainSeeds[chain];
    RNG rng = mltCreateSampler(seedIndex, fc.mltSeed, chain * MLT_MAX_DIMS, fc.mltSigma);
    uint base = chain * MLT_RECORD_SLOTS;
    MltSample s = mltEvaluate(rng, base);

    MltChainMeta meta;
    meta.rngState = pcgHash(chain + fc.mltSeed * 9781u + 0x9E3779B9u);
    meta.currentIteration = 0u;
    meta.lastLargeStepIteration = 0u;
    meta.seedIndex = seedIndex;
    meta.cCurrent = s.c;
    meta.nRecords = s.nRecords;
    meta.pad0 = 0u; meta.pad1 = 0u;
    mltChainMeta[chain] = meta;
    [loop]
    for (uint i = 0u; i < s.nRecords; i++)
        mltCurrentRecords[base + i] = mltProposalRecords[base + i];
}

// Phase M (mutate): one thread = one chain = ONE Metropolis iteration.
// The host dispatches this `iterations` times per frame (flush-bounded on
// Metal — dispatch hygiene), each: propose → dual splat → accept/reject.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfMltMutate(uint3 tid: SV_DispatchThreadID)
{
    uint chain = wfTile.streamBase + tid.x;
    if (tid.x >= wfTile.streamSize || chain >= fc.mltNumChains) return;

    MltChainMeta meta = mltChainMeta[chain];

    RNG rng;
    rng.state = meta.rngState;
    rng.xBase = chain * MLT_MAX_DIMS;
    rng.currentIteration = meta.currentIteration;
    rng.lastLargeStepIteration = meta.lastLargeStepIteration;
    rng.largeStep = 0u;
    rng.streamIndex = 0u;
    rng.sampleIndex = 0u;
    rng.sigma = fc.mltSigma;
    rng.maxDim = 0u;   // per-invocation reject() scan bound

    rng.startIteration(fc.mltLargeStepProb);
    uint base = chain * MLT_RECORD_SLOTS;
    MltSample prop = mltEvaluate(rng, base);

    float cCurrent = meta.cCurrent;
    // Seeding guarantees cCurrent > 0 (bootstrap resample over positive
    // weights); the max() below is a defensive NaN/zero guard only.
    float a = (cCurrent > 0.0)
        ? min(1.0, prop.c / cCurrent)
        : (prop.c > 0.0 ? 1.0 : 0.0);

    // Dual splat (pbrt MLTIntegrator::Render): proposal × a/c_p, current ×
    // (1−a)/c_c — every captured contribution at its own raster position.
    // Decision-independent (uses a, not the accept draw), so it stays here.
    if (a > 0.0 && prop.c > 0.0)
    {
        float wp = a / prop.c;
        [loop]
        for (uint i = 0u; i < prop.nRecords; i++)
        {
            MltRecord pr = mltProposalRecords[base + i];
            mltFilmSplat(pr.pixel, float3(pr.r, pr.g, pr.b) * wp);
        }
    }
    if (cCurrent > 0.0 && a < 1.0)
    {
        float wc = (1.0 - a) / cCurrent;
        [loop]
        for (uint i = 0u; i < meta.nRecords; i++)
        {
            MltRecord rr = mltCurrentRecords[base + i];
            mltFilmSplat(rr.pixel, float3(rr.r, rr.g, rr.b) * wc);
        }
    }

    // Accept / reject (the decision draw is mutation-stream, never PSS).
    if (rng.nextMutation() < a)
    {
        rng.accept();
        meta.cCurrent = prop.c;
        meta.nRecords = prop.nRecords;
        [loop]
        for (uint i = 0u; i < prop.nRecords; i++)
            mltCurrentRecords[base + i] = mltProposalRecords[base + i];
    }
    else
    {
        rng.reject();
    }

    meta.rngState = rng.state;
    meta.currentIteration = rng.currentIteration;
    meta.lastLargeStepIteration = rng.lastLargeStepIteration;
    mltChainMeta[chain] = meta;
}


// Phase R (resolve): fold the frame's splats into the accumulation image,
// scaled b / mpp_actual (pbrt WriteImage(b / mutationsPerPixel), applied as a
// per-frame film average — design D4), and clear the splat buffer for the
// next frame. One thread per pixel, full-frame dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfMltResolve(uint3 tid: SV_DispatchThreadID)
{
    uint lane = tid.x;
    if (lane >= fc.width * fc.height) return;
    uint base = lane * 3u;

    float scale = fc.mltB / (MLT_SPLAT_SCALE * max(fc.mltMppActual, 1e-8));
    float3 frameL = float3(float(lightSplatBuffer[base + 0u]),
                           float(lightSplatBuffer[base + 1u]),
                           float(lightSplatBuffer[base + 2u])) * scale;
    lightSplatBuffer[base + 0u] = 0u;
    lightSplatBuffer[base + 1u] = 0u;
    lightSplatBuffer[base + 2u] = 0u;

    uint2 p = uint2(lane % fc.width, lane / fc.width);
    uint n = fc.accumFrame;
    float3 accum = frameL;
    if (n != 0u)
    {
        let prev = accumBuffer[p].rgb;
        accum = (prev * float(n) + frameL) / float(n + 1u);
    }
    accumBuffer[p] = float4(accum, 1.0);
    wfWriteDisplay(p, accum);
}
