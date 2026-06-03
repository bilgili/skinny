// ReSTIR DI primary-direct passes (wavefront). Run once per lane at the primary
// hit (bounce 0), scheduled by the reuse hook in WavefrontPathPass before shade
// (whose depth-0 reuseDirect is gated to zero, so ReSTIR is the sole primary-NEE
// source). The estimate ADDS into the path-state radiance.
//
// Split into a fill pass (initial RIS → per-pixel reservoir buffer) and a
// resolve pass (read reservoir → one shadow ray → shade), so the reservoir
// persists in VRAM for spatial / temporal reuse passes to read. Behaviour is
// identical to the single-pass M2 RIS (same candidates, same survivor, same
// shadow ray) — it still CONVERGES to reuse=none. Spatial/temporal passes slot
// between fill and resolve and merge neighbour / prev-frame reservoirs.

import common;          // RNG, pcgHash, worldToTangent, buildBasis
import bindings;        // set 0: fc, materialTypes, lights, env, materials
import interfaces;      // HitInfo, IMaterial
import materials.flat.flat_material;   // FlatMaterial, loadFlatMaterial
import environment;                     // sampleEnvDir, EnvSample (spatial re-eval)
import restir.reservoir;                // Reservoir
import restir.light_ris;                // restirFillReservoir / Resolve / Directional / restirEvalRef
import wavefront.wavefront_state;       // WavefrontPathState, pathAlive

// ReSTIR set 1: wavefront path-state (0) + hit (1) [shared with WavefrontPathPass]
// Per-pixel G-buffer for the spatial-neighbour domain check.
struct GBuf { float3 pos; float3 normal; };

// + ping-pong reservoir buffers A/B (2,3) + a G-buffer (4) [ReSTIR-owned].
//   fill: writes reservoirA + G-buffer ;  spatial: reservoirA → reservoirB ;
//   resolve: reads reservoirB.
[[vk::binding(0, 1)]] RWStructuredBuffer<WavefrontPathState> wfState;
[[vk::binding(1, 1)]] RWStructuredBuffer<HitInfo>            wfHits;
[[vk::binding(2, 1)]] RWStructuredBuffer<Reservoir>          wfReservoirA;
[[vk::binding(3, 1)]] RWStructuredBuffer<Reservoir>          wfReservoirB;
[[vk::binding(4, 1)]] RWStructuredBuffer<GBuf>              wfGBuffer;

// ReSTIR config (push constant, 32 B scalar — renderer-driven via RestirDiReuse).
// flags: bit0 = spatial reuse, bit1 = temporal reuse.
struct RestirPC
{
    uint  streamSize;
    uint  flags;
    uint  mLight;        // initial light-sampled candidates
    uint  spatialK;      // spatial neighbours
    float spatialRadius; // screen px
    float normalThresh;  // dot(N_i, N_j) min
    float depthThresh;   // |pos_i - pos_j| max (world)
    uint  mCap;          // temporal history cap
    uint  mBsdf;         // initial BSDF-sampled candidates
};
[[vk::push_constant]] ConstantBuffer<RestirPC> rpc;

static const uint RESTIR_FLAG_SPATIAL  = 1u;
static const uint RESTIR_FLAG_TEMPORAL = 2u;
static const uint RESTIR_FLAG_BIASED   = 4u;   // ΣM combination (faster, skips GRIS)

// Shared per-lane setup: returns false unless this lane is a live flat primary
// hit. Fills the shading frame + outgoing direction.
bool restirLoadLane(uint i, out WavefrontPathState s, out HitInfo h,
                    out FlatMaterial mat, out float3 N, out float3 T, out float3 B,
                    out float3 wo)
{
    s = wfState[i];
    mat = (FlatMaterial)0; N = float3(0,0,1); T = float3(1,0,0); B = float3(0,1,0);
    wo = float3(0,0,1); h = (HitInfo)0;
    if (i >= rpc.streamSize || !pathAlive(s) || s.depth != 0u)
        return false;
    h = wfHits[i];
    if (!h.hit || (materialTypes[h.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
        return false;
    mat = loadFlatMaterial(h);
    N = mat.data.N;
    if (h.hasTangent)
    {
        T = normalize(h.tangent - N * dot(h.tangent, N));
        B = cross(N, T);
    }
    else
    {
        buildBasis(N, T, B);
    }
    wo = worldToTangent(-s.rayDir, T, B, N);
    return true;
}

// Fill: initial RIS over the light set → reservoirA[i] + the G-buffer.
[shader("compute")]
[numthreads(64, 1, 1)]
void restirFill(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    WavefrontPathState s; HitInfo h; FlatMaterial mat; float3 N, T, B, wo;
    if (!restirLoadLane(i, s, h, mat, N, T, B, wo))
        return;
    // Separate RNG stream so we don't consume the shade kernel's cursor.
    RNG rng;
    rng.state = pcgHash(s.rngState ^ 0x9e3779b9u);
    wfReservoirA[i] = restirFillReservoir(mat, h, N, T, B, wo, rpc.mLight, rpc.mBsdf, rng);
    GBuf g; g.pos = h.position; g.normal = N;
    wfGBuffer[i] = g;
}

// Max combined sources: self + spatial neighbours + temporal.
static const uint RESTIR_MAX_SRC = 12u;

// Spatial + temporal reuse with the UNBIASED generalized balance heuristic
// (GRIS, Lin 2022 / Bitterli 2020 unbiased). Each source sample z_s is combined
// with MIS weight
//     m_s = M_s·p̂_s(z_s) / Σ_j M_j·p̂_j(z_s)
// where p̂_j(z_s) is z_s's target re-evaluated in source j's OWN domain (its
// shading point + material, re-loaded from wfHits[j] via restirLoadLane). A
// sample that many domains could have produced is down-weighted, which BOUNDS the
// spatial→temporal feedback that the biased (ΣM) combination let explode on
// glossy surfaces. DI reuses the same world light point (uv) ⇒ the reconnection
// shift is identity (Jacobian 1); the geometry/BSDF change is captured by the
// per-domain p̂ re-eval. The resampling weight is w_s = m_s·p̂_q(z_s)·W_s at the
// canonical pixel q; W_out = Σw_s / p̂_q(Y) (the m_s normalize — no 1/M, no 1/Z).
[shader("compute")]
[numthreads(64, 1, 1)]
void restirSpatial(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    WavefrontPathState s; HitInfo h; FlatMaterial mat; float3 N, T, B, wo;
    if (!restirLoadLane(i, s, h, mat, N, T, B, wo))
    {
        if (i < rpc.streamSize)
            wfReservoirB[i] = wfReservoirA[i];     // passthrough non-ReSTIR lanes
        return;
    }
    GBuf gi = wfGBuffer[i];
    Reservoir prev = wfReservoirB[i];              // last frame's reservoir (read before write)
    bool hasPrev = (fc.accumFrame > 0u) && (prev.W > 0.0)
                 && ((rpc.flags & RESTIR_FLAG_TEMPORAL) != 0u);

    // ── Collect sources: self + domain-checked spatial neighbours + temporal ──
    uint           sLane[RESTIR_MAX_SRC];
    LightSampleRef sRef[RESTIR_MAX_SRC];
    float          sW[RESTIR_MAX_SRC];
    float          sM[RESTIR_MAX_SRC];
    uint K = 0u;

    Reservoir self = wfReservoirA[i];
    sLane[K] = i; sRef[K] = self.y; sW[K] = self.W; sM[K] = float(self.M); K++;

    uint px = i % fc.width;
    uint py = i / fc.width;
    RNG rng;
    rng.state = pcgHash((i * 9781u) ^ 0x85ebca6bu ^ fc.frameIndex);
    bool doSpatial = (rpc.flags & RESTIR_FLAG_SPATIAL) != 0u;
    for (uint k = 0u; doSpatial && k < rpc.spatialK && K < RESTIR_MAX_SRC - 1u; k++)
    {
        float2 d = (rng.next2() * 2.0 - 1.0) * rpc.spatialRadius;
        int nx = int(px) + int(round(d.x));
        int ny = int(py) + int(round(d.y));
        if (nx < 0 || ny < 0 || uint(nx) >= fc.width || uint(ny) >= fc.height)
            continue;
        uint j = uint(ny) * fc.width + uint(nx);
        if (j == i || j >= rpc.streamSize)
            continue;
        GBuf gj = wfGBuffer[j];
        if (dot(gi.normal, gj.normal) < rpc.normalThresh ||
            length(gi.pos - gj.pos) > rpc.depthThresh)
            continue;
        Reservoir nb = wfReservoirA[j];
        if (nb.W <= 0.0)
            continue;
        sLane[K] = j; sRef[K] = nb.y; sW[K] = nb.W; sM[K] = float(nb.M); K++;
    }
    if (hasPrev && K < RESTIR_MAX_SRC)
    {
        sLane[K] = i; sRef[K] = prev.y; sW[K] = prev.W;
        sM[K] = float(min(prev.M, rpc.mCap)); K++;        // M-capped temporal history
    }

    // Canonical target p̂_q(z_s) of each source's sample at THIS pixel.
    float pCanon[RESTIR_MAX_SRC];
    for (uint a = 0u; a < K; a++)
    {
        RCand c = restirEvalRef(mat, refType(sRef[a]), refId(sRef[a]), sRef[a].uv, h, N, T, B, wo);
        pCanon[a] = (c.valid) ? c.pHat : 0.0;
    }

    // BIASED toggle: the simple ΣM combination (Bitterli 2020 Alg. 4) — each
    // source weighted by p̂_q(z_s)·W_s·M_s, normalized by ΣM. Skips the O(K²)
    // per-domain p̂ re-eval (the GRIS denominator) → faster, but biased
    // (discontinuity darkening; with TEMPORAL on glossy it over-brightens via
    // feedback — keep it to spatial-only). Default is the unbiased GRIS below.
    if ((rpc.flags & RESTIR_FLAG_BIASED) != 0u)
    {
        Reservoir outb;
        reservoirReset(outb);
        float mTot = 0.0;
        for (uint a = 0u; a < K; a++)
        {
            mTot += sM[a];
            if (pCanon[a] <= 0.0 || sW[a] <= 0.0)
                continue;
            float w = pCanon[a] * sW[a] * sM[a];
            outb.wSum += w;
            if (w > 0.0 && rng.next() * outb.wSum < w)
            {
                outb.y = sRef[a];
                outb.pHat = pCanon[a];
            }
        }
        outb.M = uint(mTot);
        outb.W = (outb.pHat > 0.0 && mTot > 0.0) ? outb.wSum / (mTot * outb.pHat) : 0.0;
        wfReservoirB[i] = outb;
        return;
    }

    // MIS denominators D_s = Σ_j M_j·p̂_j(z_s) + own-domain target p̂_s(z_s).
    // Load each source domain ONCE (outer j), evaluate every sample in it.
    float D[RESTIR_MAX_SRC];
    float pOwn[RESTIR_MAX_SRC];
    for (uint a = 0u; a < K; a++) { D[a] = 0.0; pOwn[a] = 0.0; }
    for (uint j = 0u; j < K; j++)
    {
        WavefrontPathState sj; HitInfo hj; FlatMaterial matj; float3 Nj, Tj, Bj, woj;
        if (!restirLoadLane(sLane[j], sj, hj, matj, Nj, Tj, Bj, woj))
            continue;                               // domain can't be reconstructed ⇒ contributes 0
        for (uint a = 0u; a < K; a++)
        {
            RCand c = restirEvalRef(matj, refType(sRef[a]), refId(sRef[a]), sRef[a].uv,
                                    hj, Nj, Tj, Bj, woj);
            float p = (c.valid) ? c.pHat : 0.0;
            D[a] += sM[j] * p;
            if (j == a) pOwn[a] = p;
        }
    }

    // Combine: select Y ∝ w_s = m_s·p̂_q(z_s)·W_s, m_s = M_s·p̂_s(z_s)/D_s.
    Reservoir outr;
    reservoirReset(outr);
    float mTotal = 0.0;
    for (uint a = 0u; a < K; a++)
    {
        mTotal += sM[a];
        if (D[a] <= 0.0 || pCanon[a] <= 0.0 || sW[a] <= 0.0)
            continue;
        float m_s = sM[a] * pOwn[a] / D[a];
        float w_s = m_s * pCanon[a] * sW[a];
        outr.wSum += w_s;
        if (w_s > 0.0 && rng.next() * outr.wSum < w_s)
        {
            outr.y = sRef[a];
            outr.pHat = pCanon[a];
        }
    }
    outr.M = uint(mTotal);
    outr.W = (outr.pHat > 0.0) ? outr.wSum / outr.pHat : 0.0;   // m_s normalize ⇒ no 1/M
    wfReservoirB[i] = outr;
}

// Resolve: read the spatially-reused reservoir (B) → one shadow ray → add direct
// (+ directional NEE) into the path-state radiance.
[shader("compute")]
[numthreads(64, 1, 1)]
void restirResolve(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    WavefrontPathState s; HitInfo h; FlatMaterial mat; float3 N, T, B, wo;
    if (!restirLoadLane(i, s, h, mat, N, T, B, wo))
        return;
    Reservoir r = wfReservoirB[i];
    float3 direct = restirResolveReservoir(mat, h, N, T, B, wo, r)
                  + restirDirectional(mat, h, N, T, B, wo);
    s.radiance += s.throughput * direct;
    wfState[i] = s;
}
