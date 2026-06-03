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
import restir.light_ris;                // restirFillReservoir / Resolve / Directional / EnvPHat
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
    uint  mLight;        // initial light candidates
    uint  spatialK;      // spatial neighbours
    float spatialRadius; // screen px
    float normalThresh;  // dot(N_i, N_j) min
    float depthThresh;   // |pos_i - pos_j| max (world)
    uint  mCap;          // temporal history cap
};
[[vk::push_constant]] ConstantBuffer<RestirPC> rpc;

static const uint RESTIR_FLAG_SPATIAL  = 1u;
static const uint RESTIR_FLAG_TEMPORAL = 2u;

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
    wfReservoirA[i] = restirFillReservoir(mat, h, N, T, B, wo, rpc.mLight, rng);
    GBuf g; g.pos = h.position; g.normal = N;
    wfGBuffer[i] = g;
}

// Spatial reuse: merge self + k domain-checked neighbour reservoirs (reservoirA)
// into reservoirB[i]. Env samples are directional ⇒ the reconnection Jacobian is
// 1; the neighbour's sample is re-evaluated at THIS pixel's frame (restirEnvPHat).
// BIASED combination (sum M-weighted) — the unbiased m_i/1-Z is a follow-up.
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
    // TEMPORAL: bufB[i] holds last frame's reservoir (this pass overwrites it
    // below). Static camera ⇒ same pixel/geometry, so p̂ is unchanged. Read it
    // before any write; only valid once accumulation has a prior frame.
    Reservoir prev = wfReservoirB[i];
    bool hasPrev = (fc.accumFrame > 0u) && (prev.W > 0.0)
                 && ((rpc.flags & RESTIR_FLAG_TEMPORAL) != 0u);

    Reservoir merged;
    reservoirReset(merged);
    Reservoir self = wfReservoirA[i];
    reservoirMerge(merged, self, self.pHat, 0.0);   // self always taken first (rand 0)

    // Single-tile assumption: lane index == pixel index (stream_size ≥ pixels),
    // so screen neighbours are reachable by lane offset. Cross-tile reuse is a
    // later refinement.
    uint px = i % fc.width;
    uint py = i / fc.width;
    RNG rng;
    rng.state = pcgHash((i * 9781u) ^ 0x85ebca6bu ^ fc.frameIndex);
    bool doSpatial = (rpc.flags & RESTIR_FLAG_SPATIAL) != 0u;
    for (uint k = 0u; doSpatial && k < rpc.spatialK; k++)
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
        if (nb.W <= 0.0 || (nb.y.packed & 0x3u) != 2u)   // RLIGHT_ENV
            continue;
        // Re-evaluate the neighbour's env sample at THIS pixel's frame.
        EnvSample es = sampleEnvDir(nb.y.uv, fc.envIntensity);
        float pHatI = restirEnvPHat(mat, es, N, T, B, wo);
        if (pHatI <= 0.0)
            continue;
        reservoirMerge(merged, nb, pHatI, rng.next());
    }

    // TEMPORAL merge: fold last frame's reservoir, M-capped to bound lag.
    // reservoirMerge weights by W·M, so clamping M scales the prev contribution.
    if (hasPrev)
    {
        prev.M = min(prev.M, rpc.mCap);
        reservoirMerge(merged, prev, prev.pHat, rng.next());
    }

    reservoirFinalize(merged);
    wfReservoirB[i] = merged;
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
