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
import restir.reservoir;                // Reservoir
import restir.light_ris;                // restirFillReservoir / Resolve / Directional
import wavefront.wavefront_state;       // WavefrontPathState, pathAlive

// ReSTIR set 1: wavefront path-state (0) + hit (1) [shared with WavefrontPathPass]
// + the per-pixel reservoir buffer (2) [ReSTIR-owned, persistent].
[[vk::binding(0, 1)]] RWStructuredBuffer<WavefrontPathState> wfState;
[[vk::binding(1, 1)]] RWStructuredBuffer<HitInfo>            wfHits;
[[vk::binding(2, 1)]] RWStructuredBuffer<Reservoir>          wfReservoir;

struct RestirPC { uint streamSize; };
[[vk::push_constant]] ConstantBuffer<RestirPC> rpc;

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

// Fill: initial RIS over the light set → wfReservoir[i].
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
    wfReservoir[i] = restirFillReservoir(mat, h, N, T, B, wo, rng);
}

// Resolve: read the (possibly reused) reservoir → one shadow ray → add direct
// (+ directional NEE) into the path-state radiance.
[shader("compute")]
[numthreads(64, 1, 1)]
void restirResolve(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    WavefrontPathState s; HitInfo h; FlatMaterial mat; float3 N, T, B, wo;
    if (!restirLoadLane(i, s, h, mat, N, T, B, wo))
        return;
    Reservoir r = wfReservoir[i];
    float3 direct = restirResolveReservoir(mat, h, N, T, B, wo, r)
                  + restirDirectional(mat, h, N, T, B, wo);
    s.radiance += s.throughput * direct;
    wfState[i] = s;
}
