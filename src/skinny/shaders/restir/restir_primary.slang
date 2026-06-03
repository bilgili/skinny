// ReSTIR DI primary-direct pass (wavefront). Runs once per lane at the primary
// hit (bounce 0), BEFORE the shade kernel, scheduled by the reuse hook in
// WavefrontPathPass. It computes the primary-hit direct lighting and ADDS it
// into the path-state radiance; the shade kernel's depth-0 reuseDirect is gated
// to zero (reuseMode == RESTIR_DI) so this is the sole source of primary NEE.
//
// MILESTONE 1 (plumbing): the estimate is plain allLightsNEE (the NEE light-
// sampling half), drawn on a SEPARATE RNG stream from shade. Combined with
// shade's still-active BSDF-sampled sphere/env-miss term (the BSDF half, MIS-
// weighted as before), the total equals stock NEE — so the image CONVERGES to
// reuse=none, validating buffers / hook / descriptor sets / radiance injection
// before any reservoir math. MILESTONE 2 replaces allLightsNEE with reservoir
// RIS over the unified light set (restir/reservoir.slang).

import common;          // RNG, pcgHash, worldToTangent, buildBasis
import bindings;        // set 0: fc, materialTypes, lights, env, materials
import interfaces;      // HitInfo, IMaterial
import materials.flat.flat_material;   // FlatMaterial, loadFlatMaterial
import restir.light_ris;               // restirDirectLighting (reservoir RIS, M2)
import wavefront.wavefront_state;       // WavefrontPathState, pathAlive

// ReSTIR set 1: the wavefront path-state + hit buffers (same buffers as
// WavefrontPathPass binds; ReSTIR only needs these two).
[[vk::binding(0, 1)]] RWStructuredBuffer<WavefrontPathState> wfState;
[[vk::binding(1, 1)]] RWStructuredBuffer<HitInfo>            wfHits;

struct RestirPC { uint streamSize; };
[[vk::push_constant]] ConstantBuffer<RestirPC> rpc;

[shader("compute")]
[numthreads(64, 1, 1)]
void restirPrimaryDirect(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= rpc.streamSize)
        return;

    WavefrontPathState s = wfState[i];
    if (!pathAlive(s) || s.depth != 0u)   // primary vertex only
        return;

    HitInfo h = wfHits[i];
    if (!h.hit)
        return;

    // Milestone 1 handles flat / MaterialX-graph materials (the demo scene).
    // Skin / python primary hits keep their own shading (no reuseDirect gate).
    uint matType = materialTypes[h.materialId] & 0xFFu;
    if (matType != MATERIAL_TYPE_FLAT)
        return;

    FlatMaterial mat = loadFlatMaterial(h);
    float3 N = mat.data.N;
    float3 T, B;
    if (h.hasTangent)
    {
        T = normalize(h.tangent - N * dot(h.tangent, N));
        B = cross(N, T);
    }
    else
    {
        buildBasis(N, T, B);
    }
    float3 V  = -s.rayDir;
    float3 wo = worldToTangent(V, T, B, N);

    // Separate RNG stream so we don't consume the shade kernel's cursor (its
    // rngState is left untouched in wfState[i]).
    RNG rng;
    rng.state = pcgHash(s.rngState ^ 0x9e3779b9u);

    // M2: reservoir RIS over the light set (MIS-weighted target) instead of the
    // single-sample allLightsNEE. Composes with shade's BSDF-direct half.
    float3 direct = restirDirectLighting(mat, h, N, T, B, wo, rng);

    s.radiance += s.throughput * direct;
    wfState[i] = s;
}
