// Reuse seam for direct lighting (the scene-sampling reuse hook, realized).
//
// Identity (reuseMode 0) forwards to stock NEE at every vertex — bit-identical
// to the pre-ReSTIR renderer. ReSTIR DI (reuseMode 1, wavefront-only) supplies
// primary-hit direct via separate reservoir passes; at the primary vertex
// (depth 0) the inline estimate is GATED to zero so it isn't double-counted,
// and the ReSTIR resolve pass injects the direct radiance instead. depth ≥ 1
// always uses stock NEE.
//
// The reservoir passes land in later commits. Until then reuseMode stays 0
// (the reuse selector only offers "None"), so the gate below is inert and
// reuseDirect always forwards — the routing is parity-preserving.

import common;        // RNG, HitInfo
import interfaces;    // IMaterial
import bindings;      // fc.reuseMode
import nee;           // allLightsNEE

static const uint REUSE_IDENTITY  = 0u;
static const uint REUSE_RESTIR_DI = 1u;

// Direct lighting at a shading vertex, routed through the reuse seam.
// `neuralActive` flags a lane whose bounce mixture includes the neural proposal
// (a flat wavefront lane); NEE's MIS companion then includes the neural term so
// direct + indirect stay coupled. Always false on the megakernel and non-flat
// lanes (neural is wavefront flat-only).
// DOC:gate#body start
float3 reuseDirect<TM : IMaterial>(
    TM mat, HitInfo h, float3 N, float3 T, float3 B, float3 wo,
    uint depth, bool neuralActive, inout RNG rng)
{
    // Primary-hit direct owned by ReSTIR: gate the inline estimate at the
    // primary vertex so the reservoir resolve pass is its sole source.
    if (fc.reuseMode == REUSE_RESTIR_DI && depth == 0u)
        return float3(0.0);
    return allLightsNEE(mat, h, N, T, B, wo, neuralActive, rng);
}
// DOC:gate#body end
