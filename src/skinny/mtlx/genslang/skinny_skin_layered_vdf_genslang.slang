// Layered scattering VDF for the delta-tracked volume estimator.
//
// MaterialX VDFs are intentionally one-shot: given a slab and a path
// segment, return (response, throughput). Skinny's full estimator is a
// multi-bounce delta-tracking random walk through a depth-stratified
// medium with Henyey-Greenstein phase per layer (volume_render.slang).
// That walk is owned by the integrator, not the VDF — it cannot be
// expressed as a single closure call.
//
// Phase A delivers the structural shape and a Beer-Lambert throughput
// approximation using the τ-weighted layer extinction, so the
// MaterialX shader generator round-trips through this nodedef. The
// integrator continues to use volume_render.slang directly until
// Phase E side-channels the per-layer state into the renderer's volume
// pass.

#include "lib/mx_closure_type.glsl"

void skinny_skin_layered_vdf(
    ClosureData closureData,

    float3 top_absorption,
    float3 top_scattering,
    float  top_anisotropy,
    float  top_thickness,
    float  top_ior,

    float3 middle_absorption,
    float3 middle_scattering,
    float  middle_anisotropy,
    float  middle_thickness,
    float  middle_ior,

    float3 bottom_absorption,
    float3 bottom_scattering,
    float  bottom_anisotropy,
    float  bottom_thickness,
    float  bottom_ior,

    float4 pigment,

    inout VDF vdf)
{
    // τ-weighted blend of layer extinctions (mm^-1) over the total slab.
    float3 sigmaT_t = top_absorption    + top_scattering;
    float3 sigmaT_m = middle_absorption + middle_scattering;
    float3 sigmaT_b = bottom_absorption + bottom_scattering;

    float totalThickness = max(top_thickness + middle_thickness + bottom_thickness, 1e-6);
    float3 sigmaT =
        (sigmaT_t * top_thickness
       + sigmaT_m * middle_thickness
       + sigmaT_b * bottom_thickness)
        / totalThickness;

    if (closureData.closureType == CLOSURE_TYPE_TRANSMISSION)
    {
        // Beer-Lambert through the full slab in mm units. The integrator
        // converts world-units distance to mm via the scene scale.
        vdf.throughput = exp(-sigmaT * totalThickness);
        vdf.response   = float3(0.0);
    }
    else
    {
        vdf.throughput = float3(1.0);
        vdf.response   = float3(0.0);
    }
}
