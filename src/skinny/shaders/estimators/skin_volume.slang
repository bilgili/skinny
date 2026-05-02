// Skin volume estimator (term §4 from skin_material.slang). Thin wrapper
// around volumeRenderSkin that applies the BSSRDF/volume MIS-style weight
// and gates on scatterBits.

import common;
import bindings;
import volume_render;
import skin_shading;

float3 skinVolumeEstimator(
    Ray ray, HitInfo hit, SkinEvalContext ctx,
    float3 analyticLightDir, float3 analyticLightRad,
    int volumeBounces, float volumeWeight, bool useVolume,
    FrameConstants fc, inout RNG rng)
{
    if (!useVolume)
        return float3(0.0);
    return volumeRenderSkin(
        ray, hit, analyticLightDir, analyticLightRad,
        ctx.sp, ctx.ink, fc.envIntensity, fc.furnaceMode,
        fc.integratorMode >= 2u ? fc.integratorMode : 0u,
        fc.mmPerUnit, volumeBounces, rng) * volumeWeight;
}
