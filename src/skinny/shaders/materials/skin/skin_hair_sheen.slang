// Vellus hair sheen estimator (term §6). Anisotropic rim BRDF — analytic
// and IBL components. Reuses envIrradiance from the IBL diffuse estimator
// to keep RNG order matching Phase 1.

import common;
import bindings;
import materials.skin.skin_bssrdf;
import materials.skin.detail;
import materials.skin.skin_shading;

struct SkinHairContribution
{
    float3 direct;
    float3 envSheen;
};

SkinHairContribution skinHairSheenEstimator(
    float3 N, float3 V, SkinEvalContext ctx,
    bool useDirect, bool volumeOnly,
    float3 analyticLightDir, float3 analyticLightRad,
    float3 envIrradiance, FrameConstants fc)
{
    SkinHairContribution out_;
    out_.direct   = float3(0.0);
    out_.envSheen = float3(0.0);
    if (!volumeOnly && effectiveFurnaceMode(ctx.materialId) == 0u && ctx.sp.hairDensity > 1e-3)
    {
        float3 hairBRDF = vellusHairBRDF(N, V, float3(1.0), ctx.sp.hairDensity, ctx.sp.hairTilt);
        if (useDirect)
        {
            float NdotLh = max(dot(N, analyticLightDir), 0.0);
            out_.direct  = hairBRDF * NdotLh * analyticLightRad;
        }
        out_.envSheen = PI * hairBRDF * envIrradiance;
    }
    return out_;
}
