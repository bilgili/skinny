// Skin thin-geometry translucency estimator (term §5). Front-back
// hit pair from scene_trace produces a skin-slab thickness; if smaller
// than (epi+derm+subcut), the back side gets diffusion-weighted direct
// + cosine-IBL contribution.

import common;
import bindings;
import scene_trace;
import scene_lights;
import environment;
import skin_bssrdf;
import skin_shading;

float3 skinTransmissionEstimator(
    HitInfo hit, SkinEvalContext ctx,
    bool useDirect, float3 analyticLightDir, float3 lightRadiance,
    FrameConstants fc, inout RNG rng)
{
    float3 transmission = float3(0.0);
    if (hit.backT > 0.0 && effectiveFurnaceMode(hit.materialId) == 0u)
    {
        float localThicknessMM = (hit.backT - hit.t) * fc.mmPerUnit;
        float skinSlabMM = ctx.sp.epidermisThickness + ctx.sp.dermisThickness + ctx.sp.subcutThickness;
        if (localThicknessMM < skinSlabMM)
        {
            SkinLayer epidermis = buildEpidermis(ctx.sp);
            SkinLayer dermis    = buildDermisInked(ctx.sp, ctx.ink);
            float3 sigmaSprime_e = epidermis.sigmaS * (1.0 - epidermis.g);
            float3 sigmaSprime_d = dermis.sigmaS    * (1.0 - dermis.g);
            float3 sigmaTr_e = sqrt(3.0 * epidermis.sigmaA * (epidermis.sigmaA + sigmaSprime_e));
            float3 sigmaTr_d = sqrt(3.0 * dermis.sigmaA    * (dermis.sigmaA    + sigmaSprime_d));
            float wE = ctx.sp.epidermisThickness;
            float wD = ctx.sp.dermisThickness;
            float3 sigmaTr = (sigmaTr_e * wE + sigmaTr_d * wD) / max(wE + wD, 1e-6);
            float3 tr = exp(-sigmaTr * localThicknessMM);
            float3 backN = -hit.normal;
            if (useDirect)
            {
                float backNdotL = max(dot(backN, analyticLightDir), 0.0);
                transmission += tr * lightRadiance * backNdotL;
            }
            float3 backDiffDir = sampleCosineHemisphere(rng.next2(), backN);
            float3 backEnv     = sceneEnvRadiance(backDiffDir, fc.envIntensity, effectiveFurnaceMode(hit.materialId));
            transmission += tr * backEnv;
            transmission /= PI;
        }
    }
    return transmission;
}
