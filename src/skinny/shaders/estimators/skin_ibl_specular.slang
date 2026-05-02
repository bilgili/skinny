// Skin IBL specular estimator (term §2 from skin_material.slang).
// Generic over ISampler — caller picks the primary sampling strategy
// (typically GGXSampler). MIS companion is always uniform sphere,
// gated by fc.integratorMode != 0.

import common;
import bindings;
import interfaces;
import skin_bssrdf;
import environment;
import skin_shading;
import samplers.uniform_sphere;
import samplers.mis_combine;

float3 skinIBLSpecularEstimator<TS : ISampler>(
    TS sampler, SkinEvalContext ctx, float3 V,
    bool volumeOnly, FrameConstants fc, inout RNG rng)
{
    if (volumeOnly)
        return float3(0.0);

    UniformSphereSampler uniSampler;
    float uniPdf = uniSampler.pdf(float3(0.0));

    float3 primaryDir = sampler.sampleDirection(rng.next2());
    float  primaryPdf = sampler.pdf(primaryDir);
    float  NdotL      = max(dot(ctx.Nspec, primaryDir), 0.0);

    float3 envSpec = float3(0.0);
    if (NdotL > 0.0 && primaryPdf > 1e-12)
    {
        float3 brdf   = evaluateSkinSpecular(ctx.Nspec, V, primaryDir, ctx.roughSpec, ctx.sp.ior);
        float3 envRad = sceneEnvRadiance(primaryDir, fc.envIntensity, fc.furnaceMode);
        float  wMis   = fc.integratorMode != 0u
            ? misPrimaryWeight(sampler, uniSampler, primaryDir)
            : 1.0;
        envSpec = brdf * NdotL * envRad * wMis / primaryPdf;
    }

    if (fc.integratorMode != 0u)
    {
        float3 specLightDir = uniSampler.sampleDirection(rng.next2());
        float  specNdotL    = max(dot(ctx.Nspec, specLightDir), 0.0);
        if (specNdotL > 0.0)
        {
            float  wMis   = misCompanionWeight(sampler, uniSampler, specLightDir);
            float3 specEnv = sceneEnvRadiance(specLightDir, fc.envIntensity, fc.furnaceMode);
            envSpec += evaluateSkinSpecular(ctx.Nspec, V, specLightDir, ctx.roughSpec, ctx.sp.ior)
                     * specEnv * specNdotL * (wMis / uniPdf);
        }
    }

    return envSpec;
}
