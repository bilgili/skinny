// Skin IBL specular estimator (term S2 from skin_material.slang).
// Generic over ISampler — caller picks the primary sampling strategy
// (typically GGXSampler). MIS with uniform sphere companion.

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

    float3 N = ctx.Nspec;
    float3 T, B;
    buildBasis(N, T, B);
    float3 wo = worldToTangent(V, T, B, N);

    UniformSphereSampler uniSampler;
    float uniPdf = uniSampler.pdf(float3(0, 0, 1), wo);

    float3 wi        = sampler.sample(wo, rng.next2());
    float3 primaryDir = tangentToWorld(wi, T, B, N);
    float  primaryPdf = sampler.pdf(wi, wo);
    float  NdotL      = max(dot(N, primaryDir), 0.0);

    float3 envSpec = float3(0.0);
    if (NdotL > 0.0 && primaryPdf > 1e-12)
    {
        float3 brdf   = evaluateSkinSpecular(N, V, primaryDir, ctx.roughSpec, ctx.sp.ior);
        float3 envRad = sceneEnvRadiance(primaryDir, fc.envIntensity, effectiveFurnaceMode(ctx.materialId));
        float  wMis   = misPrimaryWeight(sampler, uniSampler, wi, wo);
        envSpec = brdf * NdotL * envRad * wMis / primaryPdf;
    }

    // MIS companion arm (uniform sphere)

    {
        float3 wiUni       = uniSampler.sample(wo, rng.next2());
        float3 specLightDir = tangentToWorld(wiUni, T, B, N);
        float  specNdotL    = max(dot(N, specLightDir), 0.0);
        if (specNdotL > 0.0)
        {
            float  wMis   = misCompanionWeight(sampler, uniSampler, wiUni, wo);
            float3 specEnv = sceneEnvRadiance(specLightDir, fc.envIntensity, effectiveFurnaceMode(ctx.materialId));
            envSpec += evaluateSkinSpecular(N, V, specLightDir, ctx.roughSpec, ctx.sp.ior)
                     * specEnv * specNdotL * (wMis / uniPdf);
        }
    }

    return envSpec;
}
