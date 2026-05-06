// Skin IBL diffuse estimator (term §3 from skin_material.slang).
// Generic over ISampler — caller picks the primary diffuse sampling
// strategy (typically LambertSampler). Returns envIrradiance + diffDir
// so the hair-sheen estimator (term §6) can reuse them.

import common;
import bindings;
import interfaces;
import skin_bssrdf;
import environment;
import skin_shading;
import samplers.uniform_sphere;
import samplers.mis_combine;

struct SkinIBLDiffuseResult
{
    float3 envSSS;
    float3 envIrradiance;
    float3 diffDir;
};

float3 _evalDiffuseBSSRDF(
    SkinEvalContext ctx, HitInfo hit, float3 N,
    float3 envRad, float bssrdfWeight)
{
    return evaluateSubsurfaceScattering(ctx.sp, hit.position, N, 0.0, envRad, ctx.ink)
         * PI * bssrdfWeight;
}

SkinIBLDiffuseResult skinIBLDiffuseEstimator<TS : ISampler>(
    TS sampler, SkinEvalContext ctx, HitInfo hit, float3 N,
    float bssrdfWeight, FrameConstants fc, inout RNG rng)
{
    SkinIBLDiffuseResult r;
    r.diffDir       = sampler.sampleDirection(rng.next2());
    r.envIrradiance = sceneEnvRadiance(r.diffDir, fc.envIntensity, effectiveFurnaceMode(hit.materialId));

    UniformSphereSampler uniSampler;
    float uniPdf = uniSampler.pdf(float3(0.0));

    r.envSSS = ctx.useBSSRDF
        ? _evalDiffuseBSSRDF(ctx, hit, N, r.envIrradiance, bssrdfWeight)
              * (fc.integratorMode != 0u
                    ? misPrimaryWeight(sampler, uniSampler, r.diffDir)
                    : 1.0)
        : float3(0.0);

    if (ctx.useBSSRDF && fc.integratorMode != 0u)
    {
        float3 diffLightDir = uniSampler.sampleDirection(rng.next2());
        float  diffNdotL    = max(dot(N, diffLightDir), 0.0);
        if (diffNdotL > 0.0)
        {
            float3 diffEnv  = sceneEnvRadiance(diffLightDir, fc.envIntensity, effectiveFurnaceMode(hit.materialId));
            float  wMis     = misCompanionWeight(sampler, uniSampler, diffLightDir);
            r.envSSS += evaluateSubsurfaceScattering(ctx.sp, hit.position, N, 0.0, diffEnv, ctx.ink)
                      * diffNdotL * (wMis / uniPdf) * bssrdfWeight;
        }
    }
    return r;
}
