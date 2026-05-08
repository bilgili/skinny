// Skin direct-light estimator — generic over ILight. Handles both delta
// lights (directional, pdfArea == 0) and area lights (sphere, emissive
// triangle) through a single code path.

import common;
import bindings;
import scene_trace;
import scene_lights;
import interfaces;
import skin_bssrdf;
import skin_shading;
import lights.sphere_light;
import lights.emissive_triangle_light;
import lights.directional_light;

float3 skinLightEstimator<TL : ILight>(
    TL light, HitInfo h, float3 N, float3 V,
    SkinEvalContext ctx, FrameConstants fc, float2 u)
{
    LightSample ls = light.samplePoint(h.position, u);
    if (!ls.valid)
        return float3(0.0);

    float3 toLight = ls.point - h.position;
    float d2 = dot(toLight, toLight);
    if (d2 <= 1e-8)
        return float3(0.0);
    float3 dir = toLight / sqrt(d2);

    float NdotL = max(dot(N, dir), 0.0);
    if (NdotL <= 0.0)
        return float3(0.0);

    if (ls.pdfArea == 0.0)
    {
        if (!visibleDirectional(fc, h.position, N, dir))
            return float3(0.0);
        if (ctx.useBSSRDF)
            return evalSkinBSDF(dir, V, h.position, h.tangent, ctx) * ls.radiance;
        return evaluateSkinSpecular(ctx.Nspec, V, dir, ctx.roughSpec, ctx.sp.ior)
             * NdotL * ls.radiance;
    }

    float cosLight = max(dot(ls.normal, -dir), 0.0);
    if (cosLight <= 0.0)
        return float3(0.0);
    if (!visibleSegment(fc, h.position, ls.point))
        return float3(0.0);

    float invPdfOmega = cosLight / (d2 * ls.pdfArea);

    if (ctx.useBSSRDF)
        return evalSkinBSDF(dir, V, h.position, h.tangent, ctx)
             * ls.radiance * invPdfOmega;
    return evaluateSkinSpecular(ctx.Nspec, V, dir, ctx.roughSpec, ctx.sp.ior)
         * NdotL * ls.radiance * invPdfOmega;
}

float3 skinAllLightsEstimator(
    HitInfo h, float3 N, float3 V,
    SkinEvalContext ctx, bool useDirect, bool volumeOnly,
    FrameConstants fc, inout RNG rng)
{
    float3 total = float3(0.0);

    if (useDirect && !volumeOnly)
    {
        DirectionalLightImpl dirLight = loadDirectionalLightImpl(
            lightDirection, lightRadiance);
        total += skinLightEstimator(dirLight, h, N, V, ctx, fc, float2(0.0));
    }

    if (volumeOnly)
        return total;

    for (uint i = 0u; i < fc.numSphereLights; i++)
    {
        SphereLightImpl sl = loadSphereLightImpl(sphereLights[i]);
        total += skinLightEstimator(sl, h, N, V, ctx, fc, rng.next2());
    }

    if (fc.numEmissiveTriangles > 0u)
    {
        uint idx = min(uint(rng.next() * float(fc.numEmissiveTriangles)),
                       fc.numEmissiveTriangles - 1u);
        EmissiveTriangleLightImpl el = loadEmissiveTriangleLightImpl(
            emissiveTriangles[idx], fc.numEmissiveTriangles);
        total += skinLightEstimator(el, h, N, V, ctx, fc, rng.next2());
    }

    return total;
}
