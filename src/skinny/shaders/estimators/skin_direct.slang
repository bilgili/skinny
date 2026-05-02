// Skin direct-light estimators: analytic directional, sphere area lights,
// emissive triangles. Extracted from skin_material.slang term §1. Bodies
// are byte-identical to the original inline blocks — orchestrator calls
// them in the same RNG order so output stays pixel-identical to Phase 1.

import common;
import bindings;
import scene_trace;
import scene_lights;
import skin_bssrdf;
import skin_shading;

// Analytic directional light (BSSRDF or specular branch).
float3 skinAnalyticDirectEstimator(
    HitInfo hit, SkinEvalContext ctx,
    float3 N, float3 V,
    bool useDirect, bool volumeOnly,
    float3 analyticLightDir, float3 analyticLightRad)
{
    if (volumeOnly || !useDirect)
        return float3(0.0);

    if (ctx.useBSSRDF)
    {
        return evalSkinBSDF(
            analyticLightDir, V, hit.position, hit.tangent, ctx
        ) * analyticLightRad;
    }

    float NdotL = max(dot(N, analyticLightDir), 0.0);
    return evaluateSkinSpecular(
        ctx.Nspec, V, analyticLightDir, ctx.roughSpec, ctx.sp.ior
    ) * analyticLightRad * NdotL;
}

float3 skinSphereLightsEstimator(
    HitInfo h, float3 N, float3 V,
    SkinEvalContext ctx, FrameConstants fc, inout RNG rng)
{
    float3 total = float3(0.0);
    if (fc.numSphereLights == 0u)
        return total;
    for (uint i = 0u; i < fc.numSphereLights; i++)
    {
        SphereLight L = sphereLights[i];
        float r = max(L.radius, 1e-4);
        float2 u  = rng.next2();
        float z   = 2.0 * u.x - 1.0;
        float phi = 2.0 * PI * u.y;
        float st  = sqrt(max(0.0, 1.0 - z * z));
        float3 lN = float3(st * cos(phi), st * sin(phi), z);
        float3 P  = L.position + r * lN;

        float3 toLight = P - h.position;
        float d2 = dot(toLight, toLight);
        if (d2 <= 1e-8) continue;
        float dist = sqrt(d2);
        float3 dir = toLight / dist;
        float NdotL    = max(dot(N, dir), 0.0);
        float cosLight = max(dot(lN, -dir), 0.0);
        if (NdotL <= 0.0 || cosLight <= 0.0) continue;
        if (!visibleSegment(fc, h.position, P)) continue;

        float area = 4.0 * PI * r * r;
        float geom = cosLight * area / d2;

        if (ctx.useBSSRDF)
        {
            float3 response = evalSkinBSDF(dir, V, h.position, h.tangent, ctx);
            total += response * L.radiance * geom;
        }
        else
        {
            float3 spec = evaluateSkinSpecular(
                ctx.Nspec, V, dir, ctx.roughSpec, ctx.sp.ior) * NdotL;
            total += spec * L.radiance * geom;
        }
    }
    return total;
}

float3 skinEmissiveTrianglesEstimator(
    HitInfo h, float3 N, float3 V,
    SkinEvalContext ctx, FrameConstants fc, inout RNG rng)
{
    float3 total = float3(0.0);
    if (fc.numEmissiveTriangles == 0u)
        return total;

    uint idx = min(uint(rng.next() * float(fc.numEmissiveTriangles)),
                   fc.numEmissiveTriangles - 1u);
    EmissiveTriangle tri = emissiveTriangles[idx];

    float2 u = rng.next2();
    float su = sqrt(u.x);
    float b0 = 1.0 - su;
    float b1 = u.y * su;
    float3 P = tri.v0 * b0 + tri.v1 * b1 + tri.v2 * (1.0 - b0 - b1);

    float3 e1 = tri.v1 - tri.v0;
    float3 e2 = tri.v2 - tri.v0;
    float3 triN = normalize(cross(e1, e2));

    float3 toLight = P - h.position;
    float d2 = dot(toLight, toLight);
    if (d2 <= 1e-8) return total;
    float dist = sqrt(d2);
    float3 dir = toLight / dist;

    float NdotL    = max(dot(N, dir), 0.0);
    float cosLight = max(dot(triN, -dir), 0.0);
    if (NdotL <= 0.0 || cosLight <= 0.0) return total;
    if (!visibleSegment(fc, h.position, P)) return total;

    float geom = float(fc.numEmissiveTriangles) * tri.area * cosLight / d2;

    if (ctx.useBSSRDF)
    {
        float3 response = evalSkinBSDF(dir, V, h.position, h.tangent, ctx);
        total = response * tri.emission * geom;
    }
    else
    {
        float3 spec = evaluateSkinSpecular(
            ctx.Nspec, V, dir, ctx.roughSpec, ctx.sp.ior) * NdotL;
        total = spec * tri.emission * geom;
    }
    return total;
}
