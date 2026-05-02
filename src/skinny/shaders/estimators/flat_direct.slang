// Flat (StandardSurface) direct-light estimators: analytic directional,
// sphere area lights with NEE+MIS, and emissive triangle area light.
// Bodies lifted verbatim from the original flat_material.slang inline
// helpers (flatDirectContribution / flatSphereLightsContribution /
// flatEmissiveTrianglesContribution).

import common;
import bindings;
import scene_trace;
import scene_lights;
import mtlx_closures;
import mtlx_std_surface;
import flat_shading;

float3 flatSphereLightsEstimator(
    HitInfo h, float3 N, float3 V, StdSurfaceParams sp,
    FrameConstants fc, inout RNG rng)
{
    float3 total = float3(0.0);
    if (fc.numSphereLights == 0u)
        return total;
    float3 T = h.hasTangent ? h.tangent : float3(1, 0, 0);
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

        float NdotL = max(dot(N, dir), 0.0);
        float cosLight = max(dot(lN, -dir), 0.0);
        if (NdotL <= 0.0 || cosLight <= 0.0) continue;
        if (!visibleSegment(fc, h.position, P)) continue;

        ClosureData cd = makeClosureData(CLOSURE_TYPE_REFLECTION, dir, V, N, h.position, 1.0);
        BSDF bsdfResult = evalStdSurfaceBSDF(cd, sp, N, T);

        float area = 4.0 * PI * r * r;
        float lightGeom = cosLight * area / d2;
        float pdf_light_sa = d2 / max(cosLight * area, 1e-12);
        float pdf_bsdf_sa  = stdSurfaceSamplingPdf(N, V, dir, sp);
        float w_nee = powerHeuristic(pdf_light_sa, pdf_bsdf_sa);
        total += bsdfResult.response * L.radiance * lightGeom * w_nee;
    }
    return total;
}

float3 flatEmissiveTrianglesEstimator(
    HitInfo h, float3 N, float3 V, StdSurfaceParams sp,
    FrameConstants fc, inout RNG rng)
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

    float NdotL = max(dot(N, dir), 0.0);
    float cosLight = max(dot(triN, -dir), 0.0);
    if (NdotL <= 0.0 || cosLight <= 0.0) return total;

    if (!visibleSegment(fc, h.position, P)) return total;

    float3 T = h.hasTangent ? h.tangent : float3(1, 0, 0);
    ClosureData cd = makeClosureData(CLOSURE_TYPE_REFLECTION, dir, V, N, h.position, 1.0);
    BSDF bsdfResult = evalStdSurfaceBSDF(cd, sp, N, T);

    float lightGeom = float(fc.numEmissiveTriangles) * tri.area * cosLight / d2;
    total = bsdfResult.response * tri.emission * lightGeom;
    return total;
}

float3 flatDirectEstimator(HitInfo h, float3 N, float3 V, StdSurfaceParams sp,
                           FrameConstants fc, float3 lightDirection, float3 lightRadiance)
{
    if (fc.useDirectLight == 0u)
        return float3(0.0);
    float3 L = lightDirection;
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0)
        return float3(0.0);
    if (!visibleDirectional(fc, h.position, N, L))
        return float3(0.0);

    float3 T = h.hasTangent ? h.tangent : float3(1, 0, 0);
    ClosureData cd = makeClosureData(CLOSURE_TYPE_REFLECTION, L, V, N, h.position, 1.0);
    BSDF result = evalStdSurfaceBSDF(cd, sp, N, T);
    return result.response * lightRadiance;
}
