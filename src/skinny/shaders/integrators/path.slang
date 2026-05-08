// Unidirectional path tracer with MIS. Owns the bounce loop, Russian
// roulette, NEE, and BSDF-path sphere-light MIS. Materials provide
// BSDF operations via IMaterial in tangent space (N=+Z); this module
// handles world↔tangent conversions. Dispatch is tag-switch monomorphized
// through evaluateBounce() — no existential types.

import common;
import bindings;
import interfaces;
import scene_trace;
import flat_shading;
import scene_lights;
import environment;
import flat_material;
import skin_material;
import debug_normal_material;
import lights.sphere_light;
import lights.emissive_triangle_light;
import lights.directional_light;

static const uint MAX_BOUNCES = 6;

// ── Generic NEE ─────────────────────────────────────────────────

float3 neeLightEstimator<TM : IMaterial, TL : ILight>(
    TM mat, TL light, HitInfo h, float3 N, float3 T, float3 B,
    float3 wo, FrameConstants fc, float2 u)
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

    float3 wi = worldToTangent(dir, T, B, N);
    BSDFEval bsdf = mat.evaluate(wo, wi);

    if (ls.pdfArea == 0.0)
    {
        if (!visibleDirectional(fc, h.position, N, dir))
            return float3(0.0);
        return bsdf.response * ls.radiance;
    }

    float cosLight = max(dot(ls.normal, -dir), 0.0);
    if (cosLight <= 0.0)
        return float3(0.0);
    if (!visibleSegment(fc, h.position, ls.point))
        return float3(0.0);

    float invPdfOmega = cosLight / (d2 * ls.pdfArea);
    float pdfLightSA  = d2 * ls.pdfArea / max(cosLight, 1e-12);
    float wNEE        = powerHeuristic(pdfLightSA, bsdf.pdf);

    return bsdf.response * ls.radiance * invPdfOmega * wNEE;
}

float3 allLightsNEE<TM : IMaterial>(
    TM mat, HitInfo h, float3 N, float3 T, float3 B,
    float3 wo, inout RNG rng)
{
    float3 total = float3(0.0);

    if (fc.useDirectLight != 0u)
    {
        DirectionalLightImpl dirLight = loadDirectionalLightImpl(
            lightDirection, lightRadiance);
        total += neeLightEstimator(mat, dirLight, h, N, T, B, wo, fc, float2(0.0));
    }

    for (uint i = 0u; i < fc.numSphereLights; i++)
    {
        SphereLightImpl sl = loadSphereLightImpl(sphereLights[i]);
        total += neeLightEstimator(mat, sl, h, N, T, B, wo, fc, rng.next2());
    }

    if (fc.numEmissiveTriangles > 0u)
    {
        uint idx = min(uint(rng.next() * float(fc.numEmissiveTriangles)),
                       fc.numEmissiveTriangles - 1u);
        EmissiveTriangleLightImpl el = loadEmissiveTriangleLightImpl(
            emissiveTriangles[idx], fc.numEmissiveTriangles);
        total += neeLightEstimator(mat, el, h, N, T, B, wo, fc, rng.next2());
    }

    return total;
}

// ── Per-bounce material dispatch ────────────────────────────────

BounceResult evaluateBounce(HitInfo h, Ray r, uint bounce, inout RNG rng)
{
    uint matFlags = materialTypes[h.materialId];
    uint matType  = matFlags & 0xFFu;
    float3 V      = -r.direction;

    BounceResult br;
    br.directLight  = float3(0.0);
    br.fullRadiance = float3(0.0);
    br.bsdfSample.valid       = false;
    br.bsdfSample.transmitted = false;
    br.bsdfSample.pdf         = 0.0;
    br.bsdfSample.emission    = float3(0.0);
    br.bsdfSample.wi          = float3(0.0);
    br.bsdfSample.weight      = float3(0.0);

    switch (matType)
    {
        case MATERIAL_TYPE_FLAT:
        {
            FlatMaterial mat = loadFlatMaterial(h);

            float3 N = mat.data.N;
            float3 T, B;
            if (h.hasTangent)
            {
                T = normalize(h.tangent - N * dot(h.tangent, N));
                B = cross(N, T);
            }
            else
            {
                buildBasis(N, T, B);
            }
            float3 wo = worldToTangent(V, T, B, N);

            br.directLight = allLightsNEE(mat, h, N, T, B, wo, rng);

            BSDFSample bs = mat.sample(wo, rng);
            if (bs.valid)
                bs.wi = tangentToWorld(bs.wi, T, B, N);
            br.bsdfSample = bs;
            break;
        }
        case MATERIAL_TYPE_SKIN:
        {
            uint scatterBits = (matFlags >> 8) & 0x3u;
            br.fullRadiance = evalSkinRadiance(h, r, rng, scatterBits);
            break;
        }
        default:
        {
            br.fullRadiance = 0.5 + 0.5 * h.normal;
            break;
        }
    }
    return br;
}

// ── Bounce loop ─────────────────────────────────────────────────

struct PathTracer : IIntegrator
{
    float3 estimateRadiance(Ray ray, HitInfo firstHit, inout RNG rng)
    {
        float3 radiance   = float3(0.0);
        float3 throughput = float3(1.0);
        HitInfo h         = firstHit;
        Ray r             = ray;

        // Skip cutout-transparent surfaces before the bounce loop begins.
        // Offset along the ray direction (not -normal) so back-facing
        // double-sided quads are cleared correctly.
        for (uint skip = 0u; skip < 16u && h.hit && isCutoutTransparent(h); skip++)
        {
            r.origin = h.position + r.direction * 0.002;
            h = traceScene(fc, r);
        }

        for (uint bounce = 0u; bounce < MAX_BOUNCES; bounce++)
        {
            if (!h.hit)
            {
                radiance += throughput * sceneEnvRadiance(
                    r.direction, fc.envIntensity,
                    effectiveFurnaceMode(h.materialId));
                break;
            }

            BounceResult br = evaluateBounce(h, r, bounce, rng);

            radiance += throughput * br.fullRadiance;

            if (bounce == 0u || fc.numEmissiveTriangles == 0u)
                radiance += throughput * br.bsdfSample.emission;

            radiance += throughput * br.directLight;

            if (!br.bsdfSample.valid)
                break;

            throughput *= br.bsdfSample.weight;

            if (bounce > 0u)
            {
                float p = clamp(
                    max(max(throughput.x, throughput.y), throughput.z),
                    0.05, 0.95);
                if (rng.next() > p)
                    break;
                throughput /= p;
            }

            float3 spawnOffset = br.bsdfSample.transmitted
                ? -h.normal * 0.002
                :  h.normal * 0.002;
            r.origin    = h.position + spawnOffset;
            r.direction = br.bsdfSample.wi;

            NearestSphereHit sph = intersectSphereLights(
                r, 1e30, fc.numSphereLights);
            HitInfo hNext = traceScene(fc, r);
            float geom_t = hNext.hit ? hNext.t : 1e30;
            if (sph.idx != 0xFFFFFFFFu && sph.t < geom_t)
            {
                SphereLight light = sphereLights[sph.idx];
                float w_bsdf = 1.0;
                // Non-delta reflective bounce: NEE also samples this light,
                // so combine via power heuristic. Delta lobes (pdf == 0)
                // and transmitted paths have no NEE counterpart for this
                // direction — take the BSDF estimate at full weight.
                if (!br.bsdfSample.transmitted && br.bsdfSample.pdf > 0.0)
                {
                    float area = 4.0 * PI * light.radius * light.radius;
                    float pdf_light_sa = (sph.t * sph.t) /
                        max(sph.cos_y * area, 1e-12);
                    w_bsdf = powerHeuristic(br.bsdfSample.pdf, pdf_light_sa);
                }
                radiance += throughput * light.radiance * w_bsdf;
                break;
            }
            h = hNext;

            for (uint skip = 0u; skip < 16u && h.hit && isCutoutTransparent(h); skip++)
            {
                r.origin = h.position + r.direction * 0.002;
                h = traceScene(fc, r);
            }
        }

        return radiance;
    }
}
