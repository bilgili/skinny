// Unidirectional path tracer with MIS. Owns the bounce loop, Russian
// roulette, NEE, and BSDF-path sphere-light MIS. Materials provide
// BSDF operations via IMaterial in tangent space (N=+Z); this module
// handles world↔tangent conversions. Dispatch is tag-switch monomorphized
// through evaluateBounce() — no existential types.

import common;
import bindings;
import interfaces;
import scene_trace;
import materials.flat.flat_shading;
import scene_lights;
import environment;
import materials.flat.flat_material;
import materials.skin.skin_material;
import materials.debug_normal_material;
import python_materials_dispatcher;
import lights.sphere_light;
import lights.emissive_triangle_light;
import lights.directional_light;
// NEE (neeLightEstimator / allLightsNEE) extracted to a material-free module so
// per-material wavefront shade kernels can reuse it without the material tree.
import nee;
// Pluggable directional-proposal seam — the bounce draws its next direction
// through sampleBounceDirection() (BSDF-only ⇒ identical to mat.sample()).
import sampling.proposal;

static const uint MAX_BOUNCES = 6;

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

            ProposalContext pctx;
            pctx.woT = wo; pctx.N = N; pctx.T = T; pctx.B = B;
            pctx.position = h.position; pctx.materialId = h.materialId;
            BSDFSample bs = sampleBounceDirection(mat, pctx, rng);
            if (bs.valid)
                bs.wi = tangentToWorld(bs.wi, T, B, N);
            br.bsdfSample = bs;
            break;
        }
        case MATERIAL_TYPE_PYTHON:
        {
            uint pyId = pythonMaterialId(h.materialId);
            PythonMaterial mat = loadPythonMaterial(h, pyId);

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

            ProposalContext pctx;
            pctx.woT = wo; pctx.N = N; pctx.T = T; pctx.B = B;
            pctx.position = h.position; pctx.materialId = h.materialId;
            BSDFSample bs = sampleBounceDirection(mat, pctx, rng);
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

        // BSDF pdf of the bounce that spawned the current ray, for MIS against
        // env NEE on a miss. -1 ⇒ no env-NEE counterpart (primary ray, delta
        // lobe, transmitted, or furnace) ⇒ take the env at full weight.
        float misBsdfPdf = -1.0;

        // Skip cutout-transparent surfaces before the bounce loop begins.
        // Offset along the ray direction (not -normal) so back-facing
        // double-sided quads are cleared correctly.
        for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
        {
            r.origin = h.position + r.direction * 0.002;
            h = traceScene(fc, r);
        }


        for (uint bounce = 0u; bounce < MAX_BOUNCES; bounce++)
        {
            if (!h.hit)
            {
                float3 envR = sceneEnvRadianceCell(
                    r.direction, fc.envIntensity,
                    effectiveFurnaceMode(h.materialId));
                float w = 1.0;
                if (misBsdfPdf > 0.0)      // env NEE also sampled this ⇒ MIS
                    w = powerHeuristic(misBsdfPdf, envPdf(r.direction));
                radiance += throughput * envR * w;
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

            // Will the env-miss at the next bounce have an env-NEE partner to
            // MIS against? Only when this surface ran env NEE (not furnace),
            // the lobe is non-delta (pdf > 0), and the ray reflected (not
            // transmitted — env NEE only samples the upper hemisphere).
            bool envNeeActive = (effectiveFurnaceMode(h.materialId) == 0u);
            misBsdfPdf = (envNeeActive && !br.bsdfSample.transmitted
                          && br.bsdfSample.pdf > 0.0)
                       ? br.bsdfSample.pdf : -1.0;

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

            for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
            {
                r.origin = h.position + r.direction * 0.002;
                h = traceScene(fc, r);
            }
        }

        return radiance;
    }
}
