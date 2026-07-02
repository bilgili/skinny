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
import materials.subsurface.medium;            // Medium, resolveMedium
import materials.subsurface.subsurface_walk;   // subsurfaceRadiance (interior walk)
import materials.subsurface.volume_walk;       // volumeWalkRadiance (free-standing medium)
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
// Reuse seam — primary-hit direct routes through reuseDirect (identity ⇒
// stock allLightsNEE; ReSTIR DI gates depth-0 + injects via reservoir passes).
import sampling.reuse;

static const uint MAX_BOUNCES = 6;

// ── Per-bounce material dispatch ────────────────────────────────
//
// `h` is inout for ONE consumer: the free-standing-volume case rewrites
// h.position/h.normal to the walk's boundary EXIT point + escape direction, so
// the caller's standard `h.position + h.normal * 0.002` spawn continues the
// path from the far side of the medium (a volume is the only material whose
// continuation does not leave from the entry hit). Every other case leaves `h`
// untouched — behavior there is byte-identical.

BounceResult evaluateBounce(inout HitInfo h, Ray r, uint bounce, inout RNG rng)
{
    uint matFlags = materialTypes[h.materialId];
    uint matType  = matFlags & 0xFFu;
    float3 V      = -r.direction;

    BounceResult br;
    br.directLight  = float3(0.0);
    br.fullRadiance = float3(0.0);
    br.preserveMisState       = false;
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

            br.directLight = reuseDirect(mat, h, N, T, B, wo, bounce, false, rng);

            ProposalContext pctx;
            pctx.woT = wo; pctx.N = N; pctx.T = T; pctx.B = B;
            pctx.position = h.position; pctx.materialId = h.materialId;
            pctx.neuralValid = false;   // megakernel: neural is wavefront-only
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

            br.directLight = reuseDirect(mat, h, N, T, B, wo, bounce, false, rng);

            ProposalContext pctx;
            pctx.woT = wo; pctx.N = N; pctx.T = T; pctx.B = B;
            pctx.position = h.position; pctx.materialId = h.materialId;
            pctx.neuralValid = false;   // megakernel: neural is wavefront-only
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
        case MATERIAL_TYPE_SUBSURFACE:
        {
            // pbrt subsurface: refract through the dielectric boundary and
            // random-walk the homogeneous interior medium (subsurface_walk.slang).
            // The medium is resolved by handle from the inline FlatMaterialParams
            // pack (binding 13). Like the skin volume, the walk's NEE uses a
            // single analytic direct light — the first distant light, or zero —
            // plus the environment; it returns the full radiance and terminates
            // the bounce loop (no BSDF continuation at this vertex).
            Medium medium = resolveMedium(h.materialId);
            float3 primaryDir = float3(0.0);
            float3 primaryRad = float3(0.0);
            if (fc.numDistantLights > 0u)
            {
                DistantLight L0 = distantLights[0];
                primaryDir = L0.direction;
                primaryRad = L0.radiance;
            }
#if defined(SKINNY_WAVEFRONT)
            // Wavefront: 3D interior walk — per-segment `traceScene` to the real
            // mesh boundary, so the path length follows the geometry.
            br.fullRadiance = subsurfaceRadiance3D(
                r, h, medium, fc, primaryDir, primaryRad,
                fc.envIntensity, fc.furnaceMode, fc.mmPerUnit, rng);
#else
            // Megakernel: 1D-slab approximation — the whole walk runs in one
            // dispatch, where a per-segment BVH trace would risk the GPU watchdog.
            br.fullRadiance = subsurfaceRadiance(
                r, h, medium, primaryDir, primaryRad,
                fc.envIntensity, fc.furnaceMode, fc.mmPerUnit, rng);
#endif
            break;
        }
        case MATERIAL_TYPE_VOLUME:
        {
            // Free-standing heterogeneous medium (nanovdb-volume-rendering):
            // index-matched pass-through boundary — no Fresnel, no refraction,
            // no BSDF. The walk returns its in-walk NEE radiance (distant +
            // env, MIS-weighted) plus an escape ray; the loop applies its
            // running throughput to br.fullRadiance and continues along the
            // escape. The continuation pdf is the walk's last HG phase pdf —
            // the loop's env-miss powerHeuristic(bsdfPdf, envPdf) is then the
            // exact MIS complement of the in-walk env NEE. A zero-scatter
            // pass-through carries pdf 0 (delta ⇒ env at full weight; no env
            // NEE ran for it). Both execution modes share this one case
            // (megakernel + wavefront catch-all shade), the subsurface
            // phase-4 precedent; only the scatter caps differ.
            Medium medium = resolveMedium(h.materialId);
            float3 primaryDir = float3(0.0);
            float3 primaryRad = float3(0.0);
            if (fc.numDistantLights > 0u)
            {
                DistantLight L0 = distantLights[0];
                primaryDir = L0.direction;
                primaryRad = L0.radiance;
            }
            VolumeWalkResult vw = volumeWalkRadiance(
                r, h, medium, fc, primaryDir, primaryRad, fc.mmPerUnit, rng);
            br.fullRadiance = vw.radiance;
            if (vw.escaped)
            {
                // Rewrite the hit so the caller's spawn (h.position +
                // h.normal * 0.002) leaves from the boundary exit point along
                // the escape direction. Emission stays 0 so the caller's
                // emissive-MIS block (which reads h.normal/h.t) never fires.
                h.position = vw.escapeRay.origin;
                h.normal   = vw.escapeRay.direction;
                br.bsdfSample.valid       = true;
                br.bsdfSample.wi          = vw.escapeRay.direction; // world space
                br.bsdfSample.weight      = vw.throughput;
                br.bsdfSample.pdf         = vw.lastPhasePdf;
                br.bsdfSample.transmitted = false; // offset along +h.normal (= wi)
                // Zero-scatter pass-through (lastPhasePdf == 0): the SAME ray
                // continues, only attenuated — keep the spawning bounce's MIS
                // state instead of turning it into a delta (which took the env
                // sun-disk at full weight → ground fireflies).
                br.preserveMisState       = (vw.lastPhasePdf == 0.0);
            }
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

        // Was the current ray spawned by a delta/perfectly-specular bounce
        // (pdf <= 0 — a smooth dielectric's mirror reflect/refract)? A delta lobe
        // has no NEE partner, so an emissive triangle it lands on must be taken at
        // full weight (mirrors the sphere-light delta branch + bdpt deltaBounce).
        // False for the primary ray.
        bool spawnedBySpecular = false;

        // BSDF pdf (solid-angle) of the bounce that spawned the current ray —
        // the MIS companion for an emissive triangle this ray lands on. -1 ⇒ no
        // NEE partner (primary ray or delta lobe) ⇒ take the emission at full
        // weight. Distinct from misBsdfPdf, which is additionally gated on env NEE.
        float prevBsdfPdf = -1.0;

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

            // Emission at the current hit. Bounce 0 (camera ray), a delta-spawned
            // ray, or a scene with no emissive triangles has no NEE partner ⇒ full
            // weight. Otherwise a non-delta bounce landed on an emissive triangle
            // that allLightsNEE (power heuristic) also sampled, so add only the
            // BSDF-sampling strategy's MIS complement — otherwise this energy is
            // double-counted (old code dropped it entirely, biasing area lights
            // dim by a solid-angle-dependent factor). The NEE solid-angle pdf for
            // this hit is reconstructed without the triangle index: the per-tri
            // area cancels in the power-weighted (area·lum) CDF, leaving
            //   pdfLightSA = Rec709-lum(emission)·d² / (emissiveTotalPower·cosLight).
            // See emissive-triangle-bsdf-hit-mis.
            if (bounce == 0u || fc.numEmissiveTriangles == 0u || spawnedBySpecular)
            {
                radiance += throughput * br.bsdfSample.emission;
            }
            else if (any(br.bsdfSample.emission > 0.0))
            {
                float cosLight = max(dot(h.normal, -r.direction), 0.0);
                float lum      = dot(br.bsdfSample.emission,
                                     float3(0.2126, 0.7152, 0.0722));
                float pdfLightSA = (cosLight > 1e-6 && fc.emissiveTotalPower > 0.0)
                    ? lum * (h.t * h.t) / (fc.emissiveTotalPower * cosLight)
                    : 1e30;   // grazing / degenerate ⇒ NEE owns it ⇒ wBSDF → 0
                float wBSDF = powerHeuristic(prevBsdfPdf, pdfLightSA);
                radiance += throughput * br.bsdfSample.emission * wBSDF;
            }

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
            if (!br.preserveMisState)
            {
                misBsdfPdf = (envNeeActive && !br.bsdfSample.transmitted
                              && br.bsdfSample.pdf > 0.0)
                           ? br.bsdfSample.pdf : -1.0;

                // The ray spawned below carries this bounce's delta-ness to the
                // next iteration's emissive-triangle gate (delta ⇒ no NEE partner
                // ⇒ take emission at full weight). prevBsdfPdf carries the
                // companion pdf for the emissive-hit MIS weight (−1 for a delta
                // lobe ⇒ full weight).
                spawnedBySpecular = (br.bsdfSample.pdf <= 0.0);
                prevBsdfPdf = spawnedBySpecular ? -1.0 : br.bsdfSample.pdf;
            }
            // else: volume zero-scatter pass-through — same ray, attenuated;
            // misBsdfPdf / spawnedBySpecular / prevBsdfPdf carry over unchanged.

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
