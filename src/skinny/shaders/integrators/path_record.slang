// Offline training-record dump for the neural directional proposal (task 5.1).
//
// A second megakernel entry (`mainImageRecord`, declared in main_pass.slang)
// runs this RR-free path tracer: it traces the SAME `{bsdf,env}` paths the
// renderer would, and for every flat/graph bounce it records one training
// sample `(position, normal, wo, wiLocal, contribution)` to a flat GPU append
// buffer the host reads back (renderer.dump_path_records → a `.nrec` file).
// The offline `spline_flow` trainer fits the proposal flow to these records.
//
// Why a separate entry, not an `fc` flag in `mainImage`: keeps the default
// `mainImage` byte-identical (no new `fc` field, no dead branch, no register
// pressure, isolates the MoltenVK big-kernel limit). Why megakernel, not
// wavefront: one thread owns the WHOLE path, so the tail radiance `Li` along
// each sampled `wi` is known at loop end and attributed back trivially from a
// local register stack — in wavefront the path is smeared across kernel
// dispatches, so `Li` would need a per-lane VRAM vertex stack + a
// terminate-time splat. The dump is offline, so megakernel-only costs nothing
// at render time (inference stays wavefront-only).
//
// CONDITION ENCODING: the recorded `(position, normal, wo)` are the raw inputs
// to `sampling/neural_proposal.slang::neuralCondition`; the scene AABB lives in
// the file header (host-written from `fc.sceneBoundsMin/Extent`) so the trainer
// normalises position byte-for-byte identically. `wiLocal` is the sampled world
// direction projected onto the SAME shading frame the wavefront neural pass uses
// at inference (N = HitInfo.normal; T,B from the hit tangent or buildBasis) —
// flow-local axes (x,y,z) = (T,N,B), y-up — so train↔infer frames match exactly.

import common;        // RNG, Ray, PI, buildBasis, powerHeuristic
import bindings;      // fc, materialTypes
import interfaces;    // HitInfo, BounceResult, BSDFSample
import scene_trace;   // traceScene
import materials.flat.flat_shading;   // isCutoutTransparent
import scene_lights;  // intersectSphereLights, NearestSphereHit
import environment;   // sceneEnvRadianceCell, envPdf
import lights.sphere_light;   // SphereLight
import integrators.path;      // evaluateBounce — the shared per-bounce material dispatch
// Shared record primitives — the ONE source of truth for PathRecord, bindings
// 36/37, emitRecord, REC_MAX_BOUNCES, and the recordContrib attribution weight
// (also imported by the wavefront-native emitter wavefront/wf_records.slang).
import integrators.path_record_common;

// RR-free unidirectional path tracer that emits one training record per
// flat/graph reflective bounce. Mirrors integrators.path::PathTracer.
// estimateRadiance exactly EXCEPT: (1) no Russian roulette (deterministic
// throughput → clean per-vertex `beta_in`, full tail always traced), and
// (2) the per-vertex record stack + terminate-time backward attribution.
void estimateRadianceRecord(Ray ray, HitInfo firstHit, inout RNG rng)
{
    float3 radiance   = float3(0.0);
    float3 throughput = float3(1.0);
    HitInfo h         = firstHit;
    Ray r             = ray;
    float misBsdfPdf  = -1.0;
    // Delta-ness of the bounce that spawned the current ray (see path.slang):
    // a delta lobe has no NEE partner, so an emissive triangle it hits is taken
    // at full weight. False for the primary ray.
    bool spawnedBySpecular = false;

    // Per-vertex stack: geometry + (radiance, beta_in) snapshots for the
    // backward attribution `contrib_k = (L_final - L_k)/beta_in_k`.
    float3 recPos[REC_MAX_BOUNCES];
    float3 recN[REC_MAX_BOUNCES];
    float3 recWo[REC_MAX_BOUNCES];
    float3 recWi[REC_MAX_BOUNCES];   // flow-local
    float3 recL[REC_MAX_BOUNCES];    // radiance after this vertex's local terms (L_k)
    float3 recBeta[REC_MAX_BOUNCES]; // throughput ENTERING this vertex (beta_in_k)
    uint   recDepth[REC_MAX_BOUNCES];
    uint   recCount = 0u;

    for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
    {
        r.origin = h.position + r.direction * 0.002;
        h = traceScene(fc, r);
    }

    for (uint bounce = 0u; bounce < REC_MAX_BOUNCES; bounce++)
    {
        if (!h.hit)
        {
            float3 envR = sceneEnvRadianceCell(
                r.direction, fc.envIntensity, effectiveFurnaceMode(h.materialId));
            float w = 1.0;
            if (misBsdfPdf > 0.0)
                w = powerHeuristic(misBsdfPdf, envPdf(r.direction));
            radiance += throughput * envR * w;
            break;
        }

        uint matType = materialTypes[h.materialId] & 0xFFu;
        BounceResult br = evaluateBounce(h, r, bounce, rng);

        radiance += throughput * br.fullRadiance;
        if (bounce == 0u || fc.numEmissiveTriangles == 0u || spawnedBySpecular)
            radiance += throughput * br.bsdfSample.emission;
        radiance += throughput * br.directLight;

        if (!br.bsdfSample.valid)
            break;

        // Snapshot BEFORE the throughput update so beta_in_k = throughput
        // entering this vertex; the training weight (L_final-L_k)/beta_in_k is
        // then f·cos/pdf · Li = the integrand value per unit incoming
        // throughput (so the contribution-weighted MLE fits q ∝ f·Li·cos).
        bool guideable = (matType == MATERIAL_TYPE_FLAT || matType == MATERIAL_TYPE_PYTHON)
                       && !br.bsdfSample.transmitted && br.bsdfSample.pdf > 0.0;
        if (guideable && recCount < REC_MAX_BOUNCES)
        {
            float3 N = h.normal;
            float3 T, B;
            if (h.hasTangent) { T = normalize(h.tangent - N * dot(h.tangent, N)); B = cross(N, T); }
            else              { buildBasis(N, T, B); }
            float3 wiW = br.bsdfSample.wi;
            float3 wiL = float3(dot(wiW, T), dot(wiW, N), dot(wiW, B));
            if (wiL.y > 1e-4)   // upper hemisphere only (the flow's domain)
            {
                recPos[recCount]   = h.position;
                recN[recCount]     = N;
                recWo[recCount]    = -normalize(r.direction);
                recWi[recCount]    = wiL;
                recL[recCount]     = radiance;      // L_k (local terms added above)
                recBeta[recCount]  = throughput;    // beta_in_k (pre-update)
                recDepth[recCount] = bounce;
                recCount++;
            }
        }

        throughput *= br.bsdfSample.weight;
        // NO Russian roulette — fixed-depth, deterministic throughput.

        bool envNeeActive = (effectiveFurnaceMode(h.materialId) == 0u);
        misBsdfPdf = (envNeeActive && !br.bsdfSample.transmitted && br.bsdfSample.pdf > 0.0)
                   ? br.bsdfSample.pdf : -1.0;

        // Carry this bounce's delta-ness to the next iteration's emission gate.
        spawnedBySpecular = (br.bsdfSample.pdf <= 0.0);

        float3 spawnOffset = br.bsdfSample.transmitted ? -h.normal * 0.002 : h.normal * 0.002;
        r.origin    = h.position + spawnOffset;
        r.direction = br.bsdfSample.wi;

        NearestSphereHit sph = intersectSphereLights(r, 1e30, fc.numSphereLights);
        HitInfo hNext = traceScene(fc, r);
        float geom_t = hNext.hit ? hNext.t : 1e30;
        if (sph.idx != 0xFFFFFFFFu && sph.t < geom_t)
        {
            SphereLight light = sphereLights[sph.idx];
            float w_bsdf = 1.0;
            if (!br.bsdfSample.transmitted && br.bsdfSample.pdf > 0.0)
            {
                float area = 4.0 * PI * light.radius * light.radius;
                float pdf_light_sa = (sph.t * sph.t) / max(sph.cos_y * area, 1e-12);
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

    // Backward attribution: the shared `recordContrib` folds this vertex's
    // f·cos/pdf into the tail radiance so the stored weight ∝ the integrand the
    // proposal should match. Drop non-finite samples. The wavefront-native
    // emitter calls the SAME helper — identical training weight.
    float3 Lfinal = radiance;
    for (uint k = 0u; k < recCount; k++)
    {
        float3 contrib = recordContrib(Lfinal, recL[k], recBeta[k]);
        if (any(isnan(contrib)) || any(isinf(contrib)))
            continue;
        PathRecord rec;
        rec.pos     = recPos[k];
        rec.normal  = recN[k];
        rec.wo      = recWo[k];
        rec.wiLocal = recWi[k];
        rec.contrib = contrib;
        rec.depth   = recDepth[k];
        emitRecord(rec);
    }
}
