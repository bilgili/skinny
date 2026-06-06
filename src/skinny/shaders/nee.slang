// Next-event estimation (direct lighting) — material-free so per-material
// wavefront shade kernels can import NEE without pulling in the whole material
// tree (skin/python). The estimator is generic over the material via IMaterial
// (mat.evaluate), instantiated at the call site. Extracted verbatim from
// integrators/path.slang so the megakernel and the wavefront shade share one
// definition.

import common;        // RNG, worldToTangent, powerHeuristic
import bindings;      // fc, distantLights / sphereLights / emissiveTriangles, effectiveFurnaceMode
import interfaces;    // IMaterial, ILight, BSDFSample, LightSample, HitInfo
import scene_trace;   // visibleDirectional, visibleSegment
import environment;   // sampleEnvDir, EnvSample
import scene_lights;
import lights.sphere_light;
import lights.emissive_triangle_light;
import lights.directional_light;
// Proposal-mixture pdf so NEE's MIS companion weight matches the bounce
// sampler (the NEE-coupling rule). BSDF-only ⇒ the plain bsdf pdf.
import sampling.proposal;

// ── Generic NEE ─────────────────────────────────────────────────

float3 neeLightEstimator<TM : IMaterial, TL : ILight>(
    TM mat, TL light, HitInfo h, float3 N, float3 T, float3 B,
    float3 wo, bool neuralActive, FrameConstants fc, float2 u)
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
    BSDFSample bsdf = mat.evaluate(wo, wi);

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
    // MIS companion = the bounce proposal mixture's pdf for this direction,
    // not the bare bsdf pdf — otherwise enabling a non-BSDF proposal biases
    // direct lighting. BSDF-only mixture ⇒ exactly bsdf.pdf (parity).
    ProposalContext pc;
    pc.woT = wo; pc.N = N; pc.T = T; pc.B = B;
    pc.position = h.position; pc.materialId = h.materialId;
    pc.neuralValid = neuralActive;   // include the neural term iff this lane uses neural
    float misPdf = mixtureProposalPdf(mat, pc, wi);
    float wNEE   = powerHeuristic(pdfLightSA, misPdf);

    return bsdf.response * ls.radiance * invPdfOmega * wNEE;
}

float3 allLightsNEE<TM : IMaterial>(
    TM mat, HitInfo h, float3 N, float3 T, float3 B,
    float3 wo, bool neuralActive, inout RNG rng)
{
    float3 total = float3(0.0);

    for (uint d = 0u; d < fc.numDistantLights; d++)
    {
        DirectionalLightImpl dirLight = loadDirectionalLightImpl(distantLights[d]);
        total += neeLightEstimator(mat, dirLight, h, N, T, B, wo, neuralActive, fc, float2(0.0));
    }

    for (uint i = 0u; i < fc.numSphereLights; i++)
    {
        SphereLightImpl sl = loadSphereLightImpl(sphereLights[i]);
        total += neeLightEstimator(mat, sl, h, N, T, B, wo, neuralActive, fc, rng.next2());
    }

    if (fc.numEmissiveTriangles > 0u)
    {
        uint idx = min(uint(rng.next() * float(fc.numEmissiveTriangles)),
                       fc.numEmissiveTriangles - 1u);
        EmissiveTriangleLightImpl el = loadEmissiveTriangleLightImpl(
            emissiveTriangles[idx], fc.numEmissiveTriangles);
        total += neeLightEstimator(mat, el, h, N, T, B, wo, neuralActive, fc, rng.next2());
    }

    // Environment NEE. Importance-sample the env distribution and combine
    // with the BSDF-sampled env-miss contribution via the power heuristic.
    // Skipped under furnace mode: the env is constant white there, so it has
    // no high-variance features and the BSDF-miss path alone suffices.
    if (effectiveFurnaceMode(h.materialId) == 0u)
    {
        EnvSample es = sampleEnvDir(rng.next2(), fc.envIntensity);
        if (es.pdf > 0.0 && dot(N, es.dir) > 0.0 &&
            visibleDirectional(fc, h.position, N, es.dir))
        {
            float3 wi = worldToTangent(es.dir, T, B, N);
            BSDFSample b = mat.evaluate(wo, wi);
            // Companion pdf = the bounce proposal mixture, so the env-miss and
            // env-NEE strategies stay MIS-consistent when an env proposal is
            // active. BSDF-only ⇒ misPdf == b.pdf (parity); the guard then
            // matches the old `b.pdf > 0` delta check.
            ProposalContext pc;
            pc.woT = wo; pc.N = N; pc.T = T; pc.B = B;
            pc.position = h.position; pc.materialId = h.materialId;
            pc.neuralValid = neuralActive;   // env-NEE companion: match the bounce mixture
            float misPdf = mixtureProposalPdf(mat, pc, wi);
            if (misPdf > 0.0)
            {
                float w = powerHeuristic(es.pdf, misPdf);
                total += b.response * es.radiance / es.pdf * w;
            }
        }
    }

    return total;
}
