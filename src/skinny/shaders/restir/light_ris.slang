// ReSTIR DI — initial RIS over the light set at the primary hit (M2).
//
// Reservoir-resamples M candidate light samples per pixel and returns the
// MIS-weighted direct-lighting estimate with deferred visibility (one shadow
// ray for the survivor). The target is the UNSHADOWED, MIS-weighted light
// contribution
//     p̂ = luminance(f·NdotL · Le · wNEE),  wNEE = powerHeuristic(pdfLightSA, bsdfPdf)
// so it composes with the shade kernel's still-active BSDF-sampled direct half
// exactly like stock NEE — no shade gating, and it CONVERGES to reuse=none.
//
// M2-first covers the environment (the demo's area light) + directional (delta,
// added as plain NEE outside the RIS). Sphere / emissive-triangle candidates
// follow the identical pattern (pick a technique ∝ 1/T, sample, p̂, sourcePdf =
// perLightPdfSA / T, store a reproducible ref) and slot into the same place.

import common;          // RNG, worldToTangent, powerHeuristic
import bindings;        // fc, distantLights, effectiveFurnaceMode
import interfaces;      // IMaterial, BSDFSample, HitInfo, LightSample
import scene_trace;     // visibleDirectional
import environment;     // sampleEnvDir, EnvSample
import lights.directional_light;
import restir.reservoir;

static const uint RLIGHT_ENV     = 2u;   // LightSampleRef.packed type tag (env)
static const uint RESTIR_M_LIGHT = 8u;   // initial light candidates (config UBO later)

float _luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// Reservoir RIS over the light set + deferred-visibility resolve + directional
// (delta) NEE. Returns the primary-hit direct radiance (RGB). Generic over the
// material so the wavefront flat path instantiates it with FlatMaterial.
float3 restirDirectLighting<TM : IMaterial>(
    TM mat, HitInfo h, float3 N, float3 T, float3 B, float3 wo, inout RNG rng)
{
    float3 pos = h.position;
    // Env is a resampled technique only when not in furnace mode (matches
    // allLightsNEE: furnace env is constant white, handled by the BSDF miss).
    bool envActive = (effectiveFurnaceMode(h.materialId) == 0u);
    uint techniques = envActive ? 1u : 0u;

    Reservoir r;
    reservoirReset(r);

    // ── Initial RIS: M env candidates (unshadowed, MIS-weighted target) ──
    for (uint k = 0u; k < RESTIR_M_LIGHT && techniques > 0u; k++)
    {
        float2 u = rng.next2();
        EnvSample es = sampleEnvDir(u, fc.envIntensity);
        if (es.pdf <= 0.0 || dot(N, es.dir) <= 0.0)
            continue;

        float3 wi = worldToTangent(es.dir, T, B, N);
        BSDFSample b = mat.evaluate(wo, wi);
        float wNEE = powerHeuristic(es.pdf, b.pdf);
        float3 contrib = b.response * es.radiance * wNEE;   // unshadowed, MIS-weighted
        float pHat = _luminance(contrib);
        if (pHat <= 0.0)
            continue;

        float sourcePdf = es.pdf / float(techniques);
        LightSampleRef ref;
        ref.packed = RLIGHT_ENV;
        ref.uv = u;
        reservoirUpdate(r, ref, pHat / sourcePdf, pHat, rng.next());
    }
    reservoirFinalize(r);

    // ── Resolve the survivor: one shadow ray, integrand·V·W ──
    float3 direct = float3(0.0);
    if (r.W > 0.0 && (r.y.packed & 0x3u) == RLIGHT_ENV)
    {
        EnvSample es = sampleEnvDir(r.y.uv, fc.envIntensity);   // reproduce the sample
        if (es.pdf > 0.0 && dot(N, es.dir) > 0.0 &&
            visibleDirectional(fc, pos, N, es.dir))
        {
            float3 wi = worldToTangent(es.dir, T, B, N);
            BSDFSample b = mat.evaluate(wo, wi);
            float wNEE = powerHeuristic(es.pdf, b.pdf);
            direct += b.response * es.radiance * wNEE * r.W;
        }
    }

    // ── Directional (delta) lights: plain NEE, outside the RIS (no MIS) ──
    for (uint d = 0u; d < fc.numDistantLights; d++)
    {
        DirectionalLightImpl dl = loadDirectionalLightImpl(distantLights[d]);
        LightSample ls = dl.samplePoint(pos, float2(0.0));
        if (!ls.valid)
            continue;
        float3 toL = ls.point - pos;
        float3 dir = toL / max(length(toL), 1e-8);
        if (dot(N, dir) <= 0.0 || !visibleDirectional(fc, pos, N, dir))
            continue;
        float3 wi = worldToTangent(dir, T, B, N);
        BSDFSample b = mat.evaluate(wo, wi);
        direct += b.response * ls.radiance;
    }

    return direct;
}
