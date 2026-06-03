// ReSTIR DI — initial RIS over the unified light set at the primary hit.
//
// Reservoir-resamples M candidate light samples per pixel over the unified light
// domain (sphere + emissive-triangle + environment) and returns the MIS-weighted
// direct-lighting estimate with deferred visibility (one shadow ray for the
// survivor). The target is the UNSHADOWED, MIS-weighted light contribution
//     p̂ = luminance(f·Le · wNEE),  wNEE = powerHeuristic(pdfLightSA, misPdf)
// matching nee.slang's neeLightEstimator exactly, so ReSTIR re-adds the gated
// allLightsNEE light-sampled half for EVERY area/env light (the reuse seam gates
// the whole allLightsNEE to zero at depth 0). Directional lights are delta (no
// MIS) → plain NEE outside the RIS. CONVERGES to reuse=none.
//
// A reservoir sample is a reproducible reference (lightType, lightId, uv): the
// sampling u maps to a fixed world point on the light (sphere surface / triangle
// barycentric / env direction), INDEPENDENT of the shading point — so resolve
// and spatial-neighbour reuse re-derive the same light point at any pixel (the
// DI reconnection).

import common;          // RNG, worldToTangent, powerHeuristic
import bindings;        // fc, distantLights, effectiveFurnaceMode
import interfaces;      // IMaterial, BSDFSample, HitInfo, LightSample, ILight
import scene_trace;     // visibleDirectional, visibleSegment
import scene_lights;    // sphereLights, emissiveTriangles
import environment;     // sampleEnvDir, EnvSample
import lights.sphere_light;
import lights.emissive_triangle_light;
import lights.directional_light;
import sampling.proposal;   // mixtureProposalPdf (MIS companion = bounce mixture)
import restir.reservoir;

// LightSampleRef.packed: type:2 | id:30.
static const uint RLIGHT_SPHERE = 0u;
static const uint RLIGHT_TRI    = 1u;
static const uint RLIGHT_ENV    = 2u;

uint refType(LightSampleRef r) { return r.packed & 0x3u; }
uint refId(LightSampleRef r)   { return r.packed >> 2; }
LightSampleRef makeRef(uint type, uint id, float2 uv)
{
    LightSampleRef r;
    r.packed = (id << 2) | (type & 0x3u);
    r.uv = uv;
    return r;
}

float _luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// One evaluated light candidate at a shading point: the unshadowed integrand
// (f·Le·wNEE — the RGB that resolve multiplies by V·W), its target p̂, the
// light-sampling solid-angle pdf (the RIS source measure), and the geometry for
// the deferred shadow ray.
struct RCand
{
    float3 integrand;   // f·Le·wNEE (RGB)
    float  pHat;        // luminance(integrand)
    float  pdfSA;       // light-sampling solid-angle pdf
    float3 toPoint;     // world point on light (area lights) — shadow-segment end
    float3 dir;         // world-space direction to the light
    bool   isEnv;       // env (directional shadow ray) vs area (segment ray)
    bool   valid;
};

float _misPdf<TM : IMaterial>(
    TM mat, HitInfo h, float3 N, float3 T, float3 B, float3 wo, float3 wi)
{
    ProposalContext pc;
    pc.woT = wo; pc.N = N; pc.T = T; pc.B = B;
    pc.position = h.position; pc.materialId = h.materialId;
    return mixtureProposalPdf(mat, pc, wi);
}

// Evaluate a reproducible (type, id, uv) light reference at a shading point.
// Visibility is deferred (the caller casts the shadow ray). Used by fill (build
// the reservoir), resolve (the survivor), and spatial reuse (re-evaluate a
// neighbour's sample at THIS pixel — the reconnection).
RCand restirEvalRef<TM : IMaterial>(
    TM mat, uint type, uint id, float2 uv, HitInfo h,
    float3 N, float3 T, float3 B, float3 wo)
{
    RCand c;
    c.integrand = float3(0.0); c.pHat = 0.0; c.pdfSA = 0.0;
    c.toPoint = float3(0.0); c.dir = float3(0.0); c.isEnv = false; c.valid = false;

    float3 dir; float3 Le; float pdfSA;
    if (type == RLIGHT_ENV)
    {
        EnvSample es = sampleEnvDir(uv, fc.envIntensity);
        if (es.pdf <= 0.0)
            return c;
        dir = es.dir; Le = es.radiance; pdfSA = es.pdf;
        c.isEnv = true;
    }
    else
    {
        LightSample ls;
        if (type == RLIGHT_SPHERE)
            ls = loadSphereLightImpl(sphereLights[id]).samplePoint(h.position, uv);
        else
            ls = loadEmissiveTriangleLightImpl(
                     emissiveTriangles[id], fc.numEmissiveTriangles).samplePoint(h.position, uv);
        if (!ls.valid || ls.pdfArea <= 0.0)
            return c;
        float3 toL = ls.point - h.position;
        float d2 = dot(toL, toL);
        if (d2 <= 1e-8)
            return c;
        dir = toL / sqrt(d2);
        float cosLight = max(dot(ls.normal, -dir), 0.0);
        if (cosLight <= 0.0)
            return c;
        pdfSA = d2 * ls.pdfArea / cosLight;     // area → solid-angle measure
        Le = ls.radiance;
        c.toPoint = ls.point;
    }
    if (pdfSA <= 0.0 || dot(N, dir) <= 0.0)
        return c;

    float3 wi = worldToTangent(dir, T, B, N);
    BSDFSample b = mat.evaluate(wo, wi);
    float wNEE = powerHeuristic(pdfSA, _misPdf(mat, h, N, T, B, wo, wi));
    c.integrand = b.response * Le * wNEE;
    c.pHat = _luminance(c.integrand);
    c.pdfSA = pdfSA;
    c.dir = dir;
    c.valid = true;
    return c;
}

// Initial RIS over the unified light set → a Reservoir (no shadow ray; deferred
// to resolve). Each candidate picks a technique uniformly among the active ones
// (env + each sphere + the emissive-triangle set), so sourcePdf = pdfSA / nTech.
// The survivor's reproducible ref is stored for resolve / neighbour re-eval.
Reservoir restirFillReservoir<TM : IMaterial>(
    TM mat, HitInfo h, float3 N, float3 T, float3 B, float3 wo, uint mLight, inout RNG rng)
{
    bool envActive = (effectiveFurnaceMode(h.materialId) == 0u);
    uint nSphere = fc.numSphereLights;
    uint nTri    = (fc.numEmissiveTriangles > 0u) ? 1u : 0u;
    uint nTech   = nSphere + nTri + (envActive ? 1u : 0u);

    Reservoir r;
    reservoirReset(r);
    for (uint k = 0u; k < mLight && nTech > 0u; k++)
    {
        uint t = min(uint(rng.next() * float(nTech)), nTech - 1u);
        uint type; uint id; float2 uv = rng.next2();
        if (t < nSphere)
        {
            type = RLIGHT_SPHERE; id = t;
        }
        else if (t < nSphere + nTri)
        {
            type = RLIGHT_TRI;
            id = min(uint(rng.next() * float(fc.numEmissiveTriangles)),
                     fc.numEmissiveTriangles - 1u);
        }
        else
        {
            type = RLIGHT_ENV; id = 0u;
        }

        // Every drawn candidate must count toward M (RIS averages over all
        // draws), even zero-weight ones — else W is inflated. Invalid / below-
        // horizon samples stream with weight 0 (never selected, but M++).
        RCand c = restirEvalRef(mat, type, id, uv, h, N, T, B, wo);
        float w = 0.0, pHat = 0.0;
        if (c.valid && c.pHat > 0.0)
        {
            float sourcePdf = c.pdfSA / float(nTech);
            if (sourcePdf > 0.0)
            {
                pHat = c.pHat;
                w = c.pHat / sourcePdf;
            }
        }
        reservoirUpdate(r, makeRef(type, id, uv), w, pHat, rng.next());
    }
    reservoirFinalize(r);
    return r;
}

// Resolve a reservoir at a shading point: one shadow ray for the survivor →
// integrand·V·W (RGB), over any light type.
float3 restirResolveReservoir<TM : IMaterial>(
    TM mat, HitInfo h, float3 N, float3 T, float3 B, float3 wo, Reservoir r)
{
    if (r.W <= 0.0)
        return float3(0.0);
    RCand c = restirEvalRef(mat, refType(r.y), refId(r.y), r.y.uv, h, N, T, B, wo);
    if (!c.valid || c.pHat <= 0.0)
        return float3(0.0);
    bool vis = c.isEnv ? visibleDirectional(fc, h.position, N, c.dir)
                       : visibleSegment(fc, h.position, c.toPoint);
    if (!vis)
        return float3(0.0);
    return c.integrand * r.W;
}

// Directional (delta) lights: plain NEE, outside the RIS (no MIS).
float3 restirDirectional<TM : IMaterial>(
    TM mat, HitInfo h, float3 N, float3 T, float3 B, float3 wo)
{
    float3 pos = h.position;
    float3 direct = float3(0.0);
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
