// ReSTIR DI — initial RIS over the unified light set at the primary hit
// (canonical integration, Decision 5: RIS owns ALL of primary direct).
//
// Reservoir-resamples M = M_light + M_bsdf candidates per pixel, mixing
// LIGHT-sampled candidates (sphere + emissive-triangle + env, drawn ~ uniform
// over the active techniques) and BSDF-sampled candidates (the BSDF/proposal
// direction traced to the sphere lights / environment). The target is the
// UNSHADOWED, UNWEIGHTED light contribution
//     p̂ = luminance(f · Le)
// and the per-candidate source pdf is the balance-heuristic MIXTURE pdf over the
// two techniques
//     p_mix(wi) = ( M_light·p_light(wi) + M_bsdf·p_bsdf(wi)·[wi→sphere|env] ) / M
// so light- and BSDF-sampled candidates combine into one unbiased estimator
// (mixture sampling = balance-heuristic MIS). The path tracer's own depth-0
// BSDF-hits-sphere / env-miss direct terms are GATED OFF (wf_shade_common +
// wavefront_path) since the RIS now owns them. Emissive triangles are NEE-only
// in the stock renderer (no BSDF-tri MIS term), so they are sampled by the light
// technique only ([wi→sphere|env] excludes them) — still unbiased, converging to
// the same direct integral. Directional lights are delta → plain NEE, outside
// the RIS. Deferred visibility: one shadow ray for the survivor at resolve.
//
// A reservoir sample is a reproducible reference (lightType, lightId, uv): for
// sphere/tri the uv maps to a fixed world point (recoverable from a BSDF hit
// point too); for env the uv is the octahedral-encoded direction. The mapping is
// INDEPENDENT of the shading point, so resolve and spatial/temporal reuse
// re-derive the same light at any pixel (the DI reconnection).

import common;          // RNG, worldToTangent, tangentToWorld, Ray
import bindings;        // fc, distantLights, effectiveFurnaceMode
import interfaces;      // IMaterial, BSDFSample, HitInfo, LightSample, ILight
import scene_trace;     // visibleDirectional, visibleSegment, traceScene
import scene_lights;    // sphereLights, emissiveTriangles, intersectSphereLights, NearestSphereHit
import environment;     // sampleEnvDir, sceneEnvRadiance, envPdf, EnvSample
import lights.sphere_light;
import lights.emissive_triangle_light;
import lights.directional_light;
import sampling.proposal;   // sampleBounceDirection, mixtureProposalPdf
import restir.reservoir;

// LightSampleRef.packed: type:2 | id:30.
static const uint RLIGHT_SPHERE = 0u;
static const uint RLIGHT_TRI    = 1u;
static const uint RLIGHT_ENV    = 2u;
static const uint RLIGHT_NONE   = 3u;   // BSDF ray hit non-emissive geometry (w=0)

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

// Octahedral unit-vector encode/decode (env directions stored in ref.uv).
float2 octEncode(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 e = n.xy;
    if (n.z < 0.0)
        e = (float2(1.0, 1.0) - abs(float2(n.y, n.x)))
          * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return e * 0.5 + 0.5;
}
float3 octDecode(float2 f)
{
    float2 e = f * 2.0 - 1.0;
    float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

// Recover the sphere-sampling uv (samplePoint inverse) from a world point P on
// the sphere, so a BSDF-sampled sphere hit becomes a reproducible light ref.
float2 sphereUVFromPoint(SphereLight raw, float3 P)
{
    float3 n = normalize(P - raw.position);
    float phi = atan2(n.y, n.x);
    if (phi < 0.0) phi += 2.0 * PI;
    return float2((n.z + 1.0) * 0.5, phi / (2.0 * PI));
}

// One evaluated candidate at a shading point: the unshadowed UNWEIGHTED integrand
// (f·Le — the RGB resolve multiplies by V·W), its target p̂, the light-technique
// solid-angle pdf and the BSDF/proposal pdf at this direction (for the mixture
// source pdf), the geometry for the deferred shadow ray, and isSE (sphere|env:
// the BSDF technique targets these → they take the mixed source pdf).
struct RCand
{
    float3 integrand;
    float  pHat;
    float  pdfLightSA;   // light-technique SA pdf (single light, NOT /nTech)
    float  pdfBsdf;      // mixtureProposalPdf at dir
    float3 toPoint;      // area-light point (shadow segment end)
    float3 dir;          // world-space direction to the light
    bool   isEnv;        // env (directional shadow ray) vs area (segment ray)
    bool   isSE;         // sphere or env (full MIS with the BSDF technique)
    bool   valid;
}

float _misPdf<TM : IMaterial>(
    TM mat, HitInfo h, float3 N, float3 T, float3 B, float3 wo, float3 wi)
{
    ProposalContext pc;
    pc.woT = wo; pc.N = N; pc.T = T; pc.B = B;
    pc.position = h.position; pc.materialId = h.materialId;
    return mixtureProposalPdf(mat, pc, wi);
}

// Evaluate a reproducible (type, id, uv) reference at a shading point. Visibility
// deferred. Used by fill (build), resolve (survivor), and spatial/temporal reuse
// (re-evaluate a neighbour's sample at THIS pixel — the reconnection).
RCand restirEvalRef<TM : IMaterial>(
    TM mat, uint type, uint id, float2 uv, HitInfo h,
    float3 N, float3 T, float3 B, float3 wo)
{
    RCand c;
    c.integrand = float3(0.0); c.pHat = 0.0; c.pdfLightSA = 0.0; c.pdfBsdf = 0.0;
    c.toPoint = float3(0.0); c.dir = float3(0.0);
    c.isEnv = false; c.isSE = false; c.valid = false;
    if (type == RLIGHT_NONE)
        return c;

    float3 dir; float3 Le; float pdfLightSA;
    if (type == RLIGHT_ENV)
    {
        dir = octDecode(uv);
        pdfLightSA = envPdf(dir);
        if (pdfLightSA <= 0.0)
            return c;
        // Cell radiance (not the point lookup) to match env NEE (sampleEnvDir)
        // + the gated env-miss term (sceneEnvRadianceCell) — MIS-consistent with
        // envPdf's cell distribution.
        Le = sceneEnvRadianceCell(dir, fc.envIntensity, effectiveFurnaceMode(h.materialId));
        c.isEnv = true; c.isSE = true;
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
        pdfLightSA = d2 * ls.pdfArea / cosLight;     // area → solid-angle measure
        Le = ls.radiance;
        c.toPoint = ls.point;
        c.isSE = (type == RLIGHT_SPHERE);            // tri = NEE-only (light technique)
    }
    if (pdfLightSA <= 0.0 || dot(N, dir) <= 0.0)
        return c;

    float3 wi = worldToTangent(dir, T, B, N);
    BSDFSample b = mat.evaluate(wo, wi);
    c.integrand = b.response * Le;                   // UNWEIGHTED (MIS lives in p_mix)
    c.pHat = _luminance(c.integrand);
    c.pdfLightSA = pdfLightSA;
    c.pdfBsdf = _misPdf(mat, h, N, T, B, wo, wi);
    c.dir = dir;
    c.valid = true;
    return c;
}

// Mixture source pdf for a candidate (balance heuristic over light + BSDF).
float _mixPdf(RCand c, uint nTech, uint mLight, uint mBsdf)
{
    float pLight = c.pdfLightSA / float(max(nTech, 1u));
    float pBsdf  = c.isSE ? c.pdfBsdf : 0.0;
    return (float(mLight) * pLight + float(mBsdf) * pBsdf) / float(max(mLight + mBsdf, 1u));
}

// Initial RIS over the unified light set → a Reservoir (no shadow ray; deferred
// to resolve). M_light light candidates (technique ~ uniform over the active
// sphere / tri / env) + M_bsdf BSDF candidates (proposal direction traced to a
// sphere light / the env). Every drawn candidate counts toward M (RIS averages
// over all draws); invalid / occluded ones stream with weight 0.
Reservoir restirFillReservoir<TM : IMaterial>(
    TM mat, HitInfo h, float3 N, float3 T, float3 B, float3 wo,
    uint mLight, uint mBsdf, inout RNG rng)
{
    bool envActive = (effectiveFurnaceMode(h.materialId) == 0u);
    uint nSphere = fc.numSphereLights;
    uint nTri    = (fc.numEmissiveTriangles > 0u) ? 1u : 0u;
    uint nTech   = nSphere + nTri + (envActive ? 1u : 0u);

    Reservoir r;
    reservoirReset(r);

    // ── Light-sampled candidates ────────────────────────────────────
    for (uint k = 0u; k < mLight; k++)
    {
        if (nTech == 0u)
        {
            reservoirUpdate(r, makeRef(RLIGHT_NONE, 0u, float2(0.0)), 0.0, 0.0, rng.next());
            continue;
        }
        uint t = min(uint(rng.next() * float(nTech)), nTech - 1u);
        uint type; uint id; float2 uv = rng.next2();
        if (t < nSphere)                       { type = RLIGHT_SPHERE; id = t; }
        else if (t < nSphere + nTri)
        {
            type = RLIGHT_TRI;
            id = min(uint(rng.next() * float(fc.numEmissiveTriangles)),
                     fc.numEmissiveTriangles - 1u);
        }
        else
        {
            // Importance-sample the env, store the direction (octahedral).
            EnvSample es = sampleEnvDir(uv, fc.envIntensity);
            type = RLIGHT_ENV; id = 0u; uv = octEncode(es.dir);
        }

        RCand c = restirEvalRef(mat, type, id, uv, h, N, T, B, wo);
        float w = 0.0, pHat = 0.0;
        if (c.valid && c.pHat > 0.0)
        {
            float src = _mixPdf(c, nTech, mLight, mBsdf);
            if (src > 0.0) { pHat = c.pHat; w = c.pHat / src; }
        }
        reservoirUpdate(r, makeRef(type, id, uv), w, pHat, rng.next());
    }

    // ── BSDF-sampled candidates (traced to sphere lights / env) ──────
    for (uint k = 0u; k < mBsdf; k++)
    {
        ProposalContext pc;
        pc.woT = wo; pc.N = N; pc.T = T; pc.B = B;
        pc.position = h.position; pc.materialId = h.materialId;
        BSDFSample bs = sampleBounceDirection(mat, pc, rng);
        if (!bs.valid || bs.transmitted || bs.pdf <= 0.0)
        {
            reservoirUpdate(r, makeRef(RLIGHT_NONE, 0u, float2(0.0)), 0.0, 0.0, rng.next());
            continue;
        }
        float3 dirW = tangentToWorld(bs.wi, T, B, N);

        Ray ray;
        ray.origin    = h.position + N * 0.002;
        ray.direction = dirW;
        NearestSphereHit sph = intersectSphereLights(ray, 1e30, fc.numSphereLights);
        HitInfo hg = traceScene(fc, ray);
        float geomT = hg.hit ? hg.t : 1e30;

        uint type; uint id; float2 uv;
        if (sph.idx != 0xFFFFFFFFu && sph.t < geomT)
        {
            float3 P = ray.origin + dirW * sph.t;
            type = RLIGHT_SPHERE; id = sph.idx; uv = sphereUVFromPoint(sphereLights[sph.idx], P);
        }
        else if (!hg.hit && (effectiveFurnaceMode(h.materialId) == 0u))
        {
            type = RLIGHT_ENV; id = 0u; uv = octEncode(dirW);
        }
        else
        {
            reservoirUpdate(r, makeRef(RLIGHT_NONE, 0u, float2(0.0)), 0.0, 0.0, rng.next());
            continue;
        }

        RCand c = restirEvalRef(mat, type, id, uv, h, N, T, B, wo);
        float w = 0.0, pHat = 0.0;
        if (c.valid && c.pHat > 0.0)
        {
            float src = _mixPdf(c, nTech, mLight, mBsdf);
            if (src > 0.0) { pHat = c.pHat; w = c.pHat / src; }
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
