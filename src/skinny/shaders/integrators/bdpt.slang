// Bidirectional path tracer (BDPT) — Veach §10 / pbrt-v3 §16.3.
//
// V1 simplifications (kept tractable for shader compile time):
//   - FlatMaterial first-hit only; non-flat hits fall through to PathTracer.
//   - Connections (s ≥ 1, t ≥ 1) and NEE evaluate the full std_surface BSDF.
//     Each surface vertex caches the fields needed to rebuild its FlatMaterial
//     (bdptSurface → loadFlatMaterial) so connectGeneric / connectT1 / the
//     light-tracer splat call FlatMaterial.evaluate() — matching PathTracer
//     NEE. Metals and coated surfaces (e.g. brass, whose colour lives in
//     coat_color over a white metallic base) therefore reflect correctly.
//     Reverse-pdf terms use the real BSDF pdf too; the only Lambertian
//     fallback left is the reverse pdf across a delta bounce (MIS weight
//     only — the estimator stays unbiased regardless).
//   - Subpaths capped at 4 vertices each (eye + light) — fewer (s, t)
//     pairs keeps the SPIR-V small enough for fast driver compile.
//   - Light tracer (s = 1) IS implemented: every non-delta light vertex is
//     projected onto camera and atomic-added into lightSplatBuffer. This
//     is the path that produces caustic spots on diffuse floors (pure-spec
//     connections via the t≥2 generic path are zero on co-planar surfaces).
//   - Directional light only via t=1 NEE (delta — no light walk).
//   - Env (IBL): eye-walk env NEE (importance-sample the env distribution) +
//     the BSDF-sampled escape ray, combined via the power heuristic — matching
//     PathTracer. Env NEE is the variance fix for bright-sun HDRIs, where the
//     BSDF-only escape ray alone left BDPT dark + firefly-noisy vs PathTracer.

import common;
import bindings;
import interfaces;
import scene_trace;
import scene_lights;
import materials.flat.flat_shading;
import materials.flat.flat_material;
import environment;
import lights.sphere_light;
import lights.emissive_triangle_light;
import lights.directional_light;

// Must match path.slang's MAX_BOUNCES. The eye subpath can hold up to
// BDPT_MAX_DEPTH surface vertices, so this caps how many bounces the camera
// walk explores. When it was lower than MAX_BOUNCES, delta-transmissive
// materials (glass, oil) came out dimmer than the path tracer: solid
// dielectrics need enter+exit refraction plus internal/secondary bounces to
// carry the transmitted radiance to the environment, and the shorter BDPT
// walk truncated those longer paths. At parity BDPT matches the path tracer
// exactly for these scenes. (Costs more per-thread vertex storage — keep in
// sync with MAX_BOUNCES rather than raising blindly.)
static const uint BDPT_MAX_DEPTH = 6u;
static const uint BDPT_MAX_VERTS = BDPT_MAX_DEPTH + 1u;   // 7

// ── Vertex ──────────────────────────────────────────────────────

static const uint BDPT_VK_CAMERA       = 0u;
static const uint BDPT_VK_SURFACE      = 1u;
static const uint BDPT_VK_LIGHT_AREA   = 2u;
static const uint BDPT_VK_LIGHT_DIR    = 3u;

struct BDPTVertex
{
    uint   kind;
    float3 position;
    float3 N;
    float3 throughput;    // β accumulated up to and including this vertex
    float3 emission;      // Le at this vertex (zero if not emissive)
    float  pdfFwd;        // area-measure pdf of arriving here from origin side
    float  pdfRev;        // area-measure pdf of arriving here from far side
    bool   isDelta;
    bool   onLight;

    // Material-reload fields — enough to rebuild the exact FlatMaterial at
    // connection time via loadFlatMaterial(). Connections evaluate the real
    // std_surface BSDF (matching PathTracer NEE) instead of a Lambertian
    // albedo/π approximation, so metals/coats (e.g. brass, whose colour lives
    // in coat_color while base_color is white) reflect correctly. Only valid
    // for kind == BDPT_VK_SURFACE; light/camera vertices leave them defaulted.
    uint   matId;
    float2 uv;
    float3 posObject;     // object-space hit (solid-texture graphs read this)
    float3 geoN;          // geometric normal pre normal-map (BSDF reload base)
    float3 tangent;       // surface tangent (+U)
    bool   hasTangent;
};

// Convert solid-angle pdf at `from` for direction toward `to` to area-measure
// pdf of landing at `to`. p_A = p_ω · |cosθ| / d².
float convertSAtoArea(float pdfOmega, float3 from, BDPTVertex to)
{
    if (to.kind == BDPT_VK_LIGHT_DIR)
        return pdfOmega;
    float3 d = to.position - from;
    float d2 = dot(d, d);
    if (d2 <= 1e-20) return 0.0;
    float invDist = 1.0 / sqrt(d2);
    float cosTo = abs(dot(to.N, -d * invDist));
    return pdfOmega * cosTo / d2;
}

// Lambertian solid-angle pdf for outgoing dir wWorld at vertex v.
float lambertSApdf(BDPTVertex v, float3 wWorld)
{
    float c = max(dot(v.N, wWorld), 0.0);
    return c / PI;
}

// ── Real-BSDF connection support ────────────────────────────────
//
// Reconstruct the exact FlatMaterial a surface vertex was created with, plus
// its tangent frame, so connections / NEE can evaluate the full std_surface
// BSDF rather than the Lambertian albedo/π approximation. loadFlatMaterial
// re-runs the MaterialX graph + texture fetches; connections are few per
// pixel so the cost is acceptable for an offline-quality integrator.
struct BdptSurface
{
    FlatMaterial mat;
    float3 T;
    float3 B;
};

BdptSurface bdptSurface(BDPTVertex v)
{
    HitInfo h;
    h.t              = 0.0;
    h.backT          = -1.0;
    h.position       = v.position;
    h.positionObject = v.posObject;
    h.normal         = v.geoN;
    h.tangent        = v.tangent;
    h.uv             = v.uv;
    h.hit            = true;
    h.hasTangent     = v.hasTangent;
    h.materialId     = v.matId;

    BdptSurface s;
    s.mat = loadFlatMaterial(h);
    if (v.hasTangent)
    {
        s.T = normalize(v.tangent - v.N * dot(v.tangent, v.N));
        s.B = cross(v.N, s.T);
    }
    else
    {
        buildBasis(v.N, s.T, s.B);
    }
    return s;
}

// Evaluate the std_surface BSDF at a reconstructed surface vertex for a view
// direction woWorld (toward the origin-side neighbour) and a connection
// direction wiWorld. Returns response = f·cosθ_i and the solid-angle pdf.
BSDFSample bdptEval(BdptSurface s, float3 N, float3 woWorld, float3 wiWorld)
{
    float3 wo = worldToTangent(woWorld, s.T, s.B, N);
    float3 wi = worldToTangent(wiWorld, s.T, s.B, N);
    return s.mat.evaluate(wo, wi);
}

// ── Environment next-event estimation ───────────────────────────
//
// Mirror of path.allLightsNEE's env block. The eye walk's only IBL strategy
// was a BSDF-sampled escape ray at full weight; for a high-dynamic-range env
// (bright sun) that has enormous variance — most glossy-surface samples miss
// the sun, the rest fire as fireflies — so BDPT read far dimmer/noisier than
// the path tracer at equal samples. Importance-sampling the env distribution
// here (and MIS-weighting the escape ray, see randomWalk) restores parity.
// Evaluates the real std_surface BSDF (wo_ts = tangent-space view dir) so the
// metallic/coat response matches connectT1 NEE. Returns radiance per unit of
// arriving throughput; the caller scales by the vertex's incoming throughput.
// Skipped under furnace mode (constant-white env has no features to sample).
float3 bdptEnvNEE(FlatMaterial mat, float3 P, float3 N, float3 T, float3 B,
                  float3 wo_ts, uint matId, inout RNG rng)
{
    if (effectiveFurnaceMode(matId) != 0u) return float3(0.0);
    EnvSample es = sampleEnvDir(rng.next2(), fc.envIntensity);
    if (es.pdf <= 0.0 || dot(N, es.dir) <= 0.0) return float3(0.0);
    if (!visibleDirectional(fc, P, N, es.dir)) return float3(0.0);
    float3 wi_ts = worldToTangent(es.dir, T, B, N);
    BSDFSample b = mat.evaluate(wo_ts, wi_ts);
    if (b.pdf <= 0.0) return float3(0.0);     // delta lobes have no NEE partner
    float w = powerHeuristic(es.pdf, b.pdf);
    return b.response * es.radiance / es.pdf * w;
}

// ── Light-side endpoint sampling ────────────────────────────────

struct LightOriginSample
{
    BDPTVertex vertex;
    float3     dirOut;
    float3     beta;
    float      pdfPos;
    float      pdfDir;
    bool       valid;
};

LightOriginSample sampleLightOrigin(inout RNG rng)
{
    LightOriginSample r;
    r.valid = false;
    r.vertex.kind = BDPT_VK_LIGHT_AREA;
    r.vertex.position = float3(0.0);
    r.vertex.N = float3(0.0, 1.0, 0.0);
    r.vertex.throughput = float3(0.0);
    r.vertex.emission = float3(0.0);
    r.vertex.pdfFwd = 0.0;
    r.vertex.pdfRev = 0.0;
    r.vertex.isDelta = false;
    r.vertex.onLight = true;
    r.vertex.matId = 0u;
    r.vertex.uv = float2(0.0);
    r.vertex.posObject = float3(0.0);
    r.vertex.geoN = float3(0.0, 1.0, 0.0);
    r.vertex.tangent = float3(1.0, 0.0, 0.0);
    r.vertex.hasTangent = false;
    r.dirOut = float3(0.0, 1.0, 0.0);
    r.beta = float3(0.0);
    r.pdfPos = 0.0;
    r.pdfDir = 0.0;

    uint nSphere = fc.numSphereLights;
    uint nTri    = fc.numEmissiveTriangles;
    uint nDir    = fc.numDistantLights;
    uint nGroups = (nDir > 0u ? 1u : 0u) + (nSphere > 0u ? 1u : 0u) + (nTri > 0u ? 1u : 0u);
    if (nGroups == 0u) return r;

    float groupPdf = 1.0 / float(nGroups);
    uint  group    = uint(min(rng.next() * float(nGroups), float(nGroups - 1u)));

    uint chosen = 0xFFFFFFFFu;
    uint cursor = 0u;
    if (nSphere > 0u) { if (cursor == group) chosen = 0u; cursor++; }
    if (nTri    > 0u) { if (cursor == group) chosen = 1u; cursor++; }
    if (nDir    > 0u) { if (cursor == group) chosen = 2u; cursor++; }
    if (chosen == 0xFFFFFFFFu) return r;

    if (chosen == 2u)
    {
        // Pick one distant light uniformly. Power-proportional weighting
        // would converge faster but every NEE/connectT1 loop iterates
        // *all* distant lights, so the BDPT sampleLightOrigin choice only
        // matters for the s≥1 light-walk strategy.
        uint idx = min(uint(rng.next() * float(nDir)), nDir - 1u);
        DistantLight L = distantLights[idx];
        float3 dir = normalize(L.direction);
        float lightPickPdf = groupPdf / float(nDir);
        r.vertex.kind = BDPT_VK_LIGHT_DIR;
        r.vertex.N = -dir;
        r.vertex.emission = L.radiance;
        r.vertex.throughput = L.radiance;
        r.vertex.pdfFwd = lightPickPdf;
        r.vertex.isDelta = true;
        r.dirOut = dir;
        r.beta = L.radiance;
        r.pdfPos = lightPickPdf;
        r.pdfDir = 1.0;
        r.valid = true;
        return r;
    }

    float3 lpos, lN, lRad;
    float pdfArea;

    if (chosen == 0u)
    {
        uint idx = min(uint(rng.next() * float(nSphere)), nSphere - 1u);
        SphereLightImpl L = loadSphereLightImpl(sphereLights[idx]);
        LightSample ls = L.samplePoint(float3(0.0), rng.next2());
        if (!ls.valid) return r;
        lpos = ls.point; lN = ls.normal; lRad = ls.radiance;
        pdfArea = ls.pdfArea;
    }
    else
    {
        uint idx = min(uint(rng.next() * float(nTri)), nTri - 1u);
        EmissiveTriangle raw = emissiveTriangles[idx];
        EmissiveTriangleLightImpl L = loadEmissiveTriangleLightImpl(raw, nTri);
        LightSample ls = L.samplePoint(float3(0.0), rng.next2());
        if (!ls.valid) return r;
        lpos = ls.point; lN = ls.normal; lRad = ls.radiance;
        // L.pdfArea folds in 1/numTris already; strip and re-apply via groupPdf.
        pdfArea = ls.pdfArea * float(nTri);
    }

    float3 T, B;
    buildBasis(lN, T, B);
    float3 dirLocal = sampleCosineHemisphereTS(rng.next2());
    float cosOut = max(dirLocal.z, 0.0);
    if (cosOut <= 0.0) return r;
    float3 dirWorld = T * dirLocal.x + B * dirLocal.y + lN * dirLocal.z;

    float pdfPos = groupPdf * pdfArea;
    float pdfDir = cosOut / PI;

    r.vertex.kind = BDPT_VK_LIGHT_AREA;
    r.vertex.position = lpos;
    r.vertex.N = lN;
    r.vertex.emission = lRad;
    r.vertex.throughput = lRad / max(pdfPos, 1e-20);
    r.vertex.pdfFwd = pdfPos;
    r.dirOut = dirWorld;
    r.beta = lRad * cosOut / max(pdfPos * pdfDir, 1e-20);
    r.pdfPos = pdfPos;
    r.pdfDir = pdfDir;
    r.valid = true;
    return r;
}

// ── Random walk (eye or light) ──────────────────────────────────
//
// Builds vertices into path[startIdx ...]. Returns past-the-end count.
// pdfDirOmegaIn = solid-angle pdf of the initial direction from path[startIdx-1].
// Vertices created here use loadFlatMaterial(.sample) for correct importance
// sampling, but cache only albedo + emission for later Lambertian-eval
// connection / pdfRev computation.

int randomWalk(
    inout BDPTVertex path[BDPT_MAX_VERTS],
    int startIdx,
    Ray ray,
    float3 throughput,
    float pdfDirOmegaIn,
    inout RNG rng,
    bool isEyeWalk,
    inout float3 escapedRadiance,
    float misBsdfPdfIn)
{
    int idx = startIdx;
    float pdfFwdOmega = pdfDirOmegaIn;
    // BSDF-strategy pdf of the bounce that spawned `ray`, for MIS against env
    // NEE when the ray escapes. -1 ⇒ no env-NEE counterpart (delta/transmitted/
    // furnace) ⇒ take the escaped env at full weight. Mirrors path.slang.
    float misBsdfPdf = misBsdfPdfIn;

    [loop]
    while (idx < int(BDPT_MAX_VERTS))
    {
        HitInfo h = traceScene(fc, ray);
        [loop]
        for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
        {
            ray.origin = h.position + ray.direction * 0.002;
            h = traceScene(fc, ray);
        }

        // Eye walks see sphere-light geometry as emissive: if the bounce
        // ray hits a sphere light before any mesh, deposit its radiance
        // here and stop the walk. Light walks skip this — they're already
        // building paths originating ON a light. Mirrors path.slang:217.
        if (isEyeWalk && fc.numSphereLights > 0u)
        {
            float t_max = h.hit ? h.t : 1e30;
            NearestSphereHit sph = intersectSphereLights(
                ray, t_max, fc.numSphereLights);
            if (sph.idx != 0xFFFFFFFFu)
            {
                SphereLight L = sphereLights[sph.idx];
                // MIS against connectT1's NEE strategy. Delta bounces
                // (pdfFwdOmega is the unit-omega sentinel = 1.0 set in
                // the previous iteration) have no NEE counterpart, so
                // we take the BSDF estimate at full weight.
                float w_bsdf = 1.0;
                BDPTVertex prev = path[idx - 1];
                if (!prev.isDelta && pdfFwdOmega > 0.0)
                {
                    float area = 4.0 * PI * L.radius * L.radius;
                    float pdf_light_sa = (sph.t * sph.t) /
                        max(sph.cos_y * area, 1e-12);
                    w_bsdf = powerHeuristic(pdfFwdOmega, pdf_light_sa);
                }
                escapedRadiance += throughput * L.radiance * w_bsdf;
                break;
            }
        }

        if (!h.hit)
        {
            if (isEyeWalk)
            {
                // MIS against env NEE: when the spawning bounce was non-delta
                // and env-active, env NEE also sampled this direction — combine
                // via the power heuristic. Delta/transmitted/furnace bounces
                // (misBsdfPdf < 0) have no NEE partner ⇒ full weight.
                float w = 1.0;
                if (misBsdfPdf > 0.0)
                    w = powerHeuristic(misBsdfPdf, envPdf(ray.direction));
                escapedRadiance += throughput * sceneEnvRadiance(
                    ray.direction, fc.envIntensity,
                    effectiveFurnaceMode(h.materialId)) * w;
            }
            break;
        }
        if ((materialTypes[h.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT) break;

        FlatMaterial mat = loadFlatMaterial(h);
        FlatHitData fd = mat.data;

        BDPTVertex v;
        v.kind = BDPT_VK_SURFACE;
        v.position = h.position;
        v.N = fd.N;
        v.throughput = throughput;
        v.emission = fd.mat.emission;
        v.pdfFwd = 0.0;
        v.pdfRev = 0.0;
        v.isDelta = false;
        v.onLight = any(fd.mat.emission > float3(0.0));
        v.matId = h.materialId;
        v.uv = h.uv;
        v.posObject = h.positionObject;
        v.geoN = h.normal;
        v.tangent = h.tangent;
        v.hasTangent = h.hasTangent;

        BDPTVertex prev = path[idx - 1];
        v.pdfFwd = convertSAtoArea(pdfFwdOmega, prev.position, v);

        int prevIdx = idx - 1;
        path[idx] = v;
        idx++;
        if (idx >= int(BDPT_MAX_VERTS)) break;

        // Sample next bounce with the full BSDF.
        float3 wo_ts;
        float3 T, B;
        if (h.hasTangent)
        {
            T = normalize(h.tangent - v.N * dot(h.tangent, v.N));
            B = cross(v.N, T);
        }
        else
        {
            buildBasis(v.N, T, B);
        }
        wo_ts = worldToTangent(-ray.direction, T, B, v.N);

        // Environment NEE at this eye vertex. `throughput` is the radiance
        // arriving at v (not yet scaled by this vertex's bounce weight), which
        // is the correct prefix for a direct env connection.
        if (isEyeWalk)
            escapedRadiance += throughput *
                bdptEnvNEE(mat, v.position, v.N, T, B, wo_ts, v.matId, rng);

        BSDFSample bs = mat.sample(wo_ts, rng);
        if (!bs.valid) break;
        float3 wi_world = tangentToWorld(bs.wi, T, B, v.N);

        bool deltaBounce = (bs.pdf <= 0.0);
        if (deltaBounce)
        {
            // BSDF sampling at v was a delta event — flag v itself so MIS /
            // connections skip BSDF eval at it. Next vertex stays normal.
            path[idx - 1].isDelta = true;
        }

        // prev.pdfRev: area-pdf of reaching prev from v along the reversed
        // path. Uses the real std_surface BSDF pdf at v with the continuation
        // direction (v→next, = wi_world) as the view side. Delta bounces have
        // no finite scatter pdf — fall back to the Lambertian surrogate (MIS
        // weight only; the estimator stays unbiased either way).
        if (prev.kind != BDPT_VK_CAMERA && !prev.isDelta)
        {
            float3 toPrev = prev.position - v.position;
            float toPrevLen2 = dot(toPrev, toPrev);
            if (toPrevLen2 > 1e-20)
            {
                float3 toPrevDir = toPrev / sqrt(toPrevLen2);
                float pdfRevOmega;
                if (deltaBounce)
                {
                    pdfRevOmega = lambertSApdf(v, toPrevDir);
                }
                else
                {
                    float3 woRev = worldToTangent(wi_world, T, B, v.N);
                    float3 wiRev = worldToTangent(toPrevDir, T, B, v.N);
                    pdfRevOmega = mat.evaluate(woRev, wiRev).pdf;
                }
                path[prevIdx].pdfRev = convertSAtoArea(pdfRevOmega, v.position, prev);
            }
        }

        throughput *= bs.weight;

        if (idx >= 2)
        {
            float p = clamp(max(max(throughput.x, throughput.y), throughput.z),
                            0.05, 0.95);
            if (rng.next() > p) break;
            throughput /= p;
        }

        float3 spawnOffset = bs.transmitted ? -v.N * 0.002 : v.N * 0.002;
        ray.origin = v.position + spawnOffset;
        ray.direction = wi_world;
        // Carry a unit-omega sentinel through delta segments so the next
        // vertex's area-measure pdfFwd remains finite (= cos/d²).
        pdfFwdOmega = deltaBounce ? 1.0 : bs.pdf;
        // Will the next bounce's env-miss have an env-NEE partner to MIS
        // against? Only when this bounce is non-delta, reflected (env NEE
        // samples the upper hemisphere only), and env-active. Mirrors path.slang.
        misBsdfPdf = (isEyeWalk && !deltaBounce && !bs.transmitted
                      && effectiveFurnaceMode(v.matId) == 0u)
                   ? bs.pdf : -1.0;
    }
    return idx;
}

// ── MIS weight (power heuristic, β = 2) ─────────────────────────

float remap0(float x) { return x == 0.0 ? 1.0 : x; }

float misWeight(
    BDPTVertex eye[BDPT_MAX_VERTS], int s,
    BDPTVertex lit[BDPT_MAX_VERTS], int t,
    float pdf_zsm1_rev,
    float pdf_zsm2_rev,
    float pdf_ytm1_rev,
    float pdf_ytm2_rev)
{
    if (s + t == 2) return 1.0;

    BDPTVertex eyeC[BDPT_MAX_VERTS];
    BDPTVertex litC[BDPT_MAX_VERTS];
    [loop] for (int i = 0; i < int(BDPT_MAX_VERTS); i++) eyeC[i] = eye[i];
    [loop] for (int i = 0; i < int(BDPT_MAX_VERTS); i++) litC[i] = lit[i];

    if (s >= 1) eyeC[s - 1].pdfRev = pdf_zsm1_rev;
    if (s >= 2) eyeC[s - 2].pdfRev = pdf_zsm2_rev;
    if (t >= 1) litC[t - 1].pdfRev = pdf_ytm1_rev;
    if (t >= 2) litC[t - 2].pdfRev = pdf_ytm2_rev;

    float sumRi = 0.0;
    float ri = 1.0;

    [loop]
    for (int i = s - 1; i > 0; i--)
    {
        ri *= remap0(eyeC[i].pdfRev) / remap0(eyeC[i].pdfFwd);
        if (!eyeC[i].isDelta && !eyeC[i - 1].isDelta)
            sumRi += ri * ri;
    }

    ri = 1.0;
    [loop]
    for (int i = t - 1; i >= 0; i--)
    {
        ri *= remap0(litC[i].pdfRev) / remap0(litC[i].pdfFwd);
        bool deltaLight = (i > 0) ? litC[i - 1].isDelta : false;
        if (!litC[i].isDelta && !deltaLight)
            sumRi += ri * ri;
    }

    return 1.0 / (1.0 + sumRi);
}

// ── Connections ─────────────────────────────────────────────────

// Connection contribution between eye[s-1] and lit[t-1] using Lambertian
// BSDFs at both endpoints. Returns (contribution, MIS pdfs).
struct ConnectResult
{
    float3 contribution;
    float  pdf_zsm1_rev;
    float  pdf_zsm2_rev;
    float  pdf_ytm1_rev;
    float  pdf_ytm2_rev;
    bool   nonzero;
};

ConnectResult connectGeneric(
    BDPTVertex eye[BDPT_MAX_VERTS], int s,
    BDPTVertex lit[BDPT_MAX_VERTS], int t)
{
    ConnectResult r;
    r.contribution = float3(0.0);
    r.pdf_zsm1_rev = 0.0;
    r.pdf_zsm2_rev = 0.0;
    r.pdf_ytm1_rev = 0.0;
    r.pdf_ytm2_rev = 0.0;
    r.nonzero = false;

    BDPTVertex z = eye[s - 1];
    BDPTVertex y = lit[t - 1];
    if (z.isDelta || y.isDelta) return r;
    if (z.kind != BDPT_VK_SURFACE) return r;

    float3 d = y.position - z.position;
    float d2 = dot(d, d);
    if (d2 <= 1e-12) return r;
    float dist = sqrt(d2);
    float3 dir = d / dist;
    float cosZ = max(dot(z.N, dir), 0.0);
    float cosY = max(dot(y.N, -dir), 0.0);
    if (cosZ <= 0.0 || cosY <= 0.0) return r;
    if (!visibleSegment(fc, z.position, y.position)) return r;

    // Eye-side endpoint: evaluate the real std_surface BSDF at z for the
    // connection direction (z→y), viewed from its origin-side neighbour.
    // response_z = f_z·cosZ already folds in the cosine, so the geometry
    // term drops to 1/d² (the cosZ/cosY of G live inside the responses).
    BdptSurface zs = bdptSurface(z);
    float3 woZ = normalize(eye[s - 2].position - z.position);
    BSDFSample ez = bdptEval(zs, z.N, woZ, dir);
    float3 respZ = ez.response;
    float  pdf_z_sa = ez.pdf;           // pdf at z for dir z→y

    // Light-side endpoint: real BSDF when y is a surface, else a cosine-
    // weighted diffuse-emitter lobe (f = 1/π) for the light-origin vertex.
    float3 respY;
    float  pdf_y_sa;                    // pdf at y for dir y→z
    if (y.kind == BDPT_VK_SURFACE)
    {
        BdptSurface ys = bdptSurface(y);
        float3 woY = normalize(lit[t - 2].position - y.position);
        BSDFSample ey = bdptEval(ys, y.N, woY, -dir);
        respY = ey.response;
        pdf_y_sa = ey.pdf;

        if (t >= 2)
        {
            float3 toPrev = lit[t - 2].position - y.position;
            float l2 = dot(toPrev, toPrev);
            if (l2 > 1e-20)
            {
                float3 dp = toPrev / sqrt(l2);
                float pr = bdptEval(ys, y.N, -dir, dp).pdf;
                r.pdf_ytm2_rev = convertSAtoArea(pr, y.position, lit[t - 2]);
            }
        }
    }
    else
    {
        respY = float3(cosY / PI);
        pdf_y_sa = cosY / PI;
    }

    r.contribution = z.throughput * respZ * respY / d2 * y.throughput;

    r.pdf_zsm1_rev = convertSAtoArea(pdf_y_sa, y.position, z);
    r.pdf_ytm1_rev = convertSAtoArea(pdf_z_sa, z.position, y);

    if (s >= 2)
    {
        float3 toPrev = eye[s - 2].position - z.position;
        float l2 = dot(toPrev, toPrev);
        if (l2 > 1e-20)
        {
            float3 dp = toPrev / sqrt(l2);
            float pr = bdptEval(zs, z.N, dir, dp).pdf;
            r.pdf_zsm2_rev = convertSAtoArea(pr, z.position, eye[s - 2]);
        }
    }
    r.nonzero = any(r.contribution > float3(0.0));
    return r;
}

// t = 1 NEE: connect eye[s-1] to a freshly sampled light point.
// Standard NEE with power heuristic against the BSDF strategy. Evaluates the
// real std_surface BSDF at z (matching path.slang neeLightEstimator) so the
// metallic/coat response is correct rather than a Lambertian albedo/π.
float3 connectT1(BDPTVertex eye[BDPT_MAX_VERTS], int s, inout RNG rng)
{
    BDPTVertex z = eye[s - 1];
    if (z.isDelta || z.kind != BDPT_VK_SURFACE) return float3(0.0);

    float3 total = float3(0.0);
    BdptSurface zs = bdptSurface(z);
    float3 woWorld = normalize(eye[s - 2].position - z.position);

    for (uint d = 0u; d < fc.numDistantLights; d++)
    {
        DistantLight L = distantLights[d];
        float3 dir = normalize(L.direction);
        float NdotL = max(dot(z.N, dir), 0.0);
        if (NdotL > 0.0 && visibleDirectional(fc, z.position, z.N, dir))
        {
            float3 resp = bdptEval(zs, z.N, woWorld, dir).response;
            total += z.throughput * resp * L.radiance;
        }
    }

    [loop]
    for (uint i = 0u; i < fc.numSphereLights; i++)
    {
        SphereLightImpl L = loadSphereLightImpl(sphereLights[i]);
        LightSample ls = L.samplePoint(z.position, rng.next2());
        if (!ls.valid) continue;
        float3 toL = ls.point - z.position;
        float d2 = dot(toL, toL);
        if (d2 <= 1e-8) continue;
        float dist = sqrt(d2);
        float3 dir = toL / dist;
        float NdotL = max(dot(z.N, dir), 0.0);
        if (NdotL <= 0.0) continue;
        float cosLight = max(dot(ls.normal, -dir), 0.0);
        if (cosLight <= 0.0) continue;
        if (!visibleSegment(fc, z.position, ls.point)) continue;

        BSDFSample b = bdptEval(zs, z.N, woWorld, dir);
        float invPdfOmega = cosLight / (d2 * ls.pdfArea);
        float pdfLightSA = d2 * ls.pdfArea / max(cosLight, 1e-12);
        float wNEE = powerHeuristic(pdfLightSA, b.pdf);
        total += z.throughput * b.response * ls.radiance * invPdfOmega * wNEE;
    }

    if (fc.numEmissiveTriangles > 0u)
    {
        uint idx = min(uint(rng.next() * float(fc.numEmissiveTriangles)),
                       fc.numEmissiveTriangles - 1u);
        EmissiveTriangleLightImpl L = loadEmissiveTriangleLightImpl(
            emissiveTriangles[idx], fc.numEmissiveTriangles);
        LightSample ls = L.samplePoint(z.position, rng.next2());
        if (ls.valid)
        {
            float3 toL = ls.point - z.position;
            float d2 = dot(toL, toL);
            if (d2 > 1e-8)
            {
                float dist = sqrt(d2);
                float3 dir = toL / dist;
                float NdotL = max(dot(z.N, dir), 0.0);
                float cosLight = max(dot(ls.normal, -dir), 0.0);
                if (NdotL > 0.0 && cosLight > 0.0
                    && visibleSegment(fc, z.position, ls.point))
                {
                    BSDFSample b = bdptEval(zs, z.N, woWorld, dir);
                    float invPdfOmega = cosLight / (d2 * ls.pdfArea);
                    float pdfLightSA = d2 * ls.pdfArea / max(cosLight, 1e-12);
                    float wNEE = powerHeuristic(pdfLightSA, b.pdf);
                    total += z.throughput * b.response * ls.radiance * invPdfOmega * wNEE;
                }
            }
        }
    }

    return total;
}

// ── Light tracer (s = 1) — camera splat ────────────────────────

static const float BDPT_SPLAT_SCALE = 1024.0;     // Q22.10 fixed-point

void atomicSplatRadiance(int2 pixel, float3 radiance)
{
    if (any(radiance != radiance)) return;        // NaN guard
    if (any(isinf(radiance))) return;
    radiance = max(radiance, float3(0.0));
    uint base = (uint(pixel.y) * fc.width + uint(pixel.x)) * 3u;
    uint orig;
    InterlockedAdd(lightSplatBuffer[base + 0u], uint(radiance.x * BDPT_SPLAT_SCALE), orig);
    InterlockedAdd(lightSplatBuffer[base + 1u], uint(radiance.y * BDPT_SPLAT_SCALE), orig);
    InterlockedAdd(lightSplatBuffer[base + 2u], uint(radiance.z * BDPT_SPLAT_SCALE), orig);
}

// Splat a single light-walk vertex onto the camera. Implements the s = 1
// BDPT strategy: the camera vertex is the deterministic camera position,
// connected to a non-delta light-walk surface vertex. The camera model
// (`PinholeCamera` or `ThickLensCamera`) is supplied as a generic
// parameter so dispatch is monomorphized at compile time.
void splatLightVertex<TC : ICamera>(BDPTVertex y, float3 prevPos, inout RNG rng, TC cam)
{
    if (y.isDelta || y.kind != BDPT_VK_SURFACE) return;
    int2 pixel;
    float cosCam;
    float distCam;
    if (!cam.sampleWi(y.position, rng, pixel, cosCam, distCam)) return;

    float3 toCam = (fc.camera.position - y.position) / distCam;
    float cosY = max(dot(y.N, toCam), 0.0);
    if (cosY <= 0.0) return;
    if (!visibleSegment(fc, y.position, fc.camera.position)) return;

    // Camera importance for a pinhole: We = 1 / (A · cos⁴θ) where A is the
    // image-plane area in normalized camera units (at distance 1 from camera
    // along forward).  Slang uses degrees for the existing fov field elsewhere
    // — but the perspective matrix is built with radians; check renderer's
    // _perspective: takes degrees. The shader's existing ray-gen multiplies
    // through the inverse projection, never reading fov, so we must convert
    // degrees → radians ourselves.
    float fovRad = fc.camera.fov * (PI / 180.0);
    float halfH = tan(fovRad * 0.5);
    float halfW = halfH * (float(fc.width) / float(fc.height));
    float A_image = 4.0 * halfW * halfH;
    float cos4 = cosCam * cosCam * cosCam * cosCam;
    float We = 1.0 / max(A_image * cos4, 1e-12);

    // Per-splat raw contribution (pbrt-v3 convention). main_pass.slang
    // composites by dividing the cumulative splatBuffer by frame count, which
    // averages the per-iteration light-path estimator across frames. The real
    // std_surface BSDF at y replaces the Lambertian albedo/π; response already
    // folds in cosY, so the geometry term drops to 1/distCam².
    BdptSurface ys = bdptSurface(y);
    float3 woWorld = normalize(prevPos - y.position);
    float3 respY = bdptEval(ys, y.N, woWorld, toCam).response;
    float3 contrib = y.throughput * respY / (distCam * distCam) * We;
    atomicSplatRadiance(pixel, contrib);
}

void splatLightWalk<TC : ICamera>(BDPTVertex lightPath[BDPT_MAX_VERTS],
                                  int lightLen, inout RNG rng, TC cam)
{
    [loop]
    for (int i = 1; i < lightLen; i++)
    {
        splatLightVertex<TC>(lightPath[i], lightPath[i - 1].position, rng, cam);
    }
}

// ── Integrator ──────────────────────────────────────────────────

struct BDPTIntegrator<TC : ICamera> : IIntegrator
{
    TC cam;

    float3 estimateRadiance(Ray ray, HitInfo firstHit, inout RNG rng)
    {
        BDPTVertex eye[BDPT_MAX_VERTS];

        BDPTVertex camVtx;
        camVtx.kind = BDPT_VK_CAMERA;
        camVtx.position = ray.origin;
        camVtx.N = -ray.direction;
        camVtx.throughput = float3(1.0);
        camVtx.emission = float3(0.0);
        camVtx.pdfFwd = 1.0;
        camVtx.pdfRev = 0.0;
        camVtx.isDelta = false;
        camVtx.onLight = false;
        camVtx.matId = 0u;
        camVtx.uv = float2(0.0);
        camVtx.posObject = float3(0.0);
        camVtx.geoN = -ray.direction;
        camVtx.tangent = float3(1.0, 0.0, 0.0);
        camVtx.hasTangent = false;
        eye[0] = camVtx;

        // Skip cutout-transparent surfaces before constructing the first
        // eye vertex. Without this, BDPT renders cutout-alpha texels as
        // solid opaque (firstHit is the raw traceScene result from
        // main_pass.slang, never cutout-resolved). Mirrors path.slang.
        Ray r = ray;
        for (uint skip = 0u; skip < 32u && firstHit.hit && isCutoutTransparent(firstHit); skip++)
        {
            r.origin = firstHit.position + r.direction * 0.002;
            firstHit = traceScene(fc, r);
        }
        if (!firstHit.hit)
        {
            return sceneEnvRadiance(r.direction, fc.envIntensity,
                                    effectiveFurnaceMode(firstHit.materialId));
        }
        if ((materialTypes[firstHit.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
        {
            // Non-flat first hit after skip — bail back to env (path
            // tracer covers the skin/python material paths).
            return float3(0.0);
        }

        FlatMaterial fmat = loadFlatMaterial(firstHit);
        BDPTVertex z1;
        z1.kind = BDPT_VK_SURFACE;
        z1.position = firstHit.position;
        z1.N = fmat.data.N;
        z1.throughput = float3(1.0);
        z1.emission = fmat.data.mat.emission;
        z1.pdfFwd = 1.0;
        z1.pdfRev = 0.0;
        z1.isDelta = false;
        z1.onLight = any(fmat.data.mat.emission > float3(0.0));
        z1.matId = firstHit.materialId;
        z1.uv = firstHit.uv;
        z1.posObject = firstHit.positionObject;
        z1.geoN = firstHit.normal;
        z1.tangent = firstHit.tangent;
        z1.hasTangent = firstHit.hasTangent;
        eye[1] = z1;
        int eyeLen = 2;

        // Continue eye walk: sample full FlatMaterial BSDF at z1.
        float3 T, B;
        if (firstHit.hasTangent)
        {
            T = normalize(firstHit.tangent - z1.N * dot(firstHit.tangent, z1.N));
            B = cross(z1.N, T);
        }
        else
        {
            buildBasis(z1.N, T, B);
        }
        float3 wo_ts = worldToTangent(-ray.direction, T, B, z1.N);

        // Environment NEE at the first eye vertex z1 (throughput = 1).
        float3 escapedRadiance = bdptEnvNEE(
            fmat, z1.position, z1.N, T, B, wo_ts, firstHit.materialId, rng);

        BSDFSample bs = fmat.sample(wo_ts, rng);
        if (bs.valid)
        {
            if (bs.pdf <= 0.0)
                eye[1].isDelta = true;
            float3 wi_world = tangentToWorld(bs.wi, T, B, z1.N);
            float3 throughput = bs.weight;
            Ray r2;
            float3 spawnOffset = bs.transmitted ? -z1.N * 0.002 : z1.N * 0.002;
            r2.origin = z1.position + spawnOffset;
            r2.direction = wi_world;
            float pdfFwdOmega = (bs.pdf > 0.0) ? bs.pdf : 1.0;
            // Initial env-miss MIS partner: z1's bounce pdf, when non-delta,
            // reflected, and env-active.
            float misBsdfPdfIn = (bs.pdf > 0.0 && !bs.transmitted
                && effectiveFurnaceMode(firstHit.materialId) == 0u)
                ? bs.pdf : -1.0;
            eyeLen = randomWalk(eye, 2, r2, throughput, pdfFwdOmega, rng,
                                true, escapedRadiance, misBsdfPdfIn);
        }

        // Light subpath.
        BDPTVertex lightPath[BDPT_MAX_VERTS];
        int lightLen = 0;

        LightOriginSample lo = sampleLightOrigin(rng);
        if (lo.valid)
        {
            lightPath[0] = lo.vertex;
            lightLen = 1;
            if (lo.vertex.kind != BDPT_VK_LIGHT_DIR)
            {
                Ray rL;
                rL.origin = lo.vertex.position + lo.vertex.N * 0.002;
                rL.direction = lo.dirOut;
                float3 unusedEscape = float3(0.0);
                lightLen = randomWalk(lightPath, 1, rL, lo.beta, lo.pdfDir, rng,
                                      false, unusedEscape, -1.0);
            }
        }

        // s = 1 light tracer: project each non-delta light vertex onto the
        // camera and atomic-add the contribution into lightSplatBuffer.
        // main_pass.slang composites the splat at end-of-frame.
        splatLightWalk<TC>(lightPath, lightLen, rng, cam);

        float3 L = escapedRadiance;

        // (s ≥ 2, t = 0): emissive eye-vertex hits, MIS-gated against the t = 1
        // NEE (connectT1) exactly as path.slang gates its BSDF-sampled emission.
        // An emissive triangle reached by a non-delta eye bounce is ALSO sampled
        // by connectT1's NEE from the previous eye vertex; adding the BSDF-hit at
        // full weight here on top of the (already power-heuristic-weighted) NEE
        // double-counts direct area-light illumination — BDPT read ~1.75× brighter
        // than pbrt on diffuse area-light scenes. Take the BSDF hit at full weight
        // only when no NEE partner exists: the primary/first hit (s == 2, analog
        // of path.slang's bounce == 0), a delta/specular bounce into z
        // (eye[s - 2].isDelta), or a scene with no emissive-triangle NEE.
        [loop]
        for (int s = 2; s <= eyeLen; s++)
        {
            BDPTVertex z = eye[s - 1];
            if (!z.onLight) continue;
            bool noNeePartner = (s == 2)
                              || (fc.numEmissiveTriangles == 0u)
                              || eye[s - 2].isDelta;
            if (noNeePartner)
                L += z.throughput * z.emission;
        }

        // (s ≥ 2, t = 1): NEE.
        [loop]
        for (int s = 2; s <= eyeLen; s++)
        {
            L += connectT1(eye, s, rng);
        }

        // (s ≥ 2, t ≥ 2): generic Lambertian-eval connections + MIS.
        [loop]
        for (int s = 2; s <= eyeLen; s++)
        {
            [loop]
            for (int t = 2; t <= lightLen; t++)
            {
                if (s + t - 2 > int(BDPT_MAX_DEPTH)) continue;
                ConnectResult cr = connectGeneric(eye, s, lightPath, t);
                if (!cr.nonzero) continue;
                float w = misWeight(eye, s, lightPath, t,
                                    cr.pdf_zsm1_rev, cr.pdf_zsm2_rev,
                                    cr.pdf_ytm1_rev, cr.pdf_ytm2_rev);
                L += cr.contribution * w;
            }
        }

        return L;
    }
}
