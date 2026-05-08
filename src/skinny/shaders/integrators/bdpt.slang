// Bidirectional path tracer (BDPT) — Veach §10 / pbrt-v3 §16.3.
//
// V1 simplifications (kept tractable for shader compile time):
//   - FlatMaterial first-hit only; non-flat hits fall through to PathTracer.
//   - Connections (s ≥ 1, t ≥ 1) use Lambertian BSDF approximation:
//     f(z, y) ≈ albedo(z) / π,  pdf(ω) ≈ cos(θ) / π.
//     Walks still use the full FlatMaterial.sample() for correct
//     importance-sampled bounces; only the connection / reverse-pdf
//     evaluations use the diffuse approximation. Pure-spec connections
//     are therefore weaker than full BDPT but the estimator stays
//     unbiased (MIS reweights against the BSDF-sampled t = 0 strategy).
//   - Subpaths capped at 4 vertices each (eye + light) — fewer (s, t)
//     pairs keeps the SPIR-V small enough for fast driver compile.
//   - Light tracer (s = 1) IS implemented: every non-delta light vertex is
//     projected onto camera and atomic-added into lightSplatBuffer. This
//     is the path that produces caustic spots on diffuse floors (pure-spec
//     connections via the t≥2 generic path are zero on co-planar surfaces).
//   - Directional light only via t=1 NEE (delta — no light walk).
//   - No env contribution from intermediate eye-walk escapes.

import common;
import bindings;
import interfaces;
import scene_trace;
import scene_lights;
import flat_shading;
import flat_material;
import lights.sphere_light;
import lights.emissive_triangle_light;
import lights.directional_light;

static const uint BDPT_MAX_DEPTH = 4u;
static const uint BDPT_MAX_VERTS = BDPT_MAX_DEPTH + 1u;   // 5

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
    float3 albedo;        // cached diffuse coefficient for Lambertian connection eval
    float3 emission;      // Le at this vertex (zero if not emissive)
    float  pdfFwd;        // area-measure pdf of arriving here from origin side
    float  pdfRev;        // area-measure pdf of arriving here from far side
    bool   isDelta;
    bool   onLight;
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
    r.vertex.albedo = float3(0.0);
    r.vertex.emission = float3(0.0);
    r.vertex.pdfFwd = 0.0;
    r.vertex.pdfRev = 0.0;
    r.vertex.isDelta = false;
    r.vertex.onLight = true;
    r.dirOut = float3(0.0, 1.0, 0.0);
    r.beta = float3(0.0);
    r.pdfPos = 0.0;
    r.pdfDir = 0.0;

    uint nSphere = fc.numSphereLights;
    uint nTri    = fc.numEmissiveTriangles;
    bool hasDir  = (fc.useDirectLight != 0u);
    uint nGroups = (hasDir ? 1u : 0u) + (nSphere > 0u ? 1u : 0u) + (nTri > 0u ? 1u : 0u);
    if (nGroups == 0u) return r;

    float groupPdf = 1.0 / float(nGroups);
    uint  group    = uint(min(rng.next() * float(nGroups), float(nGroups - 1u)));

    uint chosen = 0xFFFFFFFFu;
    uint cursor = 0u;
    if (nSphere > 0u) { if (cursor == group) chosen = 0u; cursor++; }
    if (nTri    > 0u) { if (cursor == group) chosen = 1u; cursor++; }
    if (hasDir)       { if (cursor == group) chosen = 2u; cursor++; }
    if (chosen == 0xFFFFFFFFu) return r;

    if (chosen == 2u)
    {
        float3 dir = normalize(lightDirection);
        r.vertex.kind = BDPT_VK_LIGHT_DIR;
        r.vertex.N = -dir;
        r.vertex.emission = lightRadiance;
        r.vertex.throughput = lightRadiance;
        r.vertex.pdfFwd = groupPdf;
        r.vertex.isDelta = true;
        r.dirOut = dir;
        r.beta = lightRadiance;
        r.pdfPos = groupPdf;
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
    inout RNG rng)
{
    int idx = startIdx;
    float pdfFwdOmega = pdfDirOmegaIn;

    [loop]
    while (idx < int(BDPT_MAX_VERTS))
    {
        HitInfo h = traceScene(fc, ray);
        [loop]
        for (uint skip = 0u; skip < 8u && h.hit && isCutoutTransparent(h); skip++)
        {
            ray.origin = h.position + ray.direction * 0.002;
            h = traceScene(fc, ray);
        }
        if (!h.hit) break;
        if ((materialTypes[h.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT) break;

        FlatMaterial mat = loadFlatMaterial(h);
        FlatHitData fd = mat.data;

        BDPTVertex v;
        v.kind = BDPT_VK_SURFACE;
        v.position = h.position;
        v.N = fd.N;
        v.throughput = throughput;
        v.albedo = fd.mat.albedo;
        v.emission = fd.mat.emission;
        v.pdfFwd = 0.0;
        v.pdfRev = 0.0;
        v.isDelta = false;
        v.onLight = any(fd.mat.emission > float3(0.0));

        BDPTVertex prev = path[idx - 1];
        v.pdfFwd = convertSAtoArea(pdfFwdOmega, prev.position, v);

        // Update prev.pdfRev using Lambertian approximation for the
        // backward direction at v.
        if (prev.kind != BDPT_VK_CAMERA && !prev.isDelta)
        {
            float3 toPrev = prev.position - v.position;
            float toPrevLen2 = dot(toPrev, toPrev);
            if (toPrevLen2 > 1e-20)
            {
                float3 toPrevDir = toPrev / sqrt(toPrevLen2);
                float pdfRevOmega = lambertSApdf(v, toPrevDir);
                path[idx - 1].pdfRev = convertSAtoArea(pdfRevOmega, v.position, prev);
            }
        }

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

    float G = cosZ * cosY / d2;
    float3 fz = z.albedo / PI;
    float3 fy;
    if (y.kind == BDPT_VK_SURFACE)
        fy = y.albedo / PI;
    else
        fy = float3(1.0 / PI);

    r.contribution = z.throughput * fz * G * fy * y.throughput;

    float pdf_z_sa = cosZ / PI;
    float pdf_y_sa = cosY / PI;
    r.pdf_zsm1_rev = convertSAtoArea(pdf_y_sa, y.position, z);
    r.pdf_ytm1_rev = convertSAtoArea(pdf_z_sa, z.position, y);

    if (s >= 2)
    {
        float3 toPrev = eye[s - 2].position - z.position;
        float l2 = dot(toPrev, toPrev);
        if (l2 > 1e-20)
        {
            float3 dp = toPrev / sqrt(l2);
            r.pdf_zsm2_rev = convertSAtoArea(lambertSApdf(z, dp),
                                             z.position, eye[s - 2]);
        }
    }
    if (t >= 2)
    {
        float3 toPrev = lit[t - 2].position - y.position;
        float l2 = dot(toPrev, toPrev);
        if (l2 > 1e-20)
        {
            float3 dp = toPrev / sqrt(l2);
            r.pdf_ytm2_rev = convertSAtoArea(lambertSApdf(y, dp),
                                             y.position, lit[t - 2]);
        }
    }
    r.nonzero = any(r.contribution > float3(0.0));
    return r;
}

// t = 1 NEE: connect eye[s-1] to a freshly sampled light point.
// Standard NEE with power heuristic against the BSDF strategy.
float3 connectT1(BDPTVertex eye[BDPT_MAX_VERTS], int s, inout RNG rng)
{
    BDPTVertex z = eye[s - 1];
    if (z.isDelta || z.kind != BDPT_VK_SURFACE) return float3(0.0);

    float3 total = float3(0.0);
    float3 fz = z.albedo / PI;

    if (fc.useDirectLight != 0u)
    {
        float3 dir = normalize(lightDirection);
        float NdotL = max(dot(z.N, dir), 0.0);
        if (NdotL > 0.0 && visibleDirectional(fc, z.position, z.N, dir))
        {
            total += z.throughput * fz * NdotL * lightRadiance;
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

        float invPdfOmega = cosLight / (d2 * ls.pdfArea);
        float pdfLightSA = d2 * ls.pdfArea / max(cosLight, 1e-12);
        float wNEE = powerHeuristic(pdfLightSA, NdotL / PI);
        total += z.throughput * fz * NdotL * ls.radiance * invPdfOmega * wNEE;
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
                    float invPdfOmega = cosLight / (d2 * ls.pdfArea);
                    float pdfLightSA = d2 * ls.pdfArea / max(cosLight, 1e-12);
                    float wNEE = powerHeuristic(pdfLightSA, NdotL / PI);
                    total += z.throughput * fz * NdotL * ls.radiance * invPdfOmega * wNEE;
                }
            }
        }
    }

    return total;
}

// ── Light tracer (s = 1) — camera splat ────────────────────────

static const float BDPT_SPLAT_SCALE = 1024.0;     // Q22.10 fixed-point

// Project world-space `p` to raster pixel coords. Returns true if visible
// in the camera frustum and within image bounds. Out: pixel (px,py),
// cosCam = dot(camera_forward, -wToCam) for camera-importance term.
bool projectToCamera(float3 p, out int2 outPixel, out float cosCam, out float distToCam)
{
    outPixel = int2(0);
    cosCam = 0.0;
    distToCam = 0.0;
    float3 toP = p - fc.camera.position;
    distToCam = length(toP);
    if (distToCam <= 1e-4) return false;
    // _look_at builds V with row-2 = -forward, so visible points (in front
    // of camera) have view-space z < 0. Likewise the projection matrix sets
    // clip.w = -viewSp.z so visible points yield clip.w > 0.
    float4 viewSp = mul(fc.camera.view, float4(p, 1.0));
    float4 clip = mul(fc.camera.proj, viewSp);
    if (clip.w <= 1e-4) return false;
    float3 ndc = clip.xyz / clip.w;
    if (any(abs(ndc.xy) > float2(1.0))) return false;
    // Match generateCameraRay's NDC.y flip.
    float xs = (ndc.x * 0.5 + 0.5) * float(fc.width);
    float ys = (-ndc.y * 0.5 + 0.5) * float(fc.height);
    if (xs < 0.0 || ys < 0.0 || xs >= float(fc.width) || ys >= float(fc.height))
        return false;
    outPixel = int2(int(xs), int(ys));

    // Camera forward in world: ray-gen reconstructs forward as
    // `viewInverse · normalize(projInverse · (0, 0, 1, 1))`. The +Z hemisphere
    // of view space holds nothing visible (visible viewSp.z < 0), so the
    // canonical world-space forward is `viewInverse · (0, 0, -1, 0)`.
    float3 camForward = normalize(mul(fc.camera.viewInverse, float4(0, 0, -1, 0)).xyz);
    cosCam = max(dot(camForward, toP / distToCam), 0.0);
    return cosCam > 1e-4;
}

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
// BDPT strategy: the camera vertex is the deterministic camera position
// (pinhole), connected to a non-delta light-walk surface vertex.
void splatLightVertex(BDPTVertex y)
{
    if (y.isDelta || y.kind != BDPT_VK_SURFACE) return;
    int2 pixel;
    float cosCam;
    float distCam;
    if (!projectToCamera(y.position, pixel, cosCam, distCam)) return;

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
    // averages the per-iteration light-path estimator across frames.
    float3 fy = y.albedo / PI;
    float3 contrib = y.throughput * fy * cosY / (distCam * distCam) * We;
    atomicSplatRadiance(pixel, contrib);
}

void splatLightWalk(BDPTVertex lightPath[BDPT_MAX_VERTS], int lightLen)
{
    [loop]
    for (int i = 1; i < lightLen; i++)
    {
        splatLightVertex(lightPath[i]);
    }
}

// ── Integrator ──────────────────────────────────────────────────

struct BDPTIntegrator : IIntegrator
{
    float3 estimateRadiance(Ray ray, HitInfo firstHit, inout RNG rng)
    {
        BDPTVertex eye[BDPT_MAX_VERTS];

        BDPTVertex cam;
        cam.kind = BDPT_VK_CAMERA;
        cam.position = ray.origin;
        cam.N = -ray.direction;
        cam.throughput = float3(1.0);
        cam.albedo = float3(0.0);
        cam.emission = float3(0.0);
        cam.pdfFwd = 1.0;
        cam.pdfRev = 0.0;
        cam.isDelta = false;
        cam.onLight = false;
        eye[0] = cam;

        FlatMaterial fmat = loadFlatMaterial(firstHit);
        BDPTVertex z1;
        z1.kind = BDPT_VK_SURFACE;
        z1.position = firstHit.position;
        z1.N = fmat.data.N;
        z1.throughput = float3(1.0);
        z1.albedo = fmat.data.mat.albedo;
        z1.emission = fmat.data.mat.emission;
        z1.pdfFwd = 1.0;
        z1.pdfRev = 0.0;
        z1.isDelta = false;
        z1.onLight = any(fmat.data.mat.emission > float3(0.0));
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
            eyeLen = randomWalk(eye, 2, r2, throughput, pdfFwdOmega, rng);
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
                lightLen = randomWalk(lightPath, 1, rL, lo.beta, lo.pdfDir, rng);
            }
        }

        // s = 1 light tracer: project each non-delta light vertex onto the
        // camera and atomic-add the contribution into lightSplatBuffer.
        // main_pass.slang composites the splat at end-of-frame.
        splatLightWalk(lightPath, lightLen);

        float3 L = float3(0.0);

        // (s ≥ 2, t = 0): emissive eye-vertex hits.
        [loop]
        for (int s = 2; s <= eyeLen; s++)
        {
            BDPTVertex z = eye[s - 1];
            if (z.onLight)
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
