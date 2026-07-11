// Wavefront bidirectional path tracer (P1 §8 / design §P1-G). The megakernel
// BDPTIntegrator.estimateRadiance, decomposed into compute dispatches that store
// the camera + light subpath vertices in VRAM between stages. Three subpath-build
// strategies (selected per the pass's walk_mode; all produce the identical image):
//
//   megakernel — wfBdptWalk: build both subpaths + the s=1 splat in one kernel.
//   eye        — wfBdptGenEye + per-bounce wfBdptBounceEye (staged eye walk),
//                then wfBdptLightTail (megakernel light walk + splat).
//   eye_light  — staged eye walk + staged light walk (gen/bounce-light) +
//                standalone wfBdptSplat.
//
// All modes then share: wfBdptClassify → build_args → scatter → split connect
// (wfBdptConnectNee = emissive+t=1 NEE; wfBdptConnectFull adds t≥2 generic+MIS,
// each indirect-dispatched over only its lanes) → wfBdptResolve (running mean +
// display). The staged bounce kernels reimplement integrators/bdpt.slang's
// randomWalk body verbatim; the megakernel + light-tail call it directly — so
// every mode is the same estimator as the megakernel bdpt (A/B-verified, exact).
// The rng cursor is carried across the staged dispatches so the sequence matches.
// Scope: flat first-hit only, pinhole camera. Set 0 is the megakernel scene
// descriptor set (incl. lightSplatBuffer for the splat); set 1 holds the per-lane
// subpath + aux + counting-sort buffers this owns.

import common;            // RNG, Ray, PI
import bindings;          // fc, accumBuffer, lightSplatBuffer, materialTypes…
import interfaces;        // BSDFSample
import scene_trace;       // traceScene
import scene_lights;      // intersectSphereLights, NearestSphereHit
import lights.sphere_light; // SphereLight
import environment;       // sceneEnvRadiance, envPdf
import cameras.pinhole;   // PinholeCamera
import materials.flat.flat_material;   // FlatMaterial, loadFlatMaterial
import materials.flat.flat_shading;    // isCutoutTransparent
import integrators.bdpt;  // BDPTVertex, randomWalk, sampleLightOrigin,
                          // connectT1, connectGeneric, misWeight, ConnectResult,
                          // bdptEnvNEE, splatLightWalk, BDPT_MAX_VERTS, BDPT_VK_*
import integrators.path;  // PathTracer — non-flat first-hit fallback (parity
                          // with main_pass.slang's useBdpt=FLAT-only gate)
import wavefront.wf_display;  // wfWriteDisplay (tonemap → output image)
#if defined(SKINNY_SPECTRAL)
import spectrum;                          // SampledWavelengths, sampleWavelengths, terminateSecondary, planckSpectrum, spectrumResolveToLinearSRGB
import integrators.spectral_flat_common;  // SpectralFlatColors, upsampleFlatColors, flatResponseS, flatResponseNEE
import lights.emissive_triangle_light;    // EmissiveTriangleLightImpl, loadEmissiveTriangleLightImpl (imports are not transitive)
import lights.directional_light;          // DirectionalLightImpl for spectral distant-light origins
// bindings.slang (already imported) supplies upsampleIlluminantBound, namedMetalEtaK,
// materialBlackbody, emitterBlackbody, distantLightSpd, planckSpectrum in the spectral build.
#endif

static const uint WF_BDPT_ALIVE = 1u;   // ewFlags bit: eye walk still extending

struct WfBdptAux
{
    int    eyeLen;
    int    lightLen;
    uint   rngState;     // rng cursor (carried across eye-walk bounces → light tail → connect)
    float  lensWeight;   // realistic-camera weight applied to the eye-side L
    uint   pixel;        // global pixel lane for this slot (SENTINEL = out-of-frame)
    // Color roles carry `Spectrum` (= float3 RGB → byte-identical layout; = float4
    // under SKINNY_SPECTRAL, per-λ). Resolved to linear sRGB at the film write
    // (wfBdptResolve) / per-splat (wfBdptSplat).
    Spectrum escaped;    // escaped env/emissive radiance gathered during the eye walk
    Spectrum radiance;   // final eye-side L (written by connect, read by resolve)
    // Transient eye-walk loop state (S2): carried across bounce-eye dispatches,
    // unused once the eye walk completes (eyeLen frozen, ewFlags ALIVE cleared).
    float3 ewRayO;       // current bounce ray origin
    float3 ewRayD;       // current bounce ray direction
    Spectrum ewThroughput; // path throughput arriving at the next vertex
    float  ewPdfFwdOmega;// sa pdf of the bounce that spawned the current ray
    float  ewMisBsdfPdf; // env-NEE MIS partner pdf (-1 = none)
    uint   ewFlags;      // WF_BDPT_ALIVE while the eye walk is still extending
#if defined(SKINNY_SPECTRAL)
    SampledWavelengths sw;  // hero wavelengths + pdfs, drawn once at wfBdptGenEye,
                            // shared by the eye AND light subpath of this sample
#endif
};

// Tiled streaming: the eye/light/aux buffers hold a fixed-size stream (slot
// index = tid.x), not one entry per pixel. The CPU loops `streamBase` over the
// frame; slot i maps to pixel `streamBase + i`, and connect/resolve recover the
// pixel from the stored `aux.pixel`. Out-of-frame tail slots are flagged dead.
static const uint WF_PIXEL_SENTINEL = 0xFFFFFFFFu;
// streamBase: per-tile pixel base (walk). shadeSlot: which connection slot an
// indirect-dispatched connect kernel processes. streamSize: slots in the stream
// buffers — bounds the full-stream kernels (the dispatch rounds up to 64).
struct WfTilePC { uint streamBase; uint shadeSlot; uint streamSize; };
[[vk::push_constant]] ConstantBuffer<WfTilePC> wfTile;

[[vk::binding(0, 1)]] RWStructuredBuffer<BDPTVertex> wfEye;     // stream × BDPT_MAX_VERTS
[[vk::binding(1, 1)]] RWStructuredBuffer<BDPTVertex> wfLight;   // stream × BDPT_MAX_VERTS
[[vk::binding(2, 1)]] RWStructuredBuffer<WfBdptAux>  wfAux;     // stream

// Connect-stage counting sort (P1 §8 compaction port). After the walk, each
// live lane is classified by subpath shape into one connection slot; the heavy
// generic+MIS double loop runs only for lanes whose light subpath has ≥2
// vertices. Mirrors the path tracer's wfLaneSlot/…/wfIndirectArgs (set 1,
// bindings 2..7) — here at bindings 3..8 (0..2 are the subpath/aux buffers).
static const uint WF_BDPT_NUM_SLOTS = 2u;
static const uint WF_BDPT_SLOT_NEE  = 0u;   // emissive + connectT1 (lightLen < 2)
static const uint WF_BDPT_SLOT_FULL = 1u;   // + generic t≥2 + MIS (lightLen ≥ 2)
static const uint WF_BDPT_GROUP     = 64u;  // connect [numthreads(64,1,1)]

[[vk::binding(3, 1)]] RWStructuredBuffer<uint> wfLaneSlot;     // [stream]  slot per lane (SENTINEL = dead)
[[vk::binding(4, 1)]] RWStructuredBuffer<uint> wfSlotCount;    // [NUM_SLOTS] live lanes per slot
[[vk::binding(5, 1)]] RWStructuredBuffer<uint> wfSlotOffset;   // [NUM_SLOTS] queue base per slot
[[vk::binding(6, 1)]] RWStructuredBuffer<uint> wfSlotQueue;    // [stream]  lane indices, grouped by slot
[[vk::binding(7, 1)]] RWStructuredBuffer<uint> wfSlotCursor;   // [NUM_SLOTS] scatter write cursor
[[vk::binding(8, 1)]] RWStructuredBuffer<uint> wfIndirectArgs; // [NUM_SLOTS*3] (x,y,z) per slot

// Resolve the stream slot an indirect-dispatched connect kernel processes, or
// 0xFFFFFFFF past the slot's queue (the lane early-outs). `qpos` is the dispatch
// thread index within the slot's slice.
uint wfBdptQueueLane(uint slot, uint qpos)
{
    if (qpos >= wfSlotCount[slot]) return 0xFFFFFFFFu;
    return wfSlotQueue[wfSlotOffset[slot] + qpos];
}

#if defined(SKINNY_SPECTRAL)
// ── Spectral transport helpers (Group 3) ────────────────────────
//
// Hero-wavelength recolor for the staged wavefront BDPT, mirroring the megakernel
// integrators/bdpt_spectral.slang but operating on `BDPTVertex` directly — the D1
// retype gave BDPTVertex `Spectrum` throughput/emission while keeping all scalar
// geometry/pdf fields, so the scalar bdpt.slang machinery (bdptSurface / bdptEval /
// convertSAtoArea / misWeight / splatMisWeight) is reused verbatim (no `asRgb`
// projection) and only the color-carrying evaluations recolor per wavelength via
// spectral_flat_common. Directions/pdfs/MIS are the SAME scalars the RGB kernels
// compute — the single MIS source (D3). Compiled only under SKINNY_SPECTRAL.

// Widen a float3 into `Spectrum` — for the non-flat PathTracer fallback whose RGB
// return is stored into the now-`Spectrum` `escaped` (that branch is unreachable
// under spectral: non-flat scenes are refused at startup — the widen only keeps it
// type-checking).
#define WF_SPEC3(c) Spectrum((c).x, (c).y, (c).z, 0.0)

struct WfSpectralEval { Spectrum response; float pdf; };

SpectralFlatColors wfColsFor(FlatMaterial mat, uint matId, SampledWavelengths sw)
{
    return upsampleFlatColors(mat, sw, flatMaterials[matId].conductorMetalId);
}

// f·cos·opacity per wavelength + the SAME scalar pdf the RGB integrator uses.
WfSpectralEval wfSpectralBdptEval(BdptSurface s, SpectralFlatColors cols, float3 N,
                                  float3 woWorld, float3 wiWorld)
{
    WfSpectralEval e;
    float3 wo = worldToTangent(woWorld, s.T, s.B, N);
    float3 wi = worldToTangent(wiWorld, s.T, s.B, N);
    e.response = flatResponseNEE(s.mat, cols, wo, wi);
    e.pdf      = s.mat.evaluate(wo, wi).pdf;
    return e;
}

// Environment NEE (spectral), mirror of bdpt.bdptEnvNEE / bdpt_spectral.spectralBdptEnvNEE.
Spectrum wfSpectralEnvNEE(FlatMaterial mat, SpectralFlatColors cols,
                          float3 P, float3 N, float3 T, float3 B,
                          float3 wo_ts, uint matId, SampledWavelengths sw, inout RNG rng)
{
    if (effectiveFurnaceMode(matId) != 0u) return Spectrum(0.0);
    EnvSample es = sampleEnvDir(rng.next2(), fc.envIntensity);
    if (es.pdf <= 0.0 || dot(N, es.dir) <= 0.0) return Spectrum(0.0);
    if (!visibleDirectional(fc, P, N, es.dir)) return Spectrum(0.0);
    float3   wi_ts = worldToTangent(es.dir, T, B, N);
    float    pdfB  = mat.evaluate(wo_ts, wi_ts).pdf;
    if (pdfB <= 0.0) return Spectrum(0.0);
    Spectrum resp  = flatResponseNEE(mat, cols, wo_ts, wi_ts);
    float    w     = powerHeuristic(es.pdf, pdfB);
    Spectrum Li    = upsampleIlluminantBound(es.radiance, sw);
    return resp * Li * (w / es.pdf);
}

// Light-side endpoint sampling (spectral β) — mirror of bdpt_spectral.sampleLightOriginS
// but producing a `BDPTVertex` (Spectrum color fields) so it drops straight into the
// stream buffers the staged kernels persist.
struct WfSpectralLightOrigin
{
    BDPTVertex vertex;
    float3   dirOut;
    Spectrum beta;
    float    pdfPos;
    float    pdfDir;
    bool     valid;
};

WfSpectralLightOrigin wfSampleLightOriginS(SampledWavelengths sw, inout RNG rng)
{
    WfSpectralLightOrigin r;
    r.valid = false;
    r.vertex.kind = BDPT_VK_LIGHT_AREA;
    r.vertex.position = float3(0.0);
    r.vertex.N = float3(0.0, 1.0, 0.0);
    r.vertex.throughput = Spectrum(0.0);
    r.vertex.emission = Spectrum(0.0);
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
    r.beta = Spectrum(0.0);
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
        uint idx = min(uint(rng.next() * float(nDir)), nDir - 1u);
        DistantLight L = distantLights[idx];
        float3 dir = normalize(L.direction);
        float lightPickPdf = groupPdf / float(nDir);
        Spectrum Le = upsampleIlluminantBound(L.radiance, sw);
        r.vertex.kind = BDPT_VK_LIGHT_DIR;
        r.vertex.N = -dir;
        r.vertex.emission = Le;
        r.vertex.throughput = Le;
        r.vertex.pdfFwd = lightPickPdf;
        r.vertex.isDelta = true;
        r.dirOut = dir;
        r.beta = Le;
        r.pdfPos = lightPickPdf;
        r.pdfDir = 1.0;
        r.valid = true;
        return r;
    }

    float3 lpos, lN, lRadRgb;
    float pdfArea;
    Spectrum Li;

    if (chosen == 0u)
    {
        uint idx = min(uint(rng.next() * float(nSphere)), nSphere - 1u);
        SphereLightImpl L = loadSphereLightImpl(sphereLights[idx]);
        LightSample ls = L.samplePoint(float3(0.0), rng.next2());
        if (!ls.valid) return r;
        lpos = ls.point; lN = ls.normal; lRadRgb = ls.radiance;
        pdfArea = ls.pdfArea;
        Li = upsampleIlluminantBound(lRadRgb, sw);
    }
    else
    {
        uint idx = min(uint(rng.next() * float(nTri)), nTri - 1u);
        EmissiveTriangle raw = emissiveTriangles[idx];
        EmissiveTriangleLightImpl L = loadEmissiveTriangleLightImpl(raw, nTri);
        LightSample ls = L.samplePoint(float3(0.0), rng.next2());
        if (!ls.valid) return r;
        lpos = ls.point; lN = ls.normal; lRadRgb = ls.radiance;
        pdfArea = ls.pdfArea * float(nTri);
        float2 bb = emitterBlackbody(idx);
        Li = (bb.x > 0.0) ? planckSpectrum(sw, bb.x) * bb.y
                          : upsampleIlluminantBound(lRadRgb, sw);
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
    r.vertex.emission = Li;
    r.vertex.throughput = Li / max(pdfPos, 1e-20);
    r.vertex.pdfFwd = pdfPos;
    r.dirOut = dirWorld;
    r.beta = Li * (cosOut / max(pdfPos * pdfDir, 1e-20));
    r.pdfPos = pdfPos;
    r.pdfDir = pdfDir;
    r.valid = true;
    return r;
}

// Spectral random walk (eye or light) — mirror of bdpt_spectral.randomWalkS on
// BDPTVertex. Used by the megakernel walk-mode kernels (wfBdptWalk / wfBdptLightTail)
// that build a whole subpath in one dispatch; the staged bounce kernels inline the
// same body per dispatch. pdf/MIS bookkeeping identical to the RGB walk.
int wfRandomWalkS(
    inout BDPTVertex path[BDPT_MAX_VERTS],
    int startIdx,
    Ray ray,
    Spectrum throughput,
    float pdfDirOmegaIn,
    inout SampledWavelengths sw,
    inout RNG rng,
    bool isEyeWalk,
    inout Spectrum escapedRadiance,
    float misBsdfPdfIn)
{
    int idx = startIdx;
    float pdfFwdOmega = pdfDirOmegaIn;
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

        if (isEyeWalk && fc.numSphereLights > 0u)
        {
            float t_max = h.hit ? h.t : 1e30;
            NearestSphereHit sph = intersectSphereLights(ray, t_max, fc.numSphereLights);
            if (sph.idx != 0xFFFFFFFFu)
            {
                SphereLight L = sphereLights[sph.idx];
                float w_bsdf = 1.0;
                BDPTVertex prev = path[idx - 1];
                if (!prev.isDelta && pdfFwdOmega > 0.0)
                {
                    float area = 4.0 * PI * L.radius * L.radius;
                    float pdf_light_sa = (sph.t * sph.t) / max(sph.cos_y * area, 1e-12);
                    w_bsdf = powerHeuristic(pdfFwdOmega, pdf_light_sa);
                }
                escapedRadiance += throughput * upsampleIlluminantBound(L.radiance, sw) * w_bsdf;
                break;
            }
        }

        if (!h.hit)
        {
            if (isEyeWalk)
            {
                float w = 1.0;
                if (misBsdfPdf > 0.0)
                    w = powerHeuristic(misBsdfPdf, envPdf(ray.direction));
                float3 envRgb = sceneEnvRadiance(ray.direction, fc.envIntensity,
                                                 effectiveFurnaceMode(h.materialId));
                escapedRadiance += throughput * upsampleIlluminantBound(envRgb, sw) * w;
            }
            break;
        }
        if ((materialTypes[h.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT) break;

        FlatMaterial mat = loadFlatMaterial(h);
        FlatHitData fd = mat.data;
        SpectralFlatColors cols = wfColsFor(mat, h.materialId, sw);

        BDPTVertex v;
        v.kind = BDPT_VK_SURFACE;
        v.position = h.position;
        v.N = fd.N;
        v.throughput = throughput;
        float2 matBB = materialBlackbody(h.materialId);
        v.emission = (matBB.x > 0.0) ? planckSpectrum(sw, matBB.x) * matBB.y : cols.emission;
        v.pdfFwd = 0.0;
        v.pdfRev = 0.0;
        v.isDelta = false;
        v.onLight = any(fd.mat.emission > float3(0.0)) || (matBB.x > 0.0);
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
        float3 wo_ts = worldToTangent(-ray.direction, T, B, v.N);

        if (isEyeWalk)
            escapedRadiance += throughput *
                wfSpectralEnvNEE(mat, cols, v.position, v.N, T, B, wo_ts, v.matId, sw, rng);

        BSDFSample bs = mat.sample(wo_ts, rng);
        if (!bs.valid) break;
        float3 wi_world = tangentToWorld(bs.wi, T, B, v.N);

        float glassB = flatMaterials[h.materialId].glassCauchyB;
        if (bs.pdf == 0.0 && glassB > 0.0 && wo_ts.z * bs.wi.z < 0.0)
        {
            float  lam0um   = sw.lambda.x * 1e-3;
            float  nHero    = mat.data.mat.ior + glassB / (lam0um * lam0um);
            bool   entering = wo_ts.z > 0.0;
            float3 faceN    = entering ? float3(0, 0, 1) : float3(0, 0, -1);
            float  etaH     = entering ? (1.0 / nHero) : nHero;
            float3 nd       = refract(-wo_ts, faceN, etaH);
            if (all(nd == float3(0.0)))
                nd = reflect(-wo_ts, faceN);
            bs.wi          = nd;
            bs.transmitted = nd.z < 0.0;
            wi_world       = tangentToWorld(bs.wi, T, B, v.N);
            sw             = terminateSecondary(sw);
        }

        bool deltaBounce = (bs.pdf <= 0.0);
        if (deltaBounce)
            path[idx - 1].isDelta = true;

        if (prev.kind != BDPT_VK_CAMERA && !prev.isDelta)
        {
            float3 toPrev = prev.position - v.position;
            float toPrevLen2 = dot(toPrev, toPrev);
            if (toPrevLen2 > 1e-20)
            {
                float3 toPrevDir = toPrev / sqrt(toPrevLen2);
                float pdfRevOmega;
                if (deltaBounce)
                    pdfRevOmega = lambertSApdf(v, toPrevDir);
                else
                {
                    float3 woRev = worldToTangent(wi_world, T, B, v.N);
                    float3 wiRev = worldToTangent(toPrevDir, T, B, v.N);
                    pdfRevOmega = mat.evaluate(woRev, wiRev).pdf;
                }
                path[prevIdx].pdfRev = convertSAtoArea(pdfRevOmega, v.position, prev);
            }
        }

        Spectrum weightS;
        if (bs.pdf > 0.0)
            weightS = flatResponseS(mat, cols, wo_ts, bs.wi) / bs.pdf;
        else
            weightS = cols.transmission;
        throughput *= weightS;

        if (idx >= 2)
        {
            float mx = max(max(throughput.x, throughput.y), max(throughput.z, throughput.w));
            float p = clamp(mx, 0.05, 0.95);
            if (rng.next() > p) break;
            throughput /= p;
        }

        float3 spawnOffset = bs.transmitted ? -v.N * 0.002 : v.N * 0.002;
        ray.origin = v.position + spawnOffset;
        ray.direction = wi_world;
        pdfFwdOmega = deltaBounce ? 1.0 : bs.pdf;
        misBsdfPdf = (isEyeWalk && !deltaBounce && !bs.transmitted
                      && effectiveFurnaceMode(v.matId) == 0u)
                   ? bs.pdf : -1.0;
    }
    return idx;
}

// Generic connection (t ≥ 2), mirror of bdpt_spectral.connectGenericS. `eye`/`lit`
// are BDPTVertex[]; scalar MIS partners are the RGB geometry pdfs.
struct WfSpectralConnectResult
{
    Spectrum contribution;
    float pdf_zsm1_rev;
    float pdf_zsm2_rev;
    float pdf_ytm1_rev;
    float pdf_ytm2_rev;
    bool  nonzero;
};

WfSpectralConnectResult wfConnectGenericS(
    BDPTVertex eye[BDPT_MAX_VERTS], int s,
    BDPTVertex lit[BDPT_MAX_VERTS], int t,
    SampledWavelengths sw)
{
    WfSpectralConnectResult r;
    r.contribution = Spectrum(0.0);
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

    BdptSurface zs = bdptSurface(z);
    SpectralFlatColors zcols = wfColsFor(zs.mat, z.matId, sw);
    float3 woZ = normalize(eye[s - 2].position - z.position);
    WfSpectralEval ez = wfSpectralBdptEval(zs, zcols, z.N, woZ, dir);
    Spectrum respZ = ez.response;
    float  pdf_z_sa = ez.pdf;

    Spectrum respY;
    float  pdf_y_sa;
    if (y.kind == BDPT_VK_SURFACE)
    {
        BdptSurface ys = bdptSurface(y);
        SpectralFlatColors ycols = wfColsFor(ys.mat, y.matId, sw);
        float3 woY = normalize(lit[t - 2].position - y.position);
        WfSpectralEval ey = wfSpectralBdptEval(ys, ycols, y.N, woY, -dir);
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
        respY = Spectrum(cosY / PI);
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
    r.nonzero = any(r.contribution > Spectrum(0.0));
    return r;
}

// t = 1 NEE (spectral), mirror of bdpt_spectral.connectT1S.
Spectrum wfConnectT1S(BDPTVertex eye[BDPT_MAX_VERTS], int s,
                      SampledWavelengths sw, inout RNG rng)
{
    BDPTVertex z = eye[s - 1];
    if (z.isDelta || z.kind != BDPT_VK_SURFACE) return Spectrum(0.0);

    Spectrum total = Spectrum(0.0);
    BdptSurface zs = bdptSurface(z);
    SpectralFlatColors zcols = wfColsFor(zs.mat, z.matId, sw);
    float3 woWorld = normalize(eye[s - 2].position - z.position);

    for (uint di = 0u; di < fc.numDistantLights; di++)
    {
        DistantLight L = distantLights[di];
        float3 dir = normalize(L.direction);
        float NdotL = max(dot(z.N, dir), 0.0);
        if (NdotL > 0.0 && visibleDirectional(fc, z.position, z.N, dir))
        {
            WfSpectralEval e = wfSpectralBdptEval(zs, zcols, z.N, woWorld, dir);
            int slot = int(L.spdIndex);
            Spectrum Li = (slot >= 0) ? distantLightSpd(slot, sw)
                                      : upsampleIlluminantBound(L.radiance, sw);
            total += z.throughput * e.response * Li;
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

        WfSpectralEval e = wfSpectralBdptEval(zs, zcols, z.N, woWorld, dir);
        float invPdfOmega = cosLight / (d2 * ls.pdfArea);
        float pdfLightSA = d2 * ls.pdfArea / max(cosLight, 1e-12);
        float wNEE = powerHeuristic(pdfLightSA, e.pdf);
        Spectrum Li = upsampleIlluminantBound(ls.radiance, sw);
        total += z.throughput * e.response * Li * (invPdfOmega * wNEE);
    }

    if (fc.numEmissiveTriangles > 0u)
    {
        uint idx = sampleEmissiveTriangle(rng.next(), fc.numEmissiveTriangles);
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
                    WfSpectralEval e = wfSpectralBdptEval(zs, zcols, z.N, woWorld, dir);
                    float invPdfOmega = cosLight / (d2 * ls.pdfArea);

                    BDPTVertex lit0;
                    lit0.kind = BDPT_VK_LIGHT_AREA;
                    lit0.position = ls.point;
                    lit0.N = ls.normal;
                    lit0.throughput = Spectrum(0.0);
                    lit0.emission = Spectrum(0.0);
                    lit0.pdfFwd = ls.pdfArea;
                    lit0.pdfRev = 0.0;
                    lit0.isDelta = false;
                    lit0.onLight = true;
                    lit0.matId = 0u;
                    lit0.uv = float2(0.0);
                    lit0.posObject = float3(0.0);
                    lit0.geoN = ls.normal;
                    lit0.tangent = float3(1.0, 0.0, 0.0);
                    lit0.hasTangent = false;
                    BDPTVertex litArr[BDPT_MAX_VERTS];
                    litArr[0] = lit0;

                    float pdf_ytm1_rev = e.pdf * cosLight / d2;
                    float pdf_zsm1_rev = (cosLight / PI) * NdotL / d2;
                    float pdf_zsm2_rev = 0.0;
                    if (s >= 3)
                    {
                        float3 toPrev = eye[s - 2].position - z.position;
                        float l2 = dot(toPrev, toPrev);
                        if (l2 > 1e-20)
                        {
                            float3 dp = toPrev / sqrt(l2);
                            float pr = bdptEval(zs, z.N, dir, dp).pdf;
                            pdf_zsm2_rev = convertSAtoArea(pr, z.position, eye[s - 2]);
                        }
                    }

                    float w = misWeight(eye, s, litArr, 1,
                                        pdf_zsm1_rev, pdf_zsm2_rev,
                                        pdf_ytm1_rev, 0.0);
                    float2 bb = emitterBlackbody(idx);
                    Spectrum Li = (bb.x > 0.0) ? planckSpectrum(sw, bb.x) * bb.y
                                               : upsampleIlluminantBound(ls.radiance, sw);
                    total += z.throughput * e.response * Li * (invPdfOmega * w);
                }
            }
        }
    }

    return total;
}

// s = 1 light-tracer camera splat (spectral), mirror of bdpt_spectral.splatLightVertexS.
// Resolves λ → linear sRGB BEFORE the atomic add (D4). NOTE: the per-splat resolve
// applies the CIE film gamut clamp per splat (clamp-per-splat ≠ clamp-of-sum) — an
// accepted v1 bias for out-of-gamut hero-collapsed dispersion splats. If the
// dispersion-caustic gate shows hue/energy bias, switch lightSplatBuffer to
// signed-XYZ accumulation (3-wide) and convert at composite.
void wfSplatLightVertexS<TC : ICamera>(BDPTVertex lightPath[BDPT_MAX_VERTS], int i,
                                       SampledWavelengths sw, inout RNG rng, TC cam)
{
    BDPTVertex y = lightPath[i];
    if (y.isDelta || y.kind != BDPT_VK_SURFACE) return;
    int2 pixel;
    float cosCam;
    float distCam;
    if (!cam.sampleWi(y.position, rng, pixel, cosCam, distCam)) return;

    float3 toCam = (fc.camera.position - y.position) / distCam;
    float cosY = max(dot(y.N, toCam), 0.0);
    if (cosY <= 0.0) return;
    if (!visibleSegment(fc, y.position, fc.camera.position)) return;

    float fovRad = fc.camera.fov * (PI / 180.0);
    float halfH = tan(fovRad * 0.5);
    float halfW = halfH * (float(fc.width) / float(fc.height));
    float A_image = 4.0 * halfW * halfH;
    float cos4 = cosCam * cosCam * cosCam * cosCam;
    float We = 1.0 / max(A_image * cos4, 1e-12);

    BdptSurface ys = bdptSurface(y);
    SpectralFlatColors ycols = wfColsFor(ys.mat, y.matId, sw);
    float3 woWorld = normalize(lightPath[i - 1].position - y.position);
    WfSpectralEval e = wfSpectralBdptEval(ys, ycols, y.N, woWorld, toCam);
    if (!any(e.response > Spectrum(0.0))) return;

    int t = i + 1;
    float pdfCamDirSA = We * cosCam;
    float pdf_ytm1_rev = pdfCamDirSA * cosY / (distCam * distCam);

    float pdf_ytm2_rev = 0.0;
    float3 toPrev = lightPath[i - 1].position - y.position;
    float l2 = dot(toPrev, toPrev);
    if (l2 > 1e-20)
    {
        float3 dp = toPrev / sqrt(l2);
        float pr = bdptEval(ys, y.N, toCam, dp).pdf;
        pdf_ytm2_rev = convertSAtoArea(pr, y.position, lightPath[i - 1]);
    }

    float w = splatMisWeight(lightPath, t, pdf_ytm1_rev, pdf_ytm2_rev);

    Spectrum contribS = y.throughput * e.response / (distCam * distCam) * (We * w);
    atomicSplatRadiance(pixel, spectrumResolveToLinearSRGB(sw, contribS));
}

void wfSplatLightWalkS<TC : ICamera>(BDPTVertex lightPath[BDPT_MAX_VERTS],
                                     int lightLen, SampledWavelengths sw, inout RNG rng, TC cam)
{
    [loop]
    for (int i = 1; i < lightLen; i++)
        wfSplatLightVertexS<TC>(lightPath, i, sw, rng, cam);
}
#endif // SKINNY_SPECTRAL

// ── Megakernel walk variant (S1 walk_mode) ──────────────────────
//
// The original single-kernel subpath build (eye randomWalk + light randomWalk +
// s=1 splat), used by walk_mode = megakernel (the default). Thin wrappers over
// integrators/bdpt.slang's randomWalk/sampleLightOrigin/splatLightWalk — no
// estimator logic duplicated. The staged kernels below (eye / eye_light modes)
// reimplement randomWalk's body per bounce; only one set is dispatched per the
// pass's walk_mode. wfBdptLightTail is also used by walk_mode = eye (staged eye
// walk + megakernel light tail).
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptWalk(uint3 tid: SV_DispatchThreadID)
{
    uint slot = tid.x;                          // stream slot
    if (slot >= wfTile.streamSize) return;
    uint base = slot * BDPT_MAX_VERTS;
    uint lane = wfTile.streamBase + slot;        // global pixel lane
    if (lane >= fc.width * fc.height)
    {
        wfAux[slot].pixel = WF_PIXEL_SENTINEL;   // out-of-frame tail → skip later
        return;
    }
    uint2 pixel = uint2(lane % fc.width, lane / fc.width);

#if defined(SKINNY_SPECTRAL)
    // Spectral megakernel-walk mode: hero-λ eye + light subpaths + s=1 splat, per
    // bdpt_spectral.slang (draw sw after generateRay; both subpaths share sw).
    RNG rng = createRNG(pixel, fc.frameIndex);
    PinholeCamera cam;
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);
    SampledWavelengths sw = sampleWavelengths(rng.next());

    WfBdptAux aux;
    aux.eyeLen = 0;
    aux.lightLen = 0;
    aux.lensWeight = lensWeight;
    aux.pixel = lane;
    aux.escaped = Spectrum(0.0);
    aux.radiance = Spectrum(0.0);
    aux.ewRayO = float3(0.0); aux.ewRayD = float3(0.0); aux.ewThroughput = Spectrum(0.0);
    aux.ewPdfFwdOmega = 0.0; aux.ewMisBsdfPdf = -1.0; aux.ewFlags = 0u;
    aux.sw = sw;

    BDPTVertex eye[BDPT_MAX_VERTS];
    BDPTVertex camVtxS;
    camVtxS.kind = BDPT_VK_CAMERA;
    camVtxS.position = ray.origin;
    camVtxS.N = -ray.direction;
    camVtxS.throughput = Spectrum(1.0);
    camVtxS.emission = Spectrum(0.0);
    camVtxS.pdfFwd = 1.0;
    camVtxS.pdfRev = 0.0;
    camVtxS.isDelta = false;
    camVtxS.onLight = false;
    camVtxS.matId = 0u;
    camVtxS.uv = float2(0.0);
    camVtxS.posObject = float3(0.0);
    camVtxS.geoN = -ray.direction;
    camVtxS.tangent = float3(1.0, 0.0, 0.0);
    camVtxS.hasTangent = false;
    eye[0] = camVtxS;

    HitInfo firstHit = traceScene(fc, ray);
    Ray r = ray;
    for (uint skip = 0u; skip < 32u && firstHit.hit && isCutoutTransparent(firstHit); skip++)
    {
        r.origin = firstHit.position + r.direction * 0.002;
        firstHit = traceScene(fc, r);
    }
    if (!firstHit.hit)
    {
        aux.escaped = upsampleIlluminantBound(
            sceneEnvRadiance(r.direction, fc.envIntensity,
                             effectiveFurnaceMode(firstHit.materialId)), sw);
        aux.rngState = rng.state;
        wfAux[slot] = aux;
        return;
    }
    if ((materialTypes[firstHit.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
    {
        // Non-flat first hit is refused at startup under spectral — unreachable;
        // widen the RGB fallback only to keep it type-checking.
        PathTracer integrator;
        aux.escaped = WF_SPEC3(integrator.estimateRadiance(r, firstHit, rng));
        aux.rngState = rng.state;
        wfAux[slot] = aux;
        return;
    }

    FlatMaterial fmat = loadFlatMaterial(firstHit);
    SpectralFlatColors fcols = wfColsFor(fmat, firstHit.materialId, sw);
    float2 firstBB = materialBlackbody(firstHit.materialId);
    BDPTVertex z1;
    z1.kind = BDPT_VK_SURFACE;
    z1.position = firstHit.position;
    z1.N = fmat.data.N;
    z1.throughput = Spectrum(1.0);
    z1.emission = (firstBB.x > 0.0) ? planckSpectrum(sw, firstBB.x) * firstBB.y : fcols.emission;
    z1.pdfFwd = 1.0;
    z1.pdfRev = 0.0;
    z1.isDelta = false;
    z1.onLight = any(fmat.data.mat.emission > float3(0.0)) || (firstBB.x > 0.0);
    z1.matId = firstHit.materialId;
    z1.uv = firstHit.uv;
    z1.posObject = firstHit.positionObject;
    z1.geoN = firstHit.normal;
    z1.tangent = firstHit.tangent;
    z1.hasTangent = firstHit.hasTangent;
    eye[1] = z1;
    int eyeLen = 2;

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

    Spectrum escapedRadiance = wfSpectralEnvNEE(
        fmat, fcols, z1.position, z1.N, T, B, wo_ts, firstHit.materialId, sw, rng);

    BSDFSample bs = fmat.sample(wo_ts, rng);
    if (bs.valid)
    {
        if (bs.pdf <= 0.0)
            eye[1].isDelta = true;
        float3 wi_world = tangentToWorld(bs.wi, T, B, z1.N);
        float glassB = flatMaterials[firstHit.materialId].glassCauchyB;
        if (bs.pdf == 0.0 && glassB > 0.0 && wo_ts.z * bs.wi.z < 0.0)
        {
            float  lam0um   = sw.lambda.x * 1e-3;
            float  nHero    = fmat.data.mat.ior + glassB / (lam0um * lam0um);
            bool   entering = wo_ts.z > 0.0;
            float3 faceN    = entering ? float3(0, 0, 1) : float3(0, 0, -1);
            float  etaH     = entering ? (1.0 / nHero) : nHero;
            float3 nd       = refract(-wo_ts, faceN, etaH);
            if (all(nd == float3(0.0)))
                nd = reflect(-wo_ts, faceN);
            bs.wi          = nd;
            bs.transmitted = nd.z < 0.0;
            wi_world       = tangentToWorld(bs.wi, T, B, z1.N);
            sw             = terminateSecondary(sw);
        }
        Spectrum throughput;
        if (bs.pdf > 0.0)
            throughput = flatResponseS(fmat, fcols, wo_ts, bs.wi) / bs.pdf;
        else
            throughput = fcols.transmission;
        Ray r2;
        float3 spawnOffset = bs.transmitted ? -z1.N * 0.002 : z1.N * 0.002;
        r2.origin = z1.position + spawnOffset;
        r2.direction = wi_world;
        float pdfFwdOmega = (bs.pdf > 0.0) ? bs.pdf : 1.0;
        float misBsdfPdfIn = (bs.pdf > 0.0 && !bs.transmitted
            && effectiveFurnaceMode(firstHit.materialId) == 0u) ? bs.pdf : -1.0;
        eyeLen = wfRandomWalkS(eye, 2, r2, throughput, pdfFwdOmega, sw, rng,
                               true, escapedRadiance, misBsdfPdfIn);
    }

    BDPTVertex lightPath[BDPT_MAX_VERTS];
    int lightLen = 0;
    WfSpectralLightOrigin lo = wfSampleLightOriginS(sw, rng);
    if (lo.valid)
    {
        lightPath[0] = lo.vertex;
        lightLen = 1;
        if (lo.vertex.kind != BDPT_VK_LIGHT_DIR)
        {
            Ray rL;
            rL.origin = lo.vertex.position + lo.vertex.N * 0.002;
            rL.direction = lo.dirOut;
            Spectrum unusedEscape = Spectrum(0.0);
            lightLen = wfRandomWalkS(lightPath, 1, rL, lo.beta, lo.pdfDir, sw, rng,
                                     false, unusedEscape, -1.0);
        }
    }

    wfSplatLightWalkS<PinholeCamera>(lightPath, lightLen, sw, rng, cam);

    for (int i = 0; i < int(BDPT_MAX_VERTS); i++)
    {
        if (i < eyeLen)   wfEye[base + uint(i)]   = eye[i];
        if (i < lightLen) wfLight[base + uint(i)] = lightPath[i];
    }
    aux.eyeLen = eyeLen;
    aux.lightLen = lightLen;
    aux.escaped = escapedRadiance;
    aux.sw = sw;
    aux.rngState = rng.state;   // connect resumes from here
    wfAux[slot] = aux;
    return;
#else
    RNG rng = createRNG(pixel, fc.frameIndex);
    PinholeCamera cam;
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);

    WfBdptAux aux;
    aux.eyeLen = 0;
    aux.lightLen = 0;
    aux.lensWeight = lensWeight;
    aux.pixel = lane;
    aux.escaped = float3(0.0);
    aux.radiance = float3(0.0);
    aux.ewRayO = float3(0.0); aux.ewRayD = float3(0.0); aux.ewThroughput = float3(0.0);
    aux.ewPdfFwdOmega = 0.0; aux.ewMisBsdfPdf = -1.0; aux.ewFlags = 0u;

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

    HitInfo firstHit = traceScene(fc, ray);
    Ray r = ray;
    for (uint skip = 0u; skip < 32u && firstHit.hit && isCutoutTransparent(firstHit); skip++)
    {
        r.origin = firstHit.position + r.direction * 0.002;
        firstHit = traceScene(fc, r);
    }
    if (!firstHit.hit)
    {
        aux.escaped = sceneEnvRadiance(r.direction, fc.envIntensity,
                                       effectiveFurnaceMode(firstHit.materialId));
        aux.rngState = rng.state;
        wfAux[slot] = aux;
        return;
    }
    uint firstMatType = materialTypes[firstHit.materialId] & 0xFFu;
    if (firstMatType != MATERIAL_TYPE_FLAT)
    {
        // Non-flat first hit → path fallback for EVERY non-flat type, mirroring
        // the megakernel: main_pass.slang gates `useBdpt` on MATERIAL_TYPE_FLAT
        // and routes every non-flat first hit to PathTracer.estimateRadiance, so
        // BDPT's own non-flat "bail to black" branch is dead code there. The
        // terminal types (SUBSURFACE / SKIN) evaluate one vertex and stop; the
        // non-terminal types (VOLUME / PYTHON) run the full multi-bounce path.
        // The heavy multi-bounce case is kept within the macOS GPU watchdog by
        // the host bounding the Metal eye submit per tile (record_bdpt_loop's
        // `bound_heavy_eye`, change wavefront-nonflat-tiled-fallback) — the
        // row-band discipline of metal-megakernel-watchdog-tiling. The lane
        // builds no eye/light subpath (eyeLen/lightLen stay 0, as for a miss),
        // so the connect stage skips it; s=1 splats from other lanes still
        // composite via lightSplatBuffer.
        PathTracer integrator;
        aux.escaped = integrator.estimateRadiance(r, firstHit, rng);
        aux.rngState = rng.state;
        wfAux[slot] = aux;
        return;
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
        float misBsdfPdfIn = (bs.pdf > 0.0 && !bs.transmitted
            && effectiveFurnaceMode(firstHit.materialId) == 0u) ? bs.pdf : -1.0;
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

    // s = 1 light tracer splat (atomic into lightSplatBuffer, set 0).
    splatLightWalk<PinholeCamera>(lightPath, lightLen, rng, cam);

    // Store both subpaths + bookkeeping for the connection stage.
    for (int i = 0; i < int(BDPT_MAX_VERTS); i++)
    {
        if (i < eyeLen)   wfEye[base + uint(i)]   = eye[i];
        if (i < lightLen) wfLight[base + uint(i)] = lightPath[i];
    }
    aux.eyeLen = eyeLen;
    aux.lightLen = lightLen;
    aux.escaped = escapedRadiance;
    aux.rngState = rng.state;   // connect resumes from here
    wfAux[slot] = aux;
#endif // SKINNY_SPECTRAL
}


// Light subpath + s=1 splat as one kernel (walk_mode = eye). Runs for flat-eye
// lanes (eyeLen >= 2); resumes the rng cursor left by the staged eye walk.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptLightTail(uint3 tid: SV_DispatchThreadID)
{
    uint slot = tid.x;
    if (slot >= wfTile.streamSize)
        return;
    WfBdptAux aux = wfAux[slot];
    if (aux.pixel == WF_PIXEL_SENTINEL || aux.eyeLen < 2)
        return;
    uint base = slot * BDPT_MAX_VERTS;

    RNG rng; rng.state = aux.rngState;
    PinholeCamera cam;

#if defined(SKINNY_SPECTRAL)
    SampledWavelengths sw = aux.sw;   // shared with the (staged) eye walk
    BDPTVertex lightPath[BDPT_MAX_VERTS];
    int lightLen = 0;
    WfSpectralLightOrigin lo = wfSampleLightOriginS(sw, rng);
    if (lo.valid)
    {
        lightPath[0] = lo.vertex;
        lightLen = 1;
        if (lo.vertex.kind != BDPT_VK_LIGHT_DIR)
        {
            Ray rL;
            rL.origin = lo.vertex.position + lo.vertex.N * 0.002;
            rL.direction = lo.dirOut;
            Spectrum unusedEscape = Spectrum(0.0);
            lightLen = wfRandomWalkS(lightPath, 1, rL, lo.beta, lo.pdfDir, sw, rng,
                                     false, unusedEscape, -1.0);
        }
    }

    wfSplatLightWalkS<PinholeCamera>(lightPath, lightLen, sw, rng, cam);

    [loop]
    for (int k = 0; k < int(BDPT_MAX_VERTS); k++)
        if (k < lightLen) wfLight[base + uint(k)] = lightPath[k];

    aux.lightLen = lightLen;
    aux.sw = sw;
    aux.rngState = rng.state;   // connect resumes from here
    wfAux[slot] = aux;
#else
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

    splatLightWalk<PinholeCamera>(lightPath, lightLen, rng, cam);

    [loop]
    for (int k = 0; k < int(BDPT_MAX_VERTS); k++)
        if (k < lightLen) wfLight[base + uint(k)] = lightPath[k];

    aux.lightLen = lightLen;
    aux.rngState = rng.state;   // connect resumes from here
    wfAux[slot] = aux;
#endif // SKINNY_SPECTRAL
}


// ── Staged eye walk (S2) ────────────────────────────────────────
//
// The eye subpath is built across per-bounce dispatches instead of one
// megakernel randomWalk: wfBdptGenEye seeds eye[0]=camera + eye[1]=z1 and the
// first bounce ray; then for each bounce wfBdptBounceEye (indirect over the live
// lanes gathered by wfBdptEyeClassify → build_args → scatter) extends one vertex
// — a verbatim relocation of integrators/bdpt.slang::randomWalk's loop body, so
// the estimator is unchanged. Vertices live in wfEye (VRAM); only the small
// transient loop state (WfBdptAux.ew*) is carried between bounces. The light
// subpath + s=1 splat stay in wfBdptLightTail (megakernel; S3 stages them).

// Seed eye[0]=camera, eye[1]=z1, the z1 env-NEE + first BSDF sample, and the
// transient loop state. Mirrors the head of estimateRadiance. Dead lanes (miss /
// non-flat / out-of-frame) are finalised here (escaped only, eyeLen < 2, not
// alive); flat-hit lanes start the bounce loop alive.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptGenEye(uint3 tid: SV_DispatchThreadID)
{
    uint slot = tid.x;
    if (slot >= wfTile.streamSize)
        return;
    uint base = slot * BDPT_MAX_VERTS;
    uint lane = wfTile.streamBase + slot;
    if (lane >= fc.width * fc.height)
    {
        wfAux[slot].pixel = WF_PIXEL_SENTINEL;   // out-of-frame tail → skip later
        wfAux[slot].ewFlags = 0u;
        return;
    }
    uint2 pixel = uint2(lane % fc.width, lane / fc.width);

#if defined(SKINNY_SPECTRAL)
    // Spectral eye-walk seed: draw the hero wavelengths after generateRay (3.1),
    // store them in the record so the staged bounce/connect/light stages share
    // them; seed eye[0]=camera + eye[1]=z1 spectrally (mirror bdpt_spectral).
    RNG rng = createRNG(pixel, fc.frameIndex);
    PinholeCamera cam;
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);
    SampledWavelengths sw = sampleWavelengths(rng.next());

    WfBdptAux aux;
    aux.eyeLen = 0;
    aux.lightLen = 0;
    aux.lensWeight = lensWeight;
    aux.pixel = lane;
    aux.escaped = Spectrum(0.0);
    aux.radiance = Spectrum(0.0);
    aux.ewRayO = float3(0.0);
    aux.ewRayD = float3(0.0);
    aux.ewThroughput = Spectrum(0.0);
    aux.ewPdfFwdOmega = 0.0;
    aux.ewMisBsdfPdf = -1.0;
    aux.ewFlags = 0u;
    aux.sw = sw;

    BDPTVertex camVtxS;
    camVtxS.kind = BDPT_VK_CAMERA;
    camVtxS.position = ray.origin;
    camVtxS.N = -ray.direction;
    camVtxS.throughput = Spectrum(1.0);
    camVtxS.emission = Spectrum(0.0);
    camVtxS.pdfFwd = 1.0;
    camVtxS.pdfRev = 0.0;
    camVtxS.isDelta = false;
    camVtxS.onLight = false;
    camVtxS.matId = 0u;
    camVtxS.uv = float2(0.0);
    camVtxS.posObject = float3(0.0);
    camVtxS.geoN = -ray.direction;
    camVtxS.tangent = float3(1.0, 0.0, 0.0);
    camVtxS.hasTangent = false;
    wfEye[base + 0u] = camVtxS;

    HitInfo firstHit = traceScene(fc, ray);
    Ray r = ray;
    for (uint skip = 0u; skip < 32u && firstHit.hit && isCutoutTransparent(firstHit); skip++)
    {
        r.origin = firstHit.position + r.direction * 0.002;
        firstHit = traceScene(fc, r);
    }
    if (!firstHit.hit)
    {
        aux.escaped = upsampleIlluminantBound(
            sceneEnvRadiance(r.direction, fc.envIntensity,
                             effectiveFurnaceMode(firstHit.materialId)), sw);
        aux.rngState = rng.state;
        wfAux[slot] = aux;
        return;
    }
    if ((materialTypes[firstHit.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
    {
        PathTracer integrator;   // unreachable under spectral (non-flat refused); widen to compile
        aux.escaped = WF_SPEC3(integrator.estimateRadiance(r, firstHit, rng));
        aux.rngState = rng.state;
        wfAux[slot] = aux;
        return;
    }

    FlatMaterial fmat = loadFlatMaterial(firstHit);
    SpectralFlatColors fcols = wfColsFor(fmat, firstHit.materialId, sw);
    float2 firstBB = materialBlackbody(firstHit.materialId);
    BDPTVertex z1;
    z1.kind = BDPT_VK_SURFACE;
    z1.position = firstHit.position;
    z1.N = fmat.data.N;
    z1.throughput = Spectrum(1.0);
    z1.emission = (firstBB.x > 0.0) ? planckSpectrum(sw, firstBB.x) * firstBB.y : fcols.emission;
    z1.pdfFwd = 1.0;
    z1.pdfRev = 0.0;
    z1.isDelta = false;
    z1.onLight = any(fmat.data.mat.emission > float3(0.0)) || (firstBB.x > 0.0);
    z1.matId = firstHit.materialId;
    z1.uv = firstHit.uv;
    z1.posObject = firstHit.positionObject;
    z1.geoN = firstHit.normal;
    z1.tangent = firstHit.tangent;
    z1.hasTangent = firstHit.hasTangent;

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

    aux.escaped = wfSpectralEnvNEE(fmat, fcols, z1.position, z1.N, T, B, wo_ts,
                                   firstHit.materialId, sw, rng);

    BSDFSample bs = fmat.sample(wo_ts, rng);
    aux.eyeLen = 2;
    if (bs.valid)
    {
        if (bs.pdf <= 0.0)
            z1.isDelta = true;
        float3 wi_world = tangentToWorld(bs.wi, T, B, z1.N);
        float glassB = flatMaterials[firstHit.materialId].glassCauchyB;
        if (bs.pdf == 0.0 && glassB > 0.0 && wo_ts.z * bs.wi.z < 0.0)
        {
            float  lam0um   = sw.lambda.x * 1e-3;
            float  nHero    = fmat.data.mat.ior + glassB / (lam0um * lam0um);
            bool   entering = wo_ts.z > 0.0;
            float3 faceN    = entering ? float3(0, 0, 1) : float3(0, 0, -1);
            float  etaH     = entering ? (1.0 / nHero) : nHero;
            float3 nd       = refract(-wo_ts, faceN, etaH);
            if (all(nd == float3(0.0)))
                nd = reflect(-wo_ts, faceN);
            bs.wi          = nd;
            bs.transmitted = nd.z < 0.0;
            wi_world       = tangentToWorld(bs.wi, T, B, z1.N);
            sw             = terminateSecondary(sw);
        }
        Spectrum throughput;
        if (bs.pdf > 0.0)
            throughput = flatResponseS(fmat, fcols, wo_ts, bs.wi) / bs.pdf;
        else
            throughput = fcols.transmission;
        float3 spawnOffset = bs.transmitted ? -z1.N * 0.002 : z1.N * 0.002;
        aux.ewRayO = z1.position + spawnOffset;
        aux.ewRayD = wi_world;
        aux.ewThroughput = throughput;
        aux.ewPdfFwdOmega = (bs.pdf > 0.0) ? bs.pdf : 1.0;
        aux.ewMisBsdfPdf = (bs.pdf > 0.0 && !bs.transmitted
            && effectiveFurnaceMode(firstHit.materialId) == 0u) ? bs.pdf : -1.0;
        aux.ewFlags = WF_BDPT_ALIVE;
    }
    wfEye[base + 1u] = z1;
    aux.sw = sw;               // dispersion collapse from the first bounce
    aux.rngState = rng.state;
    wfAux[slot] = aux;
    return;
#else
    RNG rng = createRNG(pixel, fc.frameIndex);
    PinholeCamera cam;
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);

    WfBdptAux aux;
    aux.eyeLen = 0;
    aux.lightLen = 0;
    aux.lensWeight = lensWeight;
    aux.pixel = lane;
    aux.escaped = float3(0.0);
    aux.radiance = float3(0.0);
    aux.ewRayO = float3(0.0);
    aux.ewRayD = float3(0.0);
    aux.ewThroughput = float3(0.0);
    aux.ewPdfFwdOmega = 0.0;
    aux.ewMisBsdfPdf = -1.0;
    aux.ewFlags = 0u;

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
    wfEye[base + 0u] = camVtx;

    HitInfo firstHit = traceScene(fc, ray);
    Ray r = ray;
    for (uint skip = 0u; skip < 32u && firstHit.hit && isCutoutTransparent(firstHit); skip++)
    {
        r.origin = firstHit.position + r.direction * 0.002;
        firstHit = traceScene(fc, r);
    }
    if (!firstHit.hit)
    {
        aux.escaped = sceneEnvRadiance(r.direction, fc.envIntensity,
                                       effectiveFurnaceMode(firstHit.materialId));
        aux.rngState = rng.state;
        wfAux[slot] = aux;
        return;
    }
    uint firstMatType = materialTypes[firstHit.materialId] & 0xFFu;
    if (firstMatType != MATERIAL_TYPE_FLAT)
    {
        // Non-flat first hit → path fallback for every non-flat type (see
        // wfBdptWalk for the full rationale). Terminal (SUBSURFACE / SKIN) = one
        // eval; non-terminal (VOLUME / PYTHON) = full multi-bounce path, kept in
        // the watchdog by the host's per-tile Metal eye submit (bound_heavy_eye).
        // No eye/light subpath built.
        PathTracer integrator;
        aux.escaped = integrator.estimateRadiance(r, firstHit, rng);
        aux.rngState = rng.state;
        wfAux[slot] = aux;
        return;
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

    aux.escaped = bdptEnvNEE(fmat, z1.position, z1.N, T, B, wo_ts, firstHit.materialId, rng);

    BSDFSample bs = fmat.sample(wo_ts, rng);
    aux.eyeLen = 2;
    if (bs.valid)
    {
        if (bs.pdf <= 0.0)
            z1.isDelta = true;
        float3 wi_world = tangentToWorld(bs.wi, T, B, z1.N);
        float3 spawnOffset = bs.transmitted ? -z1.N * 0.002 : z1.N * 0.002;
        aux.ewRayO = z1.position + spawnOffset;
        aux.ewRayD = wi_world;
        aux.ewThroughput = bs.weight;
        aux.ewPdfFwdOmega = (bs.pdf > 0.0) ? bs.pdf : 1.0;
        aux.ewMisBsdfPdf = (bs.pdf > 0.0 && !bs.transmitted
            && effectiveFurnaceMode(firstHit.materialId) == 0u) ? bs.pdf : -1.0;
        aux.ewFlags = WF_BDPT_ALIVE;             // bounce loop extends eye[2..]
    }
    wfEye[base + 1u] = z1;
    aux.rngState = rng.state;
    wfAux[slot] = aux;
#endif // SKINNY_SPECTRAL
}

// Mark lanes whose walk (eye or light) is still extending into slot 0 and count
// them, so build_args + scatter gather them for the next indirect bounce
// dispatch. Shared by the eye and light bounce loops (both flag liveness in
// ewFlags). Full-stream dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptWalkClassify(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= wfTile.streamSize)
        return;
    if ((wfAux[i].ewFlags & WF_BDPT_ALIVE) == 0u)
    {
        wfLaneSlot[i] = WF_PIXEL_SENTINEL;       // dead / done → not enqueued
        return;
    }
    wfLaneSlot[i] = 0u;
    uint prev;
    InterlockedAdd(wfSlotCount[0u], 1u, prev);
}

// Extend the eye subpath by one vertex for one live lane — a verbatim port of
// integrators/bdpt.slang::randomWalk's loop body. Indirect-dispatched over the
// live queue (slot 0). Reads/writes the lane's transient state in wfAux and its
// vertices in wfEye. Clears ALIVE when the walk terminates (miss, sphere-light
// hit, non-flat, invalid/RR-killed bounce, or max depth).
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptBounceEye(uint3 tid: SV_DispatchThreadID)
{
    uint slotLane = wfBdptQueueLane(0u, tid.x);
    if (slotLane == 0xFFFFFFFFu)
        return;
    uint i = slotLane;
    uint base = i * BDPT_MAX_VERTS;

    WfBdptAux aux = wfAux[i];
#if defined(SKINNY_SPECTRAL)
    // Spectral eye bounce: one iteration of wfRandomWalkS's loop body, staged.
    int idx = aux.eyeLen;
    Ray rayS; rayS.origin = aux.ewRayO; rayS.direction = aux.ewRayD;
    Spectrum throughput = aux.ewThroughput;
    float pdfFwdOmega = aux.ewPdfFwdOmega;
    float misBsdfPdf = aux.ewMisBsdfPdf;
    Spectrum escaped = aux.escaped;
    SampledWavelengths sw = aux.sw;
    RNG rng; rng.state = aux.rngState;

    HitInfo h = traceScene(fc, rayS);
    [loop]
    for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
    {
        rayS.origin = h.position + rayS.direction * 0.002;
        h = traceScene(fc, rayS);
    }

    if (fc.numSphereLights > 0u)
    {
        float t_max = h.hit ? h.t : 1e30;
        NearestSphereHit sph = intersectSphereLights(rayS, t_max, fc.numSphereLights);
        if (sph.idx != 0xFFFFFFFFu)
        {
            float w_bsdf = 1.0;
            BDPTVertex prev = wfEye[base + uint(idx - 1)];
            if (!prev.isDelta && pdfFwdOmega > 0.0)
            {
                SphereLight L = sphereLights[sph.idx];
                float area = 4.0 * PI * L.radius * L.radius;
                float pdf_light_sa = (sph.t * sph.t) / max(sph.cos_y * area, 1e-12);
                w_bsdf = powerHeuristic(pdfFwdOmega, pdf_light_sa);
            }
            escaped += throughput * upsampleIlluminantBound(sphereLights[sph.idx].radiance, sw) * w_bsdf;
            aux.escaped = escaped; aux.sw = sw; aux.rngState = rng.state; aux.ewFlags = 0u;
            wfAux[i] = aux;
            return;
        }
    }

    if (!h.hit)
    {
        float w = 1.0;
        if (misBsdfPdf > 0.0)
            w = powerHeuristic(misBsdfPdf, envPdf(rayS.direction));
        escaped += throughput * upsampleIlluminantBound(sceneEnvRadiance(
            rayS.direction, fc.envIntensity, effectiveFurnaceMode(h.materialId)), sw) * w;
        aux.escaped = escaped; aux.sw = sw; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }
    if ((materialTypes[h.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
    {
        aux.sw = sw; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }

    FlatMaterial mat = loadFlatMaterial(h);
    FlatHitData fd = mat.data;
    SpectralFlatColors cols = wfColsFor(mat, h.materialId, sw);

    BDPTVertex v;
    v.kind = BDPT_VK_SURFACE;
    v.position = h.position;
    v.N = fd.N;
    v.throughput = throughput;
    float2 matBB = materialBlackbody(h.materialId);
    v.emission = (matBB.x > 0.0) ? planckSpectrum(sw, matBB.x) * matBB.y : cols.emission;
    v.pdfFwd = 0.0;
    v.pdfRev = 0.0;
    v.isDelta = false;
    v.onLight = any(fd.mat.emission > float3(0.0)) || (matBB.x > 0.0);
    v.matId = h.materialId;
    v.uv = h.uv;
    v.posObject = h.positionObject;
    v.geoN = h.normal;
    v.tangent = h.tangent;
    v.hasTangent = h.hasTangent;

    BDPTVertex prev = wfEye[base + uint(idx - 1)];
    v.pdfFwd = convertSAtoArea(pdfFwdOmega, prev.position, v);
    int prevIdx = idx - 1;
    wfEye[base + uint(idx)] = v;
    idx++;
    if (idx >= int(BDPT_MAX_VERTS))
    {
        aux.eyeLen = idx; aux.escaped = escaped; aux.sw = sw; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }

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
    float3 wo_ts = worldToTangent(-rayS.direction, T, B, v.N);

    escaped += throughput * wfSpectralEnvNEE(mat, cols, v.position, v.N, T, B, wo_ts, v.matId, sw, rng);

    BSDFSample bs = mat.sample(wo_ts, rng);
    if (!bs.valid)
    {
        aux.eyeLen = idx; aux.escaped = escaped; aux.sw = sw; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }
    float3 wi_world = tangentToWorld(bs.wi, T, B, v.N);

    float glassB = flatMaterials[h.materialId].glassCauchyB;
    if (bs.pdf == 0.0 && glassB > 0.0 && wo_ts.z * bs.wi.z < 0.0)
    {
        float  lam0um   = sw.lambda.x * 1e-3;
        float  nHero    = mat.data.mat.ior + glassB / (lam0um * lam0um);
        bool   entering = wo_ts.z > 0.0;
        float3 faceN    = entering ? float3(0, 0, 1) : float3(0, 0, -1);
        float  etaH     = entering ? (1.0 / nHero) : nHero;
        float3 nd       = refract(-wo_ts, faceN, etaH);
        if (all(nd == float3(0.0)))
            nd = reflect(-wo_ts, faceN);
        bs.wi          = nd;
        bs.transmitted = nd.z < 0.0;
        wi_world       = tangentToWorld(bs.wi, T, B, v.N);
        sw             = terminateSecondary(sw);
    }

    bool deltaBounce = (bs.pdf <= 0.0);
    if (deltaBounce)
        wfEye[base + uint(idx - 1)].isDelta = true;

    if (prev.kind != BDPT_VK_CAMERA && !prev.isDelta)
    {
        float3 toPrev = prev.position - v.position;
        float toPrevLen2 = dot(toPrev, toPrev);
        if (toPrevLen2 > 1e-20)
        {
            float3 toPrevDir = toPrev / sqrt(toPrevLen2);
            float pdfRevOmega;
            if (deltaBounce)
                pdfRevOmega = lambertSApdf(v, toPrevDir);
            else
            {
                float3 woRev = worldToTangent(wi_world, T, B, v.N);
                float3 wiRev = worldToTangent(toPrevDir, T, B, v.N);
                pdfRevOmega = mat.evaluate(woRev, wiRev).pdf;
            }
            wfEye[base + uint(prevIdx)].pdfRev = convertSAtoArea(pdfRevOmega, v.position, prev);
        }
    }

    Spectrum weightS;
    if (bs.pdf > 0.0)
        weightS = flatResponseS(mat, cols, wo_ts, bs.wi) / bs.pdf;
    else
        weightS = cols.transmission;
    throughput *= weightS;

    // Russian roulette (idx ≥ 2 always in the bounce loop); include the .w lane.
    float mx = max(max(throughput.x, throughput.y), max(throughput.z, throughput.w));
    float p = clamp(mx, 0.05, 0.95);
    if (rng.next() > p)
    {
        aux.eyeLen = idx; aux.escaped = escaped; aux.sw = sw; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }
    throughput /= p;

    float3 spawnOffset = bs.transmitted ? -v.N * 0.002 : v.N * 0.002;
    aux.ewRayO = v.position + spawnOffset;
    aux.ewRayD = wi_world;
    aux.ewThroughput = throughput;
    aux.ewPdfFwdOmega = deltaBounce ? 1.0 : bs.pdf;
    aux.ewMisBsdfPdf = (!deltaBounce && !bs.transmitted
                        && effectiveFurnaceMode(v.matId) == 0u) ? bs.pdf : -1.0;
    aux.eyeLen = idx;
    aux.escaped = escaped;
    aux.sw = sw;
    aux.rngState = rng.state;
    aux.ewFlags = WF_BDPT_ALIVE;
    wfAux[i] = aux;
    return;
#else
    int idx = aux.eyeLen;                         // next write slot
    Ray ray; ray.origin = aux.ewRayO; ray.direction = aux.ewRayD;
    float3 throughput = aux.ewThroughput;
    float pdfFwdOmega = aux.ewPdfFwdOmega;
    float misBsdfPdf = aux.ewMisBsdfPdf;
    float3 escaped = aux.escaped;
    RNG rng; rng.state = aux.rngState;

    HitInfo h = traceScene(fc, ray);
    [loop]
    for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
    {
        ray.origin = h.position + ray.direction * 0.002;
        h = traceScene(fc, ray);
    }

    // Eye walk sees sphere-light geometry as emissive (MIS vs connectT1 NEE).
    if (fc.numSphereLights > 0u)
    {
        float t_max = h.hit ? h.t : 1e30;
        NearestSphereHit sph = intersectSphereLights(ray, t_max, fc.numSphereLights);
        if (sph.idx != 0xFFFFFFFFu)
        {
            float w_bsdf = 1.0;
            BDPTVertex prev = wfEye[base + uint(idx - 1)];
            if (!prev.isDelta && pdfFwdOmega > 0.0)
            {
                SphereLight L = sphereLights[sph.idx];
                float area = 4.0 * PI * L.radius * L.radius;
                float pdf_light_sa = (sph.t * sph.t) / max(sph.cos_y * area, 1e-12);
                w_bsdf = powerHeuristic(pdfFwdOmega, pdf_light_sa);
            }
            escaped += throughput * sphereLights[sph.idx].radiance * w_bsdf;
            aux.escaped = escaped; aux.rngState = rng.state; aux.ewFlags = 0u;
            wfAux[i] = aux;
            return;
        }
    }

    if (!h.hit)
    {
        float w = 1.0;
        if (misBsdfPdf > 0.0)
            w = powerHeuristic(misBsdfPdf, envPdf(ray.direction));
        escaped += throughput * sceneEnvRadiance(
            ray.direction, fc.envIntensity, effectiveFurnaceMode(h.materialId)) * w;
        aux.escaped = escaped; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }
    if ((materialTypes[h.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
    {
        aux.rngState = rng.state; aux.ewFlags = 0u;   // non-flat → end walk
        wfAux[i] = aux;
        return;
    }

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

    BDPTVertex prev = wfEye[base + uint(idx - 1)];
    v.pdfFwd = convertSAtoArea(pdfFwdOmega, prev.position, v);
    int prevIdx = idx - 1;
    wfEye[base + uint(idx)] = v;
    idx++;
    if (idx >= int(BDPT_MAX_VERTS))               // depth cap — vertex kept, walk ends
    {
        aux.eyeLen = idx; aux.escaped = escaped; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }

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
    float3 wo_ts = worldToTangent(-ray.direction, T, B, v.N);

    escaped += throughput * bdptEnvNEE(mat, v.position, v.N, T, B, wo_ts, v.matId, rng);

    BSDFSample bs = mat.sample(wo_ts, rng);
    if (!bs.valid)
    {
        aux.eyeLen = idx; aux.escaped = escaped; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }
    float3 wi_world = tangentToWorld(bs.wi, T, B, v.N);

    bool deltaBounce = (bs.pdf <= 0.0);
    if (deltaBounce)
        wfEye[base + uint(idx - 1)].isDelta = true;   // flag v itself

    if (prev.kind != BDPT_VK_CAMERA && !prev.isDelta)
    {
        float3 toPrev = prev.position - v.position;
        float toPrevLen2 = dot(toPrev, toPrev);
        if (toPrevLen2 > 1e-20)
        {
            float3 toPrevDir = toPrev / sqrt(toPrevLen2);
            float pdfRevOmega;
            if (deltaBounce)
                pdfRevOmega = lambertSApdf(v, toPrevDir);
            else
            {
                float3 woRev = worldToTangent(wi_world, T, B, v.N);
                float3 wiRev = worldToTangent(toPrevDir, T, B, v.N);
                pdfRevOmega = mat.evaluate(woRev, wiRev).pdf;
            }
            wfEye[base + uint(prevIdx)].pdfRev = convertSAtoArea(pdfRevOmega, v.position, prev);
        }
    }

    throughput *= bs.weight;

    // Russian roulette (idx ≥ 2 always in the bounce loop).
    float p = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05, 0.95);
    if (rng.next() > p)
    {
        aux.eyeLen = idx; aux.escaped = escaped; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }
    throughput /= p;

    float3 spawnOffset = bs.transmitted ? -v.N * 0.002 : v.N * 0.002;
    aux.ewRayO = v.position + spawnOffset;
    aux.ewRayD = wi_world;
    aux.ewThroughput = throughput;
    aux.ewPdfFwdOmega = deltaBounce ? 1.0 : bs.pdf;
    aux.ewMisBsdfPdf = (!deltaBounce && !bs.transmitted
                        && effectiveFurnaceMode(v.matId) == 0u) ? bs.pdf : -1.0;
    aux.eyeLen = idx;
    aux.escaped = escaped;
    aux.rngState = rng.state;
    aux.ewFlags = WF_BDPT_ALIVE;                  // survive to the next bounce
    wfAux[i] = aux;
#endif // SKINNY_SPECTRAL
}

// ── Staged light walk + splat (S3) ──────────────────────────────
//
// The light subpath is built across per-bounce dispatches like the eye walk,
// reusing the same alive-classify → build_args → scatter → indirect machinery
// (liveness in ewFlags, vertices in wfLight). The s=1 light-tracer splat becomes
// its own stage. Runs only for lanes with a flat eye first hit (eyeLen ≥ 2),
// matching estimateRadiance which returns before the light subpath on a primary
// miss / non-flat hit. The rng cursor flows eye-walk → gen-light → bounce-light,
// so the per-lane draw order matches the megakernel and the image is unchanged.

// Seed light[0] = light-origin vertex and the first light ray. Delta lights
// (directional) yield lightLen 1 and no walk. Reuses the ew* transient fields
// (the eye walk is done) for the light-walk loop state.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptGenLight(uint3 tid: SV_DispatchThreadID)
{
    uint slot = tid.x;
    if (slot >= wfTile.streamSize)
        return;
    WfBdptAux aux = wfAux[slot];
    aux.ewFlags = 0u;                            // default: no light walk
    if (aux.pixel == WF_PIXEL_SENTINEL || aux.eyeLen < 2)
    {
        wfAux[slot] = aux;
        return;
    }
    uint base = slot * BDPT_MAX_VERTS;
    RNG rng; rng.state = aux.rngState;

#if defined(SKINNY_SPECTRAL)
    WfSpectralLightOrigin lo = wfSampleLightOriginS(aux.sw, rng);
    aux.lightLen = 0;
    if (lo.valid)
    {
        wfLight[base + 0u] = lo.vertex;
        aux.lightLen = 1;
        if (lo.vertex.kind != BDPT_VK_LIGHT_DIR)
        {
            aux.ewRayO = lo.vertex.position + lo.vertex.N * 0.002;
            aux.ewRayD = lo.dirOut;
            aux.ewThroughput = lo.beta;
            aux.ewPdfFwdOmega = lo.pdfDir;
            aux.ewFlags = WF_BDPT_ALIVE;         // extend the light subpath
        }
    }
    aux.rngState = rng.state;
    wfAux[slot] = aux;
#else
    LightOriginSample lo = sampleLightOrigin(rng);
    aux.lightLen = 0;
    if (lo.valid)
    {
        wfLight[base + 0u] = lo.vertex;
        aux.lightLen = 1;
        if (lo.vertex.kind != BDPT_VK_LIGHT_DIR)
        {
            aux.ewRayO = lo.vertex.position + lo.vertex.N * 0.002;
            aux.ewRayD = lo.dirOut;
            aux.ewThroughput = lo.beta;
            aux.ewPdfFwdOmega = lo.pdfDir;
            aux.ewFlags = WF_BDPT_ALIVE;         // extend the light subpath
        }
    }
    aux.rngState = rng.state;
    wfAux[slot] = aux;
#endif // SKINNY_SPECTRAL
}

// Extend the light subpath by one vertex for one live lane — randomWalk's loop
// body with isEyeWalk = false (no env-NEE, no sphere-light emissive, no escaped;
// a miss just ends the walk). Indirect-dispatched over the live queue (slot 0).
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptBounceLight(uint3 tid: SV_DispatchThreadID)
{
    uint slotLane = wfBdptQueueLane(0u, tid.x);
    if (slotLane == 0xFFFFFFFFu)
        return;
    uint i = slotLane;
    uint base = i * BDPT_MAX_VERTS;

    WfBdptAux aux = wfAux[i];
#if defined(SKINNY_SPECTRAL)
    int idx = aux.lightLen;
    Ray rayS; rayS.origin = aux.ewRayO; rayS.direction = aux.ewRayD;
    Spectrum throughput = aux.ewThroughput;
    float pdfFwdOmega = aux.ewPdfFwdOmega;
    SampledWavelengths sw = aux.sw;
    RNG rng; rng.state = aux.rngState;

    HitInfo h = traceScene(fc, rayS);
    [loop]
    for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
    {
        rayS.origin = h.position + rayS.direction * 0.002;
        h = traceScene(fc, rayS);
    }

    if (!h.hit || (materialTypes[h.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
    {
        aux.sw = sw; aux.rngState = rng.state; aux.ewFlags = 0u;   // miss / non-flat → end walk
        wfAux[i] = aux;
        return;
    }

    FlatMaterial mat = loadFlatMaterial(h);
    FlatHitData fd = mat.data;
    SpectralFlatColors cols = wfColsFor(mat, h.materialId, sw);

    BDPTVertex v;
    v.kind = BDPT_VK_SURFACE;
    v.position = h.position;
    v.N = fd.N;
    v.throughput = throughput;
    float2 matBB = materialBlackbody(h.materialId);
    v.emission = (matBB.x > 0.0) ? planckSpectrum(sw, matBB.x) * matBB.y : cols.emission;
    v.pdfFwd = 0.0;
    v.pdfRev = 0.0;
    v.isDelta = false;
    v.onLight = any(fd.mat.emission > float3(0.0)) || (matBB.x > 0.0);
    v.matId = h.materialId;
    v.uv = h.uv;
    v.posObject = h.positionObject;
    v.geoN = h.normal;
    v.tangent = h.tangent;
    v.hasTangent = h.hasTangent;

    BDPTVertex prev = wfLight[base + uint(idx - 1)];
    v.pdfFwd = convertSAtoArea(pdfFwdOmega, prev.position, v);
    int prevIdx = idx - 1;
    wfLight[base + uint(idx)] = v;
    idx++;
    if (idx >= int(BDPT_MAX_VERTS))
    {
        aux.lightLen = idx; aux.sw = sw; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }

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
    float3 wo_ts = worldToTangent(-rayS.direction, T, B, v.N);

    // (no env-NEE on the light walk)
    BSDFSample bs = mat.sample(wo_ts, rng);
    if (!bs.valid)
    {
        aux.lightLen = idx; aux.sw = sw; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }
    float3 wi_world = tangentToWorld(bs.wi, T, B, v.N);

    float glassB = flatMaterials[h.materialId].glassCauchyB;
    if (bs.pdf == 0.0 && glassB > 0.0 && wo_ts.z * bs.wi.z < 0.0)
    {
        float  lam0um   = sw.lambda.x * 1e-3;
        float  nHero    = mat.data.mat.ior + glassB / (lam0um * lam0um);
        bool   entering = wo_ts.z > 0.0;
        float3 faceN    = entering ? float3(0, 0, 1) : float3(0, 0, -1);
        float  etaH     = entering ? (1.0 / nHero) : nHero;
        float3 nd       = refract(-wo_ts, faceN, etaH);
        if (all(nd == float3(0.0)))
            nd = reflect(-wo_ts, faceN);
        bs.wi          = nd;
        bs.transmitted = nd.z < 0.0;
        wi_world       = tangentToWorld(bs.wi, T, B, v.N);
        sw             = terminateSecondary(sw);
    }

    bool deltaBounce = (bs.pdf <= 0.0);
    if (deltaBounce)
        wfLight[base + uint(idx - 1)].isDelta = true;

    if (prev.kind != BDPT_VK_CAMERA && !prev.isDelta)
    {
        float3 toPrev = prev.position - v.position;
        float toPrevLen2 = dot(toPrev, toPrev);
        if (toPrevLen2 > 1e-20)
        {
            float3 toPrevDir = toPrev / sqrt(toPrevLen2);
            float pdfRevOmega;
            if (deltaBounce)
                pdfRevOmega = lambertSApdf(v, toPrevDir);
            else
            {
                float3 woRev = worldToTangent(wi_world, T, B, v.N);
                float3 wiRev = worldToTangent(toPrevDir, T, B, v.N);
                pdfRevOmega = mat.evaluate(woRev, wiRev).pdf;
            }
            wfLight[base + uint(prevIdx)].pdfRev = convertSAtoArea(pdfRevOmega, v.position, prev);
        }
    }

    Spectrum weightS;
    if (bs.pdf > 0.0)
        weightS = flatResponseS(mat, cols, wo_ts, bs.wi) / bs.pdf;
    else
        weightS = cols.transmission;
    throughput *= weightS;

    float mx = max(max(throughput.x, throughput.y), max(throughput.z, throughput.w));
    float p = clamp(mx, 0.05, 0.95);
    if (rng.next() > p)
    {
        aux.lightLen = idx; aux.sw = sw; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }
    throughput /= p;

    float3 spawnOffset = bs.transmitted ? -v.N * 0.002 : v.N * 0.002;
    aux.ewRayO = v.position + spawnOffset;
    aux.ewRayD = wi_world;
    aux.ewThroughput = throughput;
    aux.ewPdfFwdOmega = deltaBounce ? 1.0 : bs.pdf;
    aux.lightLen = idx;
    aux.sw = sw;
    aux.rngState = rng.state;
    aux.ewFlags = WF_BDPT_ALIVE;                  // survive to the next light bounce
    wfAux[i] = aux;
    return;
#else
    int idx = aux.lightLen;                       // next write slot
    Ray ray; ray.origin = aux.ewRayO; ray.direction = aux.ewRayD;
    float3 throughput = aux.ewThroughput;
    float pdfFwdOmega = aux.ewPdfFwdOmega;
    RNG rng; rng.state = aux.rngState;

    HitInfo h = traceScene(fc, ray);
    [loop]
    for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
    {
        ray.origin = h.position + ray.direction * 0.002;
        h = traceScene(fc, ray);
    }

    if (!h.hit || (materialTypes[h.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
    {
        aux.rngState = rng.state; aux.ewFlags = 0u;   // miss / non-flat → end walk
        wfAux[i] = aux;
        return;
    }

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

    BDPTVertex prev = wfLight[base + uint(idx - 1)];
    v.pdfFwd = convertSAtoArea(pdfFwdOmega, prev.position, v);
    int prevIdx = idx - 1;
    wfLight[base + uint(idx)] = v;
    idx++;
    if (idx >= int(BDPT_MAX_VERTS))
    {
        aux.lightLen = idx; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }

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
    float3 wo_ts = worldToTangent(-ray.direction, T, B, v.N);

    // (no env-NEE on the light walk)
    BSDFSample bs = mat.sample(wo_ts, rng);
    if (!bs.valid)
    {
        aux.lightLen = idx; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }
    float3 wi_world = tangentToWorld(bs.wi, T, B, v.N);

    bool deltaBounce = (bs.pdf <= 0.0);
    if (deltaBounce)
        wfLight[base + uint(idx - 1)].isDelta = true;

    if (prev.kind != BDPT_VK_CAMERA && !prev.isDelta)
    {
        float3 toPrev = prev.position - v.position;
        float toPrevLen2 = dot(toPrev, toPrev);
        if (toPrevLen2 > 1e-20)
        {
            float3 toPrevDir = toPrev / sqrt(toPrevLen2);
            float pdfRevOmega;
            if (deltaBounce)
                pdfRevOmega = lambertSApdf(v, toPrevDir);
            else
            {
                float3 woRev = worldToTangent(wi_world, T, B, v.N);
                float3 wiRev = worldToTangent(toPrevDir, T, B, v.N);
                pdfRevOmega = mat.evaluate(woRev, wiRev).pdf;
            }
            wfLight[base + uint(prevIdx)].pdfRev = convertSAtoArea(pdfRevOmega, v.position, prev);
        }
    }

    throughput *= bs.weight;

    float p = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05, 0.95);
    if (rng.next() > p)
    {
        aux.lightLen = idx; aux.rngState = rng.state; aux.ewFlags = 0u;
        wfAux[i] = aux;
        return;
    }
    throughput /= p;

    float3 spawnOffset = bs.transmitted ? -v.N * 0.002 : v.N * 0.002;
    aux.ewRayO = v.position + spawnOffset;
    aux.ewRayD = wi_world;
    aux.ewThroughput = throughput;
    aux.ewPdfFwdOmega = deltaBounce ? 1.0 : bs.pdf;
    aux.lightLen = idx;
    aux.rngState = rng.state;
    aux.ewFlags = WF_BDPT_ALIVE;                  // survive to the next light bounce
    wfAux[i] = aux;
#endif // SKINNY_SPECTRAL
}

// s=1 light-tracer splat: project each non-delta light-walk vertex onto the
// camera and atomic-add into lightSplatBuffer. Standalone stage after the light
// walk. Pinhole sampleWi draws no rng, so the cursor is untouched (connect
// resumes from the same post-light-walk state). Full-stream dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptSplat(uint3 tid: SV_DispatchThreadID)
{
    uint slot = tid.x;
    if (slot >= wfTile.streamSize)
        return;
    WfBdptAux aux = wfAux[slot];
    if (aux.pixel == WF_PIXEL_SENTINEL || aux.eyeLen < 2 || aux.lightLen < 2)
        return;                                   // need ≥2 light vertices to splat
    uint base = slot * BDPT_MAX_VERTS;

    RNG rng; rng.state = aux.rngState;            // pinhole: no draw
    PinholeCamera cam;

    BDPTVertex lightPath[BDPT_MAX_VERTS];
    [loop]
    for (int k = 0; k < int(BDPT_MAX_VERTS); k++)
        if (k < aux.lightLen) lightPath[k] = wfLight[base + uint(k)];

#if defined(SKINNY_SPECTRAL)
    wfSplatLightWalkS<PinholeCamera>(lightPath, aux.lightLen, aux.sw, rng, cam);
#else
    splatLightWalk<PinholeCamera>(lightPath, aux.lightLen, rng, cam);
#endif
}

// Classify each post-walk lane by subpath shape and count it for the connect
// counting sort. Dead lanes (out-of-frame, primary miss, or non-flat first hit
// ⇒ eyeLen < 2) carry only the escaped radiance and are finalised here so the
// connect kernels never run for them; live lanes are routed to the NEE slot
// (lightLen < 2) or the FULL slot (lightLen ≥ 2). Full-stream dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptClassify(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= wfTile.streamSize)
        return;
    WfBdptAux aux = wfAux[i];
    if (aux.pixel == WF_PIXEL_SENTINEL)         // out-of-frame tail slot
    {
        wfLaneSlot[i] = WF_PIXEL_SENTINEL;
        return;
    }
    if (aux.eyeLen < 2)                          // miss / non-flat: escaped only
    {
        aux.radiance = aux.escaped * aux.lensWeight;
        wfAux[i] = aux;
        wfLaneSlot[i] = WF_PIXEL_SENTINEL;
        return;
    }
    uint slot = (aux.lightLen >= 2) ? WF_BDPT_SLOT_FULL : WF_BDPT_SLOT_NEE;
    wfLaneSlot[i] = slot;
    uint prev;
    InterlockedAdd(wfSlotCount[slot], 1u, prev);
}

// Exclusive prefix-sum the per-slot counts into queue offsets and emit one
// VkDispatchIndirectCommand (ceil(count/64),1,1) per slot. Single thread.
[shader("compute")]
[numthreads(1, 1, 1)]
void wfBdptBuildArgs(uint3 tid: SV_DispatchThreadID)
{
    uint running = 0u;
    for (uint slot = 0u; slot < WF_BDPT_NUM_SLOTS; ++slot)
    {
        wfSlotOffset[slot] = running;
        uint c = wfSlotCount[slot];
        running += c;
        wfIndirectArgs[slot * 3u + 0u] = (c + WF_BDPT_GROUP - 1u) / WF_BDPT_GROUP;
        wfIndirectArgs[slot * 3u + 1u] = 1u;
        wfIndirectArgs[slot * 3u + 2u] = 1u;
    }
}

// Place each live lane into its slot's contiguous queue slice via an atomic
// write cursor (counting sort). Full-stream dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptScatter(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= wfTile.streamSize)
        return;
    uint slot = wfLaneSlot[i];
    if (slot == WF_PIXEL_SENTINEL)              // dead → not enqueued
        return;
    uint local;
    InterlockedAdd(wfSlotCursor[slot], 1u, local);
    wfSlotQueue[wfSlotOffset[slot] + local] = i;
}

// Shared connection body for lane `i`. `includeGeneric` gates the O(s·t)
// generic+MIS double loop (only the FULL slot runs it). The eye-side estimator
// is identical to the megakernel bdpt; a lane runs exactly one of the two
// connect kernels, so the rng consumption order (connectT1 draws, then the
// rng-free generic loop) matches the fused connect verbatim.
void wfBdptConnectLane(uint i, bool includeGeneric)
{
    WfBdptAux aux = wfAux[i];
    uint base = i * BDPT_MAX_VERTS;
    int eyeLen = aux.eyeLen;
    int lightLen = aux.lightLen;

    RNG rng; rng.state = aux.rngState;

    BDPTVertex eye[BDPT_MAX_VERTS];
    BDPTVertex lightPath[BDPT_MAX_VERTS];
    for (int k = 0; k < int(BDPT_MAX_VERTS); k++)
    {
        if (k < eyeLen)   eye[k]       = wfEye[base + uint(k)];
        if (k < lightLen) lightPath[k] = wfLight[base + uint(k)];
    }

#if defined(SKINNY_SPECTRAL)
    Spectrum L = aux.escaped;
#else
    float3 L = aux.escaped;
#endif

    // (s ≥ 2, t = 0): emissive eye-vertex hits, MIS-gated against the t = 1 NEE
    // (connectT1) exactly as the megakernel bdpt.slang / path.slang do. A
    // non-delta eye bounce onto an emissive triangle is also sampled by NEE from
    // the previous eye vertex, so adding the BSDF hit at full weight on top of
    // the power-heuristic-weighted NEE double-counts direct area-light energy
    // (~1.75× too bright vs pbrt). Full weight only with no NEE partner: the
    // primary/first hit (s == 2), a delta bounce into z (eye[s - 2].isDelta),
    // or no emissive-triangle NEE in the scene.
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
#if defined(SKINNY_SPECTRAL)
        L += wfConnectT1S(eye, s, aux.sw, rng);
#else
        L += connectT1(eye, s, rng);
#endif

    // (s ≥ 2, t ≥ 2): generic connections + MIS (FULL slot only).
    if (includeGeneric)
    {
        [loop]
        for (int s = 2; s <= eyeLen; s++)
        {
            [loop]
            for (int t = 2; t <= lightLen; t++)
            {
                if (s + t - 2 > int(BDPT_MAX_DEPTH)) continue;
#if defined(SKINNY_SPECTRAL)
                WfSpectralConnectResult cr = wfConnectGenericS(eye, s, lightPath, t, aux.sw);
#else
                ConnectResult cr = connectGeneric(eye, s, lightPath, t);
#endif
                if (!cr.nonzero) continue;
                float w = misWeight(eye, s, lightPath, t,
                                    cr.pdf_zsm1_rev, cr.pdf_zsm2_rev,
                                    cr.pdf_ytm1_rev, cr.pdf_ytm2_rev);
                L += cr.contribution * w;
            }
        }
    }

    aux.radiance = L * aux.lensWeight;
    wfAux[i] = aux;
}

// NEE slot: emissive + connectT1, no generic loop. Indirect-dispatched over the
// NEE queue (lanes with lightLen < 2 — e.g. directional-only scenes).
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptConnectNee(uint3 tid: SV_DispatchThreadID)
{
    uint i = wfBdptQueueLane(wfTile.shadeSlot, tid.x);
    if (i == 0xFFFFFFFFu)
        return;
    wfBdptConnectLane(i, false);
}

// FULL slot: emissive + connectT1 + generic + MIS. Indirect-dispatched over the
// FULL queue (lanes with a ≥2-vertex light subpath). Zero groups when no scene
// light produces a light subpath (e.g. directional-only).
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptConnectFull(uint3 tid: SV_DispatchThreadID)
{
    uint i = wfBdptQueueLane(wfTile.shadeSlot, tid.x);
    if (i == 0xFFFFFFFFu)
        return;
    wfBdptConnectLane(i, true);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptResolve(uint3 tid: SV_DispatchThreadID)
{
    uint slot = tid.x;
    if (slot >= wfTile.streamSize)              // dispatch rounds up to 64 — skip tail
        return;
    uint px = wfAux[slot].pixel;
    if (px == WF_PIXEL_SENTINEL)                // out-of-frame tail slot
        return;
    uint2 pixel = uint2(px % fc.width, px / fc.width);

#if defined(SKINNY_SPECTRAL)
    float3 sample = spectrumResolveToLinearSRGB(wfAux[slot].sw, wfAux[slot].radiance);
#else
    float3 sample = wfAux[slot].radiance;
#endif
    bool3 bad = isnan(sample) | isinf(sample);
    sample = select(bad, float3(0.0), sample);
    sample = max(sample, float3(0.0));
    // Film per-sample radiance clamp (change film-maxcomponent-clamp); no-op
    // when fc.filmMaxComponent == 0. Matches main_pass.slang / wavefront_path.
    sample = clampSampleRadiance(sample, fc.filmMaxComponent);

    // Running mean of the eye-side estimator. main_pass.slang writes exactly
    // this to accumBuffer (binding 2 = what read_accumulation reads); the s=1
    // splatBuffer composite is added only on the display/tonemap path, never
    // into accumBuffer, so the headless A/B compares eye-side estimates. The
    // splat is still accumulated in wfBdptWalk for a future windowed composite
    // (and is a no-op for the demo's directional light — no light subpath).
    uint n = fc.accumFrame;
    float3 accum;
    if (n == 0u)
        accum = sample;
    else
    {
        float3 prev = accumBuffer[pixel].rgb;
        accum = (prev * float(n) + sample) / float(n + 1u);
    }
    accumBuffer[pixel] = float4(accum, 1.0);
    wfWriteDisplay(pixel, accum);   // tonemap → offscreen/swapchain output
}
