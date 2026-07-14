// Hero-wavelength spectral bidirectional path tracer (change
// spectral-bdpt-megakernel). Compiled ONLY in the spectral variant
// (`-DSKINNY_SPECTRAL`); the RGB build never imports it.
//
// This is the bidirectional analog of integrators/path_spectral.slang: a
// SEPARATE integrator carrying Spectrum (float4) throughput/emission that reuses
// the RGB BDPT machinery (integrators/bdpt.slang) for every wavelength-
// INDEPENDENT quantity and recolors per wavelength via the shared spectral flat
// helpers (spectral_flat_common). Directions, pdfs, and MIS weights are the SAME
// scalar values the RGB integrator computes — so the ONLY thing that becomes
// spectral is the transported energy (throughput, emission, connection
// contributions, splat radiance).
//
// Reuse strategy (no MIS duplication): the pure-scalar MIS chain in bdpt.slang
// (misWeight / splatMisWeight / convertSAtoArea / bdptSurface / atomicSplat)
// reads only geometry + pdf fields, never color. A SpectralBDPTVertex therefore
// projects to a color-free BDPTVertex (asRgb) and calls the RGB functions
// directly — one source of MIS truth, no drift-guard needed.
//
// v1 scope: flat materials only (non-flat first hit falls to SpectralPathTracer
// via main_pass, exactly like RGB useBdpt falls to PathTracer). Wavelengths are
// drawn ONCE per pixel path and shared across the eye walk, light walk, all
// connections, and the splat (dispersion collapse on either walk narrows every
// subsequent strategy — pinned order: eye → light → splat → connections).

import common;
import bindings;
import interfaces;
import scene_trace;
import scene_lights;
import environment;
import materials.flat.flat_shading;
import materials.flat.flat_material;
import lights.sphere_light;
import lights.emissive_triangle_light;
import lights.directional_light;
import spectrum;                          // SampledWavelengths, sampleWavelengths, terminateSecondary, resolve, upsampleIlluminantBound, planckSpectrum, distantLightSpd
import integrators.bdpt;                  // BDPTVertex, BDPT_MAX_*, BDPT_VK_*, convertSAtoArea, misWeight, splatMisWeight, BdptSurface, bdptSurface, bdptEval, atomicSplatRadiance
import integrators.spectral_flat_common;  // SpectralFlatColors, upsampleFlatColors, flatResponseS, flatResponseNEE

// ── Spectral vertex ─────────────────────────────────────────────
// Same geometry/pdf fields as BDPTVertex; throughput/emission become Spectrum.
struct SpectralBDPTVertex
{
    uint     kind;
    float3   position;
    float3   N;
    Spectrum throughput;   // β (per wavelength)
    Spectrum emission;     // Le at this vertex (per wavelength)
    float    pdfFwd;
    float    pdfRev;
    bool     isDelta;
    bool     onLight;
    uint     matId;
    float2   uv;
    float3   posObject;
    float3   geoN;
    float3   tangent;
    bool     hasTangent;
};

// Color-free projection so the scalar MIS/geometry helpers in bdpt.slang can be
// reused verbatim (they read only these fields).
BDPTVertex asRgb(SpectralBDPTVertex v)
{
    BDPTVertex r;
    r.kind = v.kind;
    r.position = v.position;
    r.N = v.N;
    r.throughput = Spectrum(0.0);   // BDPTVertex color fields are Spectrum (D1 retype); misWeight ignores them
    r.emission = Spectrum(0.0);
    r.pdfFwd = v.pdfFwd;
    r.pdfRev = v.pdfRev;
    r.isDelta = v.isDelta;
    r.onLight = v.onLight;
    r.matId = v.matId;
    r.uv = v.uv;
    r.posObject = v.posObject;
    r.geoN = v.geoN;
    r.tangent = v.tangent;
    r.hasTangent = v.hasTangent;
    return r;
}

// Build the RGB mirror array (geometry + pdfs) for the whole subpath so misWeight
// / splatMisWeight / convertSAtoArea can be called on it.
void mirrorRgb(SpectralBDPTVertex src[BDPT_MAX_VERTS], out BDPTVertex dst[BDPT_MAX_VERTS])
{
    [loop] for (int i = 0; i < int(BDPT_MAX_VERTS); i++) dst[i] = asRgb(src[i]);
}

// Upsampled reflectances for a reconstructed surface vertex (conductor id from
// the material record), reused across the connection / splat responses.
SpectralFlatColors colsFor(FlatMaterial mat, uint matId, SampledWavelengths sw)
{
    return upsampleFlatColors(mat, sw, flatMaterials[matId].conductorMetalId);
}

// Spectral endpoint response (f·cos·opacity, per wavelength) + the SAME scalar
// pdf the RGB integrator uses. `s` is the reloaded surface, `cols` its upsampled
// reflectances; woWorld/wiWorld are world-space view / connection directions.
struct SpectralEval { Spectrum response; float pdf; }

SpectralEval spectralBdptEval(BdptSurface s, SpectralFlatColors cols, float3 N,
                              float3 woWorld, float3 wiWorld)
{
    SpectralEval e;
    float3 wo = worldToTangent(woWorld, s.T, s.B, N);
    float3 wi = worldToTangent(wiWorld, s.T, s.B, N);
    e.response = flatResponseNEE(s.mat, cols, wo, wi);
    e.pdf      = s.mat.evaluate(wo, wi).pdf;   // scalar geometry pdf (RGB machinery)
    return e;
}

// ── Environment NEE (spectral), mirror of bdpt.bdptEnvNEE ────────
Spectrum spectralBdptEnvNEE(FlatMaterial mat, SpectralFlatColors cols,
                            float3 P, float3 N, float3 T, float3 B,
                            float3 wo_ts, uint matId, SampledWavelengths sw, inout RNG rng)
{
    if (effectiveFurnaceMode(matId) != 0u) return Spectrum(0.0);
    EnvSample es = sampleEnvDir(rng.next2(), fc.envIntensity);
    if (es.pdf <= 0.0 || dot(N, es.dir) <= 0.0) return Spectrum(0.0);
    if (!visibleDirectional(fc, P, N, es.dir)) return Spectrum(0.0);
    float3   wi_ts = worldToTangent(es.dir, T, B, N);
    float    pdfB  = mat.evaluate(wo_ts, wi_ts).pdf;
    if (pdfB <= 0.0) return Spectrum(0.0);       // delta lobes have no NEE partner
    Spectrum resp  = flatResponseNEE(mat, cols, wo_ts, wi_ts);
    float    w     = powerHeuristic(es.pdf, pdfB);
    Spectrum Li    = upsampleIlluminantBound(es.radiance, sw);
    return resp * Li * (w / es.pdf);
}

// ── Light-side endpoint sampling (spectral β) ───────────────────
// Mirror of bdpt.sampleLightOrigin. Only the emitted radiance becomes spectral:
// blackbody emissive triangles seed the exact Planck SPD (walk-capable origin);
// sphere / distant seed the upsampled illuminant. Directional origins spawn no
// walk (their exactness lives on the t=1 NEE side) so their β is unused here.
struct SpectralLightOrigin
{
    SpectralBDPTVertex vertex;
    float3   dirOut;
    Spectrum beta;
    float    pdfPos;
    float    pdfDir;
    bool     valid;
};

SpectralLightOrigin sampleLightOriginS(SampledWavelengths sw, inout RNG rng)
{
    SpectralLightOrigin r;
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
        // Distant emission (change distant-light-caustic-parity): real disk-
        // origin light ray, mirroring bdpt.sampleLightOrigin's distant branch
        // — the spectral light walk carries distant radiance through specular
        // chains (glass caustics). Per-λ recolor via the authored SPD slot
        // when present (matching the SPPM photon emitter), else the RGB
        // radiance upsampled as an illuminant. isDelta stays FALSE on the
        // vertex (vertex flag ≠ delta LIGHT; misWeight / splatMisWeight
        // suppress the i == 0 eye-hit term by vertex KIND).
        uint idx = min(uint(rng.next() * float(nDir)), nDir - 1u);
        DistantLight L = distantLights[idx];
        float3 dir = normalize(L.direction);
        float lightPickPdf = groupPdf / float(nDir);
        int      slot = int(distantLights[idx].spdIndex);
        Spectrum Le   = (slot >= 0) ? distantLightSpd(slot, sw)
                                    : upsampleIlluminantBound(L.radiance, sw);
        float3 center = fc.sceneBoundsMin + 0.5 * fc.sceneBoundsExtent;
        float  Rd = max(0.5 * length(fc.sceneBoundsExtent), 1e-4);
        float3 t, b;
        buildBasis(dir, t, b);
        float2 u2 = rng.next2();
        float rr = Rd * sqrt(u2.x);
        float ph = 2.0 * PI * u2.y;
        float pdfPos = lightPickPdf / (PI * Rd * Rd);
        r.vertex.kind = BDPT_VK_LIGHT_DIR;
        r.vertex.position = center + dir * Rd + t * (rr * cos(ph)) + b * (rr * sin(ph));
        r.vertex.N = -dir;
        r.vertex.emission = Le;
        r.vertex.throughput = Le / max(pdfPos, 1e-20);
        r.vertex.pdfFwd = pdfPos;
        r.vertex.isDelta = false;
        r.dirOut = -dir;
        r.beta = Le / max(pdfPos, 1e-20);
        r.pdfPos = pdfPos;
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
        // Blackbody area light ⇒ exact Planck SPD at the hero wavelengths.
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

// ── Random walk (spectral) — mirror of bdpt.randomWalk ──────────
// Spectral throughput (flatResponseS/pdf, delta glass = upsampled transmission);
// env-miss + sphere-hit deposits upsample as illuminants; hero-λ dispersion
// collapse (shared sw) on a delta refraction. pdfFwd/pdfRev bookkeeping and the
// scalar MIS partners are IDENTICAL to the RGB walk.
int randomWalkS(
    inout SpectralBDPTVertex path[BDPT_MAX_VERTS],
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
                SpectralBDPTVertex prev = path[idx - 1];
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
        SpectralFlatColors cols = colsFor(mat, h.materialId, sw);

        SpectralBDPTVertex v;
        v.kind = BDPT_VK_SURFACE;
        v.position = h.position;
        v.N = fd.N;
        v.throughput = throughput;
        // Emission is the surface's own SPD (blackbody material ⇒ exact Planck).
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

        SpectralBDPTVertex prev = path[idx - 1];
        // Distant-light source segment: parallel-projection area density
        // (disk pdf × |N·beam|, no cos/d²) — mirrors bdpt.randomWalk's
        // DIR-source case (change distant-light-caustic-parity).
        if (prev.kind == BDPT_VK_LIGHT_DIR)
            v.pdfFwd = pdfFwdOmega * abs(dot(v.N, ray.direction));
        else
            v.pdfFwd = convertSAtoArea(pdfFwdOmega, prev.position, asRgb(v));

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
                spectralBdptEnvNEE(mat, cols, v.position, v.N, T, B, wo_ts, v.matId, sw, rng);

        BSDFSample bs = mat.sample(wo_ts, rng);
        if (!bs.valid) break;
        float3 wi_world = tangentToWorld(bs.wi, T, B, v.N);

        // Hero-λ dispersion (mirror path_spectral): recompute the delta refraction
        // at n(λ0) = ior + B/λ0² for a named glass and collapse to the hero λ.
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

        // prev.pdfRev — identical scalar geometry to the RGB walk.
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
                    pdfRevOmega = lambertSApdf(asRgb(v), toPrevDir);
                }
                else
                {
                    float3 woRev = worldToTangent(wi_world, T, B, v.N);
                    float3 wiRev = worldToTangent(toPrevDir, T, B, v.N);
                    pdfRevOmega = mat.evaluate(woRev, wiRev).pdf;
                }
                path[prevIdx].pdfRev = convertSAtoArea(pdfRevOmega, v.position, asRgb(prev));
            }
        }

        // Spectral throughput update (path_spectral weightS rule).
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

// ── Connection (t ≥ 2), mirror of bdpt.connectGeneric ───────────
struct SpectralConnectResult
{
    Spectrum contribution;
    float pdf_zsm1_rev;
    float pdf_zsm2_rev;
    float pdf_ytm1_rev;
    float pdf_ytm2_rev;
    bool  nonzero;
};

SpectralConnectResult connectGenericS(
    SpectralBDPTVertex eye[BDPT_MAX_VERTS], int s,
    SpectralBDPTVertex lit[BDPT_MAX_VERTS], int t,
    SampledWavelengths sw)
{
    SpectralConnectResult r;
    r.contribution = Spectrum(0.0);
    r.pdf_zsm1_rev = 0.0;
    r.pdf_zsm2_rev = 0.0;
    r.pdf_ytm1_rev = 0.0;
    r.pdf_ytm2_rev = 0.0;
    r.nonzero = false;

    SpectralBDPTVertex z = eye[s - 1];
    SpectralBDPTVertex y = lit[t - 1];
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

    BdptSurface zs = bdptSurface(asRgb(z));
    SpectralFlatColors zcols = colsFor(zs.mat, z.matId, sw);
    float3 woZ = normalize(eye[s - 2].position - z.position);
    SpectralEval ez = spectralBdptEval(zs, zcols, z.N, woZ, dir);
    Spectrum respZ = ez.response;
    float  pdf_z_sa = ez.pdf;

    Spectrum respY;
    float  pdf_y_sa;
    if (y.kind == BDPT_VK_SURFACE)
    {
        BdptSurface ys = bdptSurface(asRgb(y));
        SpectralFlatColors ycols = colsFor(ys.mat, y.matId, sw);
        float3 woY = normalize(lit[t - 2].position - y.position);
        SpectralEval ey = spectralBdptEval(ys, ycols, y.N, woY, -dir);
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
                r.pdf_ytm2_rev = convertSAtoArea(pr, y.position, asRgb(lit[t - 2]));
            }
        }
    }
    else
    {
        // Diffuse-emitter lobe at the light-origin vertex (f = 1/π). Achromatic —
        // the emitted SPD is already folded into y.throughput.
        respY = Spectrum(cosY / PI);
        pdf_y_sa = cosY / PI;
    }

    r.contribution = z.throughput * respZ * respY / d2 * y.throughput;

    r.pdf_zsm1_rev = convertSAtoArea(pdf_y_sa, y.position, asRgb(z));
    r.pdf_ytm1_rev = convertSAtoArea(pdf_z_sa, z.position, asRgb(y));

    if (s >= 2)
    {
        float3 toPrev = eye[s - 2].position - z.position;
        float l2 = dot(toPrev, toPrev);
        if (l2 > 1e-20)
        {
            float3 dp = toPrev / sqrt(l2);
            float pr = bdptEval(zs, z.N, dir, dp).pdf;
            r.pdf_zsm2_rev = convertSAtoArea(pr, z.position, asRgb(eye[s - 2]));
        }
    }
    r.nonzero = any(r.contribution > Spectrum(0.0));
    return r;
}

// ── t = 1 NEE (spectral), mirror of bdpt.connectT1 ──────────────
Spectrum connectT1S(SpectralBDPTVertex eye[BDPT_MAX_VERTS], int s,
                    SampledWavelengths sw, inout RNG rng)
{
    SpectralBDPTVertex z = eye[s - 1];
    if (z.isDelta || z.kind != BDPT_VK_SURFACE) return Spectrum(0.0);

    Spectrum total = Spectrum(0.0);
    BdptSurface zs = bdptSurface(asRgb(z));
    SpectralFlatColors zcols = colsFor(zs.mat, z.matId, sw);
    float3 woWorld = normalize(eye[s - 2].position - z.position);

    // Distant NEE — unified onto misWeight, mirror of bdpt.connectT1's distant
    // branch (change distant-light-caustic-parity): with the distant light walk
    // enabled, the unconditional full-weight term would double-count against
    // the s=1 splat / t≥2 connections. The weight degrades to ~1.0 whenever the
    // scene-bounds-disk walk density σ·N·L at z is negligible (pure-direct
    // distant scenes unchanged).
    for (uint d = 0u; d < fc.numDistantLights; d++)
    {
        DistantLight L = distantLights[d];
        float3 dir = normalize(L.direction);
        float NdotL = max(dot(z.N, dir), 0.0);
        if (NdotL > 0.0 && visibleDirectional(fc, z.position, z.N, dir))
        {
            SpectralEval e = spectralBdptEval(zs, zcols, z.N, woWorld, dir);
            // Authored `spectrum L` distant (slot >= 0) emits its exact SPD.
            int slot = int(L.spdIndex);
            Spectrum Li = (slot >= 0) ? distantLightSpd(slot, sw)
                                      : upsampleIlluminantBound(L.radiance, sw);

            uint nGroups = (fc.numDistantLights > 0u ? 1u : 0u)
                         + (fc.numSphereLights  > 0u ? 1u : 0u)
                         + (fc.numEmissiveTriangles > 0u ? 1u : 0u);
            float Rd = max(0.5 * length(fc.sceneBoundsExtent), 1e-4);
            float sigma = 1.0 / (float(nGroups) * float(fc.numDistantLights)
                                 * PI * Rd * Rd);

            BDPTVertex lit0;
            lit0.kind = BDPT_VK_LIGHT_DIR;
            lit0.position = z.position + dir;   // direction anchor; unused by MIS
            lit0.N = -dir;
            lit0.throughput = Spectrum(0.0);
            lit0.emission = Spectrum(0.0);
            lit0.pdfFwd = sigma;
            lit0.pdfRev = 0.0;
            lit0.isDelta = false;
            lit0.onLight = true;
            lit0.matId = 0u;
            lit0.uv = float2(0.0);
            lit0.posObject = float3(0.0);
            lit0.geoN = -dir;
            lit0.tangent = float3(1.0, 0.0, 0.0);
            lit0.hasTangent = false;
            BDPTVertex litArr[BDPT_MAX_VERTS];
            litArr[0] = lit0;

            float pdf_zsm1_rev = sigma * NdotL;
            float pdf_zsm2_rev = 0.0;
            if (s >= 3)
            {
                float3 toPrev = eye[s - 2].position - z.position;
                float l2 = dot(toPrev, toPrev);
                if (l2 > 1e-20)
                {
                    float3 dp = toPrev / sqrt(l2);
                    float pr = bdptEval(zs, z.N, dir, dp).pdf;
                    pdf_zsm2_rev = convertSAtoArea(pr, z.position, asRgb(eye[s - 2]));
                }
            }

            BDPTVertex eyeRgb[BDPT_MAX_VERTS];
            mirrorRgb(eye, eyeRgb);
            // s == 2: full weight — the t = 2 distant splat is skipped (see
            // bdpt.connectT1; codex P2-1).
            float w = (s == 2) ? 1.0
                    : misWeight(eyeRgb, s, litArr, 1,
                                pdf_zsm1_rev, pdf_zsm2_rev, 0.0, 0.0);
            total += z.throughput * e.response * Li * w;
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

        SpectralEval e = spectralBdptEval(zs, zcols, z.N, woWorld, dir);
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
                    SpectralEval e = spectralBdptEval(zs, zcols, z.N, woWorld, dir);
                    float invPdfOmega = cosLight / (d2 * ls.pdfArea);

                    // Light point as lit[0] for the unified misWeight partition
                    // (RGB geometry; color unused by misWeight).
                    BDPTVertex lit0;
                    lit0.kind = BDPT_VK_LIGHT_AREA;
                    lit0.position = ls.point;
                    lit0.N = ls.normal;
                    lit0.throughput = Spectrum(0.0);   // BDPTVertex color fields are Spectrum (D1 retype)
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
                            pdf_zsm2_rev = convertSAtoArea(pr, z.position, asRgb(eye[s - 2]));
                        }
                    }

                    BDPTVertex eyeRgb[BDPT_MAX_VERTS];
                    mirrorRgb(eye, eyeRgb);
                    float w = misWeight(eyeRgb, s, litArr, 1,
                                        pdf_zsm1_rev, pdf_zsm2_rev,
                                        pdf_ytm1_rev, 0.0);
                    // Blackbody area light ⇒ exact Planck SPD.
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

// ── Light tracer (s = 1) camera splat (spectral) ────────────────
// Mirror of bdpt.splatLightVertex; the contribution resolves λ → linear sRGB
// BEFORE the atomic add (D4). NOTE: the per-splat resolve applies the CIE film
// resolve's gamut clamp per splat (clamp-per-splat ≠ clamp-of-sum) — an accepted
// v1 bias for out-of-gamut hero-collapsed dispersion splats, consistent with the
// eye side's per-sample clamp. If the dispersion-caustic gate shows hue/energy
// bias, switch lightSplatBuffer to signed-XYZ accumulation (3-wide) and convert
// at composite.
void splatLightVertexS<TC : ICamera>(SpectralBDPTVertex lightPath[BDPT_MAX_VERTS], int i,
                                     SampledWavelengths sw, inout RNG rng, TC cam)
{
    SpectralBDPTVertex y = lightPath[i];
    if (y.isDelta || y.kind != BDPT_VK_SURFACE) return;
    // Delta-direction (distant) lights: the t = 2 splat is the pure DIRECT
    // path (light -> surface -> camera), which eye-side NEE (connectT1's
    // distant branch) samples deterministically at ~full weight. Under the
    // repo's z1.pdfFwd = 1 camera convention the two strategies cannot
    // partition consistently at grazing camera pdfs (horizon), so distant
    // DIRECT is owned by NEE outright and the splat carries only the
    // t >= 3 transport NEE cannot reach (specular caustic chains, where
    // every alternative crosses a delta vertex and this splat's MIS weight
    // is 1). Change distant-light-caustic-parity.
    if (i == 1 && lightPath[0].kind == BDPT_VK_LIGHT_DIR) return;
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

    BdptSurface ys = bdptSurface(asRgb(y));
    SpectralFlatColors ycols = colsFor(ys.mat, y.matId, sw);
    float3 woWorld = normalize(lightPath[i - 1].position - y.position);
    SpectralEval e = spectralBdptEval(ys, ycols, y.N, woWorld, toCam);
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
        pdf_ytm2_rev = convertSAtoArea(pr, y.position, asRgb(lightPath[i - 1]));
    }

    BDPTVertex litRgb[BDPT_MAX_VERTS];
    mirrorRgb(lightPath, litRgb);
    float w = splatMisWeight(litRgb, t, pdf_ytm1_rev, pdf_ytm2_rev);

    Spectrum contribS = y.throughput * e.response / (distCam * distCam) * (We * w);
    atomicSplatRadiance(pixel, spectrumResolveToLinearSRGB(sw, contribS));
}

void splatLightWalkS<TC : ICamera>(SpectralBDPTVertex lightPath[BDPT_MAX_VERTS],
                                   int lightLen, SampledWavelengths sw, inout RNG rng, TC cam)
{
    [loop]
    for (int i = 1; i < lightLen; i++)
        splatLightVertexS<TC>(lightPath, i, sw, rng, cam);
}

// ── Integrator ──────────────────────────────────────────────────
struct SpectralBDPTIntegrator<TC : ICamera>
{
    TC cam;

    float3 estimateRadiance(Ray ray, HitInfo firstHit, inout RNG rng)
    {
        // Hero-wavelength draw: shared across eye + light subpaths (dispersion
        // collapse on either narrows both). First RNG consumption in this path.
        SampledWavelengths sw = sampleWavelengths(rng.next());

        SpectralBDPTVertex eye[BDPT_MAX_VERTS];

        SpectralBDPTVertex camVtx;
        camVtx.kind = BDPT_VK_CAMERA;
        camVtx.position = ray.origin;
        camVtx.N = -ray.direction;
        camVtx.throughput = Spectrum(1.0);
        camVtx.emission = Spectrum(0.0);
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

        Ray r = ray;
        for (uint skip = 0u; skip < 32u && firstHit.hit && isCutoutTransparent(firstHit); skip++)
        {
            r.origin = firstHit.position + r.direction * 0.002;
            firstHit = traceScene(fc, r);
        }
        if (!firstHit.hit)
        {
            float3 envRgb = sceneEnvRadiance(r.direction, fc.envIntensity,
                                             effectiveFurnaceMode(firstHit.materialId));
            return spectrumResolveToLinearSRGB(sw, upsampleIlluminantBound(envRgb, sw));
        }
        if ((materialTypes[firstHit.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
            return float3(0.0);

        FlatMaterial fmat = loadFlatMaterial(firstHit);
        SpectralFlatColors fcols = colsFor(fmat, firstHit.materialId, sw);
        float2 firstBB = materialBlackbody(firstHit.materialId);

        SpectralBDPTVertex z1;
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

        Spectrum escapedRadiance = throughputScaledEnvNEE(fmat, fcols, z1, T, B, wo_ts,
                                                          firstHit.materialId, sw, rng);

        BSDFSample bs = fmat.sample(wo_ts, rng);
        if (bs.valid)
        {
            if (bs.pdf <= 0.0)
                eye[1].isDelta = true;
            float3 wi_world = tangentToWorld(bs.wi, T, B, z1.N);

            // Dispersion at the first eye vertex.
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
                && effectiveFurnaceMode(firstHit.materialId) == 0u)
                ? bs.pdf : -1.0;
            eyeLen = randomWalkS(eye, 2, r2, throughput, pdfFwdOmega, sw, rng,
                                 true, escapedRadiance, misBsdfPdfIn);
        }

        // Light subpath.
        SpectralBDPTVertex lightPath[BDPT_MAX_VERTS];
        int lightLen = 0;

        SpectralLightOrigin lo = sampleLightOriginS(sw, rng);
        if (lo.valid)
        {
            lightPath[0] = lo.vertex;
            lightLen = 1;
            // Distant lights walk too (change distant-light-caustic-parity) —
            // the DIR seed passes its AREA-measure pdfPos so randomWalkS's
            // parallel-projection case converts it at the first surface hit.
            Ray rL;
            rL.origin = lo.vertex.position + lo.vertex.N * 0.002;
            rL.direction = lo.dirOut;
            Spectrum unusedEscape = Spectrum(0.0);
            float pdfIn = (lo.vertex.kind == BDPT_VK_LIGHT_DIR)
                        ? lo.pdfPos : lo.pdfDir;
            lightLen = randomWalkS(lightPath, 1, rL, lo.beta, pdfIn, sw, rng,
                                   false, unusedEscape, -1.0);
        }

        splatLightWalkS<TC>(lightPath, lightLen, sw, rng, cam);

        Spectrum L = escapedRadiance;

        // (s ≥ 2, t = 0): emissive eye-vertex hits, MIS-gated as bdpt.slang.
        [loop]
        for (int s = 2; s <= eyeLen; s++)
        {
            SpectralBDPTVertex z = eye[s - 1];
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
            L += connectT1S(eye, s, sw, rng);

        // (s ≥ 2, t ≥ 2): generic connections + MIS.
        BDPTVertex eyeRgb[BDPT_MAX_VERTS];
        BDPTVertex litRgb[BDPT_MAX_VERTS];
        mirrorRgb(eye, eyeRgb);
        mirrorRgb(lightPath, litRgb);
        [loop]
        for (int s = 2; s <= eyeLen; s++)
        {
            [loop]
            for (int t = 2; t <= lightLen; t++)
            {
                if (s + t - 2 > int(BDPT_MAX_DEPTH)) continue;
                SpectralConnectResult cr = connectGenericS(eye, s, lightPath, t, sw);
                if (!cr.nonzero) continue;
                float w = misWeight(eyeRgb, s, litRgb, t,
                                    cr.pdf_zsm1_rev, cr.pdf_zsm2_rev,
                                    cr.pdf_ytm1_rev, cr.pdf_ytm2_rev);
                L += cr.contribution * w;
            }
        }

        return spectrumResolveToLinearSRGB(sw, L);
    }
};

// Env NEE at the first eye vertex (throughput = 1). Split out so estimateRadiance
// reads like bdpt.slang's z1 block.
Spectrum throughputScaledEnvNEE(FlatMaterial mat, SpectralFlatColors cols,
                                SpectralBDPTVertex z1, float3 T, float3 B, float3 wo_ts,
                                uint matId, SampledWavelengths sw, inout RNG rng)
{
    return spectralBdptEnvNEE(mat, cols, z1.position, z1.N, T, B, wo_ts, matId, sw, rng);
}
