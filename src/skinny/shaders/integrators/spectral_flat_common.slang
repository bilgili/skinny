// Shared hero-wavelength flat-material machinery for the spectral megakernel
// integrators (path_spectral + bdpt_spectral). Compiled ONLY in the spectral
// variant (`-DSKINNY_SPECTRAL`); the RGB build never imports it.
//
// Extracted verbatim from path_spectral.slang (change spectral-bdpt-megakernel,
// Group 1) so the bidirectional integrator reuses the SAME per-λ upsampling,
// conductor Fresnel, exact-source overrides, and NEE as the path tracer. Pure
// code motion — the spectral path output must be bit-unchanged.

import common;
import spectrum;        // SampledWavelengths, upsampleReflectance/IlluminantBound, namedMetalEtaK, planckSpectrum, distantLightSpd
import bindings;        // fc, flat/material buffers, spectral buffers, materialBlackbody
import interfaces;      // ILight, LightSample
import scene_trace;     // visibleDirectional, visibleSegment
import scene_lights;    // loadDirectionalLightImpl / loadSphereLightImpl / loadEmissiveTriangleLightImpl / sampleEmissiveTriangle / emitterBlackbody
import environment;     // sampleEnvDir, envPdf, effectiveFurnaceMode
import materials.flat.flat_shading;    // FlatHitMat
import materials.flat.flat_material;   // FlatMaterial, loadFlatMaterial
import materials.flat.flat_lobes;      // flatBsdfResponseSpectral
import lights.sphere_light;
import lights.emissive_triangle_light;
import lights.directional_light;
import sampling.proposal;              // ProposalContext, mixtureProposalPdf (scalar MIS companion)

// Upsampled flat reflectances for one hit — computed once, reused by the bounce
// weight and every NEE light so the sigmoid-table fetches are shared.
struct SpectralFlatColors
{
    Spectrum albedo;    // reflectance
    Spectrum specColor; // reflectance
    Spectrum coatColor; // reflectance
    Spectrum F0;        // Schlick base (reflectance)
    Spectrum emission;  // illuminant (× D65)
    Spectrum transmission; // reflectance (delta glass tint)
    // Named-conductor exact Fresnel (6.2): isConductor ⇒ evaluate complex-index
    // Fresnel from etaHero/kHero (eta(λ)/k(λ) at the 4 hero wavelengths) instead
    // of Schlick(F0). Off (id 0) ⇒ the RGB/dielectric Schlick path.
    bool     isConductor;
    float4   etaHero;
    float4   kHero;
};

SpectralFlatColors upsampleFlatColors(FlatMaterial mat, SampledWavelengths sw, uint metalId)
{
    FlatHitMat m = mat.data.mat;
    SpectralFlatColors c;
    c.albedo       = upsampleReflectanceBound(m.albedo, sw);
    c.specColor    = upsampleReflectanceBound(m.specularColor, sw);
    c.coatColor    = upsampleReflectanceBound(m.coatColor, sw);
    c.F0           = upsampleReflectanceBound(mat.F0, sw);
    c.emission     = upsampleIlluminantBound(m.emission, sw);
    c.transmission = upsampleReflectanceBound(m.transmissionColor, sw);
    c.isConductor  = (metalId >= 1u && metalId <= SPECTRAL_METAL_COUNT);
    if (c.isConductor)
        namedMetalEtaK(metalId, sw, c.etaHero, c.kHero);
    else
    {
        c.etaHero = float4(0.0);
        c.kHero   = float4(0.0);
    }
    return c;
}

// Spectral f·cos response at a flat hit (tangent space), per wavelength. This is
// the bare BSDF response (no opacity factor) — correct for the bounce THROUGHPUT
// weight, where opacity cancels (response·op / (pdf·op), and the sample pdf here
// carries no op). NEE consumes `response·opacity` directly (see flatResponseNEE),
// mirroring FlatMaterial.evaluate()'s `* op`.
Spectrum flatResponseS(FlatMaterial mat, SpectralFlatColors c, float3 wo, float3 wi)
{
    return flatBsdfResponseSpectral(wo, wi, mat.data.mat, mat.F0,
                                    c.albedo, c.specColor, c.coatColor, c.F0,
                                    c.isConductor, c.etaHero, c.kHero);
}

// NEE response: the analytic `f·cos·opacity` (opacity is NOT stochastically
// sampled at a direct-light connection, so it must be applied here — matching
// FlatMaterial.evaluate().response = flatBsdfResponse * op). Without it, NEE
// over-lights partial-opacity (e.g. glass) surfaces.
Spectrum flatResponseNEE(FlatMaterial mat, SpectralFlatColors c, float3 wo, float3 wi)
{
    return flatResponseS(mat, c, wo, wi) * mat.data.mat.opacity;
}

// Per-wavelength throughput weight for a direction returned by the proposal
// seam. The BSDF-only fast path returns FlatMaterial.sample()'s CONDITIONAL
// continuous-lobe pdf; stochastic branch selection supplies the opacity factor,
// so the historical flatResponseS/pdf estimator must stay unchanged. A genuine
// mixture instead returns mat.evaluate()'s UNCONDITIONAL BSDF component
// (opacity*pdf) plus the other proposal densities, so its numerator must carry
// the matching opacity factor via flatResponseNEE.
Spectrum flatProposalWeightS(
    FlatMaterial mat, SpectralFlatColors c, float3 wo, float3 wi, float proposalPdf)
{
    Spectrum response = (fc.proposalMask == PROPOSAL_BSDF)
                      ? flatResponseS(mat, c, wo, wi)
                      : flatResponseNEE(mat, c, wo, wi);
    return response / proposalPdf;
}

// ── Spectral NEE (flat-only) ─────────────────────────────────────
// Mirrors nee.slang's allLightsNEE, but the BSDF response is per wavelength
// (flatResponseS) and each RGB light/env radiance upsamples as an illuminant, so
// the response(λ)·L(λ) product is formed per wavelength (metamerism). Every pdf /
// MIS weight is the SAME scalar the RGB NEE uses (mixtureProposalPdf on the RGB
// material), so direct + indirect stay MIS-consistent with the bounce sampler.
// `liOverride`/`hasLiOverride`: when set, the sampled light's radiance is taken
// from `liOverride` (already per-wavelength) instead of upsampling ls.radiance —
// used for exact-Planck blackbody emissive triangles (Group 6.1). All geometry /
// pdf / MIS factors are identical to the RGB-upsample path.
Spectrum spectralLightNEE<TL : ILight>(
    FlatMaterial mat, SpectralFlatColors c, TL light,
    HitInfo h, float3 N, float3 T, float3 B, float3 wo,
    SampledWavelengths sw, float2 u,
    Spectrum liOverride, bool hasLiOverride)
{
    LightSample ls = light.samplePoint(h.position, u);
    if (!ls.valid)
        return Spectrum(0.0);

    float3 toLight = ls.point - h.position;
    float  d2 = dot(toLight, toLight);
    if (d2 <= 1e-8)
        return Spectrum(0.0);
    float3 dir = toLight / sqrt(d2);

    float NdotL = max(dot(N, dir), 0.0);
    if (NdotL <= 0.0)
        return Spectrum(0.0);

    float3   wi   = worldToTangent(dir, T, B, N);
    Spectrum resp = flatResponseNEE(mat, c, wo, wi);
    Spectrum Li   = hasLiOverride ? liOverride
                                  : upsampleIlluminantBound(ls.radiance, sw);

    if (ls.pdfArea == 0.0)   // delta (directional) light
    {
        if (!visibleDirectional(fc, h.position, N, dir))
            return Spectrum(0.0);
        return resp * Li;
    }

    float cosLight = max(dot(ls.normal, -dir), 0.0);
    if (cosLight <= 0.0)
        return Spectrum(0.0);
    if (!visibleSegment(fc, h.position, ls.point))
        return Spectrum(0.0);

    float invPdfOmega = cosLight / (d2 * ls.pdfArea);
    float pdfLightSA  = d2 * ls.pdfArea / max(cosLight, 1e-12);

    ProposalContext pc;
    pc.woT = wo; pc.N = N; pc.T = T; pc.B = B;
    pc.position = h.position; pc.materialId = h.materialId;
    pc.neuralValid = false;   // megakernel: neural is wavefront-only
    float misPdf = mixtureProposalPdf(mat, pc, wi);
    float wNEE   = powerHeuristic(pdfLightSA, misPdf);

    return resp * Li * (invPdfOmega * wNEE);
}

Spectrum spectralAllLightsNEE(
    FlatMaterial mat, SpectralFlatColors c, HitInfo h,
    float3 N, float3 T, float3 B, float3 wo, SampledWavelengths sw, inout RNG rng)
{
    Spectrum total = Spectrum(0.0);

    for (uint d = 0u; d < fc.numDistantLights; d++)
    {
        DistantLight raw = distantLights[d];
        DirectionalLightImpl dl = loadDirectionalLightImpl(raw);
        // Group 6.3: an authored `spectrum L` distant light (slot >= 0) emits its
        // exact SPD at the hero wavelengths instead of upsampling the RGB radiance.
        int      slot = int(raw.spdIndex);
        bool     spd  = slot >= 0;
        Spectrum spdL = spd ? distantLightSpd(slot, sw) : Spectrum(0.0);
        total += spectralLightNEE(mat, c, dl, h, N, T, B, wo, sw, float2(0.0),
                                  spdL, spd);
    }

    for (uint i = 0u; i < fc.numSphereLights; i++)
    {
        SphereLightImpl sl = loadSphereLightImpl(sphereLights[i]);
        total += spectralLightNEE(mat, c, sl, h, N, T, B, wo, sw, rng.next2(),
                                  Spectrum(0.0), false);
    }

    if (fc.numEmissiveTriangles > 0u)
    {
        uint idx = sampleEmissiveTriangle(rng.next(), fc.numEmissiveTriangles);
        EmissiveTriangleLightImpl el = loadEmissiveTriangleLightImpl(
            emissiveTriangles[idx], fc.numEmissiveTriangles);
        // Group 6.1: a blackbody area light (temperature > 0) emits the exact
        // Planck SPD at the hero wavelengths, luminance-matched by the host scale,
        // instead of upsampling its RGB reduction. spectralEmitters is parallel-
        // indexed to emissiveTriangles.
        float2   bb = emitterBlackbody(idx);
        bool     bbHit = bb.x > 0.0;
        Spectrum bbLi = bbHit ? planckSpectrum(sw, bb.x) * bb.y : Spectrum(0.0);
        total += spectralLightNEE(mat, c, el, h, N, T, B, wo, sw, rng.next2(),
                                  bbLi, bbHit);
    }

    // Environment NEE (skipped under furnace: constant white env has no features).
    if (effectiveFurnaceMode(h.materialId) == 0u)
    {
        EnvSample es = sampleEnvDir(rng.next2(), fc.envIntensity);
        if (es.pdf > 0.0 && dot(N, es.dir) > 0.0 &&
            visibleDirectional(fc, h.position, N, es.dir))
        {
            float3   wi   = worldToTangent(es.dir, T, B, N);
            Spectrum resp = flatResponseNEE(mat, c, wo, wi);
            ProposalContext pc;
            pc.woT = wo; pc.N = N; pc.T = T; pc.B = B;
            pc.position = h.position; pc.materialId = h.materialId;
            pc.neuralValid = false;
            float misPdf = mixtureProposalPdf(mat, pc, wi);
            if (misPdf > 0.0)
            {
                float    w  = powerHeuristic(es.pdf, misPdf);
                Spectrum Li = upsampleIlluminantBound(es.radiance, sw);
                total += resp * Li * (w / es.pdf);
            }
        }
    }

    return total;
}
