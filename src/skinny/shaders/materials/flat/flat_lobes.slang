// Unified lobe model for the flat / std_surface BSDF.
//
// One lobe set — {coat, spec, diffuse} — is the single source of truth that
// BOTH FlatMaterial.sample() (drawing) and FlatMaterial.evaluate() (weighting
// / pdf, used by NEE, BDPT connections, ReSTIR, and the directional-proposal
// mixture) walk. Because sample() and evaluate() share these helpers,
// sample().pdf == evaluate().pdf structurally and evaluate().response /
// evaluate().pdf reduces to the bounded native per-lobe weight (F·G₁ for the
// GGX coat/spec lobes, the diffuse albedo term for the Lambert lobe) — the
// firefly fix is preserved by construction, not by a clamp.
//
// Each lobe carries a sampler id resolved at runtime to a draw/density strategy.
// Native (id 0) is the 2023 spherical-cap VNDF for coat/spec and cosine for
// diffuse; the registered alternates are the Heitz-2018 basis-form VNDF (GGX
// lobes) and uniform-hemisphere (diffuse). sample() and evaluate() read the SAME
// per-lobe id (unpacked from fc.flatLobeSamplers), so sample().pdf ==
// evaluate().pdf for every strategy. flatBsdfResponse (= f·cos) is
// sampler-invariant: it always spells the physical f·cos via the native pdf, so
// only the {draw, pdf, weight} triple depends on the id. Host registry:
// sampling/lobe_samplers.py.

import common;                          // schlickFresnelColor, ggxSmithG1, fresnelDielectric, relativeLuminance, PI
import samplers.ggx;                    // GGXSampler (spherical-cap VNDF) + sampleGGXVNDFBasis
import samplers.lambert;                // LambertSampler (cosine)
import materials.flat.flat_shading;     // FlatHitMat

// Lobe kinds.
static const uint LOBE_COAT    = 0u;
static const uint LOBE_SPEC    = 1u;
static const uint LOBE_DIFFUSE = 2u;

// Per-lobe sampler strategies. Native (id 0) is each lobe's default: the 2023
// spherical-cap / bounded VNDF for the GGX coat/spec lobes, cosine for diffuse.
// Alternates are registered host-side (sampling/lobe_samplers.py) and dispatched
// by id below; an unrecognised (lobe, id) pair falls back to native.
static const uint FLAT_SAMPLER_NATIVE  = 0u;
static const uint FLAT_SAMPLER_BASIS   = 1u;   // GGX coat/spec: Heitz-2018 basis-form VNDF
static const uint FLAT_SAMPLER_UNIFORM = 2u;   // diffuse: uniform-hemisphere

static const float FLAT_INV_TWO_PI = 0.15915494309189535;  // 1 / (2π)

// Extract a lobe's 8-bit sampler id from the packed FrameConstants field
// (`flatLobeSamplers`: coat | spec<<8 | diff<<16). LOBE_* values are the byte
// indices, so the shift is `lobeKind * 8`.
uint flatLobeSamplerId(uint packed, uint lobeKind)
{
    return (packed >> (lobeKind * 8u)) & 0xFFu;
}

// Uniform-hemisphere draw in tangent space (N=+Z), z>0. Constant density 1/2π.
float3 flatUniformHemisphere(float2 uv)
{
    float z   = uv.x;                                   // z ∈ (0, 1]
    float r   = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * uv.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

// ── Per-lobe sampler dispatch (the runtime-pluggable seam) ──
// GGX coat/spec: native = spherical-cap VNDF (GGXSampler), id BASIS = Heitz-2018
// basis-form. Diffuse: native = cosine (LambertSampler), id UNIFORM =
// uniform-hemisphere. Unknown id → native. Adding a strategy means adding a case
// here + a registry entry — sample()/evaluate() stay untouched.
float3 flatSampleLobe(uint lobeKind, uint samplerId, float roughness,
                      float3 wo, float2 uv)
{
    if (lobeKind == LOBE_DIFFUSE)
    {
        if (samplerId == FLAT_SAMPLER_UNIFORM)
            return flatUniformHemisphere(uv);
        LambertSampler s;
        return s.sample(wo, uv);
    }
    if (samplerId == FLAT_SAMPLER_BASIS)
        return sampleGGXVNDFBasis(roughness, wo, uv);
    GGXSampler s;
    s.roughness = roughness;
    return s.sample(wo, uv);
}

// Solid-angle density of `wi`. The GGX basis-form and spherical-cap warps draw
// from the SAME visible-normal distribution, so the GGX pdf is samplerId-agnostic
// (one shared GGXSampler.pdf) — this is what makes basis-form parity structural.
// Only the diffuse lobe's density differs by strategy (cosine NdotL/π vs uniform
// 1/2π), so it is the only samplerId branch here.
float flatLobePdf(uint lobeKind, uint samplerId, float roughness,
                  float3 wi, float3 wo)
{
    if (lobeKind == LOBE_DIFFUSE)
    {
        if (samplerId == FLAT_SAMPLER_UNIFORM)
            return (wi.z > 0.0) ? FLAT_INV_TWO_PI : 0.0;
        LambertSampler s;
        return s.pdf(wi, wo);
    }
    GGXSampler s;
    s.roughness = roughness;
    return s.pdf(wi, wo);
}

// Oren-Nayar qualitative diffuse multiplier relative to Lambert (= 1 at σ=0).
// Response-only: the diffuse lobe keeps cosine sampling, so multiplying f by this
// factor leaves the pdf (NdotL/π) untouched — sample().pdf == evaluate().pdf.
// Bounded by construction (cosβ floored like the other 1e-x domain guards), so
// response/pdf = diffHue·ON stays finite — firefly-free without a clamp.
// `sigma` is `diffuseRoughness`; wo/wi are tangent-space (N=+Z, z>0). σ=0 ⇒ exact
// Lambert (early-out returns 1).
float orenNayarFactor(float3 wo, float3 wi, float sigma)
{
    if (sigma <= 0.0)
        return 1.0;
    float s2 = sigma * sigma;
    float A  = 1.0 - 0.5 * s2 / (s2 + 0.33);
    float B  = 0.45 * s2 / (s2 + 0.09);
    // Azimuthal cos(Δφ) from the tangent-plane projections of wo, wi.
    float2 woT = wo.xy;
    float2 wiT = wi.xy;
    float  lo  = length(woT);
    float  li  = length(wiT);
    float  cosPhi = (lo > 1e-5 && li > 1e-5) ? dot(woT, wiT) / (lo * li) : 0.0;
    // α = larger zenith angle (smaller cos), β = smaller angle (larger cos).
    float cosA = min(wo.z, wi.z);
    float cosB = max(wo.z, wi.z);
    float sinA = sqrt(max(0.0, 1.0 - cosA * cosA));
    float tanB = sqrt(max(0.0, 1.0 - cosB * cosB)) / max(cosB, 1e-4);
    return A + B * max(0.0, cosPhi) * sinA * tanB;
}

// Lobe-selection probability from colored Fresnel / albedo balance.
float flatLobePSpec(float NdotV, float3 albedo, float metallic, float3 F0)
{
    float3 Fview   = schlickFresnelColor(NdotV, F0);
    float3 diffHue = albedo * (float3(1.0) - Fview) * (1.0 - metallic);
    float specLum  = relativeLuminance(Fview);
    float diffLum  = relativeLuminance(diffHue);
    return clamp(specLum / max(specLum + diffLum, 1e-6), 0.05, 0.95);
}

// MIS pdf across coat + base spec + base diffuse lobes (solid-angle measure).
// The actual marginal density FlatMaterial.sample() draws `wi` from. Each lobe's
// density uses that lobe's selected sampler id, so this equals the draw density
// for ANY registered strategy (GGX ids share one pdf; diffuse cosine vs uniform
// differ). sample() and evaluate() pass the same ids → their pdfs agree.
float flatBsdfPdf(float3 wo, float3 wi,
                  float pCoat, float coatRoughness,
                  float pSpec, float baseRoughness,
                  uint coatSamplerId, uint specSamplerId, uint diffSamplerId)
{
    float coatPdf = flatLobePdf(LOBE_COAT,    coatSamplerId, coatRoughness, wi, wo);
    float specPdf = flatLobePdf(LOBE_SPEC,    specSamplerId, baseRoughness, wi, wo);
    float diffPdf = flatLobePdf(LOBE_DIFFUSE, diffSamplerId, 0.0,           wi, wo);
    float basePdf = pSpec * specPdf + (1.0 - pSpec) * diffPdf;
    return pCoat * coatPdf + (1.0 - pCoat) * basePdf;
}

// f(L,V)·NdotL for the unified BSDF — the exact integrand FlatMaterial.sample()
// is an unbiased estimator of, i.e. Σ_lobe P(lobe)·pdf_lobe·W_lobe where
// P(lobe) is the lobe-selection probability and W_lobe is sample()'s bounded
// per-lobe weight. The selection factors (pCoat, 1-pCoat) are part of the
// effective model so that response/pdf collapses to those exact weights:
//   coat:    response/pdf → coatColor·Fc·G₁          (sample()'s coat weight)
//   spec:    response/pdf → coatAtten·Fs·G₁/pSpec    (sample()'s spec weight)
//   diffuse: response/pdf → coatAtten·diffHue/(1-pSpec)
// MaterialX convention: returns f·NdotL (NEE / BDPT / ReSTIR consume this).
float3 flatBsdfResponse(float3 wo, float3 wi, FlatHitMat m, float3 F0)
{
    if (wi.z <= 0.0 || wo.z <= 0.0)
        return float3(0.0);

    float  NdotV = max(wo.z, 1e-6);
    float  NdotL = wi.z;
    float3 H     = normalize(wo + wi);
    float  VdotH = max(dot(wo, H), 0.0);

    // Lobe-selection probabilities — identical to FlatMaterial.sample().
    // Entering eta = 1/coatIOR (view ray enters coat from air); see sample().
    float  coatFresnel = (m.coat > 0.0) ? fresnelDielectric(NdotV, 1.0 / m.coatIOR) : 0.0;
    float  pCoat       = m.coat * coatFresnel;
    float  pSpec       = flatLobePSpec(NdotV, m.albedo, m.metallic, F0);
    float3 coatAtten   = (m.coat > 0.0) ? m.coatColor : float3(1.0);

    float3 resp = float3(0.0);

    // Coat lobe: f·cos = pCoat · pdf_coat · (coatColor · Fc · G₁(L)).
    if (m.coat > 0.0)
    {
        float r    = (1.0 - m.coatIOR) / (1.0 + m.coatIOR);
        float F0c  = r * r;
        float Fc   = schlickFresnelColor(VdotH, float3(F0c)).r;
        float G1L  = ggxSmithG1(NdotL, m.coatRoughness);
        float pdfC = flatLobePdf(LOBE_COAT, FLAT_SAMPLER_NATIVE, m.coatRoughness, wi, wo);
        resp += pCoat * pdfC * (m.coatColor * (Fc * G1L));
    }

    // Base spec lobe: f·cos = pdf_spec · (Fs · G₁(L) · specularColor). The
    // specularColor tint is a response-only factor — pdf unchanged. White ⇒ no
    // change.
    float3 Fs       = schlickFresnelColor(VdotH, F0);
    float  G1Ls     = ggxSmithG1(NdotL, m.roughness);
    float  pdfS     = flatLobePdf(LOBE_SPEC, FLAT_SAMPLER_NATIVE, m.roughness, wi, wo);
    float3 specResp = pdfS * (Fs * G1Ls) * m.specularColor;

    // Base diffuse lobe: f·cos = pdf_diff · diffHue · ON. Oren-Nayar (ON) is a
    // response-only factor; cosine sampling is kept so pdfD is unchanged.
    // diffuseRoughness = 0 ⇒ ON = 1 ⇒ exact Lambert.
    float3 Fview    = schlickFresnelColor(NdotV, F0);
    float3 diffHue  = m.albedo * (float3(1.0) - Fview) * (1.0 - m.metallic);
    float  pdfD     = flatLobePdf(LOBE_DIFFUSE, FLAT_SAMPLER_NATIVE, 0.0, wi, wo);
    float  on       = orenNayarFactor(wo, wi, m.diffuseRoughness);
    float3 diffResp = pdfD * diffHue * on;

    // Base lobes are selected with probability (1 - pCoat) and attenuated by
    // the coat. The pSpec / (1-pSpec) within-base split cancels against
    // sample()'s per-lobe weights, leaving the bounded ratios above.
    resp += (1.0 - pCoat) * coatAtten * (specResp + diffResp);
    return resp;
}
