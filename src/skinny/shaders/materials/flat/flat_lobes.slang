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
// Each lobe carries a sampler id resolved at runtime to an ISampler. This
// change registers ONLY the native strategies (GGX VNDF for coat/spec, cosine
// for diffuse); the dispatch below is the runtime-pluggable seam, shipped
// unpopulated. A future change (`per-lobe-sampler-registry`) adds samplerId
// cases + a host registry without touching sample()/evaluate().

import common;                          // schlickFresnelColor, ggxSmithG1, fresnelDielectric, relativeLuminance
import samplers.ggx;                    // GGXSampler (VNDF)
import samplers.lambert;                // LambertSampler (cosine)
import materials.flat.flat_shading;     // FlatHitMat

// Lobe kinds.
static const uint LOBE_COAT    = 0u;
static const uint LOBE_SPEC    = 1u;
static const uint LOBE_DIFFUSE = 2u;

// Per-lobe sampler strategies. Native is the only one registered here.
static const uint FLAT_SAMPLER_NATIVE = 0u;

// ── Per-lobe sampler dispatch (the runtime-pluggable seam) ──
// native: GGX VNDF for coat/spec, cosine for diffuse. Adding a strategy means
// adding a samplerId case here — sample()/evaluate() stay untouched.
float3 flatSampleLobe(uint lobeKind, uint samplerId, float roughness,
                      float3 wo, float2 uv)
{
    if (lobeKind == LOBE_DIFFUSE)
    {
        LambertSampler s;
        return s.sample(wo, uv);
    }
    GGXSampler s;
    s.roughness = roughness;
    return s.sample(wo, uv);
}

float flatLobePdf(uint lobeKind, uint samplerId, float roughness,
                  float3 wi, float3 wo)
{
    if (lobeKind == LOBE_DIFFUSE)
    {
        LambertSampler s;
        return s.pdf(wi, wo);
    }
    GGXSampler s;
    s.roughness = roughness;
    return s.pdf(wi, wo);
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
// The actual marginal density FlatMaterial.sample() draws `wi` from.
float flatBsdfPdf(float3 wo, float3 wi,
                  float pCoat, float coatRoughness,
                  float pSpec, float baseRoughness)
{
    float coatPdf = flatLobePdf(LOBE_COAT,    FLAT_SAMPLER_NATIVE, coatRoughness, wi, wo);
    float specPdf = flatLobePdf(LOBE_SPEC,    FLAT_SAMPLER_NATIVE, baseRoughness, wi, wo);
    float diffPdf = flatLobePdf(LOBE_DIFFUSE, FLAT_SAMPLER_NATIVE, 0.0,           wi, wo);
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
    float  coatFresnel = (m.coat > 0.0) ? fresnelDielectric(NdotV, m.coatIOR) : 0.0;
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

    // Base spec lobe: f·cos = pdf_spec · (Fs · G₁(L)).
    float3 Fs       = schlickFresnelColor(VdotH, F0);
    float  G1Ls     = ggxSmithG1(NdotL, m.roughness);
    float  pdfS     = flatLobePdf(LOBE_SPEC, FLAT_SAMPLER_NATIVE, m.roughness, wi, wo);
    float3 specResp = pdfS * (Fs * G1Ls);

    // Base diffuse lobe (Lambert): f·cos = pdf_diff · diffHue.
    float3 Fview    = schlickFresnelColor(NdotV, F0);
    float3 diffHue  = m.albedo * (float3(1.0) - Fview) * (1.0 - m.metallic);
    float  pdfD     = flatLobePdf(LOBE_DIFFUSE, FLAT_SAMPLER_NATIVE, 0.0, wi, wo);
    float3 diffResp = pdfD * diffHue;

    // Base lobes are selected with probability (1 - pCoat) and attenuated by
    // the coat. The pSpec / (1-pSpec) within-base split cancels against
    // sample()'s per-lobe weights, leaving the bounded ratios above.
    resp += (1.0 - pCoat) * coatAtten * (specResp + diffResp);
    return resp;
}
