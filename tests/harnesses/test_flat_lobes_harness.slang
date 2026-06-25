// Test harness for the unified flat / std_surface lobe BSDF — exercises the
// Stage-2 rich inputs (flat-lobes-rich-inputs): specular_color (response-only
// spec tint), diffuse_roughness (Oren-Nayar response), and the pdf-symmetry
// invariant (the rich inputs must NOT perturb flatBsdfPdf, which is what makes
// sample().pdf == evaluate().pdf hold).

import common;                          // schlickF0, fresnelDielectric
import materials.flat.flat_shading;     // FlatHitMat, schlickF0
import materials.flat.flat_lobes;       // flatBsdfResponse, flatBsdfPdf, orenNayarFactor, flatLobePSpec

struct FlatEvalResult
{
    float3 response;
    float  pdf;
};

FlatHitMat _mkFlat(float3 albedo, float roughness, float metallic,
                   float3 specularColor, float diffuseRoughness)
{
    FlatHitMat m;
    m.albedo            = albedo;
    m.roughness         = roughness;
    m.metallic          = metallic;
    m.specular          = 0.5;
    m.ior               = 1.5;
    m.opacity           = 1.0;
    m.emission          = float3(0.0);
    m.coat              = 0.0;
    m.coatRoughness     = 0.04;
    m.coatIOR           = 1.5;
    m.coatColor         = float3(1.0);
    m.transmissionColor = albedo;
    m.specularColor     = specularColor;
    m.diffuseRoughness  = diffuseRoughness;
    return m;
}

// f·cos (response) and the all-native solid-angle pdf for a flat material with
// the given rich inputs. Both are exactly what NEE / evaluate() consume.
FlatEvalResult test_flat_eval(float3 wo, float3 wi, float3 albedo, float roughness,
                              float metallic, float3 specularColor,
                              float diffuseRoughness)
{
    FlatHitMat m = _mkFlat(albedo, roughness, metallic, specularColor, diffuseRoughness);
    float3 F0    = schlickF0(m.albedo, m.metallic, m.specular, m.ior);
    float  NdotV = max(wo.z, 1e-6);
    float  pSpec = flatLobePSpec(NdotV, m.albedo, m.metallic, F0);

    FlatEvalResult r;
    r.response = flatBsdfResponse(wo, wi, m, F0);
    r.pdf      = flatBsdfPdf(wo, wi, /*pCoat*/ 0.0, m.coatRoughness,
                             pSpec, m.roughness,
                             FLAT_SAMPLER_NATIVE, FLAT_SAMPLER_NATIVE,
                             FLAT_SAMPLER_NATIVE);
    return r;
}

// Oren-Nayar multiplier (1.0 at σ=0 ⇒ exact Lambert).
float test_oren_nayar(float3 wo, float3 wi, float sigma)
{
    return orenNayarFactor(wo, wi, sigma);
}

// f·cos (response) for a grey diffuse base under a dielectric coat of the given
// weight/roughness/IOR. Exercises the coat lobe-selection Fresnel
// (fix-flat-coat-fresnel-eta): the view ray enters the coat from air, so the
// coat term must use the entering eta = 1/coatIOR. The earlier bug passed
// coatIOR raw — the coat→air direction — which gives spurious total internal
// reflection past ~42° (NdotV < 0.745), driving pCoat → 1 and zeroing the base
// diffuse/spec lobes. A unit-weight dielectric coat must instead only mildly
// attenuate the base (≈ Fresnel), so the coated response must NOT collapse.
float3 test_flat_coat_response(float3 wo, float3 wi, float coat,
                               float coatRoughness, float coatIOR)
{
    FlatHitMat m = _mkFlat(float3(0.5), /*roughness*/ 1.0, /*metallic*/ 0.0,
                           /*specularColor*/ float3(1.0), /*diffuseRoughness*/ 0.0);
    m.coat          = coat;
    m.coatRoughness = coatRoughness;
    m.coatIOR       = coatIOR;
    m.coatColor     = float3(1.0);
    float3 F0 = schlickF0(m.albedo, m.metallic, m.specular, m.ior);
    return flatBsdfResponse(wo, wi, m, F0);
}
