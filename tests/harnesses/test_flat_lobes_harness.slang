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
