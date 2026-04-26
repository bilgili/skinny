// Composite BSDF for layered skin. Combines:
//   1. GGX dielectric specular driven by `roughness` + `ior` (Fresnel-Schlick)
//   2. Layered Lambertian diffuse — point-BSSRDF estimator using the
//      τ-weighted single-scatter albedo across the three layers, with the
//      dermis pigment overlay folded in.
//   3. Vellus-hair sheen rim (additive on the diffuse side).
//
// Mirrors the per-pixel composition in main_pass.slang::mainImage. The full
// detail-map / pore / hair-tilt physics from shaders/skin_bssrdf.slang +
// shaders/detail.slang are landed incrementally in Phase E; this Phase A
// skeleton provides the right structural shape and the dominant lobes so
// the gen + integrator can be exercised end-to-end.
//
// References:
//   - Jensen et al. "A Practical Model for Subsurface Light Transport" (2001)
//   - Christensen & Burley "Approximate Reflectance Profiles" (Disney 2015)
//   - Walter et al. "Microfacet Models for Refraction" (EGSR 2007)

#include "lib/mx_closure_type.glsl"

// ─── Local helpers ─────────────────────────────────────────────────

// Reduced single-scattering albedo σ_s' / (σ_a + σ_s') for one layer.
float3 skinny_layer_albedo(float3 sigmaA, float3 sigmaS, float g)
{
    float3 sigmaSprime = sigmaS * (1.0 - g);
    float3 sigmaT      = sigmaA + sigmaSprime;
    return sigmaSprime / max(sigmaT, float3(1e-6));
}

// Schlick approximation of Fresnel at a dielectric interface.
float skinny_fresnel_schlick(float cosTheta, float ior)
{
    float r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;
    float c = 1.0 - cosTheta;
    return r0 + (1.0 - r0) * c * c * c * c * c;
}

// GGX normal distribution.
float skinny_ggx_D(float NdotH, float roughness)
{
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265358979 * d * d);
}

// Smith-G1 partial geometry.
float skinny_ggx_G1(float NdotV, float roughness)
{
    float a = roughness * roughness;
    float k = a / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// ─── BSDF entry point ──────────────────────────────────────────────

void skinny_skin_layered_bsdf(
    ClosureData closureData,

    // Top layer (epidermis-like)
    float3 top_absorption,
    float3 top_scattering,
    float  top_anisotropy,
    float  top_thickness,
    float  top_ior,

    // Middle layer (dermis-like)
    float3 middle_absorption,
    float3 middle_scattering,
    float  middle_anisotropy,
    float  middle_thickness,
    float  middle_ior,

    // Bottom layer (subcut-like)
    float3 bottom_absorption,
    float3 bottom_scattering,
    float  bottom_anisotropy,
    float  bottom_thickness,
    float  bottom_ior,

    // Pigment overlay (rgb=ink colour, a=density). Not directly read here
    // because the dermis nodedef already folds it into middle_absorption /
    // middle_scattering upstream — kept on the BSDF interface so a future
    // generic non-skin material wiring can route it through unchanged.
    float4 pigment,

    // Surface
    float  weight,
    float  roughness,
    float  ior,
    float3 N,
    float3 T,

    // Sub-mm detail (TODO Phase E: pores perturb Nspec/roughSpec; hair sheen
    // adds an additive lobe). Stub for Phase A.
    float  pore_density,
    float  pore_depth,
    float  hair_density,
    float  hair_tilt,

    inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);
    if (weight < 1e-8) { return; }

    float3 V = closureData.V;
    float3 L = closureData.L;

    // Forward-facing N for safety.
    if (dot(N, V) < 0.0) { N = -N; }
    float NdotV = clamp(dot(N, V), 1e-6, 1.0);

    // Per-layer optical depth τ = σ_t · thickness drives the τ-weighted
    // layer-albedo blend (matches `evaluateSubsurfaceScattering`).
    float3 sigmaT_t = top_absorption    + top_scattering    * (1.0 - top_anisotropy);
    float3 sigmaT_m = middle_absorption + middle_scattering * (1.0 - middle_anisotropy);
    float3 sigmaT_b = bottom_absorption + bottom_scattering * (1.0 - bottom_anisotropy);

    float3 tau_t = sigmaT_t * top_thickness;
    float3 tau_m = sigmaT_m * middle_thickness;
    float3 tau_b = sigmaT_b * bottom_thickness;
    float3 totalTau = max(tau_t + tau_m + tau_b, float3(1e-6));

    float3 albedo =
        (tau_t * skinny_layer_albedo(top_absorption,    top_scattering,    top_anisotropy)
       + tau_m * skinny_layer_albedo(middle_absorption, middle_scattering, middle_anisotropy)
       + tau_b * skinny_layer_albedo(bottom_absorption, bottom_scattering, bottom_anisotropy))
        / totalTau;

    // Fresnel split between specular and diffuse sides.
    float F = skinny_fresnel_schlick(NdotV, ior);

    if (closureData.closureType == CLOSURE_TYPE_REFLECTION)
    {
        // Direct light at L. The integrator multiplies by L-radiance.
        float3 H     = normalize(L + V);
        float NdotL  = clamp(dot(N, L), 1e-6, 1.0);
        float NdotH  = clamp(dot(N, H), 1e-6, 1.0);
        float VdotH  = clamp(dot(V, H), 1e-6, 1.0);

        // GGX specular
        float D = skinny_ggx_D(NdotH, roughness);
        float G = skinny_ggx_G1(NdotV, roughness) * skinny_ggx_G1(NdotL, roughness);
        float Fl = skinny_fresnel_schlick(VdotH, ior);
        float3 spec = float3(D * G * Fl / max(4.0 * NdotV * NdotL, 1e-6));

        // Layered Lambertian (BSSRDF estimator). Returns the BRDF; the
        // integrator multiplies by L · NdotL.
        float3 diffuse = albedo / 3.14159265358979;

        bsdf.response =
            (spec + diffuse * (1.0 - F)) * NdotL * weight * closureData.occlusion;
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT)
    {
        // IBL approximation: directional albedo × environment irradiance.
        // The full GGX + cosine-hemisphere IBL passes from main_pass.slang
        // are reinstated when the gen output is wired into the integrator
        // (Phase C/E). For now give a flat directional-albedo response.
        // mx_environment_irradiance is provided by the genslang environment
        // helper; if unresolved this branch silently produces zero.
        float3 irradiance = mx_environment_irradiance(N);
        bsdf.response = irradiance * albedo * (1.0 - F) * weight;
    }
}
