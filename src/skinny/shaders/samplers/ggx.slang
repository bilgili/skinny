// GGX microfacet importance sampler.
//
// Draws half-vectors from the GGX *visible* normal distribution (Heitz &
// d'Eon 2018) in tangent space (N = +Z) and reflects ``wo`` through them
// to produce ``wi``. VNDF sampling is unbiased and — unlike classical
// D(H) sampling — yields a finite Monte-Carlo weight at grazing
// configurations: the BRDF×cosθ/pdf ratio reduces to F·G₁(L), bounded
// above by 1.  Classical D(H) sampling produced ``VdotH/(NdotV·NdotH)``
// in the weight, which is the source of the spec-lobe fireflies the
// firefly_debug tool documented at extreme magnitudes (see commit
// history). Callers compute the colored weight outside this struct.
//
// References:
//   Heitz, "Sampling the GGX Distribution of Visible Normals",
//     JCGT 7(4), 2018. (sec. 3 derivation; sec. 4 algorithm)
//   Walter, Marschner, Li, Torrance,
//     "Microfacet Models for Refraction through Rough Surfaces",
//     EGSR 2007. (D, G₁ definitions)

import common;
import interfaces;

struct GGXSampler : ISampler
{
    float roughness;

    // VNDF importance sampling of the GGX microfacet normal in tangent
    // space, then reflection to produce ``wi``. Returns ``wi`` (caller
    // checks ``wi.z`` and handles below-horizon rejection).
    float3 sample(float3 wo, float2 uv)
    {
        float a  = max(roughness * roughness, 1e-4);

        // Transform view direction to the hemisphere configuration.
        float3 Vh = normalize(float3(wo.xy * a, max(wo.z, 0.0)));

        // Sample a spherical cap in (-Vh.z, 1] — Heitz & d'Eon 2023
        // simplified parameterisation (used here instead of the older
        // Heitz 2018 basis form because it avoids the degenerate
        // basis singularity when Vh aligns with +Z).
        float phi      = 2.0 * PI * uv.x;
        float z        = (1.0 - uv.y) * (1.0 + Vh.z) - Vh.z;
        float sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
        float3 c       = float3(sinTheta * cos(phi), sinTheta * sin(phi), z);

        float3 H = c + Vh;
        // Transform the microfacet normal back to the ellipsoid
        // configuration and clamp the z component non-negative so the
        // reflected direction never crosses the geometric horizon
        // through a numerical underflow.
        H = normalize(float3(H.xy * a, max(H.z, 0.0)));

        return reflect(-wo, H);
    }

    // PDF (solid-angle measure) of ``wi`` given ``wo`` under VNDF
    // sampling. Returns 0 for occluded or non-physical inputs so MIS
    // weights stay well-defined. Mirrors the formula in Heitz 2018
    //   p(L) = D(H) · G₁(V) / (4 · NdotV).
    float pdf(float3 wi, float3 wo)
    {
        if (wi.z <= 0.0 || wo.z <= 0.0) return 0.0;

        float3 H     = normalize(wo + wi);
        float  NdotH = H.z;
        if (NdotH <= 0.0) return 0.0;

        float a  = max(roughness * roughness, 1e-4);
        float a2 = a * a;
        float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D  = a2 / (PI * d * d);

        float G1V = ggxSmithG1(wo.z, roughness);
        return D * G1V / (4.0 * wo.z);
    }
}
