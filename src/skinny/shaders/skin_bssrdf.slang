// Physically-based skin BSSRDF using a separable multi-layer diffusion model.
//
// References:
//   [1] Jensen, Marschner, Levoy, Hanrahan,
//       "A Practical Model for Subsurface Light Transport", SIGGRAPH 2001.
//   [2] d'Eon, Irving,
//       "A Quantized-Diffusion Model for Rendering Translucent Materials",
//       ACM TOG (SIGGRAPH) 2011.
//   [3] Christensen, Burley,
//       "Approximate Reflectance Profiles for Efficient Subsurface Scattering",
//       Disney Research / SIGGRAPH 2015.
//   [4] Donner, Jensen,
//       "A Spectral BSSRDF for Shading Human Skin", EGSR 2006.
//   [5] Walter, Marschner, Li, Torrance,
//       "Microfacet Models for Refraction through Rough Surfaces", EGSR 2007.
//   [6] Schlick,
//       "An Inexpensive BRDF Model for Physically-Based Rendering", CGF 1994.

import common;

// ── Skin optical parameters ─────────────────────────────────────

struct SkinLayer
{
    float3 sigmaA;      // absorption coefficient  (mm⁻¹)
    float3 sigmaS;      // scattering coefficient  (mm⁻¹)
    float  thickness;   // layer thickness          (mm)
    float  g;           // Henyey-Greenstein anisotropy
    float  ior;         // index of refraction
};

struct SkinParams
{
    float melaninFraction;
    float hemoglobinFraction;
    float bloodOxygenation;
    float epidermisThickness;
    float dermisThickness;
    float subcutThickness;
    float3 scatteringCoeff;
    float anisotropy;
    float roughness;
    float ior;
    float poreDensity;      // 0 = off, 1 = visible cheek-like pore pattern
    float poreDepth;        // normal perturbation amplitude inside pores
    float hairDensity;      // 0 = no vellus hair, 1 = heavy peach-fuzz rim
    float hairTilt;         // 0 = hairs lying flat, 1 = standing up (strong rim)
};

/// Eumelanin absorption fit at RGB centres 611 nm / 549 nm / 464 nm [4].
///
///   σ_a(λ) = c_mel · μ_eu(λ)
///
/// where  c_mel      = melanin volume fraction ∈ [0, 1]
///        μ_eu(λ)   = {6.6, 11.0, 24.0} mm⁻¹  (eumelanin RGB coefficients)
float3 melaninAbsorption(float melaninFraction)
{
    // Eumelanin dominant; wavelength-dependent fit at RGB centres (611, 549, 464 nm)
    float3 eumelanin = float3(6.6, 11.0, 24.0);
    return melaninFraction * eumelanin;
}

/// Haemoglobin absorption from oxygenated / deoxygenated mix [4].
///
///   σ_a = c_hb · (sat · μ_HbO₂ + (1 − sat) · μ_Hb)
///
/// where  c_hb    = haemoglobin volume fraction
///        sat     = blood oxygenation ∈ [0, 1]
///        μ_HbO₂ ≈ {0.3, 7.0, 1.0} mm⁻¹  (oxy-haemoglobin RGB)
///        μ_Hb   ≈ {1.6, 8.5, 3.2} mm⁻¹  (deoxy-haemoglobin RGB)
float3 hemoglobinAbsorption(float hbFraction, float oxygenation)
{
    float3 hbO2  = float3(0.3, 7.0, 1.0);   // oxy-hemoglobin  (RGB approx)
    float3 hb    = float3(1.6, 8.5, 3.2);    // deoxy-hemoglobin
    return hbFraction * (oxygenation * hbO2 + (1.0 - oxygenation) * hb);
}

SkinLayer buildEpidermis(SkinParams p)
{
    SkinLayer layer;
    layer.sigmaA   = melaninAbsorption(p.melaninFraction) + float3(0.015);
    layer.sigmaS   = p.scatteringCoeff;
    layer.thickness = p.epidermisThickness;
    layer.g         = p.anisotropy;
    layer.ior       = p.ior;
    return layer;
}

SkinLayer buildDermis(SkinParams p)
{
    SkinLayer layer;
    layer.sigmaA   = hemoglobinAbsorption(p.hemoglobinFraction, p.bloodOxygenation)
                     + float3(0.015);
    layer.sigmaS   = p.scatteringCoeff;
    layer.thickness = p.dermisThickness;
    layer.g         = p.anisotropy;
    layer.ior       = p.ior;
    return layer;
}

// ── Tattoo ink optical contribution ─────────────────────────────
//
// Tattoo inks are micron-scale pigment particles deposited in the papillary
// dermis. Each particle absorbs strongly in its complement spectrum and
// scatters weakly in its own. We model this as an *additive* σ_a / σ_s on
// top of the baseline dermis, driven by the ink sample taken from the
// tattoo texture:
//
//   σ_a_ink = density · ( INK_ABS_MAX · (1 − rgb) + INK_ABS_MIN · rgb )
//   σ_s_ink = density · INK_SCATTER_BASE · (1 + 0.5 · luminance(rgb))
//
// "density" is texture-alpha × global tattooDensity in [0, 1]. A white-
// pigment (TiO₂) ink ends up with near-zero absorption but higher scatter
// — hence the luminance term on σ_s. The constants below are chosen so
// that a fully opaque mid-saturation colour gives a visually convincing
// tattoo without completely dominating the underlying hemoglobin.

static const float INK_ABS_MAX        = 22.0;  // mm⁻¹ in the complement channels
static const float INK_ABS_MIN        = 1.2;   // mm⁻¹ residual in the ink's own colour
static const float INK_SCATTER_BASE   = 3.5;   // mm⁻¹ baseline ink scattering

float3 inkSigmaAbsorption(float3 inkRgb, float density)
{
    float3 absContribution = INK_ABS_MAX * (1.0 - inkRgb) + INK_ABS_MIN * inkRgb;
    return density * absContribution;
}

float3 inkSigmaScattering(float3 inkRgb, float density)
{
    float luminance = dot(inkRgb, float3(0.299, 0.587, 0.114));
    return density * INK_SCATTER_BASE * (1.0 + 0.5 * luminance);
}

SkinLayer buildDermisInked(SkinParams p, float4 ink)
{
    SkinLayer layer = buildDermis(p);
    float density = clamp(ink.a, 0.0, 1.0);
    if (density > 1e-6)
    {
        layer.sigmaA += inkSigmaAbsorption(ink.rgb, density);
        layer.sigmaS += inkSigmaScattering(ink.rgb, density);
    }
    return layer;
}

SkinLayer buildSubcutaneous(SkinParams p)
{
    SkinLayer layer;
    layer.sigmaA   = float3(0.05, 0.06, 0.08);
    layer.sigmaS   = float3(1.2, 1.1, 0.95);
    layer.thickness = p.subcutThickness;
    layer.g         = 0.8;
    layer.ior       = 1.4;
    return layer;
}

struct SkinLayerStack
{
    SkinLayer epi;
    SkinLayer der;
    SkinLayer sub;
};

SkinLayerStack buildSkinLayerStackFromMtlx(MtlxSkinParams skin, float4 ink)
{
    SkinLayerStack s;

    s.epi.sigmaA    = melaninAbsorption(skin.layer_top_melanin) + float3(0.015);
    s.epi.sigmaS    = skin.layer_top_scattering_coeff;
    s.epi.thickness = skin.layer_top_thickness;
    s.epi.g         = skin.layer_top_anisotropy;
    s.epi.ior       = skin.layer_top_ior;

    s.der.sigmaA = hemoglobinAbsorption(
                       skin.layer_middle_hemoglobin,
                       skin.layer_middle_blood_oxygenation
                   ) + float3(0.015);
    s.der.sigmaS = skin.layer_middle_scattering_coeff;
    {
        float density = clamp(ink.a, 0.0, 1.0);
        if (density > 1e-6)
        {
            s.der.sigmaA += inkSigmaAbsorption(ink.rgb, density);
            s.der.sigmaS += inkSigmaScattering(ink.rgb, density);
        }
    }
    s.der.thickness = skin.layer_middle_thickness;
    s.der.g         = skin.layer_middle_anisotropy;
    s.der.ior       = skin.layer_middle_ior;

    s.sub.sigmaA    = skin.layer_bottom_absorption;
    s.sub.sigmaS    = skin.layer_bottom_scattering;
    s.sub.thickness = skin.layer_bottom_thickness;
    s.sub.g         = skin.layer_bottom_anisotropy;
    s.sub.ior       = skin.layer_bottom_ior;

    return s;
}

SkinParams skinParamsFromMtlx(MtlxSkinParams m)
{
    SkinParams p;
    p.melaninFraction    = m.layer_top_melanin;
    p.hemoglobinFraction = m.layer_middle_hemoglobin;
    p.bloodOxygenation   = m.layer_middle_blood_oxygenation;
    p.epidermisThickness = m.layer_top_thickness;
    p.dermisThickness    = m.layer_middle_thickness;
    p.subcutThickness    = m.layer_bottom_thickness;
    p.scatteringCoeff    = m.layer_top_scattering_coeff;
    p.anisotropy         = m.layer_top_anisotropy;
    p.roughness          = m.skin_bsdf_roughness;
    p.ior                = m.skin_bsdf_ior;
    p.poreDensity        = m.skin_bsdf_pore_density;
    p.poreDepth          = m.skin_bsdf_pore_depth;
    p.hairDensity        = m.skin_bsdf_hair_density;
    p.hairTilt           = m.skin_bsdf_hair_tilt;
    return p;
}

// ── Burley normalised diffusion profile ─────────────────────────

/// Burley normalised diffusion profile (Christensen & Burley [3]).
///
///   R(r) = (e^(−r/d) + e^(−r/(3d))) / (8π·d·r)
///
/// where  r = radial distance from entry point (mm)
///        d = diffuse mean free path per RGB channel (mm); d = 1/(σ_a + σ_s')
float3 burleyDiffusionProfile(float r, float3 d)
{
    float3 rd = max(float3(r), float3(1e-6)) / d;
    return (exp(-rd) + exp(-rd / 3.0)) / (8.0 * PI * d * max(float3(r), float3(1e-6)));
}

/// Reduced mean free path, corrected for anisotropic scattering.
///
///   σ_s' = σ_s · (1 − g)      (reduced scattering, anisotropy-corrected)
///   σ_t  = σ_a + σ_s'         (effective extinction coefficient, mm⁻¹)
///   d    = 1 / σ_t            (diffuse MFP per channel, mm)
///
/// where  g = Henyey-Greenstein anisotropy ∈ [−1, 1]
float3 effectiveMFP(SkinLayer layer)
{
    float3 sigmaSprime = layer.sigmaS * (1.0 - layer.g);
    float3 sigmaT      = layer.sigmaA + sigmaSprime;
    return 1.0 / max(sigmaT, float3(1e-6));
}

// ── Fresnel ─────────────────────────────────────────────────────

/// Schlick Fresnel approximation for dielectrics [6].
///
///   F(θ) = F₀ + (1 − F₀) · (1 − cosθ)⁵
///
/// where  F₀ = ((1 − n) / (1 + n))²   (normal-incidence reflectance)
///        θ  = angle between the view direction and the half-vector
float fresnelSchlick(float cosTheta, float ior)
{
    float r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;
    float c  = 1.0 - cosTheta;
    return r0 + (1.0 - r0) * c * c * c * c * c;
}

// ── GGX microfacet specular for skin surface ────────────────────

/// GGX (Trowbridge-Reitz) normal distribution function [5].
///
///   D(h) = α² / (π · ((n·h)² · (α² − 1) + 1)²)
///
/// where  h   = microfacet half-vector (normalised)
///        α   = roughness²  (Disney perceptual remapping)
///        n·h = dot(shading normal, half-vector)
float ggxDistribution(float NdotH, float roughness)
{
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

/// Microfacet specular BRDF: GGX NDF + separable Smith G + Schlick F [5, 6].
///
///   f_spec(l, v) = D(h) · G₁(v) · G₁(l) · F(v·h) / (4 · (n·l) · (n·v))
///
/// where  h = normalize(v + l)
///        D = GGX NDF (ggxDistribution)   [5]
///        G = separable Smith G₁ product  (ggxSmithG1 × ggxSmithG1)
///        F = Schlick Fresnel             [6]
float3 evaluateSkinSpecular(float3 N, float3 V, float3 L, float roughness, float ior)
{
    float3 H    = normalize(V + L);
    float NdotH = max(dot(N, H), 1e-6);
    float NdotV = max(dot(N, V), 1e-6);
    float NdotL = max(dot(N, L), 1e-6);
    float VdotH = max(dot(V, H), 1e-6);

    float D = ggxDistribution(NdotH, roughness);
    float G = ggxSmithG1(NdotV, roughness) * ggxSmithG1(NdotL, roughness);
    float F = fresnelSchlick(VdotH, ior);

    return float3(D * G * F / max(4.0 * NdotV * NdotL, 1e-6));
}

/// Sampling PDF for GGX NDF importance sampling (half-vector form) [5].
///
///   p(l) = D(h) · (n·h) / (4 · (v·h))
///
/// where  h   = normalize(v + l)
///        D   = GGX NDF evaluated at h
///        The Jacobian |∂h/∂l| = 1/(4·(v·h)) converts half-vector PDF to
///        outgoing-direction PDF.
float ggxSpecularPdf(float3 N, float3 V, float3 L, float roughness)
{
    float3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 1e-6);
    if (dot(N, L) <= 0.0 || NdotH <= 0.0)
        return 0.0;
    return ggxDistribution(NdotH, roughness) * NdotH / (4.0 * VdotH);
}

/// GGX NDF importance-sampled specular for image-based lighting [5].
///
/// Samples a half-vector h from the NDF, reflects V to get L.
/// The Monte Carlo weight simplifies to:
///
///   w = BRDF · cosθ / pdf = G₁(v) · G₁(l) · F · (v·h) / ((n·v) · (n·h))
///
/// Multiplied by L(l), this gives an unbiased one-sample estimate of
///   ∫ f_spec(ω) · L(ω) · (n·ω) dω.
/// Zero weight is returned for below-horizon samples (n·l ≤ 0).
struct SpecularIBLSample
{
    float3 direction;
    float3 weight;
};

SpecularIBLSample sampleGGXSpecularIBL(
    float3 N, float3 V, float roughness, float ior, float2 u
)
{
    float a = max(roughness * roughness, 1e-4);
    float a2 = a * a;

    float cosTheta = sqrt((1.0 - u.y) / (1.0 + (a2 - 1.0) * u.y));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi      = 2.0 * PI * u.x;

    float3 tangent   = abs(N.x) > 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 bitangent = normalize(cross(N, tangent));
    tangent          = cross(bitangent, N);

    float3 H = normalize(
        tangent   * cos(phi) * sinTheta +
        bitangent * sin(phi) * sinTheta +
        N * cosTheta
    );

    float3 L = reflect(-V, H);

    float NdotV = max(dot(N, V), 1e-6);
    float NdotL = dot(N, L);
    float NdotH = max(dot(N, H), 1e-6);
    float VdotH = max(dot(V, H), 1e-6);

    SpecularIBLSample s;
    s.direction = L;
    if (NdotL <= 0.0)
    {
        s.weight = float3(0.0);
        return s;
    }

    float G = ggxSmithG1(NdotV, roughness) * ggxSmithG1(NdotL, roughness);
    float F = fresnelSchlick(VdotH, ior);
    s.weight = float3(G * F * VdotH / (NdotV * NdotH));
    return s;
}

// ── Subsurface evaluation (disk-based sampling) ─────────────────

/// Reduced single-scatter albedo (transport theory).
///
///   ω₀ = σ_s' / (σ_a + σ_s')
///
/// where  σ_s' = σ_s · (1 − g)   (anisotropy-corrected scattering, mm⁻¹)
///        σ_a  = absorption coefficient (mm⁻¹)
/// The ratio sets the diffuse skin tint: melanin raises σ_a → darker ω₀,
/// haemoglobin shifts ω₀ toward red.
float3 layerAlbedo(SkinLayer layer)
{
    float3 sigmaSprime = layer.sigmaS * (1.0 - layer.g);
    float3 sigmaT      = layer.sigmaA + sigmaSprime;
    return sigmaSprime / max(sigmaT, float3(1e-6));
}

// Point-wise SSS: returns outgoing radiance for irradiance landing on a skin
// patch, using the layered diffuse albedo as a Lambertian BRDF. This is an
// unblurred approximation — the full Burley profile below is retained for
// disk-sampled BSSRDF if/when we add multi-tap sampling.
//
// `ink` samples a tattoo texture at this pixel: rgb is the linear ink colour,
// a is the effective density (texture-alpha × global tattooDensity). When
// a == 0 the function is identical to the bare-skin case.
float3 evaluateSubsurfaceScattering(
    SkinParams params,
    float3 position,
    float3 normal,
    float  sampleRadius,
    float3 sampleIrradiance,
    float4 ink
)
{
    SkinLayer epidermis = buildEpidermis(params);
    SkinLayer dermis    = buildDermisInked(params, ink);
    SkinLayer subcut    = buildSubcutaneous(params);

    // Weight each layer's single-scatter albedo by its per-channel optical
    // depth τ = σ_t · thickness. τ / Σ τ approximates the probability that
    // a photon's first scatter event occurs in that layer — the same
    // quantity the volume march integrates path-wise, so the two estimators
    // now agree in steady state. The previous fixed 0.2/0.6/0.2 mix was
    // insensitive to thickness and wavelength, which pushed BSSRDF toward
    // dermis-magenta regardless of how thick the dermis actually was.
    float3 sigmaT_e = epidermis.sigmaA + epidermis.sigmaS * (1.0 - epidermis.g);
    float3 sigmaT_d = dermis.sigmaA    + dermis.sigmaS    * (1.0 - dermis.g);
    float3 sigmaT_s = subcut.sigmaA    + subcut.sigmaS    * (1.0 - subcut.g);

    float3 tau_e = sigmaT_e * epidermis.thickness;
    float3 tau_d = sigmaT_d * dermis.thickness;
    float3 tau_s = sigmaT_s * subcut.thickness;
    float3 totalTau = max(tau_e + tau_d + tau_s, float3(1e-6));

    float3 skinAlbedo =
        (tau_e * layerAlbedo(epidermis)
       + tau_d * layerAlbedo(dermis)
       + tau_s * layerAlbedo(subcut)) / totalTau;

    return skinAlbedo * sampleIrradiance / PI;
}
