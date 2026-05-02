// Statistical sub-millimeter surface detail: skin pores + vellus hair sheen.
//
// Both features are too small to represent as explicit geometry in a Monte-
// Carlo path tracer, but both admit an analytic, per-pixel treatment:
//
// References:
//   [1] Worley, "A Cellular Texture Basis Function", SIGGRAPH 1996.
//   [2] Box, Muller, "A Note on the Generation of Random Normal Deviates",
//       Annals of Mathematical Statistics, 1958.
//   [3] Schlick, "An Inexpensive BRDF Model for Physically-Based Rendering",
//       CGF 1994.
//
//   * Pores perturb the macro-surface normal + roughness around feature
//     points defined by a 3D Worley noise on the shading position.
//   * Vellus hair (fine, translucent body hair) contributes an anisotropic
//     rim sheen — the expected radiance of many sub-pixel cylinders.
//
// Driven by four scalar parameters on SkinParams (poreDensity/Depth,
// hairDensity/Tilt); when density is 0 each function is a no-op.

import common;

// ── Hash helpers ────────────────────────────────────────────────
// Build a vec3 of independent floats in [0,1) from an integer cell id.
// Reuses pcgHash (common.slang) so there's one RNG family in the shader.

float _hash1(int3 cell, uint salt)
{
    uint h = pcgHash(
        uint(cell.x) * 73856093u ^
        uint(cell.y) * 19349663u ^
        uint(cell.z) * 83492791u ^
        salt
    );
    return float(h) / float(0xFFFFFFFFu);
}

float3 _hash3(int3 cell)
{
    return float3(_hash1(cell, 1u), _hash1(cell, 2u), _hash1(cell, 3u));
}

/// Box-Muller transform — two uniform variates → two independent N(0,1) variates [2].
///
///   Z₀ = √(−2 ln u₁) · cos(2π·u₂)
///   Z₁ = √(−2 ln u₁) · sin(2π·u₂)
///
/// where  u₁, u₂ ∈ (0, 1]  (uniform cell hashes; u₁ clamped away from 0)
/// Used to sample per-pore physical attributes (size, depth) from a
/// log-normal distribution: exp(μ + σ·Z) for chosen μ, σ.
float2 _sampleNormal2(int3 cell, uint saltA, uint saltB)
{
    float u1  = max(_hash1(cell, saltA), 1e-6);
    float u2  = _hash1(cell, saltB);
    float r   = sqrt(-2.0 * log(u1));
    float phi = 6.283185307 * u2;
    return float2(r * cos(phi), r * sin(phi));
}

// ── Worley F1 ───────────────────────────────────────────────────

/// 3D Worley (cellular) noise — F1 distance to the nearest jittered cell [1].
///
/// For a query point p, each integer cell c ∈ Z³ has a jittered centre:
///   centre(c) = c + hash3(c)       (per-component uniform in [0, 1))
///
/// F1(p) = min_{c ∈ 3³ neighbourhood} ‖centre(c) − p‖₂
///
/// The 3×3×3 neighbour search (27 cells) guarantees the nearest centre is
/// found for any jitter amplitude ≤ 1. Returns F1 distance plus the vector
/// to the winning centre (used to tilt the surface normal toward pore walls).
struct WorleyResult
{
    float  f1;
    float3 toCentre;
    int3   cell;       // winning cell id — feeds per-pore physical attributes
};

WorleyResult worley3D(float3 p)
{
    int3   base = int3(floor(p));
    float3 frac = p - float3(base);

    WorleyResult r;
    r.f1       = 1e9;
    r.toCentre = float3(0.0);
    r.cell     = int3(0);

    for (int z = -1; z <= 1; z++)
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++)
    {
        int3   cell   = base + int3(x, y, z);
        float3 centre = float3(cell) + _hash3(cell);
        float3 d      = centre - p;
        float  d2     = dot(d, d);
        if (d2 < r.f1)
        {
            r.f1       = d2;
            r.toCentre = d;
            r.cell     = cell;
        }
    }
    r.f1 = sqrt(r.f1);
    return r;
}

// ── Pores ───────────────────────────────────────────────────────
//
// Model a pore as a small depression centred at each Worley cell centre.
// At the bottom of the depression the surface is locally flat (normal
// unchanged); on the wall the normal tilts *away* from the centre; far
// from any pore the surface is again flat. A Gaussian-ish falloff on
// f1 keeps the perturbation localised inside the pore radius.

struct PoreSample
{
    float3 N;
    float  dRoughness;
};

PoreSample applyPores(float3 N, float3 P, float density, float depth)
{
    PoreSample ps;
    ps.N          = N;
    ps.dRoughness = 0.0;
    if (density < 1e-3 || depth < 1e-3)
        return ps;

    // Density controls cells-per-unit; the SDF head is ~2 units tall.
    // Real cheek skin has several-hundred pores per centimetre so the
    // upper end is generous. Lower end keeps sparse-pore presets useful.
    float cellsPerUnit = lerp(40.0, 400.0, density);
    WorleyResult w     = worley3D(P * cellsPerUnit);

    // Per-pore physical variation. Histological data for facial cheek skin
    // shows pore diameters ~50–300 μm following a log-normal distribution
    // (σ_log ≈ 0.45) with larger pores correlating with greater depth
    // (ρ ≈ 0.6). Sample size + depth as a correlated normal pair, then
    // exponentiate into log-normal multipliers on the baseline radius and
    // user-set depth. Clamped to keep tails sane.
    float2 zn       = _sampleNormal2(w.cell, 4u, 5u);
    float  zSize    = zn.x;
    float  zDepth   = 0.6 * zn.x + 0.8 * zn.y;       // corr(size, depth) = 0.6
    float  sizeScale  = clamp(exp(zSize  * 0.45), 0.35, 2.8);
    float  depthScale = clamp(exp(zDepth * 0.50), 0.30, 3.5);

    // Localisation: strong inside ~poreRadius of a cell centre, zero
    // between pores. Baseline 0.35 in cell units, scaled per-pore so
    // small pores stay tight and large pores open up.
    float poreRadius = 0.35 * sizeScale;
    float u          = w.f1 / poreRadius;
    float falloff    = exp(-u * u * 2.5);

    float effectiveDepth = depth * depthScale;

    // toCentre sits in cell-space; project onto the tangent plane to get
    // the in-surface direction from the shading point *toward* the pore
    // centre. For a concave cavity the outward normal at the wall tilts
    // toward the centre (same direction as this tangent), so we *add*
    // rather than subtract — flipping this sign reads as bumps rather
    // than pores. The 2.0 multiplier makes pore depth actually steer the
    // specular highlight: at depth=1 the rim normal bends ~40°, visible
    // under typical skin roughness; at depth=0.2 still a few degrees.
    float3 t    = w.toCentre - N * dot(w.toCentre, N);
    float  tlen = length(t);
    if (tlen > 1e-5)
    {
        float3 tangent = t / tlen;
        float  tilt    = effectiveDepth * falloff * min(tlen, 1.0) * 2.0;
        ps.N = normalize(N + tangent * tilt);
    }

    // Pore cavities scatter light more diffusely — bump roughness inside
    // the localised region. Depth (now per-pore) modulates the bump so
    // shallow pores remain mostly specular while deep ones read as matte
    // cavities. Capped so caller's `roughness+dRoughness ≤ 1` stays safe.
    ps.dRoughness = density * falloff * (0.08 + 0.32 * min(effectiveDepth, 1.0));

    return ps;
}

// ── Vellus hair sheen ───────────────────────────────────────────
//
// Sub-pixel translucent hairs viewed as a statistical ensemble — at
// grazing angles they backscatter incident light toward the viewer,
// giving skin its characteristic pale silhouette rim. We encode this as
// a Schlick-Fresnel-shaped rim term [3] and return it as a *BRDF*
// (no NdotL baked in) so both direct and IBL integrators can consume it
// with the correct cosine weighting.
///
/// Vellus hair sheen BRDF — Schlick-Fresnel rim approximation [3].
///
///   f_hair(v) = 0.3 · density · tiltK · (1 − n·v)⁵ · hairColour
///
/// where  n·v   = dot(shading normal, view dir) ∈ [0, 1]
///        tiltK = lerp(0.35, 1.0, tilt)   (hair population inclination)
///        (1 − n·v)⁵ = Schlick rim factor (grazing-angle back-scatter)
/// Returned as a BRDF; caller multiplies by (n·l) for direct light or
/// by π for IBL cosine-sampled integral.
//
// `tilt` rotates the hair population from lying flat (tilt=0, faint rim)
// to standing up (tilt=1, strong rim). The 0.3 leading constant was set
// so the furnace probe stays clean at density=1 under a unit-white env.
//
// Usage (caller's responsibility):
//   direct : BRDF * max(dot(N, L), 0)    * L_incoming
//   IBL    : PI   * BRDF                  * L_env(cos-hemisphere sample)
//      (cosine-sampling's pdf = NdotL/PI cancels the missing NdotL factor
//       against an implicit * NdotL, leaving the PI multiplier.)

float3 vellusHairBRDF(
    float3 N, float3 V,
    float3 hairColour,
    float  density,
    float  tilt
)
{
    if (density < 1e-3)
        return float3(0.0);

    float NdotV = max(dot(N, V), 0.0);
    float rim   = pow(1.0 - NdotV, 5.0);
    float tiltK = lerp(0.35, 1.0, clamp(tilt, 0.0, 1.0));

    return 0.3 * density * tiltK * rim * hairColour;
}
