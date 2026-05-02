// Volume ray marching for translucent skin layers.
//
// Performs delta-tracking (Woodcock tracking) through a heterogeneous
// participating medium defined by the skin layer stack.
//
// References:
//   [1] Henyey, Greenstein,
//       "Diffuse Radiation in the Galaxy", ApJ 1941.
//   [2] Woodcock et al.,
//       "Techniques Used in the GEM Code for Monte Carlo Neutronics
//       Calculations", Applications of Computing Methods to Reactor Problems,
//       Argonne National Laboratory 1965.
//   [3] Veach,
//       "Robust Monte Carlo Methods for Light Transport Simulation",
//       PhD thesis, Stanford University 1997.

import common;
import skin_bssrdf;
import environment;
static const int   MAX_VOLUME_STEPS = 128;
static const int   VOLUME_DEFAULT_BOUNCES = 8;

// ── Phase functions ─────────────────────────────────────────────

/// Henyey-Greenstein phase function [1].
///
///   p(cosθ, g) = (1 − g²) / (4π · (1 + g² − 2g·cosθ)^(3/2))
///
/// where  cosθ = dot(ω_in, ω_out)
///        g    = anisotropy ∈ (−1, 1);  g > 0 → forward, g < 0 → back,
///               g = 0 → isotropic (reduces to 1/(4π))
float henyeyGreenstein(float cosTheta, float g)
{
    float g2  = g * g;
    float num = 1.0 - g2;
    float den = pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
    return num / (4.0 * PI * den);
}

/// Importance-sample the HG phase function by CDF inversion [1].
///
/// Analytic inverse CDF for cosθ:
///   cosθ = [1 + g² − ((1−g²) / (1 − g + 2g·u))²] / (2g)   (g ≠ 0)
///   cosθ = 1 − 2·u                                          (|g| < 10⁻³)
///
/// The returned direction has pdf = p(cosθ, g), so the sampling weight
/// phase/pdf ≡ 1 — each call is a zero-variance one-sample estimator
/// of the phase-function integral over the unit sphere.
float3 sampleHenyeyGreenstein(float3 forward, float g, float2 u)
{
    float cosTheta;
    if (abs(g) > 1e-3)
    {
        float t = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
        cosTheta = (1.0 + g * g - t * t) / (2.0 * g);
    }
    else
    {
        cosTheta = 1.0 - 2.0 * u.x;
    }
    cosTheta = clamp(cosTheta, -1.0, 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi      = 2.0 * PI * u.y;

    float3 tangent   = abs(forward.x) > 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 bitangent = normalize(cross(forward, tangent));
    tangent          = cross(bitangent, forward);

    return normalize(
        tangent   * cos(phi) * sinTheta +
        bitangent * sin(phi) * sinTheta +
        forward * cosTheta
    );
}

// ── Medium sampling ─────────────────────────────────────────────

struct MediumSample
{
    float3 transmittance;
    float3 scattering;
    float  t;
    bool   scattered;
};

// Determine which skin layer a point at depth d (mm from surface) falls in.
// `ink` is only meaningful for dermis-depth samples (where tattoo pigment
// actually lives); for epidermis / subcut we return the unaltered layers.
SkinLayer layerAtDepth(float depthMM, SkinParams params, float4 ink)
{
    if (depthMM < params.epidermisThickness)
        return buildEpidermis(params);
    if (depthMM < params.epidermisThickness + params.dermisThickness)
        return buildDermisInked(params, ink);
    return buildSubcutaneous(params);
}

/// Delta-tracking (Woodcock / null-collision method) through layered skin [2].
///
/// Free-flight distance from a majorant extinction σ̄_t (mm⁻¹):
///   dt = −ln(u) / σ̄_t         (exponential free-flight)
///
/// At each candidate event, accept as a real interaction with probability:
///   p_real = σ_t(x) / σ̄_t
///
/// Otherwise it is a null collision — the local weight is unchanged and the
/// walk continues. The estimator is unbiased for any majorant ≥ σ_t(x).
///
/// Unit bridge: σ in mm⁻¹, ray t in world units; mmPerUnit converts between
/// them. Depth from surface = (t − tMin) · mmPerUnit.
MediumSample sampleMedium(
    Ray         ray,
    float       tMin,
    float       tMax,
    SkinParams  params,
    float4      ink,
    float       mmPerUnit,
    inout RNG   rng
)
{
    // Use majorant extinction = max over all layers. Tattoo ink can add up
    // to ~22 mm⁻¹ of absorption in the dermis, so bump the upper bound so
    // delta-tracking stays unbiased under the densest presets.
    float3 majorant = params.scatteringCoeff + float3(60.0); // upper bound σ_t (mm⁻¹)

    MediumSample result;
    result.transmittance = float3(1.0);
    result.scattering    = float3(0.0);
    result.t             = tMax;
    result.scattered     = false;

    float invMmPerUnit = 1.0 / max(mmPerUnit, 1e-6);
    float t = tMin;
    for (int i = 0; i < MAX_VOLUME_STEPS; i++)
    {
        // Sample free-flight distance. majorant is in mm⁻¹ so -log(u)/majorant
        // is in mm; divide by mmPerUnit to advance t in world units.
        float majorantMax = max(majorant.x, max(majorant.y, majorant.z));
        float dt = (-log(max(rng.next(), 1e-10)) / majorantMax) * invMmPerUnit;
        t += dt;

        if (t >= tMax)
            break;

        float3 pos = ray.origin + ray.direction * t;

        // Physical depth from entry: traversed world distance × mmPerUnit.
        float depthMM = (t - tMin) * mmPerUnit;

        SkinLayer layer = layerAtDepth(depthMM, params, ink);
        float3 sigmaT  = layer.sigmaA + layer.sigmaS;
        float3 sigmaS  = layer.sigmaS;

        // Null-collision probability
        float3 pReal = sigmaT / max(majorant, float3(1e-6));
        float  pRealMax = max(pReal.x, max(pReal.y, pReal.z));

        if (rng.next() < pRealMax)
        {
            // Real interaction
            result.t = t;
            result.scattered = true;
            result.scattering = sigmaS / max(sigmaT, float3(1e-6));
            break;
        }

        // Null collision — continue tracking
        result.transmittance *= 1.0 - pReal;
    }

    return result;
}

// ── Volume-integrated radiance along a ray through skin ─────────
//
// The march is a delta-tracking random walk through the layered skin slab.
// At every real scatter event we accumulate two next-event-estimation terms:
//
//   (a) Analytic directional light — the traditional "shadow ray" through the
//       rest of the slab toward `lightDir`. Skipped when lightRadiance ≈ 0
//       (pure-IBL renders, or furnace mode where analytic light is disabled).
//
//   (b) Environment light — HG-importance-sampled direction back to the
//       surface, weighted by the slab shadow transmittance and the env
//       radiance at the sampled direction. In furnace mode env radiance is
//       forced to 1 per channel by sceneEnvRadiance, which makes the volume
//       participate in the energy-conservation probe.
//
// HG importance sampling means phase(ω)/pdf(ω) ≡ 1, so each NEE tap is an
// unbiased one-sample estimator of the in-scattered integral.
float3 volumeRenderSkin(
    Ray         ray,
    HitInfo     hit,
    float3      lightDir,
    float3      lightRadiance,
    SkinParams  params,
    float4      ink,
    float       envIntensity,
    uint        furnaceMode,
    uint        integratorMode,
    float       mmPerUnit,
    int         maxBounces,
    inout RNG   rng
)
{
    if (!hit.hit)
        return float3(0.0);

    // Slab thickness in mm (params) vs world units (ray march).
    float totalThicknessMM    = params.epidermisThickness
                              + params.dermisThickness
                              + params.subcutThickness;
    float totalThicknessWorld = totalThicknessMM / max(mmPerUnit, 1e-6);

    // Clamp slab to actual mesh thickness when back-face is known.
    if (hit.backT > 0.0)
    {
        float meshThicknessWorld = hit.backT - hit.t;
        totalThicknessWorld = min(totalThicknessWorld, meshThicknessWorld);
        totalThicknessMM    = totalThicknessWorld * mmPerUnit;
    }

    float tMin = hit.t;
    float tMax = hit.t + totalThicknessWorld;

    float3 radiance   = float3(0.0);
    float3 throughput = float3(1.0);

    Ray    currentRay = ray;
    float3 slabOut    = hit.normal;                 // outward surface normal
    bool   hasDirect  = dot(lightRadiance, lightRadiance) > 1e-8;

    int bounces = clamp(maxBounces, 1, 32);
    for (int bounce = 0; bounce < bounces; bounce++)
    {
        MediumSample ms = sampleMedium(currentRay, tMin, tMax, params, ink, mmPerUnit, rng);
        throughput *= ms.transmittance;
        if (!ms.scattered)
            break;

        float3 scatterPos = currentRay.origin + currentRay.direction * ms.t;
        float  depthMM    = (ms.t - tMin) * mmPerUnit;
        SkinLayer layer   = layerAtDepth(depthMM, params, ink);
        float3 sigmaT     = layer.sigmaA + layer.sigmaS;

        // NEE (a) — analytic directional light. Phase argument is the cosine
        // of the angle between photon travel directions (incoming: -lightDir,
        // outgoing: -currentRay.direction) — equivalent to
        //   dot(currentRay.direction, lightDir).
        if (hasDirect)
        {
            float cosScatter = dot(currentRay.direction, lightDir);
            float phase      = henyeyGreenstein(cosScatter, layer.g);
            float shadowDist = totalThicknessMM - depthMM;   // mm, matches sigmaT units
            float3 shadowTr  = exp(-sigmaT * shadowDist);
            radiance += throughput * ms.scattering * phase * lightRadiance * shadowTr;
        }

        // NEE (b) — environment. Importance-sample HG around the photon's
        // current travel direction so pdf ≡ phase and the weight simplifies
        // to env × shadow transmittance. Slab shadow path is depthMM scaled
        // by the cosine of the angle the sampled ray makes with the surface
        // normal; samples pointing into the skin (cos ≤ 0) contribute zero.
        {
            float3 wEnv    = sampleHenyeyGreenstein(currentRay.direction, layer.g, rng.next2());
            float  cosN    = dot(wEnv, slabOut);
            if (cosN > 0.001)
            {
                float  exitDist = depthMM / cosN;
                float3 shadowTr = exp(-sigmaT * exitDist);
                float3 envRad   = sceneEnvRadiance(wEnv, envIntensity, furnaceMode);
                float  phasePdf = henyeyGreenstein(dot(wEnv, currentRay.direction), layer.g);
                float  wMis     = integratorMode != 0u
                    ? powerHeuristic(phasePdf, uniformSpherePdf())
                    : 1.0;
                radiance += throughput * ms.scattering * envRad * shadowTr * wMis;
            }

            if (integratorMode != 0u)
            {
                float3 wLight = sampleUniformSphere(rng.next2());
                float  cosL   = dot(wLight, slabOut);
                if (cosL > 0.001)
                {
                    float  exitDistL = depthMM / cosL;
                    float3 shadowTrL = exp(-sigmaT * exitDistL);
                    float3 envRadL   = sceneEnvRadiance(wLight, envIntensity, furnaceMode);
                    float  phasePdfL = henyeyGreenstein(dot(wLight, currentRay.direction), layer.g);
                    float  lightPdf  = uniformSpherePdf();
                    float  wLightMis = powerHeuristic(lightPdf, phasePdfL);
                    radiance += throughput * ms.scattering
                              * phasePdfL * envRadL * shadowTrL * (wLightMis / lightPdf);
                }
            }
        }

        // Continue the random walk. The previous cosine-hemisphere fallback
        // ignored the phase function; HG-sample around the current travel
        // direction instead so forward-scattering media behave correctly.
        float3 newDir = sampleHenyeyGreenstein(currentRay.direction, layer.g, rng.next2());
        currentRay.origin    = scatterPos;
        currentRay.direction = newDir;
        throughput *= ms.scattering;

        // Russian roulette
        float pContinue = max(throughput.x, max(throughput.y, throughput.z));
        if (rng.next() > pContinue)
            break;
        throughput /= pContinue;

        tMin = 0.0;
        tMax = (totalThicknessMM - depthMM) / max(mmPerUnit, 1e-6);  // remaining slab in world units
    }

    return radiance;
}
