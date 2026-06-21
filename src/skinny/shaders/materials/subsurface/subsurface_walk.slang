// Volumetric subsurface random walk (pbrt-subsurface-volumetric).
//
// pbrt `subsurface` → a smooth dielectric boundary + a homogeneous interior
// medium, transported by a delta-tracked (Woodcock) random walk: refract in at the
// boundary, null-collision march the interior with Henyey-Greenstein scattering,
// next-event-estimate direct + environment light at each real scatter, and exit
// through the boundary. The medium is read ONLY through the density seam
// (medium.slang densityAt / mediumMajorant), so a heterogeneous source slots in as
// a new kind. Structure mirrors the proven volumeRenderSkin (volume_render.slang);
// the new part vs skin is the dielectric Fresnel boundary entry/exit.
//
// NOTE on pbrt parity: pbrt's `subsurface` material is a tabulated diffusion
// BSSRDF (dipole), not a 3D random walk. This random walk is the physically-correct
// ground truth the diffusion model approximates (they agree in the diffusive,
// high-albedo limit); it produces the milky look and tracks pbrt closely but is not
// bit-parity with pbrt's dipole. A diffusion-BSSRDF path is a separate option.

import common;                          // Ray, HitInfo, fresnelDielectric? (in flat_shading), PI
import bindings;
import materials.flat.flat_shading;     // fresnelDielectric
import materials.subsurface.medium;     // Medium, densityAt, mediumMajorant, resolveMedium
import volume_render;                   // henyeyGreenstein, sampleHenyeyGreenstein
import environment;                     // sceneEnvRadiance

#if defined(SKINNY_METAL)
// Watchdog-safe caps — the whole walk runs in one megakernel dispatch per pixel
// (see volume_render.slang); an unbounded march overruns the macOS GPU watchdog.
static const int SSS_MAX_STEPS   = 24;
static const int SSS_MAX_BOUNCES = 8;
#else
static const int SSS_MAX_STEPS   = 256;
static const int SSS_MAX_BOUNCES = 64;
#endif

// ── Factored segment traversal (the reusable transport core) ────────
//
// One ray segment [tMin,tMax] through `medium`: null-collision (Woodcock) tracking
// against the seam majorant. Returns the real-scatter event (or unscattered + the
// segment transmittance). Reads the medium ONLY via densityAt / mediumMajorant —
// a future heterogeneous grid reuses this verbatim. `mmPerUnit` bridges σ (mm⁻¹)
// to world-space ray t, exactly as the skin volume does.
struct SegmentSample
{
    float3 transmittance;   // accumulated null-collision transmittance if unscattered
    float3 scatterAlbedo;   // σ_s/σ_t at the scatter point (real event)
    float  t;               // scatter distance (world units); tMax if unscattered
    bool   scattered;
};

SegmentSample traverseMediumSegment(
    Ray ray, float tMin, float tMax, Medium medium, float mmPerUnit, inout RNG rng)
{
    SegmentSample s;
    s.transmittance = float3(1.0);
    s.scatterAlbedo = float3(0.0);
    s.t             = tMax;
    s.scattered     = false;

    float3 pA = ray.origin + ray.direction * tMin;
    float3 pB = ray.origin + ray.direction * tMax;
    float3 majorant = mediumMajorant(medium, pA, pB);              // σ̄_t (mm⁻¹)
    float  majMax   = max(majorant.x, max(majorant.y, majorant.z));
    if (majMax <= 1e-8)
        return s;                                                  // vacuum

    float3 sigmaTbase = medium.sigmaA + medium.sigmaS;             // base σ_t (mm⁻¹)
    float  invMm = 1.0 / max(mmPerUnit, 1e-6);
    float  t = tMin;
    for (int i = 0; i < SSS_MAX_STEPS; i++)
    {
        // Free-flight from the majorant: -ln(u)/σ̄_t in mm → /mmPerUnit to world.
        float dt = (-log(max(rng.next(), 1e-10)) / majMax) * invMm;
        t += dt;
        if (t >= tMax)
            break;

        float3 pos    = ray.origin + ray.direction * t;
        float  dens   = densityAt(medium, pos);                    // seam (1.0 homogeneous)
        float3 sigmaT = sigmaTbase * dens;
        float3 pReal  = sigmaT / max(majorant, float3(1e-6));      // accept prob per channel
        float  pRealMax = max(pReal.x, max(pReal.y, pReal.z));

        if (rng.next() < pRealMax)
        {
            s.t = t;
            s.scattered = true;
            s.scatterAlbedo = medium.sigmaS * dens / max(sigmaT, float3(1e-6));  // σ_s/σ_t
            return s;
        }
        s.transmittance *= (1.0 - pReal);                         // null collision
    }
    return s;
}

// ── Boundary helper ─────────────────────────────────────────────────
// Refract `wo` (pointing away from the surface) into the medium across normal `n`
// with relative IOR eta = 1/ior (entering). Returns false on total internal
// reflection (shouldn't happen entering from vacuum).
bool refractInto(float3 wi, float3 n, float eta, out float3 wt)
{
    float cosI = dot(-wi, n);
    float sin2T = eta * eta * max(0.0, 1.0 - cosI * cosI);
    if (sin2T >= 1.0) { wt = float3(0.0); return false; }
    float cosT = sqrt(1.0 - sin2T);
    wt = eta * wi + (eta * cosI - cosT) * n;
    return true;
}

// ── Subsurface radiance via the interior random walk ─────────────────
//
// At a subsurface hit, refract through the boundary (energy-conserving Fresnel
// split: reflect contributes the env in the mirror direction; refract enters the
// interior), random-walk the slab between the front hit and the back face
// (hit.backT, like the skin volume), NEE direct + env at each real scatter through
// the boundary transmittance, HG-continue, Russian-roulette. Returns the radiance.
float3 subsurfaceRadiance(
    Ray ray, HitInfo hit, Medium medium,
    float3 lightDir, float3 lightRadiance,
    float envIntensity, uint furnaceMode, float mmPerUnit, inout RNG rng)
{
    if (!hit.hit)
        return float3(0.0);

    float3 N    = hit.normal;
    float  ior  = max(medium.eta, 1.0);
    float  NdotV = max(dot(-ray.direction, N), 1e-4);

    // Energy-conserving Fresnel boundary split.
    float Fr = fresnelDielectric(NdotV, 1.0 / ior);               // reflectance entering
    float3 radiance = float3(0.0);

    // Reflected component: env seen in the mirror direction (surface specular).
    {
        float3 wr = reflect(ray.direction, N);
        radiance += Fr * sceneEnvRadiance(wr, envIntensity, furnaceMode);
    }

    // Refracted component: enter the interior with (1 - Fr) throughput.
    float3 wt;
    if (!refractInto(ray.direction, N, 1.0 / ior, wt))
        return radiance;                                          // TIR (degenerate)

    float3 throughput = float3(1.0 - Fr);

    // 1D-slab random walk. The interior is modelled as a homogeneous slab of
    // perpendicular thickness `Tmm` (mm) bounded by the front face (z=0) and the
    // back face (z=Tmm); `zMM` is the perpendicular depth of the current vertex.
    // Closed parity meshes (sssdragon, SSS sphere) set hit.backT; the open-mesh
    // fallback is a several-MFP span. Each step marches to whichever face the ray
    // heads toward, so the walk can actually *exit* (the prior full-slab tMax never
    // took the escape branch ⇒ lost energy). Env is collected ONLY on escape and
    // the analytic direct light ONLY via per-scatter NEE — disjoint sources, so the
    // two never double-count, and σ_a→0 conserves energy (furnace ≈ unity).
    float sigmaTmax = max(medium.sigmaA.x + medium.sigmaS.x,
                      max(medium.sigmaA.y + medium.sigmaS.y,
                          medium.sigmaA.z + medium.sigmaS.z));
    float fallbackSlab = 10.0 / (max(sigmaTmax, 1e-4) * max(mmPerUnit, 1e-6));
    float slabWorld = (hit.backT > hit.t) ? (hit.backT - hit.t) : fallbackSlab;
    float Tmm       = slabWorld * mmPerUnit;                      // slab thickness (mm)
    float zMM       = 0.0;                                        // depth below front face
    Ray   cur; cur.origin = ray.origin + ray.direction * hit.t; cur.direction = wt;
    bool  hasDirect = dot(lightRadiance, lightRadiance) > 1e-8;
    float3 sigmaTmm = medium.sigmaA + medium.sigmaS;             // mm⁻¹ (homogeneous shadow)

    for (int b = 0; b < SSS_MAX_BOUNCES; b++)
    {
        // Perpendicular cosine: +deeper (toward back), −shallower (toward front).
        float mu = dot(cur.direction, -N);
        float exitWorld;                                          // ray distance to a face
        if (mu > 1e-4)        exitWorld = ((Tmm - zMM) / mu)  / max(mmPerUnit, 1e-6);
        else if (mu < -1e-4)  exitWorld = (zMM        / -mu)  / max(mmPerUnit, 1e-6);
        else                  exitWorld = (8.0 * Tmm)         / max(mmPerUnit, 1e-6);

        SegmentSample ss = traverseMediumSegment(cur, 0.0, exitWorld, medium, mmPerUnit, rng);
        throughput *= ss.transmittance;
        if (!ss.scattered)
        {
            // Escaped through a boundary face: refract out and carry the
            // environment in the travel direction. This is the walk's env term.
            float cosExit = max(abs(mu), 1e-3);
            float Ft = 1.0 - fresnelDielectric(cosExit, ior);
            radiance += throughput
                      * sceneEnvRadiance(cur.direction, envIntensity, furnaceMode) * Ft;
            break;
        }

        // Advance the perpendicular depth to the scatter vertex.
        zMM = clamp(zMM + mu * ss.t * mmPerUnit, 0.0, Tmm);
        float3 scatterPos = cur.origin + cur.direction * ss.t;

        // NEE — analytic directional light, refracting OUT through the front face.
        if (hasDirect)
        {
            float cosL = dot(lightDir, N);
            if (cosL > 1e-3)
            {
                float  phase     = henyeyGreenstein(dot(cur.direction, lightDir), medium.g);
                float  shadowDist = zMM / cosL;                  // mm to the front boundary
                float3 shadowTr   = exp(-sigmaTmm * shadowDist);
                // Exit Fresnel transmittance (D1): 1 on TIR ⇒ tap auto-zeros.
                float  Ft = 1.0 - fresnelDielectric(cosL, ior);
                radiance += throughput * ss.scatterAlbedo * phase * lightRadiance * shadowTr * Ft;
            }
        }

        // HG continuation.
        cur.origin    = scatterPos;
        cur.direction = sampleHenyeyGreenstein(cur.direction, medium.g, rng.next2());
        throughput   *= ss.scatterAlbedo;

        // Russian roulette.
        float pCont = max(throughput.x, max(throughput.y, throughput.z));
        if (rng.next() > pCont)
            break;
        throughput /= max(pCont, 1e-6);
    }
    return radiance;
}
