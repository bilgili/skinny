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
import scene_trace;                     // traceScene (3D-walk prototype)

#if defined(SKINNY_WAVEFRONT)
// 3D interior walk — the production wavefront path for large scenes. Each interior
// segment traces the real mesh boundary (`traceScene`), so per-bounce cost carries
// a BVH traversal. The bounce cap was swept (task 3.2) on the furnace sphere and
// the 28.8M-tri sssdragon (Metal wavefront):
//   cap   furnace η=1.0/1.2/1.5 @τ20     dragon mean   dragon wall   watchdog
//   16    (proto)                        0.108         83 s          safe
//   32    0.842 / 0.781 / 0.656          —             —             —
//   48    —                              0.111         86 s          safe
//   96    0.974 / 0.960 / 0.911          0.111         84 s          safe
// A high-albedo random walk needs ~τ² scatter events to diffuse out of a thick
// medium, so energy rises steeply with the cap; the η-spread (boundary internal
// reflection traps η>1 paths longer) collapses 0.186→0.063 from cap 32→96. The
// dragon is cap-INSENSITIVE (its medium is far thinner than the τ20 furnace, most
// pixels escape in <16 bounces) — wall-time and mean are flat 16→96, and 96 never
// trips the watchdog. So 96 is nearly free on real scenes yet near-unity on the
// furnace. (Headroom: 96 stayed watchdog-safe; the energy ceiling at τ200 is still
// ~0.85 — fundamental to a bounded-cap brute-force walk, not closable cheaply.)
static const int SSS_MAX_STEPS   = 256;
static const int SSS_MAX_BOUNCES = 96;
#elif defined(SKINNY_METAL)
// megakernel slab on Metal — the whole walk runs in ONE dispatch per pixel, so an
// unbounded march overruns the macOS GPU watchdog. The slab is cheap (no per-step
// trace); the old `8` dropped ~half the energy of a high-optical-depth medium
// (a high-albedo random walk needs ~τ² scatter events to diffuse out). 16 bounces
// / 24 steps conserves energy on the slab while staying watchdog-safe.
static const int SSS_MAX_STEPS   = 24;
static const int SSS_MAX_BOUNCES = 16;
#else
// megakernel slab on Vulkan — no single-dispatch watchdog headroom issue here.
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

// `maxSteps` is the null-collision budget for THIS segment. It defaults to the
// subsurface tuning (SSS_MAX_STEPS); the free-standing volume walk passes its
// own much larger budget (volume_walk.slang VOLUME_MAX_STEPS) because a sparse
// production cloud needs thousands of majorant steps per crossing (σ̄·d ≫ the
// skin slab) — with the small cap the loop exhausted silently and the segment
// read as transparent (the invisible-disney-cloud bug).
SegmentSample traverseMediumSegment(
    Ray ray, float tMin, float tMax, Medium medium, float mmPerUnit, inout RNG rng,
    int maxSteps = SSS_MAX_STEPS)
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
    for (int i = 0; i < maxSteps; i++)
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
        // Null collision — analog delta tracking: the null branch was SAMPLED
        // with probability (1 − pRealMax), so the per-channel weight is the
        // ratio (1 − pReal)/(1 − pRealMax), which is exactly 1 for grey σ. The
        // old `*= (1 − pReal)` alone double-counted the attenuation (sampled
        // AND multiplied) — invisible for homogeneous media, whose exact
        // majorant makes every tentative collision real (pReal ≡ 1, this
        // branch unreachable), but it squared the transmittance of every
        // dens < 1 grid voxel and rendered the WDAS cloud several× too dark.
        // Same estimator serves shadow rays: E[step] = pRealMax·0 +
        // (1 − pRealMax)·(1 − pReal)/(1 − pRealMax) = (1 − pReal) → e^(−τ).
        s.transmittance *= (float3(1.0) - pReal) / max(1.0 - pRealMax, 1e-6);
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
            // Reached a boundary face. Fresnel-split the energy: with probability
            // Ft transmit (carry the environment out), else TOTAL/partial internal
            // reflection — bounce the ray back into the medium and keep walking.
            // Importance-sampling the split keeps throughput unbiased (no Ft / 1−Ft
            // weights). The old code added only the Ft transmitted part and dropped
            // the (1−Ft) reflected part, which discards most of the interior
            // multiple-scattering at a dielectric boundary (η=1.5 loses ~half) and
            // makes the medium far too dark.
            float cosExit = max(abs(mu), 1e-3);
            float Ft = 1.0 - fresnelDielectric(cosExit, ior);
            if (rng.next() < Ft)
            {
                radiance += throughput
                          * sceneEnvRadiance(cur.direction, envIntensity, furnaceMode);
                break;
            }
            // Internal reflection: flip the boundary-normal component of the
            // direction and resume from the face (zMM is already clamped to 0/Tmm).
            cur.origin    = cur.origin + cur.direction * exitWorld;
            cur.direction = cur.direction - 2.0 * dot(cur.direction, N) * N;
            continue;
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

// True 3D interior walk (wavefront path). Trace each scattered direction to the
// real mesh boundary (`traceScene`) instead of a flat slab, so path length follows
// the geometry — the slab over-estimated path length on curved/complex meshes and
// rendered the sssdragon too dark/red. Distant-light NEE is shadow-traced through
// the real boundary at each scatter; the environment is collected on escape with a
// Snell-refracted exit direction (env and the delta distant light are disjoint
// sources, so they never double-count). Unbiased: single-pdf, bounded throughput
// (no clamp), Russian-roulette-terminated — the path/BDPT/ReSTIR invariants hold.
float3 subsurfaceRadiance3D(
    Ray ray, HitInfo hit, Medium medium, FrameConstants fc,
    float3 lightDir, float3 lightRadiance,
    float envIntensity, uint furnaceMode, float mmPerUnit, inout RNG rng)
{
    if (!hit.hit)
        return float3(0.0);

    float3 N     = hit.normal;
    float  ior   = max(medium.eta, 1.0);
    float  NdotV = max(dot(-ray.direction, N), 1e-4);
    float  Fr    = fresnelDielectric(NdotV, 1.0 / ior);
    float3 radiance = float3(0.0);

    // Surface specular (entry reflection).
    {
        float3 wr = reflect(ray.direction, N);
        radiance += Fr * sceneEnvRadiance(wr, envIntensity, furnaceMode);
    }

    float3 wt;
    if (!refractInto(ray.direction, N, 1.0 / ior, wt))
        return radiance;
    float3 throughput = float3(1.0 - Fr);

    const float kEps = 1e-4;
    Ray   cur;
    cur.origin    = hit.position;
    cur.direction = wt;
    bool  hasDirect = dot(lightRadiance, lightRadiance) > 1e-8;
    float3 sigmaTmm = medium.sigmaA + medium.sigmaS;             // mm⁻¹ (homogeneous shadow)

    for (int b = 0; b < SSS_MAX_BOUNCES; b++)
    {
        Ray seg;
        seg.origin    = cur.origin + cur.direction * kEps;
        seg.direction = cur.direction;
        HitInfo bhit  = traceScene(fc, seg);
        float exitWorld = bhit.hit ? max(bhit.t, kEps) : 1e8;

        SegmentSample ss = traverseMediumSegment(seg, 0.0, exitWorld, medium, mmPerUnit, rng);
        throughput *= ss.transmittance;

        if (!ss.scattered)
        {
            if (!bhit.hit)
            {
                // No boundary ahead (open mesh / degenerate) — leave along the raw
                // interior direction.
                radiance += throughput
                          * sceneEnvRadiance(cur.direction, envIntensity, furnaceMode);
                break;
            }
            // Reached a real boundary face. Fresnel-split: with probability Ft
            // transmit (carry the environment out), else internally reflect and
            // keep walking. Importance-sampling the split keeps throughput unbiased.
            float3 bn      = bhit.normal;
            // Orient the face normal against the incident interior direction so the
            // exit refraction is well-posed (`refractInto` expects n on the inbound
            // side: dot(-cur.direction, nFace) > 0).
            float3 nFace   = (dot(cur.direction, bn) < 0.0) ? bn : -bn;
            float  cosExit = max(abs(dot(cur.direction, nFace)), 1e-3);
            float  Ft      = 1.0 - fresnelDielectric(cosExit, ior);
            if (rng.next() < Ft)
            {
                // Snell-refract the escape direction (medium→vacuum, η = ior) for
                // the env lookup instead of the raw interior direction. Ft > 0 ⇒ not
                // TIR ⇒ refractInto succeeds; guard the degenerate case anyway.
                float3 wExit;
                if (!refractInto(cur.direction, nFace, ior, wExit))
                    wExit = cur.direction;
                radiance += throughput
                          * sceneEnvRadiance(wExit, envIntensity, furnaceMode);
                break;
            }
            // Internal reflection off the real boundary; resume inside.
            cur.origin    = bhit.position;
            cur.direction = cur.direction - 2.0 * dot(cur.direction, bn) * bn;
            continue;
        }

        float3 scatterPos = seg.origin + cur.direction * ss.t;

        // NEE — distant light, shadow-traced through the real boundary. The light is
        // a delta direction (the walk can never randomly hit it), so NEE is the only
        // way to capture it. Trace toward `lightDir` to the first boundary, attenuate
        // by the in-medium transmittance over that distance × exit Fresnel × HG phase.
        if (hasDirect)
        {
            Ray sray;
            sray.origin    = scatterPos + lightDir * kEps;
            sray.direction = lightDir;
            HitInfo shit   = traceScene(fc, sray);
            if (shit.hit)
            {
                float  dMm    = shit.t * mmPerUnit;                 // in-medium dist (mm)
                float3 shTr   = exp(-sigmaTmm * dMm);
                float  cosL   = max(abs(dot(lightDir, shit.normal)), 1e-3);
                float  Ft     = 1.0 - fresnelDielectric(cosL, ior); // exit transmittance
                float  phase  = henyeyGreenstein(dot(cur.direction, lightDir), medium.g);
                radiance += throughput * ss.scatterAlbedo * phase
                          * lightRadiance * shTr * Ft;
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
