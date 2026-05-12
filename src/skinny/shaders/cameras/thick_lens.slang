// PBRT-v3 RealisticCamera port.
//
// Direct line-by-line port of `RealisticCamera::TraceLensesFromFilm`,
// `IntersectSphericalElement`, and `Refract` from
// pbrt-v3/src/cameras/realistic.cpp (Pharr, Jakob, Humphreys, "Physically
// Based Rendering" 3rd ed.). The only deliberate divergences are:
//
//  * No exit-pupil bounds precomputation — we sample the full rear-
//    element disk uniformly and accept the vignetting that PBRT's
//    `BoundExitPupil` would have removed.
//  * Throughput weight reduced to `cos⁴θ` (drop A_rear/Z² absolute
//    scale) so a real-scale scene doesn't crush into the ACES toe.
//
// Conventions (PBRT, mirrored here):
//   * Lens-local frame: film at z = 0, scene direction is -z.
//   * Element thicknesses are stored in scene-side order (index 0 =
//     front element). Each iteration of the trace decrements
//     `elementZ` by `element.thickness` *before* intersecting that
//     surface, so the rear surface is at z = -thickness[N-1] and the
//     front surface at z = -sum(thicknesses).
//   * IOR convention: `element[i].ior` is the medium *between* this
//     surface and the next one toward the film (i.e. the medium just
//     behind this surface in scene→film order).

import common;
import bindings;
import interfaces;

LensSurface unpackLensSurface(uint k)
{
    float4 e = lensElements[k];
    LensSurface s;
    s.radius       = e.x;
    s.thickness    = e.y;   // PBRT thickness, *not* a precomputed z position
    s.ior          = e.z;
    s.halfAperture = e.w;
    return s;
}

/// PBRT's IntersectSphericalElement (realistic.cpp). Solves the
/// quadratic `|o + t·d − c|² = r²` for the ray–sphere intersection,
/// then picks the geometrically correct root: closer when the ray
/// direction and the surface curvature point the same way (a ray
/// going forward into a convex surface enters at the closer hit), or
/// farther otherwise.
bool intersectSphericalElement(
    float3 origin, float3 dir,
    float radius, float zCenter,
    out float t, out float3 n)
{
    t = 0.0; n = float3(0.0, 0.0, 1.0);
    float3 o = origin - float3(0.0, 0.0, zCenter);
    float A = dot(dir, dir);
    float B = 2.0 * dot(dir, o);
    float C = dot(o, o) - radius * radius;
    float disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return false;
    float sq = sqrt(disc);
    float invA = 0.5 / A;
    float t0 = (-B - sq) * invA;
    float t1 = (-B + sq) * invA;
    bool useCloserT = (dir.z > 0.0) ^ (radius < 0.0);
    t = useCloserT ? min(t0, t1) : max(t0, t1);
    if (t < 0.0) return false;
    // Outward normal at the hit, then face-forward against the ray.
    float3 hit = o + t * dir;
    n = normalize(hit);
    if (dot(n, -dir) < 0.0) n = -n;
    return true;
}

/// PBRT's Refract (reflection.cpp). `wi` is the *incident* direction
/// pointing AWAY from the surface (PBRT calls this `-rLens.d` after
/// normalisation). Returns the transmitted direction in `wt`. Returns
/// false on total internal reflection.
bool refractRay(float3 wi, float3 n, float eta, out float3 wt)
{
    wt = float3(0.0);
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = max(0.0, 1.0 - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;
    if (sin2ThetaT >= 1.0) return false;
    float cosThetaT = sqrt(1.0 - sin2ThetaT);
    wt = eta * (-wi) + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

/// `RealisticCamera::TraceLensesFromFilm`. Walks the lens stack from
/// the rear element forward; refracts at every spherical interface;
/// rejects rays that miss the clear aperture or fail Snell's law.
/// Updates `origin`/`dir` in place; returns false when the ray is
/// blocked. Coordinates are in lens-local space (z = 0 is the film
/// plane; scene direction is -z).
bool traceLensesFromFilm(
    inout float3 origin, inout float3 dir, FrameConstants fc)
{
    float elementZ = 0.0;
    uint N = fc.numLensElements;
    for (int i = int(N) - 1; i >= 0; --i)
    {
        LensSurface e = unpackLensSurface(uint(i));
        elementZ -= e.thickness;

        float t;
        float3 n = float3(0.0, 0.0, 1.0);
        bool isStop = (e.radius == 0.0);
        if (isStop)
        {
            // Aperture stop: planar surface, ray must travel toward -z.
            if (dir.z >= 0.0) return false;
            t = (elementZ - origin.z) / dir.z;
        }
        else
        {
            float zCenter = elementZ + e.radius;
            if (!intersectSphericalElement(origin, dir, e.radius, zCenter, t, n))
                return false;
        }

        // Aperture clip (clear-aperture circle around the optical axis).
        float3 pHit = origin + t * dir;
        if (pHit.x * pHit.x + pHit.y * pHit.y >
            e.halfAperture * e.halfAperture)
            return false;
        origin = pHit;

        if (!isStop)
        {
            // PBRT: Refract(Normalize(-rLens.d), n, etaI/etaT, &w)
            float etaI = e.ior;
            float etaT = (i > 0) ? unpackLensSurface(uint(i - 1)).ior : 1.0;
            float3 wi = normalize(-dir);
            float3 wt;
            if (!refractRay(wi, n, etaI / etaT, wt)) return false;
            dir = wt;
        }
    }
    return true;
}

/// PBRT-v3 `RealisticCamera::TraceLensesFromScene`. Walks front (i=0)
/// to rear (i=N-1). Caller passes a ray heading toward the film
/// (lens-local `dir.z < 0`); returns false if the ray is blocked by
/// any element aperture or fails Snell's law. On success
/// `origin/dir` are updated to the rear-element exit point and the
/// refracted direction. Used by BDPT's `s = 1` splat to project a
/// world-space light vertex through the lens onto the correct film
/// pixel.
bool traceLensesFromScene(
    inout float3 origin, inout float3 dir, FrameConstants fc)
{
    float elementZ = -fc.frontZ;
    uint N = fc.numLensElements;
    for (uint i = 0u; i < N; ++i)
    {
        LensSurface e = unpackLensSurface(i);
        float t;
        float3 n = float3(0.0, 0.0, 1.0);
        bool isStop = (e.radius == 0.0);
        if (isStop)
        {
            if (dir.z >= 0.0) return false;       // ray must head toward film
            t = (elementZ - origin.z) / dir.z;
        }
        else
        {
            float zCenter = elementZ + e.radius;
            if (!intersectSphericalElement(origin, dir, e.radius, zCenter, t, n))
                return false;
        }
        float3 pHit = origin + t * dir;
        if (pHit.x * pHit.x + pHit.y * pHit.y >
            e.halfAperture * e.halfAperture)
            return false;
        origin = pHit;
        if (!isStop)
        {
            // IOR convention: surface[i].ior = medium between this
            // surface and the next-toward-film. Going scene→film:
            //   etaI = medium just left (between i-1 and i, or air)
            //   etaT = element[i].ior
            float etaI = (i > 0u) ? unpackLensSurface(i - 1u).ior : 1.0;
            float etaT = e.ior;
            float3 wi = normalize(-dir);
            float3 wt;
            if (!refractRay(wi, n, etaI / etaT, wt)) return false;
            dir = wt;
        }
        elementZ += e.thickness;
    }
    return true;
}

struct ThickLensCamera : ICamera
{
    Ray generateRay(uint2 pixel, inout RNG rng, out float weight)
    {
        Ray ray;
        weight = 0.0;
        ray.origin    = fc.camera.position;
        ray.direction = float3(0.0);

        if (fc.numLensElements == 0u || fc.rearAperture <= 0.0) return ray;

        // 1. Sample film point. PBRT applies a single x-flip to handle the
        //    real-camera optical image inversion (sensor is upside-down +
        //    left-right swapped relative to the scene image; the digital
        //    readout convention undoes one of those flips, leaving x). We
        //    therefore drop the pinhole's NDC y-flip in the lens path.
        float2 jitter = rng.next2() - 0.5;
        // Same zoom remap as the pinhole path so the lens viewport tracks
        // the user-selected zoom rectangle.
        float2 ndc = zoomedNDC(fc, pixel, jitter);
        ndc.y = -ndc.y;   // zoomedNDC pre-flips for the pinhole; undo before the lens code does its own
        float aspect = float(fc.width) / max(float(fc.height), 1.0);
        float filmHalfH = fc.filmHalfH;
        float filmHalfW = filmHalfH * aspect;
        float3 pFilm = float3(-ndc.x * filmHalfW, ndc.y * filmHalfH, 0.0);

        // 2. Sample point on the rear element. To keep image position the
        //    same regardless of aperture (and avoid the off-axis pixels
        //    vignetting to black at small fstops), centre the rear-plane
        //    disk on each pixel's chief ray rather than on the optical
        //    axis. Chief = line from pFilm through (0,0,-irisZ). At
        //    z = -rearZ that line crosses
        //        chief.xy = pFilm.xy · (1 − rearZ / irisZ)
        //    (rearZ and irisZ are positive distances from the film here).
        LensSurface eRear = unpackLensSurface(fc.numLensElements - 1u);
        float lensRearZ = -eRear.thickness;
        // Reconstruct iris-z (positive) by walking the stack rear→front
        // until we hit the aperture stop (radius == 0).
        float irisZpos = eRear.thickness;
        {
            float zwalk = eRear.thickness;
            for (int j = int(fc.numLensElements) - 2; j >= 0; --j)
            {
                LensSurface ej = unpackLensSurface(uint(j));
                zwalk += ej.thickness;
                if (ej.radius == 0.0) { irisZpos = zwalk; break; }
            }
        }
        float kRatio = (irisZpos > 1e-9) ? (eRear.thickness / irisZpos) : 0.0;
        float2 chiefXY = pFilm.xy * (1.0 - kRatio);
        float2 diskUV = sampleUniformDisk(rng.next2());
        float3 pRear = float3(chiefXY.x + diskUV.x * fc.rearAperture,
                              chiefXY.y + diskUV.y * fc.rearAperture,
                              lensRearZ);

        // 3. Trace film→rear ray through the lens stack.
        float3 origin = pFilm;
        float3 dir = pRear - pFilm;     // PBRT does *not* normalise here
        if (!traceLensesFromFilm(origin, dir, fc)) return ray;
        // Guard against NaN / Inf escaping into traceScene — degenerate
        // refractions or near-coincident-surface precision can produce
        // non-finite directions that cause downstream Vulkan state to
        // crash. Treat as a rejected sample.
        if (any(isnan(dir)) || any(isinf(dir))
            || any(isnan(origin)) || any(isinf(origin)))
            return ray;
        float dirLen = length(dir);
        if (dirLen < 1e-6) return ray;
        dir /= dirLen;

        // 4. Throughput weight. Full PBRT eq. 6.10 is
        //       (cos⁴θ' · A_rear) / Z²
        //    but A_rear/Z² collapses to a tiny constant in world units and
        //    visibly crushes the image. Keep just the cos⁴ vignetting
        //    falloff so axial rays land at unit weight.
        float3 filmToRear = pRear - pFilm;
        float Z = max(length(filmToRear), 1e-6);
        float cosVignette = abs(filmToRear.z) / Z;
        float cos4 = cosVignette * cosVignette * cosVignette * cosVignette;
        weight = cos4;

        // 5. Map lens-local → camera-local → world. Skinny's camera looks
        //    down -z (USD convention), and PBRT's lens-local frame also
        //    has the scene at -z, so the direction passes through without
        //    a z-flip. Translate so the front lens vertex lands at the
        //    authored camera position (camera-local origin) — `frontZ`
        //    holds |sum of thicknesses| (always ≥ 0).
        float3 oCam = float3(origin.x, origin.y, origin.z + fc.frontZ);
        float3 dCam = dir;
        ray.origin    = mul(fc.camera.viewInverse, float4(oCam, 1.0)).xyz;
        ray.direction = normalize(mul(fc.camera.viewInverse, float4(dCam, 0.0)).xyz);
        return ray;
    }

    bool sampleWi(float3 p, inout RNG rng,
                  out int2 outPixel, out float cosCam, out float distToCam)
    {
        outPixel = int2(0);
        cosCam = 0.0;
        distToCam = 0.0;
        if (fc.numLensElements == 0u) return false;

        // 1. Sample rear-element disk in lens-local frame.
        float2 diskUV = sampleUniformDisk(rng.next2());
        LensSurface eRear = unpackLensSurface(fc.numLensElements - 1u);
        float lensRearZ = -eRear.thickness;
        float3 pLens = float3(diskUV.x * fc.rearAperture,
                              diskUV.y * fc.rearAperture, lensRearZ);

        // 2. World point → camera-local → lens-local. The lens path
        //    anchors the front-element vertex at the camera's world
        //    position, with lens-local +z pointing back toward film.
        //    Camera-local (-x, y, -z) ↔ lens-local (x, y, z) (the same
        //    180° y-rotation generateRay applies on output, inverted
        //    here for input).
        float3 pCam = mul(fc.camera.view, float4(p, 1.0)).xyz;
        float3 pLensSpace = float3(-pCam.x, pCam.y, -pCam.z + fc.frontZ);

        // 3. Build the connection ray in lens-local. Heading from
        //    scene-side `p` toward sampled rear-disk point.
        float3 origin = pLensSpace;
        float3 dir = pLens - pLensSpace;
        float pLensDist = length(dir);
        if (pLensDist <= 1e-6) return false;
        dir /= pLensDist;
        if (dir.z >= 0.0) return false;        // wrong hemisphere

        // 4. Trace through lens stack scene→film. Reject if blocked.
        if (!traceLensesFromScene(origin, dir, fc)) return false;

        // 5. Continue from the rear-element exit to the film plane (z=0).
        if (dir.z >= -1e-9) return false;
        float tFilm = -origin.z / dir.z;
        if (tFilm < 0.0) return false;
        float3 pFilm = origin + tFilm * dir;

        // 6. Film → NDC → pixel. Mirrors generateRay's mapping.
        float aspect = float(fc.width) / max(float(fc.height), 1.0);
        float filmHalfH = fc.filmHalfH;
        float filmHalfW = filmHalfH * aspect;
        if (filmHalfW <= 0.0 || filmHalfH <= 0.0) return false;
        float2 ndc = float2(-pFilm.x / filmHalfW, pFilm.y / filmHalfH);
        if (any(abs(ndc) > float2(1.0))) return false;
        float xs = (ndc.x * 0.5 + 0.5) * float(fc.width);
        float ys = (ndc.y * 0.5 + 0.5) * float(fc.height);
        if (xs < 0.0 || ys < 0.0 || xs >= float(fc.width) || ys >= float(fc.height))
            return false;
        outPixel = int2(int(xs), int(ys));

        // 7. Geometric terms. distToCam = world-space distance from `p`
        //    to the camera anchor; cosCam = angle between camera forward
        //    (world-space -z in camera-local) and the world ray from
        //    camera to `p`. Approximate with the pinhole values — for a
        //    well-behaved lens these are within a few percent.
        float3 toP_world = p - fc.camera.position;
        distToCam = max(length(toP_world), 1e-6);
        float3 camForward = normalize(
            mul(fc.camera.viewInverse, float4(0, 0, -1, 0)).xyz);
        cosCam = max(dot(camForward, toP_world / distToCam), 0.0);
        return cosCam > 1e-4;
    }
}

// Vignette-debug wrapper — main_pass.slang's lens-vignette overlay path
// keeps calling `generateLensRay(fc, ...)` as a free function so the
// debug branch stays a pure ICamera::generateRay invocation without
// reaching into the integrator. Signature kept for grep stability.
Ray generateLensRay(FrameConstants fcUnused, uint2 pixel, inout RNG rng,
                    out float weight)
{
    ThickLensCamera cam;
    return cam.generateRay(pixel, rng, weight);
}
