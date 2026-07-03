// Free-standing participating-medium walk (nanovdb-volume-rendering).
//
// The volume twin of the subsurface interior walk: a shape bound to a pbrt
// `Material "interface"` (MATERIAL_TYPE_VOLUME) bounds a medium with an
// INDEX-MATCHED boundary — no Fresnel reflection, no refraction bend, no BSDF
// shading, no surface-specular env tap. Rays enter at the hit point unchanged,
// null-collision march the interior through the density seam
// (`traverseMediumSegment`, reused verbatim from subsurface_walk.slang), do
// distant-light NEE with delta-tracked transmittance at each real scatter,
// HG-continue, and on reaching the boundary without a real collision simply
// pass through and hand an ESCAPE ray back to the path loop. The environment
// is NOT collected inside the walk: the escape continuation returns to the
// integrator, whose miss branch takes the env (at full weight — the escape is
// a delta continuation) and whose trace shades the geometry behind the volume
// (the ground disks in disney-cloud / bunny-cloud depend on this).
//
// The homogeneous subsurface entry/exit paths (subsurfaceRadiance /
// subsurfaceRadiance3D) are untouched — byte-identical rendering there is a
// standing gate of this change.

import common;                          // Ray, HitInfo, RNG, powerHeuristic
import bindings;                        // fc, materialTypes, MATERIAL_TYPE_VOLUME
import materials.subsurface.medium;     // Medium, densityAt, mediumMajorant
import materials.subsurface.subsurface_walk;  // traverseMediumSegment, SegmentSample
import volume_render;                   // henyeyGreenstein, sampleHenyeyGreenstein
import scene_trace;                     // traceScene
import environment;                     // sampleEnvDir, envPdf (env NEE + MIS)

// Scatter cap per walk. Truncation is handled by Russian roulette on the
// throughput after a few scatters (unbiased); the cap is the watchdog-safety
// backstop (Metal dispatch hygiene, design D5.1). NOTE the walk runs inside ONE
// dispatch in BOTH execution modes (the wavefront catch-all shade kernel runs
// the whole walk per thread too), so Metal gets the tight caps in both.
// pbrt's disney-cloud reference uses volpath maxdepth 100 — 128 with RR is at
// parity; Metal's 32 is the watchdog compromise (re-tuned in 6.5, recorded in
// the gate baseline).
#if defined(SKINNY_METAL)
static const int VOLUME_MAX_SCATTERS = 64;   // Metal watchdog cap (both modes)
#else
static const int VOLUME_MAX_SCATTERS = 128;
#endif

// Null-collision step budget per SEGMENT (passed to traverseMediumSegment,
// overriding its subsurface-tuned SSS_MAX_STEPS default). A sparse production
// cloud needs σ̄·d·mmPerUnit majorant steps to cross — disney-cloud is ~2000
// worst case across the grid diagonal — and an exhausted budget silently reads
// as a transparent segment (the invisible-cloud bug). The uvw-slab clip below
// keeps the budget spent inside the grid only.
#if defined(SKINNY_METAL)
static const int VOLUME_MAX_STEPS = 4096;    // watchdog headroom (soaked in 6.5)
#else
static const int VOLUME_MAX_STEPS = 8192;
#endif

// Start Russian roulette after this many scatters (before it, deep forced
// truncation would bias dense clouds; after it, RR keeps truncation unbiased).
static const int VOLUME_RR_START = 3;

// Matches the loop's continuation offset (path.slang spawns the next ray at
// h.position + h.normal * 0.002 — the walk pre-compensates non-pass-through
// boundary exits by this much so interior geometry is re-hit, not tunnelled).
static const float VOLUME_SPAWN_EPS = 0.002;

// Clip a world-space segment [0, tMax] to the medium's [0,1]³ support box in
// uvw / medium-local space (an affine slab test through the SAME worldToUvw
// rows the density lookup uses). BOTH box-supported kinds share this: the
// nanovdb grid (uvw texture coords) and the procedural cloud (medium-local
// coords, pbrt's default unit-cube bounds) — densityAt returns exactly 0
// outside the box for both, so marching there only burns null-collision budget
// (on clouds.pbrt the radius-1 bounding sphere is ~76% empty margin, and an
// escape/shadow ray toward the open sky runs to tMax≈1e8 — the whole budget
// spent in vacuum, which silently reads as a transparent segment). Returns
// false when the segment misses the box entirely (pure pass-through).
// Homogeneous free-standing media have no support box: the whole segment stands.
bool clipSegmentToGrid(Medium m, Ray seg, float tMax, out float t0, out float t1)
{
    t0 = 0.0;
    t1 = tMax;
    if (m.kind != MEDIUM_NANOVDB && m.kind != MEDIUM_CLOUD)
        return true;
    float3 o = float3(dot(m.worldToUvw0.xyz, seg.origin) + m.worldToUvw0.w,
                      dot(m.worldToUvw1.xyz, seg.origin) + m.worldToUvw1.w,
                      dot(m.worldToUvw2.xyz, seg.origin) + m.worldToUvw2.w);
    float3 d = float3(dot(m.worldToUvw0.xyz, seg.direction),
                      dot(m.worldToUvw1.xyz, seg.direction),
                      dot(m.worldToUvw2.xyz, seg.direction));
    [ForceUnroll]
    for (int a = 0; a < 3; a++)
    {
        if (abs(d[a]) < 1e-12)
        {
            if (o[a] < 0.0 || o[a] > 1.0)
                return false;                    // parallel + outside the slab
        }
        else
        {
            float ta = (0.0 - o[a]) / d[a];
            float tb = (1.0 - o[a]) / d[a];
            t0 = max(t0, min(ta, tb));
            t1 = min(t1, max(ta, tb));
        }
    }
    return t1 > t0;
}

struct VolumeWalkResult
{
    float3 radiance;    // in-walk NEE (distant + env) — caller applies its throughput
    float3 throughput;  // path throughput multiplier along the escape ray
    Ray    escapeRay;   // continuation from the boundary exit point (see below)
    bool   escaped;     // false ⇒ absorbed / RR-terminated / cap-exhausted
    // HG pdf of the last real-scatter continuation direction, 0 when the walk
    // never scattered. The caller forwards it as the escape bsdfSample.pdf so
    // the path loop's env-miss MIS (powerHeuristic(bsdfPdf, envPdf)) exactly
    // complements the walk's per-scatter env NEE (same heuristic pair). A
    // zero-scatter pass-through stays a delta continuation (pdf 0 ⇒ env at
    // full weight — correct, since no env NEE ran for it).
    float  lastPhasePdf;
};

// Delta-tracked shadow transmittance toward `dir` from `pos` (grid-clipped),
// shared by the distant-light and env NEE estimators. Returns per-channel
// transmittance; zero when occluded by non-volume geometry or a real
// collision on the shadow segment (analog: E → e^(−τ)).
float3 volumeShadowTransmittance(
    float3 pos, float3 dir, Medium medium, FrameConstants fc, float mmPerUnit,
    inout RNG rng)
{
    const float kEps = 1e-4;
    Ray sray;
    sray.origin    = pos + dir * kEps;
    sray.direction = dir;
    HitInfo shit = traceScene(fc, sray);
    if (shit.hit && (materialTypes[shit.materialId] & 0xFFu) != MATERIAL_TYPE_VOLUME)
        return float3(0.0);                       // opaque occluder
    float shadowWorld = shit.hit ? max(shit.t, kEps) : 1e8;
    float sA, sB;
    if (!clipSegmentToGrid(medium, sray, shadowWorld, sA, sB))
        return float3(1.0);                       // misses the grid entirely
    SegmentSample sh = traverseMediumSegment(sray, sA, sB, medium, mmPerUnit,
                                             rng, VOLUME_MAX_STEPS);
    return sh.scattered ? float3(0.0) : sh.transmittance;
}

// Walk the medium starting at `hit` (the interface-boundary entry hit of `ray`).
// `lightDir`/`lightRadiance` are the analytic distant light (or zero), exactly
// like the subsurface case wiring. `escapeRay.origin` is chosen so that the
// path loop's `+ direction * VOLUME_SPAWN_EPS` spawn (via the rewritten
// h.position/h.normal) lands just PAST the volume's own interface shell —
// pass-through — but just BEFORE any non-volume surface met inside the medium,
// so the loop's next trace re-hits and shades it instead of tunnelling.
VolumeWalkResult volumeWalkRadiance(
    Ray ray, HitInfo hit, Medium medium, FrameConstants fc,
    float3 lightDir, float3 lightRadiance, float mmPerUnit, inout RNG rng)
{
    VolumeWalkResult res;
    res.radiance   = float3(0.0);
    res.throughput = float3(1.0);
    res.escapeRay.origin    = hit.position;
    res.escapeRay.direction = ray.direction;
    res.escaped      = false;
    res.lastPhasePdf = 0.0;
    if (!hit.hit)
        return res;

    const float kEps = 1e-4;
    Ray cur;
    cur.origin    = hit.position;      // index-matched: enter at the hit, unbent
    cur.direction = ray.direction;
    bool hasDirect = dot(lightRadiance, lightRadiance) > 1e-8;

    for (int b = 0; b < VOLUME_MAX_SCATTERS; b++)
    {
        // March to the far boundary (the interface shell — or any geometry
        // inside the medium; classified below on escape).
        Ray seg;
        seg.origin    = cur.origin + cur.direction * kEps;
        seg.direction = cur.direction;
        HitInfo bhit  = traceScene(fc, seg);
        float exitWorld = bhit.hit ? max(bhit.t, kEps) : 1e8;

        // Spend the step budget only inside the grid's support (vacuum outside).
        float tA, tB;
        SegmentSample ss;
        if (clipSegmentToGrid(medium, seg, exitWorld, tA, tB))
        {
            ss = traverseMediumSegment(seg, tA, tB, medium, mmPerUnit, rng,
                                       VOLUME_MAX_STEPS);
            if (!ss.scattered)
                ss.t = exitWorld;                // vacuum tail past the grid
        }
        else
        {
            ss.transmittance = float3(1.0);      // segment misses the grid
            ss.scatterAlbedo = float3(0.0);
            ss.t             = exitWorld;
            ss.scattered     = false;
        }
        res.throughput *= ss.transmittance;

        if (!ss.scattered)
        {
            // Reached the segment end without a real collision: pass through.
            res.escaped = true;
            res.escapeRay.direction = cur.direction;
            if (!bhit.hit)
            {
                // Open boundary (shouldn't happen inside a closed shell): the
                // loop's env-miss branch takes it from here.
                res.escapeRay.origin = seg.origin;
                return res;
            }
            bool shellHit =
                (materialTypes[bhit.materialId] & 0xFFu) == MATERIAL_TYPE_VOLUME;
            // Shell ⇒ exit just past it (the loop offsets +VOLUME_SPAWN_EPS);
            // interior opaque geometry ⇒ back off so the loop re-hits + shades
            // it (no tunnelling through objects embedded in the medium).
            res.escapeRay.origin = shellHit
                ? bhit.position
                : bhit.position - cur.direction * (2.0 * VOLUME_SPAWN_EPS);
            return res;
        }

        float3 scatterPos = seg.origin + cur.direction * ss.t;

        // NEE at the real scatter — delta-tracked shadow transmittance through
        // the SAME density seam (unbiased for the grid, where exp(−σ̄d) would
        // be wrong). No exit Fresnel — the boundary is index-matched.
        //
        //  * Distant light: delta direction, the phase walk can never hit it
        //    by chance ⇒ NEE is its only estimator, no MIS partner.
        //  * Environment: importance-sampled from the env CDF and
        //    power-heuristic-weighted against the HG phase pdf; the phase-
        //    sampled complement is the escape continuation, whose env-miss the
        //    path loop MIS-weights with the SAME heuristic via lastPhasePdf.
        //    (Emissive triangles are NOT sampled in-walk — volume scenes with
        //    area lights are a recorded follow-up, like the subsurface walk.)
        if (hasDirect)
        {
            float3 T = volumeShadowTransmittance(
                scatterPos, lightDir, medium, fc, mmPerUnit, rng);
            if (any(T > 0.0))
            {
                float phase = henyeyGreenstein(
                    dot(cur.direction, lightDir), medium.g);
                res.radiance += res.throughput * ss.scatterAlbedo * phase
                              * lightRadiance * T;
            }
        }
        {
            EnvSample es = sampleEnvDir(rng.next2(), fc.envIntensity);
            if (es.pdf > 0.0 && any(es.radiance > 0.0))
            {
                float3 T = volumeShadowTransmittance(
                    scatterPos, es.dir, medium, fc, mmPerUnit, rng);
                if (any(T > 0.0))
                {
                    float phase = henyeyGreenstein(
                        dot(cur.direction, es.dir), medium.g);
                    float w = powerHeuristic(es.pdf, phase);
                    res.radiance += res.throughput * ss.scatterAlbedo * phase
                                  * es.radiance * T * (w / es.pdf);
                }
            }
        }

        // HG continuation (pdf recorded for the escape env MIS complement).
        float3 nextDir = sampleHenyeyGreenstein(cur.direction, medium.g, rng.next2());
        res.lastPhasePdf = henyeyGreenstein(dot(cur.direction, nextDir), medium.g);
        cur.origin    = scatterPos;
        cur.direction = nextDir;
        res.throughput *= ss.scatterAlbedo;

        // Russian roulette after VOLUME_RR_START scatters — pbrt volpath
        // semantics: survival = the throughput itself, NO forced death while
        // throughput ≈ 1. A conservative 0.95 ceiling here turned the WDAS
        // cloud black: σ_a = 0 ⇒ albedo ≈ 1 ⇒ the light needs hundreds of
        // scatters to diffuse out, and a forced 5%-per-scatter death converts
        // that energy into rare huge-weight fireflies (statistically unbiased,
        // visually black at gate spp). With absorption (bunny σ_a/σ_s = 0.05)
        // throughput decays and RR engages naturally; the hard scatter cap
        // stays as the watchdog backstop (biased, recorded in the baseline).
        if (b >= VOLUME_RR_START)
        {
            float pCont = clamp(
                max(res.throughput.x, max(res.throughput.y, res.throughput.z)),
                0.05, 1.0);
            if (pCont < 1.0)
            {
                if (rng.next() > pCont)
                {
                    res.throughput = float3(0.0);
                    return res;                  // absorbed (escaped = false)
                }
                res.throughput /= pCont;
            }
        }
    }
    // Cap exhausted without escaping (biased truncation — Metal megakernel
    // watchdog trade-off, same as the SSS caps; re-tuned in 6.5).
    return res;
}
