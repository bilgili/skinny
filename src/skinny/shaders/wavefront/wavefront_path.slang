// Staged wavefront path tracer (P1 §5–§6). The megakernel's in-register
// generate→trace→shade→bounce loop, torn apart into compute dispatches that
// round-trip a per-lane path-state record through VRAM:
//
//   wfPathGenerate  — seed the camera ray + path state for every pixel lane
//   wfPathBounce ×N — one bounce: trace, NEE+MIS, BSDF sample, RR, next ray
//                     (dispatched WF_MAX_BOUNCES times by the CPU loop, with a
//                      path-state barrier between)
//   wfPathResolve   — fold each lane's accumulated radiance into the running
//                     mean in the accumulation image (matches main_pass.slang)
//
// The per-bounce body is a verbatim relocation of integrators/path.slang's
// estimateRadiance loop body + runFrame's primary-miss branch — it calls the
// SAME `evaluateBounce` (NEE over all lights + env, BSDF sample, graph/flat/
// skin/python material dispatch) and the SAME env/sphere-light MIS, so the
// accumulated image is an unbiased estimator of the same integral as the
// megakernel (A/B-verified to the renderer's own re-render noise floor).
//
// Set 0 is the megakernel's full scene descriptor set (bindings.slang); set 1
// binding 0 is the path-state buffer this owns. Reusing set 0 avoids
// re-enumerating the ~30-binding scene layout — the kernel imports the same
// declarations, so its SPIR-V reflects a subset of the megakernel's layout.

import common;            // RNG, createRNG, Ray, PI
import bindings;          // fc, accumBuffer, scene buffers, effectiveFurnaceMode, materialTypes
import scene_trace;       // traceScene, HitInfo
import environment;       // sceneEnvRadiance, sceneEnvRadianceCell, envPdf
import cameras.pinhole;   // PinholeCamera
import interfaces;        // BounceResult
import integrators.path;  // evaluateBounce (catch-all shade: skin / python / debug)
import materials.flat.flat_shading;    // isCutoutTransparent
import wavefront.wavefront_state;      // WavefrontPathState, PATH_FLAG_*
import wavefront.wf_display;           // wfWriteDisplay (tonemap → output image)
// Shared path-state bindings (wfState / wfHits / wfTile / wfDecodePixel /
// wfTerminate) + the material-free finisher wfFinishShade.
import wavefront.wf_shade_common;
// Flat-material bounce eval — imports only the flat material + NEE, so the flat
// shade kernel stays small (no skin/python). The catch-all uses evaluateBounce.
import wavefront.flat_bounce;

// Write the lane's accumulated radiance + clear ALIVE; persist the rng cursor.
void wfTerminate(uint i, inout WavefrontPathState s, float3 radiance, RNG rng)
{
    s.radiance = radiance;
    s.flags &= ~PATH_FLAG_ALIVE;
    s.rngState = rng.state;
    wfState[i] = s;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void wfPathGenerate(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;                            // slot in the fixed-size stream
    uint lane = wfTile.streamBase + i;          // global pixel lane this tile
    if (lane >= fc.width * fc.height)
    {
        // Out-of-frame slot (last tile tail): mark dead so resolve skips it.
        wfState[i].pixelIndex = WF_PIXEL_SENTINEL;
        wfState[i].flags = 0u;
        return;
    }
    uint2 pixel = wfDecodePixel(lane);

    RNG rng = createRNG(pixel, fc.frameIndex);
    PinholeCamera cam;
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);

    WavefrontPathState s;
    s.rayOrigin  = ray.origin;
    s.rayDir     = ray.direction;
    // Fold the realistic-camera weight into the initial throughput — radiance
    // is linear in it, so this reproduces main_pass's post-hoc `sample *=
    // lensWeight` (1.0 for pinhole).
    s.throughput = float3(lensWeight);
    s.radiance   = float3(0.0);
    s.pixelIndex = lane;
    s.rngState   = rng.state;
    s.depth      = 0u;
    s.flags      = PATH_FLAG_ALIVE;
    s.bsdfPdf    = -1.0;   // no env-NEE partner for the primary ray
    wfState[i] = s;
}

// Intersect stage — trace the alive lanes' current ray, handle the cutout skip
// + the miss (env), and stash the hit for the shade stage. Material-free: it
// imports only the BVH traversal + env, NOT the material/integrator tree, so it
// stays small. A lane that misses terminates here; a lane that hits stays alive
// and shade processes wfHits[i].
[shader("compute")]
[numthreads(64, 1, 1)]
void wfPathIntersect(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;                            // stream slot
    WavefrontPathState s = wfState[i];
    if (!pathAlive(s))                          // dead / out-of-frame slots skip
        return;

    RNG rng; rng.state = s.rngState;            // not consumed here; preserved
    Ray r; r.origin = s.rayOrigin; r.direction = s.rayDir;

    HitInfo h = traceScene(fc, r);
    for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
    {
        r.origin = h.position + r.direction * 0.002;
        h = traceScene(fc, r);
    }

    if (!h.hit)
    {
        float3 radiance = s.radiance;
        if (s.depth == 0u)
        {
            // runFrame primary-ray miss: env at full weight (no NEE partner).
            radiance += s.throughput * sceneEnvRadiance(
                r.direction, fc.envIntensity, fc.furnaceMode);
        }
        else
        {
            // estimateRadiance loop miss: MIS the env against the BSDF lobe
            // that spawned this ray (when it also ran env NEE).
            float3 envR = sceneEnvRadianceCell(
                r.direction, fc.envIntensity, effectiveFurnaceMode(h.materialId));
            float w = 1.0;
            if (s.bsdfPdf > 0.0)
                w = powerHeuristic(s.bsdfPdf, envPdf(r.direction));
            radiance += s.throughput * envR * w;
        }
        wfTerminate(i, s, radiance, rng);
        return;
    }

    // Hit: the cutout skip may have advanced the origin — persist it so shade
    // spawns continuation rays from the resolved surface. Stash the hit.
    s.rayOrigin = r.origin;
    wfState[i] = s;
    wfHits[i] = h;
}

// Per-material shade — flat. Handles MATERIAL_TYPE_FLAT lanes (flat + MaterialX
// graph materials). Imports only the flat material + NEE (via flat_bounce), so
// this kernel is small — the common-case scenes (no skin/python) compile only
// this, never the heavy catch-all (the MoltenVK Metal-compile fix).
[shader("compute")]
[numthreads(64, 1, 1)]
void wfPathShadeFlat(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    WavefrontPathState s = wfState[i];
    if (!pathAlive(s))
        return;
    HitInfo h = wfHits[i];
    if ((materialTypes[h.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
        return;                                 // non-flat → catch-all kernel

    RNG rng; rng.state = s.rngState;
    Ray r; r.origin = s.rayOrigin; r.direction = s.rayDir;
    BounceResult br = evaluateFlatBounce(h, r, s.depth, rng);
    wfFinishShade(i, s, rng, h, br);
}

// Per-material shade — catch-all for the non-flat material types (skin BSSRDF,
// python materials, debug). Imports the full integrator/material tree via
// evaluateBounce, so it is the heavy kernel — built (and dispatched) only when
// the scene actually contains a non-flat material.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfPathShade(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    WavefrontPathState s = wfState[i];
    if (!pathAlive(s))
        return;
    HitInfo h = wfHits[i];
    if ((materialTypes[h.materialId] & 0xFFu) == MATERIAL_TYPE_FLAT)
        return;                                 // flat → wfPathShadeFlat

    RNG rng; rng.state = s.rngState;
    Ray r; r.origin = s.rayOrigin; r.direction = s.rayDir;
    BounceResult br = evaluateBounce(h, r, s.depth, rng);
    wfFinishShade(i, s, rng, h, br);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void wfPathResolve(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;                            // stream slot
    uint pixelIndex = wfState[i].pixelIndex;
    if (pixelIndex == WF_PIXEL_SENTINEL)        // out-of-frame slot
        return;
    uint2 pixel = wfDecodePixel(pixelIndex);

    float3 sample = wfState[i].radiance;
    // Same sanitisation as main_pass.slang: kill NaN/Inf, clamp negatives.
    bool3 bad = isnan(sample) | isinf(sample);
    sample = select(bad, float3(0.0), sample);
    sample = max(sample, float3(0.0));

    // Progressive accumulation — running mean in linear HDR (main_pass.slang).
    uint n = fc.accumFrame;
    float3 accum;
    if (n == 0u)
        accum = sample;
    else
    {
        float3 prev = accumBuffer[pixel].rgb;
        accum = (prev * float(n) + sample) / float(n + 1u);
    }
    accumBuffer[pixel] = float4(accum, 1.0);
    wfWriteDisplay(pixel, accum);   // tonemap → offscreen/swapchain output
}
