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
import bindings;          // fc, accumBuffer, scene buffers, effectiveFurnaceMode
import scene_trace;       // traceScene, HitInfo
import environment;       // sceneEnvRadiance, sceneEnvRadianceCell, envPdf
import cameras.pinhole;   // PinholeCamera
import interfaces;        // BounceResult
import integrators.path;  // evaluateBounce (+ allLightsNEE, material dispatch)
import scene_lights;      // NearestSphereHit, intersectSphereLights
import lights.sphere_light;            // SphereLight
import materials.flat.flat_shading;    // isCutoutTransparent
import wavefront.wavefront_state;      // WavefrontPathState, PATH_FLAG_*
import wavefront.wf_display;           // wfWriteDisplay (tonemap → output image)

// Match integrators/path.slang::MAX_BOUNCES (the CPU loop dispatches this many
// wfPathBounce passes; keep the two in lockstep).
static const uint WF_MAX_BOUNCES = 6u;

// Tiled streaming (§P1-E): the path-state buffer holds a fixed-size stream, not
// one slot per pixel. The CPU loops `streamBase` over the frame in stream-size
// tiles; lane `i` (the slot) maps to pixel `streamBase + i`. Slots whose global
// lane falls past the frame are marked dead with a sentinel pixelIndex so the
// bounce/resolve passes skip them without needing the base. Only generate reads
// the base (the per-tile push constant); bounce/resolve work by slot + the
// stored pixelIndex.
static const uint WF_PIXEL_SENTINEL = 0xFFFFFFFFu;

struct WfTilePC { uint streamBase; };
[[vk::push_constant]] ConstantBuffer<WfTilePC> wfTile;

[[vk::binding(0, 1)]] RWStructuredBuffer<WavefrontPathState> wfState;

uint2 wfDecodePixel(uint lane) { return uint2(lane % fc.width, lane / fc.width); }

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

[shader("compute")]
[numthreads(64, 1, 1)]
void wfPathBounce(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;                            // stream slot
    WavefrontPathState s = wfState[i];
    if (!pathAlive(s))                          // dead / out-of-frame slots skip
        return;

    RNG rng; rng.state = s.rngState;
    Ray r; r.origin = s.rayOrigin; r.direction = s.rayDir;
    float3 throughput = s.throughput;
    float3 radiance   = s.radiance;
    float  misBsdfPdf = s.bsdfPdf;
    uint   depth      = s.depth;

    HitInfo h = traceScene(fc, r);
    for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
    {
        r.origin = h.position + r.direction * 0.002;
        h = traceScene(fc, r);
    }

    if (!h.hit)
    {
        if (depth == 0u)
        {
            // runFrame primary-ray miss: env at full weight (no NEE partner).
            radiance += throughput * sceneEnvRadiance(
                r.direction, fc.envIntensity, fc.furnaceMode);
        }
        else
        {
            // estimateRadiance loop miss: MIS the env against the BSDF lobe
            // that spawned this ray (when it also ran env NEE).
            float3 envR = sceneEnvRadianceCell(
                r.direction, fc.envIntensity, effectiveFurnaceMode(h.materialId));
            float w = 1.0;
            if (misBsdfPdf > 0.0)
                w = powerHeuristic(misBsdfPdf, envPdf(r.direction));
            radiance += throughput * envR * w;
        }
        wfTerminate(i, s, radiance, rng);
        return;
    }

    BounceResult br = evaluateBounce(h, r, depth, rng);
    radiance += throughput * br.fullRadiance;
    if (depth == 0u || fc.numEmissiveTriangles == 0u)
        radiance += throughput * br.bsdfSample.emission;
    radiance += throughput * br.directLight;

    if (!br.bsdfSample.valid)
    {
        wfTerminate(i, s, radiance, rng);
        return;
    }
    throughput *= br.bsdfSample.weight;

    if (depth > 0u)
    {
        float p = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05, 0.95);
        if (rng.next() > p)
        {
            wfTerminate(i, s, radiance, rng);
            return;
        }
        throughput /= p;
    }

    bool envNeeActive = (effectiveFurnaceMode(h.materialId) == 0u);
    misBsdfPdf = (envNeeActive && !br.bsdfSample.transmitted && br.bsdfSample.pdf > 0.0)
               ? br.bsdfSample.pdf : -1.0;

    float3 spawnOffset = br.bsdfSample.transmitted ? -h.normal * 0.002 : h.normal * 0.002;
    r.origin    = h.position + spawnOffset;
    r.direction = br.bsdfSample.wi;

    // BSDF-sampled sphere-light hit: MIS against the NEE estimate.
    NearestSphereHit sph = intersectSphereLights(r, 1e30, fc.numSphereLights);
    HitInfo hNext = traceScene(fc, r);
    float geom_t = hNext.hit ? hNext.t : 1e30;
    if (sph.idx != 0xFFFFFFFFu && sph.t < geom_t)
    {
        SphereLight light = sphereLights[sph.idx];
        float w_bsdf = 1.0;
        if (!br.bsdfSample.transmitted && br.bsdfSample.pdf > 0.0)
        {
            float area = 4.0 * PI * light.radius * light.radius;
            float pdf_light_sa = (sph.t * sph.t) / max(sph.cos_y * area, 1e-12);
            w_bsdf = powerHeuristic(br.bsdfSample.pdf, pdf_light_sa);
        }
        radiance += throughput * light.radiance * w_bsdf;
        wfTerminate(i, s, radiance, rng);
        return;
    }

    // Survive to the next bounce. The CPU loop dispatches exactly
    // WF_MAX_BOUNCES bounces, so a lane that reaches the last bounce simply
    // never gets its stored continuation traced — matching estimateRadiance's
    // loop exit (the depth-MAX ray is never followed).
    s.rayOrigin  = r.origin;
    s.rayDir     = r.direction;
    s.throughput = throughput;
    s.radiance   = radiance;
    s.depth      = depth + 1u;
    s.bsdfPdf    = misBsdfPdf;
    s.rngState   = rng.state;
    wfState[i] = s;
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
