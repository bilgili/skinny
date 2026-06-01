// Staged wavefront bidirectional path tracer (P1 §8 / design §P1-G). The
// megakernel BDPTIntegrator.estimateRadiance, torn into compute dispatches that
// store the camera + light subpath vertices in VRAM between stages:
//
//   wfBdptWalk     — per lane: trace the primary ray, build the eye subpath and
//                    the light subpath (the SAME randomWalk + sampleLightOrigin
//                    the megakernel uses), do the s=1 light-tracer splat, and
//                    store both subpaths + bookkeeping to VRAM.
//   wfBdptConnect  — per lane: reload both subpaths and run the connection
//                    strategies (t=0 emissive, t=1 NEE connectT1, t≥2 generic +
//                    MIS) verbatim, writing the eye-side radiance.
//   wfBdptResolve  — fold the eye-side radiance into the running mean and add
//                    the s=1 splat composite, matching main_pass.slang.
//
// Correctness is by reuse: every estimator function comes from
// integrators/bdpt.slang unchanged, so the accumulated image is the same
// estimator as the megakernel bdpt (A/B-verified). The rng cursor is carried
// across the walk/connect split so the random sequence matches. Matches the
// megakernel bdpt scope: flat first-hit only, pinhole camera. Set 0 is the
// megakernel scene descriptor set (incl. lightSplatBuffer for the splat); set 1
// holds the per-lane subpath + aux buffers this owns.

import common;            // RNG, Ray, PI
import bindings;          // fc, accumBuffer, lightSplatBuffer, materialTypes…
import interfaces;        // BSDFSample
import scene_trace;       // traceScene
import environment;       // sceneEnvRadiance
import cameras.pinhole;   // PinholeCamera
import materials.flat.flat_material;   // FlatMaterial, loadFlatMaterial
import materials.flat.flat_shading;    // isCutoutTransparent
import integrators.bdpt;  // BDPTVertex, randomWalk, sampleLightOrigin,
                          // connectT1, connectGeneric, misWeight, ConnectResult,
                          // bdptEnvNEE, splatLightWalk, BDPT_MAX_VERTS, BDPT_VK_*
import wavefront.wf_display;  // wfWriteDisplay (tonemap → output image)

struct WfBdptAux
{
    int    eyeLen;
    int    lightLen;
    uint   rngState;     // rng cursor after the walk + splat (resumed in connect)
    float  lensWeight;   // realistic-camera weight applied to the eye-side L
    uint   pixel;        // global pixel lane for this slot (SENTINEL = out-of-frame)
    float3 escaped;      // escaped env/emissive radiance gathered during the walk
    float3 radiance;     // final eye-side L (written by connect, read by resolve)
};

// Tiled streaming: the eye/light/aux buffers hold a fixed-size stream (slot
// index = tid.x), not one entry per pixel. The CPU loops `streamBase` over the
// frame; slot i maps to pixel `streamBase + i`, and connect/resolve recover the
// pixel from the stored `aux.pixel`. Out-of-frame tail slots are flagged dead.
static const uint WF_PIXEL_SENTINEL = 0xFFFFFFFFu;
// streamBase: per-tile pixel base (walk). shadeSlot: which connection slot an
// indirect-dispatched connect kernel processes. streamSize: slots in the stream
// buffers — bounds the full-stream kernels (the dispatch rounds up to 64).
struct WfTilePC { uint streamBase; uint shadeSlot; uint streamSize; };
[[vk::push_constant]] ConstantBuffer<WfTilePC> wfTile;

[[vk::binding(0, 1)]] RWStructuredBuffer<BDPTVertex> wfEye;     // stream × BDPT_MAX_VERTS
[[vk::binding(1, 1)]] RWStructuredBuffer<BDPTVertex> wfLight;   // stream × BDPT_MAX_VERTS
[[vk::binding(2, 1)]] RWStructuredBuffer<WfBdptAux>  wfAux;     // stream

// Connect-stage counting sort (P1 §8 compaction port). After the walk, each
// live lane is classified by subpath shape into one connection slot; the heavy
// generic+MIS double loop runs only for lanes whose light subpath has ≥2
// vertices. Mirrors the path tracer's wfLaneSlot/…/wfIndirectArgs (set 1,
// bindings 2..7) — here at bindings 3..8 (0..2 are the subpath/aux buffers).
static const uint WF_BDPT_NUM_SLOTS = 2u;
static const uint WF_BDPT_SLOT_NEE  = 0u;   // emissive + connectT1 (lightLen < 2)
static const uint WF_BDPT_SLOT_FULL = 1u;   // + generic t≥2 + MIS (lightLen ≥ 2)
static const uint WF_BDPT_GROUP     = 64u;  // connect [numthreads(64,1,1)]

[[vk::binding(3, 1)]] RWStructuredBuffer<uint> wfLaneSlot;     // [stream]  slot per lane (SENTINEL = dead)
[[vk::binding(4, 1)]] RWStructuredBuffer<uint> wfSlotCount;    // [NUM_SLOTS] live lanes per slot
[[vk::binding(5, 1)]] RWStructuredBuffer<uint> wfSlotOffset;   // [NUM_SLOTS] queue base per slot
[[vk::binding(6, 1)]] RWStructuredBuffer<uint> wfSlotQueue;    // [stream]  lane indices, grouped by slot
[[vk::binding(7, 1)]] RWStructuredBuffer<uint> wfSlotCursor;   // [NUM_SLOTS] scatter write cursor
[[vk::binding(8, 1)]] RWStructuredBuffer<uint> wfIndirectArgs; // [NUM_SLOTS*3] (x,y,z) per slot

// Resolve the stream slot an indirect-dispatched connect kernel processes, or
// 0xFFFFFFFF past the slot's queue (the lane early-outs). `qpos` is the dispatch
// thread index within the slot's slice.
uint wfBdptQueueLane(uint slot, uint qpos)
{
    if (qpos >= wfSlotCount[slot]) return 0xFFFFFFFFu;
    return wfSlotQueue[wfSlotOffset[slot] + qpos];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptWalk(uint3 tid: SV_DispatchThreadID)
{
    uint slot = tid.x;                          // stream slot
    uint base = slot * BDPT_MAX_VERTS;
    uint lane = wfTile.streamBase + slot;        // global pixel lane
    if (lane >= fc.width * fc.height)
    {
        wfAux[slot].pixel = WF_PIXEL_SENTINEL;   // out-of-frame tail → skip later
        return;
    }
    uint2 pixel = uint2(lane % fc.width, lane / fc.width);

    RNG rng = createRNG(pixel, fc.frameIndex);
    PinholeCamera cam;
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);

    WfBdptAux aux;
    aux.eyeLen = 0;
    aux.lightLen = 0;
    aux.lensWeight = lensWeight;
    aux.pixel = lane;
    aux.escaped = float3(0.0);
    aux.radiance = float3(0.0);

    BDPTVertex eye[BDPT_MAX_VERTS];
    BDPTVertex camVtx;
    camVtx.kind = BDPT_VK_CAMERA;
    camVtx.position = ray.origin;
    camVtx.N = -ray.direction;
    camVtx.throughput = float3(1.0);
    camVtx.emission = float3(0.0);
    camVtx.pdfFwd = 1.0;
    camVtx.pdfRev = 0.0;
    camVtx.isDelta = false;
    camVtx.onLight = false;
    camVtx.matId = 0u;
    camVtx.uv = float2(0.0);
    camVtx.posObject = float3(0.0);
    camVtx.geoN = -ray.direction;
    camVtx.tangent = float3(1.0, 0.0, 0.0);
    camVtx.hasTangent = false;
    eye[0] = camVtx;

    HitInfo firstHit = traceScene(fc, ray);
    Ray r = ray;
    for (uint skip = 0u; skip < 32u && firstHit.hit && isCutoutTransparent(firstHit); skip++)
    {
        r.origin = firstHit.position + r.direction * 0.002;
        firstHit = traceScene(fc, r);
    }
    if (!firstHit.hit)
    {
        aux.escaped = sceneEnvRadiance(r.direction, fc.envIntensity,
                                       effectiveFurnaceMode(firstHit.materialId));
        aux.rngState = rng.state;
        wfAux[slot] = aux;
        return;
    }
    if ((materialTypes[firstHit.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
    {
        // Non-flat first hit — bdpt bails to black here (the megakernel path
        // tracer covers skin/python). Matches estimateRadiance.
        aux.rngState = rng.state;
        wfAux[slot] = aux;
        return;
    }

    FlatMaterial fmat = loadFlatMaterial(firstHit);
    BDPTVertex z1;
    z1.kind = BDPT_VK_SURFACE;
    z1.position = firstHit.position;
    z1.N = fmat.data.N;
    z1.throughput = float3(1.0);
    z1.emission = fmat.data.mat.emission;
    z1.pdfFwd = 1.0;
    z1.pdfRev = 0.0;
    z1.isDelta = false;
    z1.onLight = any(fmat.data.mat.emission > float3(0.0));
    z1.matId = firstHit.materialId;
    z1.uv = firstHit.uv;
    z1.posObject = firstHit.positionObject;
    z1.geoN = firstHit.normal;
    z1.tangent = firstHit.tangent;
    z1.hasTangent = firstHit.hasTangent;
    eye[1] = z1;
    int eyeLen = 2;

    float3 T, B;
    if (firstHit.hasTangent)
    {
        T = normalize(firstHit.tangent - z1.N * dot(firstHit.tangent, z1.N));
        B = cross(z1.N, T);
    }
    else
    {
        buildBasis(z1.N, T, B);
    }
    float3 wo_ts = worldToTangent(-ray.direction, T, B, z1.N);

    float3 escapedRadiance = bdptEnvNEE(
        fmat, z1.position, z1.N, T, B, wo_ts, firstHit.materialId, rng);

    BSDFSample bs = fmat.sample(wo_ts, rng);
    if (bs.valid)
    {
        if (bs.pdf <= 0.0)
            eye[1].isDelta = true;
        float3 wi_world = tangentToWorld(bs.wi, T, B, z1.N);
        float3 throughput = bs.weight;
        Ray r2;
        float3 spawnOffset = bs.transmitted ? -z1.N * 0.002 : z1.N * 0.002;
        r2.origin = z1.position + spawnOffset;
        r2.direction = wi_world;
        float pdfFwdOmega = (bs.pdf > 0.0) ? bs.pdf : 1.0;
        float misBsdfPdfIn = (bs.pdf > 0.0 && !bs.transmitted
            && effectiveFurnaceMode(firstHit.materialId) == 0u) ? bs.pdf : -1.0;
        eyeLen = randomWalk(eye, 2, r2, throughput, pdfFwdOmega, rng,
                            true, escapedRadiance, misBsdfPdfIn);
    }

    // Light subpath.
    BDPTVertex lightPath[BDPT_MAX_VERTS];
    int lightLen = 0;
    LightOriginSample lo = sampleLightOrigin(rng);
    if (lo.valid)
    {
        lightPath[0] = lo.vertex;
        lightLen = 1;
        if (lo.vertex.kind != BDPT_VK_LIGHT_DIR)
        {
            Ray rL;
            rL.origin = lo.vertex.position + lo.vertex.N * 0.002;
            rL.direction = lo.dirOut;
            float3 unusedEscape = float3(0.0);
            lightLen = randomWalk(lightPath, 1, rL, lo.beta, lo.pdfDir, rng,
                                  false, unusedEscape, -1.0);
        }
    }

    // s = 1 light tracer splat (atomic into lightSplatBuffer, set 0).
    splatLightWalk<PinholeCamera>(lightPath, lightLen, rng, cam);

    // Store both subpaths + bookkeeping for the connection stage.
    for (int i = 0; i < int(BDPT_MAX_VERTS); i++)
    {
        if (i < eyeLen)   wfEye[base + uint(i)]   = eye[i];
        if (i < lightLen) wfLight[base + uint(i)] = lightPath[i];
    }
    aux.eyeLen = eyeLen;
    aux.lightLen = lightLen;
    aux.escaped = escapedRadiance;
    aux.rngState = rng.state;   // connect resumes from here
    wfAux[slot] = aux;
}

// Classify each post-walk lane by subpath shape and count it for the connect
// counting sort. Dead lanes (out-of-frame, primary miss, or non-flat first hit
// ⇒ eyeLen < 2) carry only the escaped radiance and are finalised here so the
// connect kernels never run for them; live lanes are routed to the NEE slot
// (lightLen < 2) or the FULL slot (lightLen ≥ 2). Full-stream dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptClassify(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= wfTile.streamSize)
        return;
    WfBdptAux aux = wfAux[i];
    if (aux.pixel == WF_PIXEL_SENTINEL)         // out-of-frame tail slot
    {
        wfLaneSlot[i] = WF_PIXEL_SENTINEL;
        return;
    }
    if (aux.eyeLen < 2)                          // miss / non-flat: escaped only
    {
        aux.radiance = aux.escaped * aux.lensWeight;
        wfAux[i] = aux;
        wfLaneSlot[i] = WF_PIXEL_SENTINEL;
        return;
    }
    uint slot = (aux.lightLen >= 2) ? WF_BDPT_SLOT_FULL : WF_BDPT_SLOT_NEE;
    wfLaneSlot[i] = slot;
    uint prev;
    InterlockedAdd(wfSlotCount[slot], 1u, prev);
}

// Exclusive prefix-sum the per-slot counts into queue offsets and emit one
// VkDispatchIndirectCommand (ceil(count/64),1,1) per slot. Single thread.
[shader("compute")]
[numthreads(1, 1, 1)]
void wfBdptBuildArgs(uint3 tid: SV_DispatchThreadID)
{
    uint running = 0u;
    for (uint slot = 0u; slot < WF_BDPT_NUM_SLOTS; ++slot)
    {
        wfSlotOffset[slot] = running;
        uint c = wfSlotCount[slot];
        running += c;
        wfIndirectArgs[slot * 3u + 0u] = (c + WF_BDPT_GROUP - 1u) / WF_BDPT_GROUP;
        wfIndirectArgs[slot * 3u + 1u] = 1u;
        wfIndirectArgs[slot * 3u + 2u] = 1u;
    }
}

// Place each live lane into its slot's contiguous queue slice via an atomic
// write cursor (counting sort). Full-stream dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptScatter(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= wfTile.streamSize)
        return;
    uint slot = wfLaneSlot[i];
    if (slot == WF_PIXEL_SENTINEL)              // dead → not enqueued
        return;
    uint local;
    InterlockedAdd(wfSlotCursor[slot], 1u, local);
    wfSlotQueue[wfSlotOffset[slot] + local] = i;
}

// Shared connection body for lane `i`. `includeGeneric` gates the O(s·t)
// generic+MIS double loop (only the FULL slot runs it). The eye-side estimator
// is identical to the megakernel bdpt; a lane runs exactly one of the two
// connect kernels, so the rng consumption order (connectT1 draws, then the
// rng-free generic loop) matches the fused connect verbatim.
void wfBdptConnectLane(uint i, bool includeGeneric)
{
    WfBdptAux aux = wfAux[i];
    uint base = i * BDPT_MAX_VERTS;
    int eyeLen = aux.eyeLen;
    int lightLen = aux.lightLen;

    RNG rng; rng.state = aux.rngState;

    BDPTVertex eye[BDPT_MAX_VERTS];
    BDPTVertex lightPath[BDPT_MAX_VERTS];
    for (int k = 0; k < int(BDPT_MAX_VERTS); k++)
    {
        if (k < eyeLen)   eye[k]       = wfEye[base + uint(k)];
        if (k < lightLen) lightPath[k] = wfLight[base + uint(k)];
    }

    float3 L = aux.escaped;

    // (s ≥ 2, t = 0): emissive eye-vertex hits.
    [loop]
    for (int s = 2; s <= eyeLen; s++)
    {
        BDPTVertex z = eye[s - 1];
        if (z.onLight)
            L += z.throughput * z.emission;
    }

    // (s ≥ 2, t = 1): NEE.
    [loop]
    for (int s = 2; s <= eyeLen; s++)
        L += connectT1(eye, s, rng);

    // (s ≥ 2, t ≥ 2): generic connections + MIS (FULL slot only).
    if (includeGeneric)
    {
        [loop]
        for (int s = 2; s <= eyeLen; s++)
        {
            [loop]
            for (int t = 2; t <= lightLen; t++)
            {
                if (s + t - 2 > int(BDPT_MAX_DEPTH)) continue;
                ConnectResult cr = connectGeneric(eye, s, lightPath, t);
                if (!cr.nonzero) continue;
                float w = misWeight(eye, s, lightPath, t,
                                    cr.pdf_zsm1_rev, cr.pdf_zsm2_rev,
                                    cr.pdf_ytm1_rev, cr.pdf_ytm2_rev);
                L += cr.contribution * w;
            }
        }
    }

    aux.radiance = L * aux.lensWeight;
    wfAux[i] = aux;
}

// NEE slot: emissive + connectT1, no generic loop. Indirect-dispatched over the
// NEE queue (lanes with lightLen < 2 — e.g. directional-only scenes).
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptConnectNee(uint3 tid: SV_DispatchThreadID)
{
    uint i = wfBdptQueueLane(wfTile.shadeSlot, tid.x);
    if (i == 0xFFFFFFFFu)
        return;
    wfBdptConnectLane(i, false);
}

// FULL slot: emissive + connectT1 + generic + MIS. Indirect-dispatched over the
// FULL queue (lanes with a ≥2-vertex light subpath). Zero groups when no scene
// light produces a light subpath (e.g. directional-only).
[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptConnectFull(uint3 tid: SV_DispatchThreadID)
{
    uint i = wfBdptQueueLane(wfTile.shadeSlot, tid.x);
    if (i == 0xFFFFFFFFu)
        return;
    wfBdptConnectLane(i, true);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void wfBdptResolve(uint3 tid: SV_DispatchThreadID)
{
    uint slot = tid.x;
    if (slot >= wfTile.streamSize)              // dispatch rounds up to 64 — skip tail
        return;
    uint px = wfAux[slot].pixel;
    if (px == WF_PIXEL_SENTINEL)                // out-of-frame tail slot
        return;
    uint2 pixel = uint2(px % fc.width, px / fc.width);

    float3 sample = wfAux[slot].radiance;
    bool3 bad = isnan(sample) | isinf(sample);
    sample = select(bad, float3(0.0), sample);
    sample = max(sample, float3(0.0));

    // Running mean of the eye-side estimator. main_pass.slang writes exactly
    // this to accumBuffer (binding 2 = what read_accumulation reads); the s=1
    // splatBuffer composite is added only on the display/tonemap path, never
    // into accumBuffer, so the headless A/B compares eye-side estimates. The
    // splat is still accumulated in wfBdptWalk for a future windowed composite
    // (and is a no-op for the demo's directional light — no light subpath).
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
