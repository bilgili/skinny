// Shared wavefront path-state bindings + the material-free shade finisher, so
// every per-material shade kernel (flat / catch-all) reuses them without
// re-declaring the set-1 layout or duplicating the post-eval body. The finisher
// is the part of the bounce that does NOT depend on the material: accumulate,
// Russian roulette, BSDF-sampled next ray, sphere-light MIS, and the state
// store. The material-specific half (NEE + BSDF sample → BounceResult) is the
// caller's; it passes the result in.

import common;          // RNG, Ray, PI, powerHeuristic
import bindings;        // fc, effectiveFurnaceMode, materialTypes
import interfaces;      // BounceResult, BSDFSample, HitInfo
import scene_trace;     // traceScene
import scene_lights;    // intersectSphereLights, NearestSphereHit
import lights.sphere_light;            // SphereLight
import wavefront.wavefront_state;      // WavefrontPathState, PATH_FLAG_*
// Wavefront-native path records: per-lane vertex stack + terminate-time splat
// (gated by fc.recordMode; no-op + byte-identical render when off).
import wavefront.wf_records;           // wfPushRecord, wfEmitRecords

static const uint WF_MAX_BOUNCES = 6u;
static const uint WF_PIXEL_SENTINEL = 0xFFFFFFFFu;
static const uint WF_SLOT_SENTINEL = 0xFFFFFFFFu;  // lane not enqueued this bounce
static const uint WF_SHADE_GROUP = 64u;            // shade [numthreads(64,1,1)]
// Per-material shade slots: 0 = flat/graph (wfPathShadeFlat), 1 = non-flat
// catch-all (wfPathShade). The counting sort groups each bounce's hit lanes by
// slot so each shade dispatch covers only its lanes (vkCmdDispatchIndirect).
static const uint WF_NUM_SLOTS = 2u;
static const uint WF_SLOT_FLAT = 0u;
static const uint WF_SLOT_OTHER = 1u;

// streamBase: per-tile pixel base (generate). shadeSlot: which material slot a
// queue-indexed shade dispatch processes. streamSize: slots in the stream
// buffers — bounds every kernel (the dispatch rounds up to a 64-multiple).
struct WfTilePC { uint streamBase; uint shadeSlot; uint streamSize; };
[[vk::push_constant]] ConstantBuffer<WfTilePC> wfTile;

[[vk::binding(0, 1)]] RWStructuredBuffer<WavefrontPathState> wfState;
[[vk::binding(1, 1)]] RWStructuredBuffer<HitInfo>            wfHits;
// Counting-sort buffers (per bounce; counts/cursor zeroed before intersect):
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> wfLaneSlot;      // [stream]  slot per lane (SENTINEL = none)
[[vk::binding(3, 1)]] RWStructuredBuffer<uint> wfSlotCount;     // [NUM_SLOTS] hits per slot
[[vk::binding(4, 1)]] RWStructuredBuffer<uint> wfSlotOffset;    // [NUM_SLOTS] queue base per slot
[[vk::binding(5, 1)]] RWStructuredBuffer<uint> wfSlotQueue;     // [stream]  lane indices, grouped by slot
[[vk::binding(6, 1)]] RWStructuredBuffer<uint> wfSlotCursor;    // [NUM_SLOTS] scatter write cursor
[[vk::binding(7, 1)]] RWStructuredBuffer<uint> wfIndirectArgs;  // [NUM_SLOTS*3] (x,y,z) per slot
// Per-lane neural-proposal forward sample, written by the neural pre-pass
// (wavefront/neural_proposal_pass.slang) and read by the flat shade kernel. A
// 1-element dummy when the neural proposal is inactive (wfNeural[i].valid == 0
// then, so the flat bounce keeps {bsdf,env}). Owned by WavefrontPathPass.
[[vk::binding(8, 1)]] RWStructuredBuffer<WfNeuralSample> wfNeural;  // [stream]

uint2 wfDecodePixel(uint lane) { return uint2(lane % fc.width, lane / fc.width); }

// Map a hit's material type to its shade slot.
uint wfSlotForType(uint matType) { return (matType == MATERIAL_TYPE_FLAT) ? WF_SLOT_FLAT : WF_SLOT_OTHER; }

// Resolve the stream slot a queue-indexed shade dispatch should process, or
// 0xFFFFFFFF past the slot's queue (the lane should early-out). `qpos` is the
// dispatch thread index within the slot's slice.
uint wfQueueLane(uint slot, uint qpos)
{
    if (qpos >= wfSlotCount[slot]) return 0xFFFFFFFFu;
    return wfSlotQueue[wfSlotOffset[slot] + qpos];
}

// Write the lane's accumulated radiance + clear ALIVE; persist the rng cursor.
// On termination `radiance` IS the path's L_final, so splat the lane's stacked
// guideable records here (gated; no-op when not recording).
void wfTerminate(uint i, inout WavefrontPathState s, float3 radiance, RNG rng)
{
    if (fc.recordMode != 0u)
        wfEmitRecords(i, radiance);
    s.radiance = radiance;
    s.flags &= ~PATH_FLAG_ALIVE;
    s.rngState = rng.state;
    wfState[i] = s;
}

// Finish a bounce given the material's BounceResult: accumulate emission +
// direct light, Russian roulette, spawn the next ray, BSDF-sampled sphere-light
// MIS, and store (or terminate). Verbatim from the fused bounce tail.
void wfFinishShade(uint i, WavefrontPathState s, inout RNG rng, HitInfo h, BounceResult br)
{
    float3 throughput = s.throughput;
    float3 radiance   = s.radiance;
    uint   depth      = s.depth;

    radiance += throughput * br.fullRadiance;
    // A delta/specular bounce spawned this ray (PATH_FLAG_SPECULAR, set below
    // when the previous bounce sampled a pdf<=0 lobe): it has no NEE partner, so
    // an emissive triangle it lands on must be taken at full weight — the
    // wavefront mirror of path.slang's `spawnedBySpecular` gate. Without it the
    // area-light reflection in a smooth dielectric and the specular caustic leg
    // are dropped (path tracer biased dark vs the pbrt reference).
    if (depth == 0u || fc.numEmissiveTriangles == 0u || pathSpecular(s))
    {
        radiance += throughput * br.bsdfSample.emission;
    }
    else if (any(br.bsdfSample.emission > float3(0.0)))
    {
        // Non-delta bounce landed on an emissive triangle that allLightsNEE
        // (power heuristic) also sampled ⇒ add only the BSDF-sampling MIS
        // complement. Mirror of path.slang's emissive-triangle-bsdf-hit-mis: the
        // NEE solid-angle pdf is reconstructed without the triangle index (area
        // cancels) — pdfLightSA = lum·d² / (emissiveTotalPower·cosLight).
        // s.bsdfPdf carries the spawning bounce's pdf (== the bare bsdf pdf for a
        // flat non-delta non-transmitted lobe, matching the megakernel).
        float cosLight = max(dot(h.normal, -s.rayDir), 0.0);
        float lum      = dot(br.bsdfSample.emission, float3(0.2126, 0.7152, 0.0722));
        float pdfLightSA = (cosLight > 1e-6 && fc.emissiveTotalPower > 0.0)
            ? lum * (h.t * h.t) / (fc.emissiveTotalPower * cosLight)
            : 1e30;   // grazing / degenerate ⇒ NEE owns it ⇒ wBSDF → 0
        float wBSDF = powerHeuristic(s.bsdfPdf, pdfLightSA);
        radiance += throughput * br.bsdfSample.emission * wBSDF;
    }
    radiance += throughput * br.directLight;

    if (!br.bsdfSample.valid)
    {
        wfTerminate(i, s, radiance, rng);
        return;
    }

    // Record-mode: snapshot this guideable vertex BEFORE the throughput update,
    // so beta_in_k = throughput entering the vertex and L_k = radiance after its
    // local terms — the same snapshot order as estimateRadianceRecord. Splatted
    // at lane termination (wfTerminate / wfPathResolve). No-op when not recording.
    if (fc.recordMode != 0u)
        wfPushRecord(i, h, s.rayDir, br.bsdfSample, radiance, throughput, depth);

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
    float misBsdfPdf = (envNeeActive && !br.bsdfSample.transmitted && br.bsdfSample.pdf > 0.0)
                     ? br.bsdfSample.pdf : -1.0;

    Ray r;
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
        // ReSTIR DI (reuseMode 1) owns primary-hit direct: at the primary vertex
        // its BSDF candidate already counted this sphere-hit term, so skip it
        // here (still terminate — the ray hit an opaque emitter). Transmitted
        // depth-0 bounces have no ReSTIR BSDF candidate → keep the stock add.
        bool restirOwns = (fc.reuseMode == 1u) && (depth == 0u) && !br.bsdfSample.transmitted;
        if (!restirOwns)
            radiance += throughput * light.radiance * w_bsdf;
        wfTerminate(i, s, radiance, rng);
        return;
    }

    // Survive to the next bounce (the CPU loop bounds the bounce count).
    s.rayOrigin  = r.origin;
    s.rayDir     = r.direction;
    s.throughput = throughput;
    s.radiance   = radiance;
    s.depth      = depth + 1u;
    s.bsdfPdf    = misBsdfPdf;
    // Carry this bounce's delta-ness to the next iteration's emission gate (a
    // delta lobe — pdf<=0 — has no NEE partner). Set/clear so a prior delta
    // bounce doesn't leak into a later non-delta one.
    if (br.bsdfSample.pdf <= 0.0)
        s.flags |= PATH_FLAG_SPECULAR;
    else
        s.flags &= ~PATH_FLAG_SPECULAR;
    s.rngState   = rng.state;
    wfState[i] = s;
}
