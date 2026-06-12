// Wavefront-native path-record emission. The megakernel `mainImageRecord` owns
// the whole path in one thread, so its tail radiance `L_final` is a register at
// loop end and the per-vertex `(L_k, beta_in_k)` snapshots live on the thread
// stack (integrators/path_record.slang). The wavefront path is torn across
// per-bounce dispatches, so neither is co-resident at the bounce that sampled
// `wi`. This module carries the snapshots in a per-lane VRAM stack and splats
// the records at lane termination — the identical arithmetic, just relocated.
//
// Storage is a SEPARATE pair of set-1 buffers (NOT inside WavefrontPathState):
// that hot struct is copied by value in every kernel, so inlining a 6-deep
// stack there would spill to scratch and ~8× its bandwidth in every kernel even
// when not training. These buffers are 1-element dummies (never touched) when
// `fc.recordMode == 0`, so the default render stays byte-identical + overhead-free.
//
// Output records go to the SAME append buffer + counter (bindings 36/37, set 0)
// the megakernel uses, via the shared `emitRecord` — so the live drain and the
// `.nrec` reader consume the wavefront stream unchanged.

import common;        // buildBasis
import bindings;      // fc, materialTypes, MATERIAL_TYPE_FLAT, MATERIAL_TYPE_PYTHON
import interfaces;    // HitInfo, BSDFSample
import integrators.path_record_common;   // PathRecord, emitRecord, recordContrib, REC_MAX_BOUNCES

// One snapshot of a guideable vertex — geometry + the (radiance, beta_in)
// pair the backward attribution needs. Scalar layout: float3 = 12 B → 76 B.
// Host mirror: skinny.wavefront_layout (REC_VERTEX_STRIDE).
struct RecVertex
{
    float3 pos;      // 0..12   world hit position
    float3 normal;   // 12..24  world shading normal (= HitInfo.normal)
    float3 wo;       // 24..36  world outgoing dir (= -rayDir)
    float3 wiLocal;  // 36..48  sampled dir in flow-local (x=T,y=N,z=B), y-up
    float3 L_k;      // 48..60  radiance after this vertex's local terms
    float3 beta_in;  // 60..72  throughput ENTERING this vertex (pre-update)
    uint   depth;    // 72..76  bounce index
};

#if defined(SKINNY_METAL) && !defined(SKINNY_METAL_RECORDS)
// Metal default build: Metal caps a kernel's argument table at 31 buffer
// slots — the wavefront program's scene + queue buffer globals already sit
// near that cap, so the four record buffers (wfRecStack/wfRecCount here +
// recordBuf/recordCounter through `emitRecord`) would overflow it ("'buffer'
// attribute parameter is out of bounds"). Stub the emitters so all four
// dead-strip from every wavefront kernel; fc.recordMode is never set, so the
// stubs are semantically the no-op path and the render is byte-identical.
// The records build (SKINNY_METAL_RECORDS, change metal-record-drain) takes
// the real emitters below and frees the slots by compiling out wavefront-dead
// globals instead (see bindings.slang). The Vulkan compile (no SKINNY_METAL)
// is byte-identical to before.
void wfResetRecords(uint i) {}
void wfPushRecord(uint i, HitInfo h, float3 rayDir, BSDFSample bs,
                  float3 L_k, float3 beta_in, uint depth) {}
void wfEmitRecords(uint i, float3 Lfinal) {}
#elif defined(SKINNY_METAL)
// Metal records build (SKINNY_METAL_RECORDS, change metal-record-drain). The
// neural build sits exactly at the 31-slot cap, so the per-lane count buffer
// is folded into the stack instead of taking its own slot: element k = 0 of
// each lane's stride-(REC_MAX_BOUNCES+1) region is a header whose `depth`
// field holds the stacked count; vertices live at k = 1..REC_MAX_BOUNCES.
// Same guard, frame, snapshot order, and attribution as the Vulkan branch
// below — keep the two in lockstep. Host sizing: metal_wavefront allocates
// [capacity * (REC_MAX_BOUNCES + 1)] elements at the reflected MSL stride
// (the stack never leaves the GPU, so its layout is internal).
static const uint WF_REC_LANE = REC_MAX_BOUNCES + 1u;
[[vk::binding(9, 1)]] RWStructuredBuffer<RecVertex> wfRecStack;  // [stream * WF_REC_LANE]

void wfResetRecords(uint i) { wfRecStack[i * WF_REC_LANE].depth = 0u; }

void wfPushRecord(uint i, HitInfo h, float3 rayDir, BSDFSample bs,
                  float3 L_k, float3 beta_in, uint depth)
{
    uint matType = materialTypes[h.materialId] & 0xFFu;
    bool guideable = (matType == MATERIAL_TYPE_FLAT || matType == MATERIAL_TYPE_PYTHON)
                   && !bs.transmitted && bs.pdf > 0.0;
    if (!guideable)
        return;

    uint k = wfRecStack[i * WF_REC_LANE].depth;
    if (k >= REC_MAX_BOUNCES)
        return;

    float3 N = h.normal;
    float3 T, B;
    if (h.hasTangent) { T = normalize(h.tangent - N * dot(h.tangent, N)); B = cross(N, T); }
    else              { buildBasis(N, T, B); }
    float3 wiL = float3(dot(bs.wi, T), dot(bs.wi, N), dot(bs.wi, B));
    if (wiL.y <= 1e-4)   // upper hemisphere only (the flow's domain)
        return;

    RecVertex v;
    v.pos     = h.position;
    v.normal  = N;
    v.wo      = -normalize(rayDir);
    v.wiLocal = wiL;
    v.L_k     = L_k;
    v.beta_in = beta_in;
    v.depth   = depth;
    wfRecStack[i * WF_REC_LANE + 1u + k] = v;
    wfRecStack[i * WF_REC_LANE].depth = k + 1u;
}

void wfEmitRecords(uint i, float3 Lfinal)
{
    uint n = wfRecStack[i * WF_REC_LANE].depth;
    for (uint k = 0u; k < n; k++)
    {
        RecVertex v = wfRecStack[i * WF_REC_LANE + 1u + k];
        float3 contrib = recordContrib(Lfinal, v.L_k, v.beta_in);
        if (any(isnan(contrib)) || any(isinf(contrib)))
            continue;
        PathRecord rec;
        rec.pos     = v.pos;
        rec.normal  = v.normal;
        rec.wo      = v.wo;
        rec.wiLocal = v.wiLocal;
        rec.contrib = contrib;
        rec.depth   = v.depth;
        emitRecord(rec);
    }
    wfRecStack[i * WF_REC_LANE].depth = 0u;
}
#else
// Per-lane record stack (binding 9) + per-lane stacked count (binding 10), in
// the path pass's own set 1. Sized [stream * REC_MAX_BOUNCES] / [stream] when
// recording; 1-element dummies otherwise. Owned by WavefrontPathPass.
[[vk::binding(9,  1)]] RWStructuredBuffer<RecVertex> wfRecStack;   // [stream * REC_MAX_BOUNCES]
[[vk::binding(10, 1)]] RWStructuredBuffer<uint>      wfRecCount;   // [stream]

// Reset a lane's stack at generate (per tile). Caller gates on fc.recordMode.
void wfResetRecords(uint i) { wfRecCount[i] = 0u; }

// Push one guideable-vertex snapshot. Same guard, frame, and snapshot order as
// the megakernel's estimateRadianceRecord: flat/graph or python material,
// reflective (non-transmitted, pdf > 0), sampled dir in the upper hemisphere of
// the HitInfo-normal frame. `L_k` is the radiance AFTER this vertex's local
// terms; `beta_in` is the throughput BEFORE the bounce's throughput update.
// Caller gates on fc.recordMode.
void wfPushRecord(uint i, HitInfo h, float3 rayDir, BSDFSample bs,
                  float3 L_k, float3 beta_in, uint depth)
{
    uint matType = materialTypes[h.materialId] & 0xFFu;
    bool guideable = (matType == MATERIAL_TYPE_FLAT || matType == MATERIAL_TYPE_PYTHON)
                   && !bs.transmitted && bs.pdf > 0.0;
    if (!guideable)
        return;

    uint k = wfRecCount[i];
    if (k >= REC_MAX_BOUNCES)
        return;

    // Flow-local frame from the HitInfo normal (matches train↔infer: the neural
    // pass uses N = HitInfo.normal). Project the world sampled dir onto (T,N,B).
    float3 N = h.normal;
    float3 T, B;
    if (h.hasTangent) { T = normalize(h.tangent - N * dot(h.tangent, N)); B = cross(N, T); }
    else              { buildBasis(N, T, B); }
    float3 wiL = float3(dot(bs.wi, T), dot(bs.wi, N), dot(bs.wi, B));
    if (wiL.y <= 1e-4)   // upper hemisphere only (the flow's domain)
        return;

    RecVertex v;
    v.pos     = h.position;
    v.normal  = N;
    v.wo      = -normalize(rayDir);
    v.wiLocal = wiL;
    v.L_k     = L_k;
    v.beta_in = beta_in;
    v.depth   = depth;
    wfRecStack[i * REC_MAX_BOUNCES + k] = v;
    wfRecCount[i] = k + 1u;
}

// Splat the lane's stacked records at termination: contrib_k = recordContrib(
// L_final, L_k, beta_in_k), dropping non-finite. Clears the count so a lane is
// never emitted twice. Caller gates on fc.recordMode.
void wfEmitRecords(uint i, float3 Lfinal)
{
    uint n = wfRecCount[i];
    for (uint k = 0u; k < n; k++)
    {
        RecVertex v = wfRecStack[i * REC_MAX_BOUNCES + k];
        float3 contrib = recordContrib(Lfinal, v.L_k, v.beta_in);
        if (any(isnan(contrib)) || any(isinf(contrib)))
            continue;
        PathRecord rec;
        rec.pos     = v.pos;
        rec.normal  = v.normal;
        rec.wo      = v.wo;
        rec.wiLocal = v.wiLocal;
        rec.contrib = contrib;
        rec.depth   = v.depth;
        emitRecord(rec);
    }
    wfRecCount[i] = 0u;
}
#endif
