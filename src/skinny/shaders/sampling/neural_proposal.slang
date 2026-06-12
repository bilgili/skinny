// Renderer-facing adapter over the pure spline-flow (sampling/neural_flow.slang).
//
// Bridges the flow's own y-up unit-square space to the renderer's shading frame
// and condition encoding, declares the frozen weight buffers, and exposes the
// two operations the scene-sampling seam needs:
//   neuralSampleWorld(...)  forward  — draw a world-space wi + its solid-angle pdf
//   neuralPdfWorld(...)     inverse  — exact solid-angle pdf of an arbitrary wi
//
// CONDITION ENCODING (neuralCondition) is canonical and MUST match the offline
// trainer (spline_flow) byte-for-byte — a mismatch raises variance silently
// rather than biasing. The renderer's record-dump (renderer.py) and the trainer
// share this exact definition: pos normalised to the scene AABB → [-1,1]³, then
// the world shading normal, then the outgoing world direction (9 = NF_COND).

import common;        // FrameConstants fc (scene bounds)
import bindings;      // fc
import sampling.neural_flow;   // NfLayerHeader, sampleNeural, pdfNeural, NF_COND

// Frozen offline-trained weights (sampling/neural_proposal.py → renderer upload).
// Placed ABOVE the MaterialX nodegraph range (GRAPH_BINDING_BASE = 25..29), the
// tool buffer (30) and the env importance CDFs (31/32) — see the Architecture.md
// binding map. The inline inverse below is statically referenced by
// proposal.slang, so these three bindings sit in the shared scene-set layout
// even on the megakernel (which never sets the NEURAL bit); the renderer binds
// 1-element dummies there until the neural proposal is activated.
// NF_WT default — Slang `#define`s do NOT cross the `import` boundary, so the
// alias from sampling.neural_flow is not visible here; this fallback keeps the
// default megakernel/wavefront build (no `-D`) compiling `StructuredBuffer<float>`.
// The host passes `-D NF_WT=half` (command-line defines ARE global to the
// compile) for the fp16 modes, which overrides this guard AND neural_flow's in
// lockstep. Keep the default in sync with neural_flow.slang.
#ifndef NF_WT
#define NF_WT float
#endif

// 33/34 element type follows NF_WT (sampling.neural_flow): `float` by default,
// `half` in the fp16-storage / fp16-compute precision modes (study change
// neural-precision-size-study). The binding slots never move; only the element
// type — and the bytes the host uploads — change. NfLayerHeader (35) is precision-
// and size-agnostic (it carries the layout the loader already validates).
//
// Metal target (change metal-wavefront-parity phase 3): the neural proposal is
// not ported until phase 6, and Metal caps a kernel's argument table at 31
// buffer slots — the wavefront flat-shade kernel's scene + queue globals sit at
// that cap, so these three weight buffers would overflow it. The SKINNY_METAL
// branch declares no buffers and stubs the two world-space entry points (pdf 0
// → the proposal MIS treats neural as absent); `fc.proposalMask`'s NEURAL bit
// is never set on Metal, so the stubs are the existing runtime no-op path. The
// Vulkan compile (no SKINNY_METAL) is byte-identical to before.
#if !defined(SKINNY_METAL)
[[vk::binding(33)]] StructuredBuffer<NF_WT>         neuralWeights;
[[vk::binding(34)]] StructuredBuffer<NF_WT>         neuralBiases;
[[vk::binding(35)]] StructuredBuffer<NfLayerHeader> neuralLayers;
#endif

// Canonical condition c = (position normalised to the scene AABB → [-1,1]³,
// shading normal, outgoing world dir). MUST match the trainer's encoding.
// DOC:cond#body start
void neuralCondition(float3 posWorld, float3 N, float3 woWorld, out float cond[NF_COND])
{
    float3 ext = max(fc.sceneBoundsExtent, float3(1e-6f));
    float3 p = (posWorld - fc.sceneBoundsMin) / ext;   // [0,1]³
    p = p * 2.0f - 1.0f;                               // [-1,1]³
    cond[0] = p.x; cond[1] = p.y; cond[2] = p.z;
    cond[3] = N.x; cond[4] = N.y; cond[5] = N.z;
    cond[6] = woWorld.x; cond[7] = woWorld.y; cond[8] = woWorld.z;
}
// DOC:cond#body end

// Flow-local frame: +Y = shading normal (the flow hemisphere is y-up). Map
// to/from world via the shading basis, so (T, N, B) ↔ flow-local (x, y, z).
float3 neuralLocalToWorld(float3 loc, float3 N, float3 T, float3 B)
{ return loc.x * T + loc.y * N + loc.z * B; }

float3 neuralWorldToLocal(float3 w, float3 N, float3 T, float3 B)
{ return float3(dot(w, T), dot(w, N), dot(w, B)); }

#if defined(SKINNY_METAL)
// Phase-6 stubs (see the weight-buffer gate above): pdf 0 ⇒ the proposal MIS
// chain treats the neural component as absent; never reachable at runtime on
// Metal because the renderer rejects the NEURAL proposal bit there.
float3 neuralSampleWorld(float3 posWorld, float3 N, float3 T, float3 B,
                         float3 woWorld, float2 u, out float pdfOmega)
{
    pdfOmega = 0.0f;
    return N;
}

float neuralPdfWorld(float3 posWorld, float3 N, float3 T, float3 B,
                     float3 woWorld, float3 wiWorld)
{
    return 0.0f;
}
#else
// Forward — draw a world-space bounce direction + its solid-angle pdf.
float3 neuralSampleWorld(float3 posWorld, float3 N, float3 T, float3 B,
                         float3 woWorld, float2 u, out float pdfOmega)
{
    float cond[NF_COND];
    neuralCondition(posWorld, N, woWorld, cond);
    float3 loc = sampleNeural(neuralWeights, neuralBiases, neuralLayers, cond, u, pdfOmega);
    return neuralLocalToWorld(loc, N, T, B);
}

// Inverse — exact solid-angle pdf the flow assigns to an arbitrary world wi.
float neuralPdfWorld(float3 posWorld, float3 N, float3 T, float3 B,
                     float3 woWorld, float3 wiWorld)
{
    float cond[NF_COND];
    neuralCondition(posWorld, N, woWorld, cond);
    float3 loc = neuralWorldToLocal(wiWorld, N, T, B);
    return pdfNeural(neuralWeights, neuralBiases, neuralLayers, cond, loc);
}
#endif
