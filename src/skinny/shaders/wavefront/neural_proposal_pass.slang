// Wavefront neural-proposal pre-pass.
//
// Scheduled between scatter and the flat shade each bounce (mirrors how the
// ReSTIR DI pass slots in): for every live lane it builds the spline-flow
// condition from the lane's HitInfo + incoming ray, draws ONE forward sample,
// and writes the per-lane (wi, pdf) the flat bounce mixture consumes. The
// arbitrary-direction INVERSE pdf is evaluated inline in the shade kernel
// (sampling/proposal.slang) — this pass only amortises the forward draw over
// the live lanes, keeping the MLP off the bounce kernel's hot path.
//
// Wavefront-only; NOT included by main_pass.slang. Set 0 is the renderer's
// shared scene set (provides fc + the neural weight buffers at 33/34/35 via
// sampling.neural_proposal); set 1 is this pass's own state / hit / output.
//
// The base sample u is hashed from the lane's GLOBAL pixel index + frame +
// bounce — independent of the shade kernel's RNG stream, so enabling neural
// never perturbs the {bsdf}/{bsdf,env} draw paths, and accumulation across
// frames still averages fresh forward samples.

import common;        // RNG, createRNG, buildBasis
import bindings;      // fc (set 0) — pulls in the neural weight buffers too
import interfaces;    // HitInfo, WfNeuralSample
import sampling.neural_proposal;        // neuralSampleWorld
import wavefront.wavefront_state;       // WavefrontPathState, PATH_FLAG_ALIVE

static const uint NP_GROUP = 64u;

struct NeuralPC { uint streamSize; uint networkVersion; uint _pad0; uint _pad1; };
[[vk::push_constant]] ConstantBuffer<NeuralPC> npc;

[[vk::binding(0, 1)]] RWStructuredBuffer<WavefrontPathState> npState;
[[vk::binding(1, 1)]] RWStructuredBuffer<HitInfo>            npHits;
[[vk::binding(2, 1)]] RWStructuredBuffer<WfNeuralSample>     npOut;

[shader("compute")]
[numthreads(64, 1, 1)]
void wfNeuralProposal(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= npc.streamSize)
        return;

    WfNeuralSample o;
    o.wi = float3(0.0, 1.0, 0.0);
    o.pdf = 0.0;
    o.version = npc.networkVersion;
    o.valid = 0u;
    o._pad0 = 0u;
    o._pad1 = 0u;

    WavefrontPathState s = npState[i];
    HitInfo h = npHits[i];

    if (h.hit && (s.flags & PATH_FLAG_ALIVE) != 0u)
    {
        // Tangent frame — identical construction to flat_bounce.slang so the
        // condition's frame matches the bounce's frame exactly.
        float3 N = h.normal;
        float3 T, B;
        if (h.hasTangent)
        {
            T = normalize(h.tangent - N * dot(h.tangent, N));
            B = cross(N, T);
        }
        else
        {
            buildBasis(N, T, B);
        }
        float3 woWorld = -normalize(s.rayDir);

        // Decorrelated base sample: global pixel + frame + bounce, NOT the
        // shade RNG state (keeps the default draw paths bit-identical).
        uint2 px = uint2(s.pixelIndex % fc.width, s.pixelIndex / fc.width);
        RNG rng = createRNG(px, fc.frameIndex * 8u + s.depth);
        float2 u = rng.next2();

        float pdf;
        float3 wi = neuralSampleWorld(h.position, N, T, B, woWorld, u, pdf);
        if (pdf > 0.0)
        {
            o.wi = wi;
            o.pdf = pdf;
            o.valid = 1u;
        }
    }

    npOut[i] = o;
}
