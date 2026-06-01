// Per-material-queue paint kernel — the minimal carrier for verifying the
// wavefront indirect-dispatch path (P1 §P1-C / tasks 3.2 / 3.3 / 9.3).
//
// After intersect tallies per-material hit counts, build_args prefix-sums them
// into per-material slice offsets + one VkDispatchIndirectCommand per material,
// and scatter groups each lane index into its material's contiguous slice of
// `materialQueue` (build_args.slang / scatter.slang — both GPU-verified). The
// shade stage then dispatches one kernel per material over ITS slice, sized by
// the indirect args. This kernel stands in for that per-material shade: it
// reads its slice from a push constant and writes the material's value to each
// queued lane's output slot.
//
// Dispatched two ways that MUST agree (the indirect-dispatch de-risk):
//   indirect — vkCmdDispatchIndirect over indirectArgs[slot] (ceil(count/64) groups)
//   direct   — conservative vkCmdDispatch(ceil(streamSize/64)) + the early-out
//              on `tid.x >= sliceCount` below (worst-case groups, empty lanes skip)

struct PaintPC
{
    uint sliceBase;   // materialOffset[slot] — base index of this material's slice
    uint sliceCount;  // materialCount[slot]  — lanes in this material's slice
    uint value;       // what to write (the material's marker / colour index)
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   materialQueue;  // [streamSize] grouped lane indices
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> paintOut;       // [streamSize] per-lane output
[[vk::push_constant]] ConstantBuffer<PaintPC>  pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void wfIndirectPaint(uint3 tid: SV_DispatchThreadID)
{
    // Early-out makes the conservative direct dispatch (worst-case groups)
    // identical to the exactly-sized indirect dispatch.
    if (tid.x >= pc.sliceCount)
        return;
    uint lane = materialQueue[pc.sliceBase + tid.x];
    paintOut[lane] = pc.value;
}
