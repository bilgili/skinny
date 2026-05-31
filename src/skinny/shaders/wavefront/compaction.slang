// Stream compaction for the wavefront bounce loop (P1 §P1-E).
//
// After `logic` clears the alive bit on terminated lanes, the survivors must be
// gathered into a dense queue so the next bounce's intersect dispatches only
// live lanes (dead lanes would otherwise waste occupancy and the live count
// would never shrink). Each surviving lane claims a slot with an atomic bump of
// the output cursor; ordering is unspecified (the atomics race) and irrelevant
// — the next intersect processes the whole `[0, outCount)` range.
//
// `outCount` is a single-element buffer that must be zero-initialised before
// dispatch; after the dispatch it holds the survivor count.

[shader("compute")]
[numthreads(64, 1, 1)]
void compactAlive(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<uint> aliveFlag,   // [numLanes] 1 = survivor, 0 = dead
    uniform RWStructuredBuffer<uint> outQueue,  // [>= survivor count] compacted lane indices
    uniform RWStructuredBuffer<uint> outCount,  // [1] zeroed; ends = survivor count
    uniform uint numLanes)
{
    uint lane = tid.x;
    if (lane >= numLanes)
        return;
    if (aliveFlag[lane] == 0u)
        return;

    uint slot;
    InterlockedAdd(outCount[0], 1u, slot);
    outQueue[slot] = lane;
}
