// Counting-sort scatter for the wavefront shade stage (P1 §P1-B).
//
// After intersect tallies hits per material and build_args prefix-sums those
// tallies into per-material base offsets (`materialOffset`), this places each
// surviving lane's index into its material's contiguous slice of
// `materialQueue`. Each lane claims its slot with an atomic bump of the
// material's write cursor, so a material's slice ends up holding exactly its
// lanes (order within a slice is unspecified — the atomics race, which is fine:
// the shade kernel processes the whole slice regardless of order).
//
// `writeCursor` must be zero-initialised before dispatch (one counter per
// material); after the dispatch it equals each material's lane count.

[shader("compute")]
[numthreads(64, 1, 1)]
void scatterByMaterial(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<uint> laneMaterial,    // [numLanes] material id per lane
    uniform StructuredBuffer<uint> materialOffset,  // [numMaterials] slice base index
    uniform RWStructuredBuffer<uint> writeCursor,   // [numMaterials] zeroed; ends = counts
    uniform RWStructuredBuffer<uint> materialQueue, // [numLanes] out: lane indices, grouped
    uniform uint numLanes)
{
    uint lane = tid.x;
    if (lane >= numLanes)
        return;

    uint mid = laneMaterial[lane];
    uint local;
    InterlockedAdd(writeCursor[mid], 1u, local);
    materialQueue[materialOffset[mid] + local] = lane;
}
