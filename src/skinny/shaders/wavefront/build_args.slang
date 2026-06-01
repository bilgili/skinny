// Build-args kernel for the wavefront shade stage (P1 §P1-C). Turns the
// per-material hit counts produced by intersect into:
//   (a) materialOffset — exclusive prefix sum of the counts, the base index
//       each material's lanes occupy in the counting-sort materialQueue, and
//   (b) indirectArgs — one VkDispatchIndirectCommand (x,y,z group counts) per
//       material, so each shade_<mid> dispatches exactly ceil(count/GROUP)
//       groups and empty materials dispatch nothing.
//
// numMaterials is small (tens), so a single-thread sequential scan is fine;
// revisit with a parallel scan only if it ever shows up in a profile.

// Group count for a stage dispatch covering `laneCount` lanes at `groupSize`
// lanes per threadgroup — ceil division. Shared by the kernel and its tests so
// there is one definition of the rounding rule.
uint wfIndirectGroupCount(uint laneCount, uint groupSize)
{
    return (laneCount + groupSize - 1u) / groupSize;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void buildArgs(
    uint3 tid: SV_DispatchThreadID,
    uniform StructuredBuffer<uint> materialCount,    // [numMaterials]
    uniform RWStructuredBuffer<uint> materialOffset, // [numMaterials] exclusive prefix sum
    uniform RWStructuredBuffer<uint> indirectArgs,   // [numMaterials*3] (x,y,z)
    uniform uint numMaterials,
    uniform uint groupSize)
{
    uint running = 0u;
    for (uint m = 0u; m < numMaterials; ++m)
    {
        materialOffset[m] = running;
        uint c = materialCount[m];
        running += c;
        indirectArgs[m * 3u + 0u] = wfIndirectGroupCount(c, groupSize);
        indirectArgs[m * 3u + 1u] = 1u;
        indirectArgs[m * 3u + 2u] = 1u;
    }
}
