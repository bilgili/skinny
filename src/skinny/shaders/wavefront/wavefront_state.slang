// Wavefront path-state record (P1 §P1-A). One element per lane in the active
// stream. Scalar layout (-fvk-use-scalar-layout): float3 = 12 B, tightly
// packed → 68 B stride. Python mirror: skinny.wavefront_layout
// (tests/test_wavefront_state.py locks the two together — field order + size).
//
// Lifetime: generate() seeds it; intersect() / shade_<mid>() / logic() mutate
// it across the bounce loop; logic() flushes `radiance` into the accumulation
// image at `pixelIndex` when a lane terminates. AoS first (Decision §P1-A);
// convert hot fields to SoA only if profiling shows a bandwidth bottleneck.

// `flags` bits — mirror skinny.wavefront_layout.PATH_FLAG_*.
static const uint PATH_FLAG_ALIVE    = 1u << 0;  // lane still bouncing
static const uint PATH_FLAG_SPECULAR = 1u << 1;  // last bounce specular → skip NEE-MIS

struct WavefrontPathState
{
    float3 rayOrigin;   // 0..12   current bounce ray origin
    float3 rayDir;      // 12..24  current bounce ray direction
    float3 throughput;  // 24..36  path throughput (beta)
    float3 radiance;    // 36..48  accumulated L for this path's pixel
    uint   pixelIndex;  // 48..52  flattened target pixel (accumulation)
    uint   rngState;    // 52..56  PCG-32 state (matches common.slang Rng.state)
    uint   depth;       // 56..60  current bounce count
    uint   flags;       // 60..64  PATH_FLAG_* bits
    float  bsdfPdf;     // 64..68  pdf of the last BSDF sample (MIS vs NEE)
};

bool pathAlive(WavefrontPathState s)    { return (s.flags & PATH_FLAG_ALIVE) != 0u; }
bool pathSpecular(WavefrontPathState s) { return (s.flags & PATH_FLAG_SPECULAR) != 0u; }
