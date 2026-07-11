// Wavefront path-state record (P1 §P1-A). One element per lane in the active
// stream. Scalar layout (-fvk-use-scalar-layout): float3 = 12 B, tightly
// packed → 68 B stride. Python mirror: skinny.wavefront_layout
// (tests/test_wavefront_state.py locks the two together — field order + size).
//
// Lifetime: generate() seeds it; intersect() / shade_<mid>() / logic() mutate
// it across the bounce loop; logic() flushes `radiance` into the accumulation
// image at `pixelIndex` when a lane terminates. AoS first (Decision §P1-A);
// convert hot fields to SoA only if profiling shows a bandwidth bottleneck.

import common;          // Spectrum typealias (float3 RGB / float4 spectral)
#if defined(SKINNY_SPECTRAL)
import spectrum;        // SampledWavelengths (spectral variant only)
#endif

// `flags` bits — mirror skinny.wavefront_layout.PATH_FLAG_*.
static const uint PATH_FLAG_ALIVE    = 1u << 0;  // lane still bouncing
static const uint PATH_FLAG_SPECULAR = 1u << 1;  // last bounce specular → skip NEE-MIS

// Color roles carry `Spectrum` (= float3 in the RGB build → byte-identical layout;
// = float4 under SKINNY_SPECTRAL). The hero wavelengths `sw` are appended only in
// the spectral variant so the RGB stride is unchanged. Python mirror:
// skinny.wavefront_layout (both variants).
struct WavefrontPathState
{
    float3   rayOrigin;   // 0..12   current bounce ray origin
    float3   rayDir;      // 12..24  current bounce ray direction
    Spectrum throughput;  // path throughput (beta)
    Spectrum radiance;    // accumulated L for this path's pixel
    uint     pixelIndex;  // flattened target pixel (accumulation)
    uint     rngState;    // PCG-32 state (matches common.slang Rng.state)
    uint     depth;       // current bounce count
    uint     flags;       // PATH_FLAG_* bits
    float    bsdfPdf;     // pdf of the last BSDF sample (MIS vs NEE)
#if defined(SKINNY_SPECTRAL)
    SampledWavelengths sw;  // hero wavelengths + pdfs, fixed for this path's life
#endif
};

bool pathAlive(WavefrontPathState s)    { return (s.flags & PATH_FLAG_ALIVE) != 0u; }
bool pathSpecular(WavefrontPathState s) { return (s.flags & PATH_FLAG_SPECULAR) != 0u; }
