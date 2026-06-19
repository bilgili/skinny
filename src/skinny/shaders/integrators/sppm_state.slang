// SPPM per-pixel state (change photon-mapping-sppm, PM-1).
//
// Two GPU-internal scratch structs, one element per pixel, sized host-side by
// skinny.wavefront_layout (VISIBLE_POINT_* / SPPM_ACCUM_*) but never packed by
// the host — only allocated, like the wavefront records (wf_records.slang).
//
// SPPM runs one stochastic pass per progressive-accumulation frame: the eye
// stage rewrites the visible-point GEOMETRY each pass, while the progressive
// estimator state (radius r, photon count N, accumulated flux tau) PERSISTS and
// is reduced once per pass by the update stage. Splitting the per-pass atomic
// deposit into a separate SppmAccum buffer keeps the deposit portable (uint
// fixed-point InterlockedAdd — neither Vulkan-core nor Metal guarantees fp
// atomics) and keeps the hot VisiblePoint struct free of atomically-mutated
// fields.
//
// Scalar layout (Vulkan, -fvk-use-scalar-layout): float3 = 12 B @ 4-byte align,
// tightly packed. MSL (Metal) pads float3 to 16 B, growing the stride.
// tests/test_sppm_state.py locks both struct layouts to these declarations.

import common;  // pcgHash for the spatial-hash helpers

// VisiblePoint.flags bits (mirror skinny.wavefront_layout.VP_ACTIVE).
static const uint VP_ACTIVE = 1u;  // a valid visible point was stored this pass

// Fixed-point scale for the atomic flux deposit. Photon flux is multiplied by
// this before the uint InterlockedAdd in the photon stage and divided back out
// in the update stage. 2^20 keeps ~6 significant decimal digits for the
// sub-unit flux magnitudes SPPM accumulates; revisit if HDR emitters overflow.
static const float SPPM_FLUX_FIXED_SCALE = 1048576.0;

// Per-pixel visible point: this-pass eye geometry + per-pass direct radiance
// fused with the persistent progressive estimator. Scalar = 88 B, MSL = 112 B.
//
// `ld` holds the FULL per-pass direct term (NEE at the visible point plus any
// emitter/env reached through the eye specular chain) — computed once in the
// eye stage so direct is added at exactly ONE site (the update stage reads it
// and never re-traces NEE), keeping SPPM's photon term disjoint from direct.
struct VisiblePoint
{
    float3 pos;         // 0..12   world hit position (rewritten each pass)
    float3 ns;          // 12..24  world shading normal (rewritten each pass)
    float3 beta;        // 24..36  eye throughput reaching the point, EXCLUDING the VP BSDF (this pass)
    float3 wo;          // 36..48  world outgoing dir toward the camera (this pass)
    float3 ld;          // 48..60  per-pass direct radiance (NEE + specular-chain emitters)
    float3 tau;         // 60..72  accumulated reflected flux (persists)
    uint   materialId;  // 72..76  flat material id at the hit
    uint   flags;       // 76..80  bit0 = VP_ACTIVE
    float  radius;      // 80..84  current search radius r_i (persists)
    float  n;           // 84..88  accumulated photon count N_i (persists; float for gamma)
};

// Per-pass photon-deposit accumulator (separate buffer, atomically written by
// the photon stage). Scalar = 16 B, MSL = 16 B.
struct SppmAccum
{
    uint phiR;  // 0..4   fixed-point accumulated flux, red
    uint phiG;  // 4..8   green
    uint phiB;  // 8..12  blue
    uint m;     // 12..16 photons deposited into this point this pass (M_i)
};

// Decode the fixed-point per-pass flux accumulator back to linear float flux.
float3 sppmDecodeFlux(SppmAccum a)
{
    let inv = 1.0 / SPPM_FLUX_FIXED_SCALE;
    return float3(float(a.phiR), float(a.phiG), float(a.phiB)) * inv;
}

// SPPM progressive reduction for one pass (Hachisuka & Jensen 2009):
//   N' = N + gamma * M
//   r' = r * sqrt(N' / (N + M))
//   tau' = (tau + Phi) * (r'/r)^2   where (r'/r)^2 = N'/(N+M)
// `phi` is the decoded per-pass flux, `m` the per-pass photon count, `gamma` the
// reduction parameter (2/3 is the standard choice). A pass with no photons
// leaves the estimator untouched.
void sppmUpdate(inout VisiblePoint vp, float3 phi, uint m, float gamma)
{
    if (m == 0u)
        return;
    let nOld = vp.n;
    let mf = float(m);
    let nNew = nOld + gamma * mf;
    let ratio = nNew / (nOld + mf);  // = (r'/r)^2, <= 1
    vp.tau = (vp.tau + phi) * ratio;
    vp.radius = vp.radius * sqrt(ratio);
    vp.n = nNew;
}

// ── Spatial-hash grid (shared by the eye/grid/photon SPPM kernels) ─────────
// Cell count = next power of two >= 2*num_pixels, so the hash masks with
// `& (numCells - 1)`. MUST match skinny.wavefront_layout.sppm_grid_cell_count.
uint sppmGridCellCount(uint numPixels)
{
    let target = max(1u, 2u * numPixels);
    uint cells = 1u;
    [loop] while (cells < target)
        cells <<= 1;
    return cells;
}

// Integer cell coordinate of a world position. cellSize is fixed at the initial
// search radius (a valid upper bound on every active VP's current radius), so a
// 3x3x3 neighbour scan never misses a photon.
int3 sppmCellCoord(float3 pos, float3 boundsMin, float cellSize)
{
    return int3(floor((pos - boundsMin) / max(cellSize, 1e-8)));
}

// Hash an (possibly offset) integer cell coordinate into [0, numCells). The
// photon stage reuses this for the 3x3x3 neighbour cells (offset coords), so it
// must be deterministic and offset-stable.
uint sppmCellHash(int3 c, uint numCells)
{
    let h = pcgHash(uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u);
    return h & (numCells - 1u);
}
