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

// Per-pixel visible point: this-pass eye geometry + per-pass direct radiance +
// the EVALUATED flat BSDF, fused with the persistent progressive estimator.
// RGB: scalar = 180 B, MSL = 240 B. Spectral (SKINNY_SPECTRAL): beta/ld widen
// float3→float4 (+8 B scalar) and conductorMetalId adds +4 B; tau stays float3
// (D5). All fields are scalar (no nested struct) so the host layout mirror +
// parse-lock test stay simple, and so sppm_state.slang need not import the
// material tree. wavefront_layout._visible_point_fields(spectral) mirrors both.
//
// `ld` holds the FULL per-pass direct term (NEE at the visible point plus any
// emitter/env reached through the eye specular chain) — computed once in the
// eye stage so direct is added at exactly ONE site (the update stage reads it
// and never re-traces NEE), keeping SPPM's photon term disjoint from direct.
//
// The albedo..diffuseRoughness block mirrors FlatHitMat (minus emission, which
// is direct not BRDF) + F0, so the photon deposit rebuilds the FlatMaterial and
// evaluates f_r at the visible point with no texture refetch, graph re-run, or
// double normal-map (pbrt's store-the-BSDF approach). EVERY FlatHitMat field
// the lobe model reads must have a slot here: a field added to FlatHitMat but
// not stored/rebuilt feeds undefined values into the deposit-time evaluate()
// (the Stage-2 rich inputs did exactly that — zeroed photon flux scene-wide,
// change fix-sppm-bathroom-black-walls).
struct VisiblePoint
{
    float3 pos;            // world hit position (rewritten each pass)
    float3 ns;             // world shading normal, post normal-map (rewritten each pass)
    float3 wo;             // world outgoing dir toward the camera (this pass)
    Spectrum beta;         // eye throughput reaching the point, EXCLUDING the VP BSDF (this pass; per-pass spectral)
    Spectrum ld;           // per-pass direct radiance (NEE + specular-chain emitters; per-pass spectral)
    float3 albedo;         // FlatHitMat.albedo
    float3 F0;             // FlatMaterial.F0
    float3 coatColor;      // FlatHitMat.coatColor
    float  roughness;      // FlatHitMat.roughness
    float  metallic;       // FlatHitMat.metallic
    float  specular;       // FlatHitMat.specular
    float  ior;            // FlatHitMat.ior
    float  opacity;        // FlatHitMat.opacity (scales the BSDF response)
    float  coat;           // FlatHitMat.coat
    float  coatRoughness;  // FlatHitMat.coatRoughness
    float  coatIOR;        // FlatHitMat.coatIOR
    float3 transmissionColor;  // FlatHitMat.transmissionColor (Stage-2 rich input)
    float3 specularColor;      // FlatHitMat.specularColor (Stage-2 rich input)
    float  diffuseRoughness;   // FlatHitMat.diffuseRoughness (Stage-2 rich input)
#if defined(SKINNY_SPECTRAL)
    // Named-conductor id (common.slang FlatMaterialParams.conductorMetalId),
    // stored so the photon deposit's per-λ recolor (upsampleFlatColors) uses the
    // EXACT conductor Fresnel (6.2) — not upsampled Schlick(F0) — for a rough
    // named-metal visible point, matching the megakernel spectral path.
    uint   conductorMetalId;
#endif
    // D5: tau is a spectral-INVARIANT progressive quantity — the per-pass per-λ
    // flux is resolved (spectrumResolveToLinearSRGB) BEFORE it folds here, so this
    // stays float3 in BOTH builds (never widened to Spectrum). Rotating the pass λ
    // across passes integrates the spectrum via the accumulation-film running mean.
    float3 tau;            // accumulated reflected flux (persists; resolved 3-wide)
    uint   flags;          // bit0 = VP_ACTIVE
    float  radius;         // current search radius r_i (persists)
    float  n;              // accumulated photon count N_i (persists; float for gamma)
};

// Per-pass photon-deposit accumulator (separate buffer, atomically written by
// the photon stage). RGB: 3 fixed-point flux channels + m (16 B). Spectral
// (SKINNY_SPECTRAL): 4 hero-λ flux channels + m (20 B) — one per shared pass
// wavelength (D5). Flux is non-negative per λ so the unsigned fixed point is
// clean (no signed-gamut concern — contrast the BDPT splat). The RGB layout is
// byte-identical (phiR/G/B → phi0/1/2 is a name-only change).
struct SppmAccum
{
    uint phiR;  // fixed-point accumulated flux — RGB red / spectral hero λ0
    uint phiG;  // RGB green / spectral λ1
    uint phiB;  // RGB blue  / spectral λ2
#if defined(SKINNY_SPECTRAL)
    uint phiW;  // spectral λ3 (RGBA hero-λ slot; absent in the RGB build so the
                // RGB SPIR-V — member names included — is byte-identical)
#endif
    uint m;     // photons deposited into this point this pass (M_i)
};

// Decode the fixed-point per-pass flux accumulator back to linear float flux
// (Spectrum: float4 spectral / float3 RGB).
Spectrum sppmDecodeFlux(SppmAccum a)
{
    let inv = 1.0 / SPPM_FLUX_FIXED_SCALE;
#if defined(SKINNY_SPECTRAL)
    return Spectrum(float(a.phiR), float(a.phiG), float(a.phiB), float(a.phiW)) * inv;
#else
    return float3(float(a.phiR), float(a.phiG), float(a.phiB)) * inv;
#endif
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
