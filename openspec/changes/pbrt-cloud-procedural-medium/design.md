# Design — pbrt Procedural `cloud` Medium

## Context

`nanovdb-volume-rendering` built the free-standing-volume machinery: a `densityAt`/`mediumMajorant`
seam dispatched on `MediumParams.kind`, the majorant/null-collision walk (`volume_walk.slang`), the
index-matched interface boundary, distant+env NEE, and the escape continuation. `MEDIUM_NANOVDB`
fills that seam with a 3D-texture density lookup.

pbrt's `clouds.pbrt` uses `MakeNamedMedium "c" "string type" "cloud"` — the built-in
`CloudMedium` (pbrt `media.h:430`), whose density is a **procedural analytic function** of the
medium-local point, no grid file. The exact pbrt density (`CloudMedium::Density`, `media.h:493`):

```
Density(p):                       # p in medium-local space, cube bounds [0,1]^3
  pp = frequency * p
  if wispiness > 0:               # domain warp (2 iters)
    vomega = 0.05*wispiness; vlambda = 10
    for i in 0..1: pp += vomega * DNoise(vlambda*pp); vomega*=0.5; vlambda*=1.99
  d = 0; omega = 0.5; lambda = 1  # fBm (5 octaves)
  for i in 0..4: d += omega * Noise(lambda*pp); omega*=0.5; lambda*=1.99
  d = clamp((1 - p.y) * 4.5 * density * d, 0, 1)   # altitude falloff
  d += 2 * max(0, 0.5 - p.y)
  return clamp(d, 0, 1)
```

`Noise`/`DNoise` are pbrt's **classic Perlin** (`util/noise.cpp`): a 256-entry `NoisePerm`
permutation table (duplicated to 512), quintic `NoiseWeight` fade `6t⁵−15t⁴+10t³`, and a `Grad`
gradient selector. `clouds.pbrt` uses `density 2` and pbrt defaults `wispiness 1`, `frequency 5`.

The same scene also uses `Material ""` (empty-string) on the `MediumInterface` shape — pbrt's
null/interface material — which the importer currently maps to grey `UsdPreviewSurface`.

## Goals / Non-Goals

**Goals**

- Render `clouds.pbrt` after `.pbrt → .usda`, comparable to pbrt v4 (dual gate: pbrt-truth +
  mega≡wave).
- Fill `MEDIUM_CLOUD` through the existing seam — **only** a new `densityAt` case + the ported
  noise; the walk, NEE, RR, majorant, boundary, integrator wiring unchanged.
- Bit-exact pbrt Perlin (`NoisePerm` + `Grad` + `NoiseWeight`) so density matches pbrt, not a
  look-alike fBm.
- `Material ""` routes to the interface/null boundary like `Material "interface"`.

**Non-Goals**

- Spectral σ (RGB only — same floor as the nanovdb scenes).
- Emissive clouds / `Lescale` blackbody term (pbrt `CloudMedium` has none; N/A).
- Other procedural media (`uniformgrid`/`rgbgrid` stay recorded skips).
- Matching pbrt's *spectral* Perlin-free σ sampling — RGB σ × scalar density is the model.
- Animated / time-varying clouds.

## Decisions

### D1 — Analytic density, no GPU resource

`MEDIUM_CLOUD` needs **no texture and no new binding** — the density is a pure function of the
world point transformed to medium-local space. The world→local affine reuses the SAME packed
`worldToUvw` rows `MEDIUM_NANOVDB` already carries in `FlatMaterialParams` (for the grid they map
world→[0,1]³ texel space; for the cloud they map world→[0,1]³ medium-local space — identical
semantics, so **no new fields**). The cloud-specific scalars (`density`, `wispiness`, `frequency`)
reuse spare medium slots: `frequency`→ reuse the grid `value_max`/majorant scalar slot, `density`
and `wispiness` fold into two currently-unused lanes of the medium pack (enumerated in tasks;
no stride growth if free lanes exist, else one float4 appended — the pack is already 240 B).

- *Why not a UsdVol.Volume with a "noise" field:* USD has no procedural fBm field schema skinny
  consumes; the params ride the interface material's `skinnyOverrides` (the channel every medium
  key already uses). The importer MAY still emit a `Volume` prim for scene-graph visibility but the
  renderer reads the params from the bound material, same as nanovdb.

### D2 — Port pbrt Perlin bit-exact

New `materials/subsurface/cloud_noise.slang`:
- `static const uint NOISE_PERM[512]` — pbrt's exact `NoisePerm` table (copied from
  `util/noise.cpp`, duplicated 256→512 as pbrt does).
- `float pbrtNoise(float3 p)` — classic Perlin: floor to lattice, quintic `noiseWeight`, `grad()`
  via the permutation, trilinear blend. Mirror pbrt `Noise(Point3f)` line-for-line.
- `float3 pbrtDNoise(float3 p)` — the analytic gradient pbrt `DNoise` returns (used by the
  wispiness warp).
- `float cloudDensity(float3 pLocal, float density, float wispiness, float frequency)` — the
  `CloudMedium::Density` body above.

Parity risk lives here: the perm table, the `Grad` sign/axis convention, and float vs double must
match pbrt. Mitigation: a CPU reference port (numpy, in the test) evaluates the SAME algorithm and
the Slang result is checked against it at a grid of points (D5) — and against pbrt's own `Noise`
if we expose it, but the numpy mirror of the identical constants is the practical oracle.

### D3 — Majorant

pbrt `CloudMedium::SampleRay` returns a **homogeneous** majorant = σ_t over the bounds (density
clamps to [0,1], so σ_t·1 bounds σ_t·density). Identical to the grid case: the packed σ_t IS the
global majorant. No macrocell needed. `mediumMajorant`'s `MEDIUM_CLOUD` case returns σ_a+σ_s, same
as the others.

### D4 — `Material ""` null boundary

`materials.py`/`api.py`: an empty-string pbrt material on a shape with a `MediumInterface` maps to
the SAME lobe-less encoding + `volume_interface` marker as `Material "interface"` (the living spec
already names "the implicit empty-material `MediumInterface` case" — this brings code to spec). A
shape with `Material ""` and **no** MediumInterface keeps today's behavior (default material).

### D5 — Parity + sanity gates

- Corpus: add `clouds` (pbrt `file` = `clouds/clouds.pbrt`), 256², spp tuned; ref via pinned pbrt
  v4. Dual gate: pbrt-truth (record measured baseline — expect the σ-cap/RGB floor) + mega≡wave.
- Noise unit gate (hostless): the Slang-ported constants mirrored in numpy, checked at ~10³ points
  vs the CPU port; asserts the fBm + altitude falloff matches at representative points.
- Zero-`density` cloud ≡ empty (invisible boundary), reusing the nanovdb zero-density scenario.

## Risks / Trade-offs

- [pbrt Perlin not bit-exact → density looks close but gate fails on structure] → port the exact
  `NoisePerm`/`Grad`/`NoiseWeight`; validate against a numpy mirror at many points before rendering.
- [Reusing spare medium pack lanes collides with a field] → audit the 240 B `FlatMaterialParams`
  layout; if no free lanes, append one float4 (pack already grew once, precedent exists) — never
  overlap a live field.
- [`Material ""` change captures non-medium empty materials] → gate strictly on the shape ALSO
  having a `MediumInterface`; existing corpus is the regression net.
- [wispiness domain warp cost (2× DNoise = 6 Perlin evals) per density sample × many null
  collisions] → clouds.pbrt `wispiness 1` is modest; the walk's majorant keeps sample counts
  bounded; Metal watchdog caps already apply. Revisit with a cheaper noise only if soak trips.

## Open Questions

- ~~Confirm the free medium-pack lanes vs appending a float4 (read the exact 240 B layout first).~~
  **ANSWERED (1.1):** the 240 B layout has exactly two free lanes (`_normalBiasPad.w` @124,
  `_specularColorPad.w` @156) but the cloud needs three scalars, and `frequency` cannot fold into
  the world→local affine (the altitude falloff reads unwarped `p.y` while the noise reads
  `frequency·p` — two different spaces). Also the assumed "grid `value_max` scalar slot" does not
  exist in the struct (value_max folds into σ at pack time). Decision: **append one float4
  `_cloudDensityWispinessFrequency`** at 240..256 (xyz = density/wispiness/frequency, w = pad) —
  stride 240→256, prefix byte-identical, zeros for every non-cloud material (precedent: the
  192→240 worldToUvw growth). The MSL stride pin (`tests/test_metal_flat_material_layout.py`) is
  updated in the same commit.
- ~~Does `clouds.pbrt`'s sphere want the same `Translate .5 .5 .5` + radius-1 bounds mapped so
  medium-local `p ∈ [0,1]³` aligns with pbrt's `bounds`?~~ **ANSWERED (1.1 recon):** the medium's
  CTM is captured at `MakeNamedMedium` (identity in `clouds.pbrt` — declared before the
  `Translate`), and pbrt clips cloud transport to the medium `bounds` (default `[0,1]³`) in
  medium-local space — so the importer packs `world→medium-local = ctm⁻¹ @ B` (12 floats on
  `skinnyOverrides["volume_world_to_uvw"]`, same row convention as the grid) and the shader's
  `MEDIUM_CLOUD` `densityAt` returns 0 outside `[0,1]³`, mirroring both pbrt's bounds clip and
  the grid case's outside-AABB zero. Non-default authored `p0`/`p1` stay an unsupported skip
  (recorded honestly; `clouds.pbrt` uses the defaults). The density formula reads medium-local `p`
  (≡ uvw for default bounds), so `p.y∈[0,1]` over the cube and the falloff reads exactly as pbrt.
