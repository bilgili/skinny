# Volumetric Subsurface Scattering

pbrt's `Material "subsurface"` (e.g. the `sssdragon`'s `Skin1` dragon) is a
**random walk in a homogeneous interior medium behind a smooth dielectric
boundary**. skinny reproduces that as a dedicated material type
(`MATERIAL_TYPE_SUBSURFACE`, type code 4) rather than the separable diffusion
BSSRDF the skin path uses. Before this change a pbrt `subsurface` material was
lowered to the flat material with `opacity = 0` and rendered as **clear glass**;
it now renders as a soft, light-diffusing (milky) translucent solid.

This is the physically-correct ground truth pbrt's tabulated dipole BSSRDF
approximates (they agree in the diffusive, high-albedo limit), so it tracks pbrt
closely but is **not bit-parity** with the dipole — see [Verification](#verification).

![subsurface random walk](diagrams/subsurface/walk.svg)

## The interior random walk

`materials/subsurface/subsurface_walk.slang` is self-integrating (returns full
radiance and terminates the bounce loop, like the skin path). At a hit it:

1. **Boundary (Fresnel split).** Reflectance `Fr = fresnelDielectric(N·V, 1/η)`.
   The reflected fraction `Fr` samples the environment in the mirror direction
   (surface specular); the refracted fraction `1−Fr` enters the interior with the
   ray bent by `refractInto`.
2. **Interior random walk — geometry depends on the execution mode** (see
   [The slab vs the 3-D walk](#the-slab-vs-the-3-d-walk) below). Each segment
   marches a homogeneous medium and either scatters (continue) or reaches a
   boundary (Fresnel-split escape/internal-reflection).
3. **Null-collision (Woodcock) tracking** per segment, in
   `traverseMediumSegment`: free-flight against the majorant `σ̄_t`, then accept a
   real scatter with probability `σ_t(p)/σ̄_t` (throughput `σ_s/σ_t`, a
   Henyey-Greenstein continuation) or a null collision. The medium is read **only
   through the density seam** (see below).
4. **Lighting.** Direct light from a single analytic distant light via per-scatter
   NEE (refracted out through the boundary, `Ft = 1 − Fresnel`); the **environment
   on escape** — when a segment leaves the medium the ray carries `throughput ·
   env(wExit) · Ft`, where `wExit` is the **Snell-refracted** exit direction
   (medium→vacuum, `refractInto` with `η = ior`), not the raw interior direction.
   Direct (NEE) and environment (escape) are disjoint sources — the distant light
   is a delta direction the env lookup never returns — so they never double-count
   and the walk is **energy-conserving** (furnace `σ_a → 0` returns ~unity).
5. **Russian roulette** bounds the walk; throughput is `float3` (per-channel σ),
   the pdf scalar.

It runs in **both execution modes** (megakernel + wavefront) and on **both
backends** (Vulkan + native Metal): a single `case MATERIAL_TYPE_SUBSURFACE` in
`integrators/path.slang` `evaluateBounce()` serves the megakernel **and** the
wavefront catch-all kernel, branching on the `SKINNY_WAVEFRONT` compile define.
BDPT excludes it (flat-only eye walk — non-flat first hits bail, so BDPT never
invokes the subsurface walk), exactly like skin; render subsurface with the path
tracer (megakernel or wavefront).

### The slab vs the 3-D walk

The interior segment length is found two different ways, gated by execution mode
(`SKINNY_WAVEFRONT`):

- **Wavefront → true 3-D walk** (`subsurfaceRadiance3D`). Each segment traces the
  **actual scene geometry** (`traceScene` from the scatter vertex) to find the
  real mesh boundary, so the path length follows the curved/complex surface. NEE
  is shadow-traced the same way — `traceScene` toward `lightDir` to the boundary,
  attenuated by `exp(−σ_t · dist)` over the in-medium distance × the exit Fresnel.
  This is the production path for large scenes (the sssdragon).
- **Megakernel → 1-D slab** (`subsurfaceRadiance`). The interior is a homogeneous
  slab of perpendicular thickness `T` (from `hit.backT`); `zMM` tracks the
  perpendicular depth and each step marches to whichever face the ray heads
  toward. The whole walk runs in **one dispatch** per pixel, where a per-segment
  BVH trace would risk the macOS GPU watchdog — so the megakernel keeps the cheap
  slab. (Large scenes OOM the megakernel first anyway; the slab is its safe path.)

**Why the slab is not enough on real geometry.** The slab models the medium as a
flat slab of perpendicular thickness, which over-estimates the path length on a
curved/complex mesh — paths travel too far through the blue-absorbing interior, so
the sssdragon renders darker and redder than pbrt. The 3-D walk follows the real
boundary, so it is brighter and closer to the reference (see
[Verification](#verification)).

**Bounce cap.** A high-albedo random walk needs ~`τ²` scatter events to diffuse
out of a thick medium, so energy rises steeply with the interior bounce cap
`SSS_MAX_BOUNCES`. The cap was swept on the furnace sphere and the 28.8 M-tri
sssdragon: the dragon is **cap-insensitive** (its medium is far thinner than the
furnace's `τ ≈ 20`; mean and wall-time are flat from 16→96 and never trip the
watchdog), while the furnace energy climbs from 0.84 (cap 32) to **0.97** (cap 96,
`τ ≈ 20`). So the wavefront path uses **cap 96** — near-free on real scenes yet
near-unity on the furnace. The megakernel slab stays at 16 (Metal) / 64 (Vulkan).

### The density seam (forward compatibility)

The walk reads the medium through exactly two functions in
`materials/subsurface/medium.slang`, each a `switch` on a `kind` tag:

- `densityAt(Medium m, p) → float` — local density multiplier (`1.0` for
  `MEDIUM_HOMOGENEOUS`).
- `mediumMajorant(Medium m, a, b) → float3` — majorant `σ̄_t` (`σ_a+σ_s` for
  homogeneous).

Because null-collision tracking **is** the heterogeneous algorithm (a constant
density is its degenerate case), a future NanoVDB grid is a new `kind` plus two
`case` bodies — the walk, NEE, RR, and integrator wiring are untouched. The
`Medium` is **handle-referenced** (`resolveMedium(matId)`), `MEDIUM_NANOVDB` is
reserved, and the boundary crossing is parameterized by mode (dielectric refract
vs index-matched) — so heterogeneous, free-standing `MediumInterface` media (e.g.
the pbrt `disney-cloud`) drop in additively. Those, plus area/emissive lights
*inside* the medium, are deliberate follow-ups.

## Coefficient derivation (importer)

The importer derives `(σ_a, σ_s, g, η)` in pbrt's own precedence
(`skinny/pbrt/subsurface.py`, `materials.py`):

1. explicit `sigma_a` + `sigma_s` (mm⁻¹) → used directly (× `scale`);
2. else a named preset (`"Skin1"`, …) → pbrt's measured
   `GetMediumScatteringProperties` table;
3. else `reflectance` + `mfp` → invert the diffuse albedo (Jensen/Christensen),
   `σ_t = 1/mfp`, `σ_s = α·σ_t`, `σ_a = σ_t − σ_s`.

The `-mtlx` / `standard_surface` inputs map to the **same** coefficients —
`subsurface_color` → albedo, `subsurface_radius` → per-channel mfp,
`subsurface_scale` → `1/mfp` scale, `subsurface_anisotropy` → `g` — so native-USD
and `-mtlx` imports agree.

**Unit storage.** pbrt media coefficients are mm⁻¹ interpreted *per scene unit*
(optical depth `τ = σ·L`). The walk, however, computes
`τ = σ_packed · L_world · mmPerUnit`, and an imported pbrt stage declares
`metersPerUnit = 1.0` (`emit.PBRT_STAGE_METERS_PER_UNIT`) → loader
`mm_per_unit = 1000`. So `media.subsurface_overrides` stores the coefficients
**per world unit**: the pbrt mm⁻¹ values divided by `mm_per_unit` (1000), so
`σ_packed · mmPerUnit` recovers pbrt's coefficients and the interior is at its
true geometric optical depth. Without this the dragon is ~1000× too dense and
renders opaque gold/brown instead of translucent. (Full pixel-mean parity with a
pbrt reference additionally needs the env-light application and high-optical-depth
walk fidelity — a separate follow-up.)

The coefficients ride on `skinnyOverrides` customData (`subsurface_sigma_a`,
`subsurface_sigma_s`, `subsurface_g`, and `ior` for the boundary η) → merged into
`Material.parameter_overrides`. The renderer packs them **inline** into
`FlatMaterialParams` (binding 13, bytes `σ_a`@160, `g`@172, `σ_s`@176,
`mediumKind`@188 — η reuses `ior`@60) so no new SSBO is needed under Metal's
31-buffer cap, and tags `MATERIAL_TYPE_SUBSURFACE` whenever `subsurface_sigma_*`
is non-zero. True `dielectric` glass (no `subsurface_sigma_*`) stays on the flat
path, untouched.

The loader's subsurface→opacity bridge (`_derive_opacity_from_subsurface`, which
drops `opacity` to `0` so the flat refraction branch fires for a pbrt subsurface
boundary) is gated on the **same** `subsurface_sigma_*` test
(`_has_subsurface_medium`, mirroring `_material_is_subsurface`). A plain Autodesk
`standard_surface` `subsurface` *weight* with no interior medium — e.g. the
`three_materials_demo` marble (`subsurface = 0.4`, no σ) — is a diffuse-SSS
shading term, so it keeps `opacity = 1` and renders as an opaque lit diffuse
surface. Without this gate it was forced to `opacity = 0` and refracted the
environment as clear glass (the "marble totally broken" regression).

## Verification

| Gate | Result |
|---|---|
| Furnace / energy, 3-D walk, cap 96 (`σ_a → 0`, white env → unity), `τ ≈ 20`, η = 1.0 / 1.2 / 1.5 | 0.974 / 0.960 / 0.911 |
| Distant-lit subsurface NEE (sphere, env off) — non-zero, isolated (light off → 0) | wired |
| PT ≡ BDPT (SSS sphere, megakernel slab — BDPT falls through to PathTracer) | relMSE 0.0000 |
| Metal ↔ Vulkan (wavefront 3-D walk, SSS sphere) | relMSE 0.000000 |
| Back-compat (`SKINNY_WAVEFRONT` gate has no effect off-subsurface — glass on vs off) | maxAbsDiff 0.0 (byte-identical) |
| sssdragon: 3-D walk **closer to pbrt than the slab** (exposure-matched) | relMSE 0.134 < 0.162; FLIP 0.067 < 0.071 |
| pbrt-v4 corpus `subsurface_infinite` (dipole vs walk, IBL-lit) | relMSE 0.079 |

pbrt's `subsurface` is a tabulated dipole BSSRDF and skinny's is a 3-D random
walk, so the corpus parity is qualitative (both milky), with a loose tolerance —
see `tests/pbrt/corpus/manifest.json`. On the sssdragon the 3-D walk is structurally
closer to pbrt than the slab (lower exposure-matched relMSE **and** FLIP); the
remaining ~2× absolute brightness (env-HDRI intensity + film ISO/exposure
calibration) is a separate, deferred follow-up — not a 3-D-walk correctness gap.
BDPT excludes subsurface in the **wavefront** path (non-flat first hits render
black — a pre-existing wavefront-BDPT limitation), so PT ≡ BDPT is gated in the
megakernel where BDPT falls through to the path tracer.
