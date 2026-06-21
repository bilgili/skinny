## Why

A pbrt scene with a `Material "subsurface"` (e.g. `sssdragon`'s Skin1 dragon)
imports into skinny and renders as **clear glass**, not the soft, milky
translucent solid pbrt produces. Side-by-side vs a pbrt v4 reference: the camera
orientation now matches (after `pbrt-camera-up-axis`), but the dragon is sharp and
refractive where pbrt is waxy and light-diffusing — the subsurface look is absent.

Root cause (verified): pbrt `subsurface` is lowered to the **flat** material with
`opacity = 0` (`map_material` / `_derive_opacity_from_subsurface`), which only
opens the flat path's **delta-refraction** branch — a dielectric boundary with no
interior transport. The richer params *are* carried — `map_material_mtlx` emits
`subsurface`, `subsurface_color`, `subsurface_radius`, and the `-mtlx`/std_surface
packer puts them in binding 19 — but they are **dead** in the production estimator
(the `mx_subsurface_bsdf` closure in `mtlx_std_surface.slang` is part of the
preview-only `evalStdSurfaceBSDF`, which `flat-bsdf-lobes` forbids in PT/BDPT).
There is no interior medium and no `MATERIAL_TYPE_SUBSURFACE`.

This is **Ch5** of the pbrt-mtlx Stage-2 roadmap — the research-grade item. pbrt-v4
`subsurface` is a **random walk in a homogeneous interior medium** behind a
dielectric boundary, so the faithful, parity-targeting fix is the same: refract
through the boundary, then delta-track (Woodcock) a volumetric random walk with
`σ_a` / `σ_s` / Henyey-Greenstein `g` until the path exits. skinny already has the
two load-bearing pieces — `volume_render.slang` (delta tracking + HG phase/
sampling) and the dielectric Fresnel boundary in `flat_shading.slang` — but they
are not wired into a per-instance interior medium or the bounce loop.

## What Changes

- A pbrt `subsurface` material SHALL import to a new **`MATERIAL_TYPE_SUBSURFACE`**
  (not flat-with-opacity-0): a dielectric boundary (`eta`) plus a homogeneous
  **interior medium** (`σ_a`, `σ_s`, HG `g`), carried per material/instance.
- The importer SHALL derive the medium coefficients from the pbrt inputs, in
  pbrt's own precedence: explicit `sigma_a`/`sigma_s` → named preset (`Skin1`, …,
  pbrt's measured table) → `reflectance` + `mfp` via the classical albedo
  inversion. The std_surface `subsurface_color` (single-scatter albedo) +
  `subsurface_radius` (per-channel mean free path) + `subsurface_scale` SHALL map
  to the same `(σ_a, σ_s)` so the `-mtlx` and native paths agree.
- The estimator SHALL transport light through the interior as a **volumetric
  random walk**: refract at the boundary (existing dielectric Fresnel), then
  delta-track through the medium (`volume_render.slang`) with HG scattering and
  per-channel `σ_t`, refracting again on exit. Single pdf, bounded throughput
  (no clamp), unbiased — preserving the integrator invariants the path/BDPT/
  ReSTIR estimators depend on.
- It SHALL run in **both execution modes** (megakernel + wavefront) and on **both
  backends** (Vulkan + native Metal), like the skin BSSRDF path.
- Back-compat: non-subsurface materials SHALL be byte-unchanged; the flat
  opacity/refraction path is untouched for true glass. The pbrt parity corpus is
  unchanged.
- Verification: the reduced `sssdragon` (subsurface Skin1) SHALL converge to a
  milky, light-diffusing dragon matching a pbrt v4 reference within a scene
  tolerance (relMSE/FLIP) — replacing today's glass look — with a furnace /
  energy-conservation check on a homogeneous SSS sphere.

## Non-Goals (separate follow-ups)

- **Reusing the skin diffusion BSSRDF** (`skin_bssrdf.slang` Burley profile) as a
  fast separable approximation — a cheaper alternative path, not the parity
  target; may be added later behind a quality switch.
- **Heterogeneous / textured media** (e.g. the `disney-cloud` NanoVDB grid),
  free-standing `MediumInterface` volumes, measured BSSRDF tables beyond pbrt's
  named presets, and spectral (wavelength-dependent) σ — constant per-channel RGB,
  dielectric-bounded interior only **in this change**. But the volume
  architecture SHALL be chosen to support these without rework: the transport is
  majorant/null-collision (Woodcock) tracking (the heterogeneous algorithm, with
  homogeneous as its degenerate case), and the medium is a handle-referenced
  registry entry (not hardwired to a material's interior), so the Disney cloud
  model — heterogeneous, free-standing, index-matched, HG — drops in later as a
  separate change reusing this walk. See design.md → Forward compatibility.
- **Nested / overlapping media** and exterior participating media (fog) — only a
  single closed dielectric-bounded interior per object.
- **Env-light intensity / exposure calibration** (the ~6–7× brightness gap seen
  in the sssdragon A/B) — a separate import-scaling matter, independent of SSS.
- **Anisotropic boundary roughness** (rough dielectric SSS boundary) — depends on
  the rough-glass BTDF (Ch4b) and is gated behind it.
