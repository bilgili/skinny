# Proposal: Faithful pbrt named-spectrum import and round-trip

## Why

pbrt-v4 scenes address ~30 built-in spectra by name — `"spectrum eta" "glass-BK7"`
on a dielectric, `"spectrum eta" "metal-CuZn-eta"` on a conductor,
`"spectrum L" "stdillum-A"` on a light. skinny's importer recognises only a
fraction of them, and the unrecognised ones fail **silently** rather than loudly:

* **Named glasses**: only `bk7` is known. Every other name (`glass-LASF9`,
  `glass-F11`, …) normalises to a `"default"` key that is the *BK7* Cauchy fit, so
  LASF9 (n≈1.85) renders as n≈1.50 with BK7's dispersion — a wrong material with no
  warning. The scalar RGB-mode IOR degrades to the generic dielectric default too.
* **Named metals**: only Ag/Al/Au/Cu are vendored. `metal-CuZn-eta` (brass),
  `metal-MgO-eta`, `metal-TiO2-eta` resolve to nothing, so the conductor falls back
  to **copper**.
* **Named illuminants**: `stdillum-A`/`D50`/`D65`/`F1`–`F12` are handled nowhere.
  The RGB reduction returns the default (white), and although the importer preserves
  the name as a `spectrum_named` payload, `usd_loader._extract_light_spd` accepts
  only `spectrum_samples` and drops it — a tungsten `stdillum-A` light renders neutral.
* **Inline spectra on materials**: `material_spectral_overrides` preserves only
  *named* identities, so an authored `"spectrum reflectance" [400 0.1 700 0.9]` on a
  material loses its SPD (lights already preserve theirs).

Scenes like *transparent-machines* lean on these names for their look. The plumbing
(the `skinnyOverrides` side-channel, the GPU's named-conductor Fresnel at binding 48,
authored illuminant SPDs at binding 50, hero-λ glass dispersion) already exists and is
unchanged by this work — the gap is data coverage and import resolution.

## What Changes

* **Vendor pbrt's full named-spectrum tables** verbatim via the existing
  `_extract_pbrt_spectra.py` dev tool run against the pinned `~/projects/pbrt-v4`:
  all 7 named glasses, all 7 named metals (eta+k), and the 16 scene-addressable illuminant
  SPDs (15 `stdillum-*` + `illum-acesD60`).
* **Named glasses get their own dispersion.** Per-glass Cauchy `(A, B)` coefficients
  fit from pbrt's tabulated eta curves, plus a per-glass scalar IOR at the sodium
  d-line (589.3 nm) for RGB mode. The GPU packing (`glass_dispersion` → `A`, `B`)
  is unchanged.
* **Named metals extended** to CuZn/MgO/TiO2 — new canonical keys, RGB
  normal-incidence reflectance, and vendored eta/k curves for spectral mode.
* **Named illuminants resolve** to an SPD: the RGB reduction gets the correct
  chromaticity (luminance-normalised, as blackbody already does), and
  `_extract_light_spd` learns to resolve a `spectrum_named` illuminant payload to
  the vendored 95-sample SPD it already binds at binding 50.
* **Inline material spectra: cut.** Nothing consumes a material SPD (`skinnyOverrides
  ["spectral"]` is read only for distant *light* prims), so preserving it would author a
  dead override that reads as a working feature. Lights keep their existing preservation.
  Real per-material reflectance SPDs need a loader field + packer + binding + shader change
  — its own change.
* **Unknown names stop failing silently.** An unrecognised `glass-*` / `metal-*` /
  `stdillum-*` name records an APPROX note in the import report naming the substituted
  fallback, instead of quietly rendering a different material.
* **Round-trip**: the authored identity survives import → `.usda` → render on both the
  plain-USD and `-mtlx` paths (both already write `skinnyOverrides`; this widens *what*
  is written, adds no new channel). No USD→pbrt writer is introduced.

Not breaking: an RGB-only scene authors no new override and imports byte-identically.
`glass-BK7` and the four existing metals keep their current keys and curves.

## Capabilities

### New Capabilities

None — this deepens existing spectrum conversion rather than adding a capability.

### Modified Capabilities

- `pbrt-spectrum-conversion`: the **Vendored spectral data tables** requirement
  broadens from "named metals + D65 + glass IOR fits" to pbrt's complete named-spectrum
  set (7 glasses with per-glass dispersion, 7 metals, 16 illuminants); the
  **Spectral payloads are preserved alongside the RGB reduction** requirement extends to
  named illuminants, and is narrowed so no payload is authored where no consumer exists
  (inline spectra on *materials*); new requirements cover
  named-illuminant RGB resolution, per-glass IOR/dispersion resolution, and explicit
  unknown-name reporting.

## Impact

* `src/skinny/pbrt/data/_extract_pbrt_spectra.py` — extract glasses, all metals, stdillums.
* `src/skinny/pbrt/data/spectral_curves.npz` — regenerated (additive; existing arrays
  unchanged).
* `src/skinny/pbrt/data/spectral_tables.py` — per-glass Cauchy table replaces the
  2-entry `_GLASS_CAUCHY`; `named_illuminant_spectrum()` added.
* `src/skinny/pbrt/spectra.py` — `named_glass_key` / `named_conductor_key` widened,
  unknown-name signalling, named-illuminant reduction in `param_to_rgb`.
* `src/skinny/pbrt/materials.py` — per-glass scalar IOR (IOR-bearing params only);
  `conductor.eta` read on `coatedconductor`.
* `src/skinny/pbrt/lights.py`, `src/skinny/usd_loader.py` — named-illuminant payload
  resolution to an SPD.
* `src/skinny/renderer.py` — the four-metal set is hard-coded twice and both must grow in
  lockstep: `_CONDUCTOR_METAL_ID` (line 570) and the `spectralMetals` upload loop (line
  4004), whose order **is** the shader's `(metalId-1)` index. Plus the now-stale
  "spectral-only" named-glass comment at lines 688-690.
* Docs: `docs/Spectral.md` (named-spectrum coverage table), `CHANGELOG.md`.
* `src/skinny/shaders/bindings.slang` + `integrators/spectral_flat_common.slang` — the
  named-conductor gate was hard-coded `metalId <= 4u`, which would have silently dropped
  CuZn/MgO/TiO2 to RGB Schlick; it now reads a `SPECTRAL_METAL_COUNT` constant pinned to the
  host upload by a test. **Spectral build only — the RGB SPIR-V is verified byte-identical
  to main.**
* Unchanged: all descriptor bindings, the `FlatMaterialParams` layout, and the import of any
  scene that authors no named spectra.
* **Changed on purpose**: named-glass scenes shift in **both** RGB and spectral modes — RGB
  more than spectral, because a named glass currently renders at the generic `eta` default
  of 1.5 (LASF9 should be 1.85). Baselines must be re-measured in both modes.
