## Why

A pbrt `Material "subsurface"` imports correctly (the `SUBSURFACE(4)` material
tag reaches the GPU on every render path — native-USD and `-mtlx`), but the
`sssdragon` scene renders as an **opaque gold/brown solid** instead of pbrt's
translucent milky look. The interior medium is ~1000× too dense.

Root cause is a unit mismatch. The pbrt importer declares
`SetStageMetersPerUnit(stage, 1.0)` (`emit.py`), so the loader derives
`mm_per_unit = metersPerUnit * 1000 = 1000` (`usd_loader.py`). The subsurface
walk computes optical depth as `τ = σ[mm⁻¹] · L_world · mm_per_unit`, so the
imported medium's optical depth is multiplied by an extra factor of 1000 that
pbrt never applies — pbrt interprets the named-medium coefficients (`Skin1`,
mm⁻¹) per **scene unit** (`τ = σ · L`). The dragon geometry is baked 1:1 with
pbrt's render space (~3 world units, transform scale 1.0), so the only
divergence is that spurious `mm_per_unit` factor.

The existing subsurface gates did not catch this because they forced
`mm_per_unit = 1.0` (probe / harness default), masking the production import
path's `mm_per_unit = 1000`.

This was confirmed empirically by re-rendering the dragon with `mm_per_unit`
forced down: 1000 → opaque (linear mean 0.039); 0.1 → translucent milky
(0.238). Geometry, material routing, and the SSS tag are all correct; only the
medium optical-density units are wrong.

## What Changes

- **Make the imported subsurface medium optical density unit-correct.** The
  importer pre-scales the packed medium coefficients (`subsurface_sigma_a`,
  `subsurface_sigma_s`) by `1 / mm_per_unit` so that the walk's
  `σ_packed · mm_per_unit` reproduces pbrt's per-scene-unit `σ` — i.e. the
  packed coefficients become per-world-unit and the spurious ×1000 cancels. No
  shader change, no new packed field; purely import-side, in
  `media.subsurface_overrides`.
- **Tie the divisor to the importer's declared stage unit** via a shared
  `emit.PBRT_STAGE_METERS_PER_UNIT` constant (× 1000 = the loader's resulting
  `mm_per_unit`), so the medium scale can never drift from the
  `SetStageMetersPerUnit` call.
- **Verify with brightness-independent gates** — a unit test on the scaled
  coefficients, an analytic optical-depth check (`τ = σ_t · L`), and the existing
  furnace energy-conservation gate (unaffected by density scaling). The dragon
  A/B confirms the medium is no longer at the catastrophic ×1000 density.

## Non-Goals (deferred follow-up)

- **Visual pixel-mean parity with the pbrt reference is NOT a gate here.** At the
  unit-correct (geometric) optical depth the dragon is still dimmer than pbrt
  because of two separate, not-yet-root-caused issues: (a) the pbrt infinite-light
  HDRI may not be active in the headless `render_linear` path (the renderer logs
  the default `Neutral Gray` env), and (b) the walk's energy / multiple-scattering
  fidelity at the dragon's high optical depth (~30–40 mean free paths). These are
  a stacked follow-up change ("subsurface brightness / env application"). This
  change only removes the confirmed 1000× unit error.
