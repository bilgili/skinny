# Confirming test scenes (`tests/assets/suite/`)

Minimal, fast-to-render scenes that each *discriminate one renderer axis* —
material lobe families, integrators, execution modes, sampling modes, and
white-furnace energy closure. They back the change `confirming-test-scenes`
(see `openspec/changes/confirming-test-scenes/`).

## Why these live here

Heavy scenes (bathroom, dragon) catch integration regressions but bury a
single-lobe defect in noise. A ~1k-triangle, 128×128 scene that isolates one
BSDF lobe or one transport path fails *precisely* when that thing breaks, and
renders in seconds. This directory is the localized regression net.

## Layout

```
tests/assets/suite/<scene>/
  <scene>.usda        # plain UsdPreviewSurface authoring (loaded directly)
  <scene>_mtlx.usda   # MaterialX authoring — references <scene>.mtlx
  <scene>.mtlx        # standard_surface / OpenPBR node graph
  <scene>.pbrt        # pbrt v4 counterpart (when pbrt-expressible) — user-runnable
```

Not every scene has every file:
- MaterialX-only materials (OpenPBR cards) have **no** plain `.usda` and **no**
  `.pbrt` — a `pbrt_skip` / `equivalence.skip` reason is recorded in the
  manifest instead of leaving a silent gap.
- Furnace scenes have no `.pbrt` (their reference is the analytic value 1.0).

## Registration and gates

Suite scenes are **registered in the single existing manifest**
`tests/pbrt/corpus/manifest.json` as `usd:`-source entries (the `usd` path is
resolved relative to the repo root, so `tests/assets/suite/<scene>/<scene>.usda`
loads directly). They are swept by the same harness as the pbrt corpus:

- **pbrt-truth** — vs the checked-in `refs/suite_<scene>.exr` (generated from the
  `.pbrt` counterpart by `tests/pbrt/regen_refs.py --scene suite`).
- **self-consistency** — every scene's megakernel ≡ wavefront and
  integrator ≡ integrator vs the `(path, wavefront)` anchor.
- **authoring-equivalence** — a `_mtlx` variant's render matches its plain-USD
  sibling within tolerance (new gate, `parity.authoring_equivalence_result`).
- **furnace-closure** — furnace scenes render to 1.0 within tolerance under
  `renderer.furnace_index = 1` (new gate, `skinny.pbrt.furnace`).

Coverage meta-tests in `tests/pbrt/test_suite.py` fail the build if a suite
scene lacks a disposition for any applicable gate class.

## Running the pbrt counterpart

```bash
~/projects/pbrt-v4/build/pbrt tests/assets/suite/<scene>/<scene>.pbrt
```

renders it unmodified. Regenerate all suite reference EXRs with:

```bash
python tests/pbrt/regen_refs.py --scene suite --res 128
```

## Budget

128×128, ≤ ~1k triangles, spp recorded per scene in the manifest. GPU sweeps
follow `metal-dispatch-hygiene` (one guarded Metal process at a time) and are
run only on explicit request.
