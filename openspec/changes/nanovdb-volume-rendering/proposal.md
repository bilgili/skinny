# NanoVDB Heterogeneous Volume Rendering

## Why

The renderer already has the volumetric machinery (delta-tracked transport, HG phase, the
`MediumParams` seam with `MEDIUM_NANOVDB` reserved in `bindings.slang`, and the additive-extension
contract in `materials/subsurface/medium.slang`), but only homogeneous media are wired up — the
pbrt importer explicitly skips heterogeneous (grid/VDB) media (`api.py:265`), so canonical volume
scenes like `disney-cloud` and `bunny-cloud` (both `MakeNamedMedium "nanovdb"` + volpath) import as
empty air. Filling the reserved seam makes skinny render real production clouds and gives the
parity harness its first heterogeneous-media gates against pbrt.

Separately, Metal compute kernels that are dispatched and then abandoned (crash, test kill,
watchdog overrun) are not cleaned up — they keep running on the GPU with no way to stop them short
of a reboot (observed repeatedly: "Metal slang-rhi WEDGED, needs reboot"). Volume marching makes
this acute (long delta-tracking loops per pixel), so shipping heterogeneous volumes without a
dispatch-hygiene/cleanup harness would make the wedge routine.

## What Changes

- **NanoVDB grid import**: parse `.nvdb` (NanoVDB) density grids in the pbrt importer, convert to a
  renderer-consumable density field, and emit the medium into the `.usda` scene (UsdVol
  `Volume`/field prim + the existing `skinnyOverrides` medium keys), instead of skipping with
  "heterogeneous media unsupported". Covers `disney-cloud.pbrt` (WDAS cloud, `sigma_a=0`,
  `sigma_s=1`, `g=0.877`, `scale=4`) and `bunny-cloud.pbrt` (`sigma_s=10`, `sigma_a=0.5`).
- **Free-standing heterogeneous medium transport**: implement the reserved `MEDIUM_NANOVDB` medium
  kind through the existing `MediumParams`/`densityAt` interface — density grid uploaded as a GPU
  3D texture, majorant-based delta tracking reusing the shipped volumetric walk. A shape with a
  `MediumInterface` and pbrt `Material "interface"` (null boundary) routes to the volume path
  instead of the flat/dielectric fallback.
- **pbrt scene coverage**: `Material "interface"` (null material), volpath integrator mapping, and
  the medium transform stack (`Translate`/`Scale` around `MakeNamedMedium`/shape) resolve correctly
  for the two target scenes.
- **Parity gates**: add `disney-cloud` and `bunny-cloud` to the pbrt corpus with pbrt-truth
  references and self-consistency gates (megakernel ≡ wavefront), same dual-gate discipline as
  every other renderer feature.
- **Metal dispatch cleanup harness**: guaranteed teardown of in-flight Metal work — bounded
  (tiled/capped) volume dispatches that stay under the macOS GPU watchdog, `wait_for_idle` +
  `destroy()` invoked on every exit path (context-manager / `atexit` / signal), and a test harness
  that verifies no kernel outlives the owning process. Applies to all Metal dispatches, not just
  volumes.

## Capabilities

### New Capabilities

- `heterogeneous-media`: grid-density participating media — NanoVDB import to USD, GPU density
  field, majorant delta-tracking transport through the `MediumParams` seam, free-standing
  (non-subsurface) medium boundaries.
- `pbrt-volume-import`: pbrt heterogeneous-media import — `.nvdb` grid parsing,
  `MakeNamedMedium "nanovdb"` → UsdVol emission, `Material "interface"` null/boundary material
  (no `pbrt-v4-scene-import` living spec exists in `openspec/specs/`, so this is a new capability
  spec rather than a delta).

### Consumed Capabilities

- `metal-dispatch-hygiene` (**extracted + landed separately**, main `c4159bb`, living spec
  `openspec/specs/metal-dispatch-hygiene/`): guaranteed teardown + kill harness now a standing
  repo requirement (CLAUDE.md). This change consumes it: the volume march must satisfy its
  watchdog-bounded requirement (task 6.5) and pass the kill harness.

### Modified Capabilities

- `subsurface-scattering`: the "Volume transport is forward-compatible with heterogeneous
  free-standing media" requirement is discharged — `densityAt`/`mediumMajorant` gain the
  heterogeneous grid source (`MEDIUM_NANOVDB`) and free-standing `MediumInterface` attachment,
  homogeneous behavior byte-unchanged.

## Impact

- **Importer**: `src/skinny/pbrt/` (`media.py`, `api.py`, new `nanovdb.py` reader, `emit.py` for
  UsdVol emission, `materials.py` for `interface`).
- **Loader/renderer**: `usd_loader.py` (volume prim ingest), `renderer.py` (density-grid upload,
  medium packing — new 3D texture resource on both backends; watch the Metal 31-slot argument
  table), `metal_compute.py`/`vk_compute.py` (3D `SampledImage` support if missing).
- **Shaders**: `materials/subsurface/medium.slang` (grid `densityAt` + majorant),
  `bindings.slang` (grid binding), integrators consume via the existing walk; `main_pass.spv`
  recompile.
- **Metal lifecycle**: `metal_context.py` (teardown paths), `renderer.py`/`app.py` (cleanup
  wiring), new test harness under `tests/`.
- **Parity harness**: `tests/pbrt/corpus/manifest.json` + regenerated refs (pinned pbrt v4 at
  `~/projects/pbrt-v4/build/pbrt`); `parity.py` validity rules if any combo is excluded.
- **Docs**: `docs/Architecture.md` (binding map), README compatibility matrix, CHANGELOG.
- **Dependencies**: NanoVDB file parsing — prefer a small pure-Python reader for the two grid
  layouts we need over a new native dependency (decided in design.md).
