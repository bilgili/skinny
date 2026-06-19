## Why

skinny has two integrators — a unidirectional path tracer and BDPT — and both
converge slowly on caustics (specular→diffuse light transport, e.g. a focused
highlight cast through a glass object onto a diffuse floor). pbrt v4 ships an
`sppm` (Stochastic Progressive Photon Mapping) integrator precisely for this
regime, and skinny's pbrt→USD importer already records `Integrator "sppm"` in
metadata but maps it to nothing — sppm scenes silently render on the path
tracer instead. A GPU SPPM integrator closes both gaps: it gives skinny a
caustic-efficient integrator and lets the importer round-trip pbrt sppm scenes
to a faithful renderer counterpart.

This is **PM-1**, the first change of a phased `photon-mapping` capability. It
delivers the core surface SPPM estimator on **flat materials only**. The
layered skin/BSSRDF photon path (PM-2) and volumetric/media photon transport
(PM-3) are deliberately deferred to follow-up changes against the same
capability spec — each is its own reviewable implementation plan and its own
research problem.

## What Changes

- **New SPPM integrator** (`INTEGRATOR_SPPM = 2`), **wavefront execution only**
  — like ReSTIR DI and neural guiding. Selectable end to end: CLI
  `--integrator sppm`, GUI mode "SPPM", `integratorType` GPU dispatch. The
  megakernel path has no global photon map, so it falls back / refuses with a
  clear message (mirrors the neural wavefront-only gate).
- **Per-pass SPPM pipeline** folded into the existing progressive-accumulation
  loop, as new wavefront stages: (1) **eye pass** stores one stochastic visible
  point per pixel at the first non-specular hit; (2) **hash-grid build** over
  visible-point positions+radii via counting sort (count → prefix-sum →
  scatter); (3) **photon pass** emits photons from lights (reusing the
  power-weighted emissive-mesh / light CDFs), traces with Russian roulette, and
  atomically deposits flux into visible points found through the grid;
  (4) **radius/flux update** shrinks each point's radius and accumulates flux per
  the SPPM update rules. Direct lighting reuses the existing NEE path.
- **Flat materials only** for PM-1 (`UsdPreviewSurface` / `standard_surface` /
  `OpenPBR` / Python flat materials) — same gating as neural guiding and the
  BDPT flat path. The layered skin estimator chain is untouched.
- **Both backends, Metal-first then Vulkan.** Native Metal is the primary
  development/verification backend in this change (the dev host resolves
  `auto`→Metal); Vulkan parity follows within the same change. New GPU buffers
  are **folded** to respect the Metal 31-slot argument-table cap (per the
  established graph-param-fold lesson), and per-pass queue compaction uses the
  Metal CPU-readback fallback already used by the wavefront driver.
- **pbrt importer mapping.** `Integrator "sppm"` is recognized: its parameters
  (`numiterations`, `maxdepth`, `photonsperiteration`, `radius`, `seed`) are
  translated to USD metadata **and** the importer selects the skinny SPPM
  integrator on load. The current "sppm / photon integrators are out of scope"
  note is lifted for the surface case.
- **Caustic parity gate.** A new caustic parity scene (glass object over a
  diffuse plane) renders under pbrt v4 `sppm` as the reference EXR and is
  compared to skinny SPPM via the existing `parity.py` harness (relMSE / FLIP
  thresholds), with a labelled side-by-side image artifact.
- **Docs:** new `docs/PhotonMapping.md` (SVG pipeline diagram + LaTeX→SVG
  equations per repo convention); updated `docs/Wavefront.md` (new stages),
  `docs/Architecture.md` (descriptor binding map), `README.md` (CLI +
  compatibility matrix), `CHANGELOG.md`.

## Capabilities

### New Capabilities
- `photon-mapping`: A GPU Stochastic Progressive Photon Mapping integrator —
  the per-pass eye/grid/photon/update pipeline, its wavefront-only execution
  and material gating, the pbrt `sppm` importer mapping, and the caustic parity
  requirement. Scoped to **surface (flat) materials** for PM-1; the spec marks
  skin/BSSRDF and volumetric transport as explicitly deferred phases so PM-2/PM-3
  extend the same capability.

### Modified Capabilities
- `render-cli`: `--integrator` gains an `sppm` choice; the cross-front-end flag
  set, persistence, and incompatibility gating are extended so that `sppm`
  requires `--execution-mode wavefront` and refuses cleanly on a megakernel /
  incompatible selection (same pattern as the existing BDPT and neural gates).

## Impact

- **New shaders:** `src/skinny/shaders/integrators/sppm.slang` (+ supporting
  modules for visible-point storage, hash-grid build, photon deposit); new
  wavefront stage kernels under `src/skinny/shaders/wavefront/`; new
  `INTEGRATOR_SPPM` constant in `common.slang`; recompiled `main_pass.spv` /
  wavefront `.spv`.
- **Renderer/host:** `renderer.py` (integrator modes, new GPU buffers,
  per-pass stage dispatch, accumulation-reset hash), `wavefront_driver.py` /
  `vk_wavefront.py` / `metal_compute.py` / `metal_wavefront.py` (new stages,
  Metal buffer fold + readback compaction), `cli_common.py`
  (`INTEGRATOR_INDEX["sppm"]`, `--integrator` choices + gating), `app.py` /
  `web_app.py` / `ui/qt/app.py` (selection wiring), GUI integrator list.
- **Descriptor bindings:** new visible-point / hash-grid / photon-record
  bindings (folded for Metal); `docs/Architecture.md` binding map updated.
- **pbrt module:** `state.py` / `metadata.py` / `emit.py` (sppm recognition +
  USD metadata + integrator selection), `report.py` (sppm now mapped not
  skipped), new caustic parity scene + reference under the pbrt test assets.
- **Tests:** SPPM unbiasedness / energy-conservation harness, hash-grid build
  unit tests, importer sppm-mapping tests, caustic parity gate (Metal + Vulkan).
- **No breaking changes.** Default integrator stays `path`; existing scenes are
  byte-unaffected.
