## 1. Worktree & scaffolding

- [x] 1.1 Create a git worktree off `main` for this change (project convention); confirm `./bin/python3.13`, `VULKAN_SDK`, `DYLD_LIBRARY_PATH` are set for headless renders (see CLAUDE.md). Tests in a worktree need `PYTHONPATH=src`. Worktree is **kept** at session exit (durable pytest harness).
- [x] 1.2 Read `parity.py`, `test_parity.py`, `manifest.json`, headless API + neural/ReSTIR/mode plumbing; confirmed validity rules + arming recipe (`proposals="bsdf,neural"`, `reuse="restir-di"` are HeadlessRenderer constructor kwargs; `_INTEGRATORS={path:0,bdpt:1,sppm:2}`; `r.execution_modes`/`r.integrator_modes` for coverage; SPPM→wavefront forced; neural=wavefront+path+flat; ReSTIR=wavefront-only).

## 2. Standardized image-metric battery (D6 — underpins §4)

- [x] 2.1 Extend `src/skinny/pbrt/metrics.py` with an `ImageMetrics` dataclass and `compute_metrics(img, ref=None) -> ImageMetrics`, keeping existing `relmse`/`flip`/`align_exposure`/`read_exr`. Error fields (exposure-aligned): `mse`, `rmse`, `mae`, `relmse`, `psnr`, `flip`. Single-image fields: `variance`, `noise_sigma` (Immerkær), `firefly_fraction` (3×3-median outlier). Pure numpy, no new deps.
- [x] 2.2 Add `tests/pbrt/test_metrics.py` (no GPU): identical images → zero error / inf PSNR; a planted firefly raises `firefly_fraction`; added noise raises `noise_sigma`; exposure-scaled copy → ~0 error after alignment; `ref=None` yields only single-image stats.

## 3. Combo model & validity table (D1)

- [x] 3.1 Add a `RenderCombo` dataclass (integrator, execution_mode, proposals: tuple, reuse: str) and extend `SceneSpec` with `usd` (asset source), `material_class` ("flat"|"subsurface"), and `megakernel_ok: bool` in `parity.py`.
- [x] 3.2 Implement `combo_is_valid(combo, scene) -> (bool, reason)` mirroring the compatibility matrix (SPPM⇒wavefront; neural⇒wavefront+path+flat; BDPT+neural invalid; ReSTIR⇒wavefront; `not megakernel_ok`⇒wavefront only). Explicit reason string for every skip.
- [x] 3.3 Implement `enumerate_combos(scene)` yielding only valid combos; define `ANCHOR = RenderCombo("path","wavefront",(),"none")`. Unit tests (no GPU) asserting the documented skips (SPPM-megakernel, neural-on-SSS, BDPT+neural, dragon-megakernel).

## 4. Dual-gate evaluate & self-consistency anchor (D2, D3, D6)

- [x] 4.1 Thread `proposals: str|None` and `reuse: str|None` through `render_linear(...)` into the `HeadlessRenderer(...)` constructor; when neural is requested, assert `renderer._neural_active()` after `_prepare`. Add a `render_combo(spec, combo, corpus_dir)` helper that maps a `RenderCombo` to a `render_linear` call.
- [x] 4.2 Extend `evaluate(spec, combo, corpus_dir)` to render the combo and return the full `ImageMetrics` vs the pbrt ref (via `compute_metrics`), plus the rendered image for self-consistency.
- [x] 4.3 Implement self-consistency: render the `ANCHOR` once per scene; compare each other valid combo to it with per-axis tolerances (mode-tight / integrator-loose / unbiasedness for neural+ReSTIR) read from the manifest. Use `compute_metrics`.
- [x] 4.4 Implement recorded-baseline handling: optional per-(scene,combo) `baseline {relmse,flip}` in the manifest; pbrt-truth assertion uses `max(tol, baseline*(1+margin))` and logs the delta; self-consistency never uses a baseline escape.

## 5. Scene source extension & reference generation (D4)

- [x] 5.1 Extend `render_linear` to load a scene from a `usd` asset path (`assets/bathroom.usda`, `assets/dragon_sss.usda`) directly (load_scene_from_stage / set_usd_scene), in addition to the existing `.pbrt`-import path; branch on which source field is set.
- [x] 5.2 Write `tests/pbrt/regen_refs.py` (offline, non-test-time): overrides the pbrt film `xresolution/yresolution` to the corpus resolution and runs the pinned pbrt binary (`~/projects/pbrt-v4/build/pbrt`) on `~/projects/pbrt-v4-scenes/contemporary-bathroom` and the dragon scene; document the exact command in the manifest `notes`.

## 6. Heavy corpus scenes + baselines (D3 §heavy)

- [x] 6.1 Generate `tests/pbrt/corpus/refs/bathroom.exr` and `dragon.exr` offline at the chosen corpus resolution (start 256², tune for CI cost); check them in.
- [x] 6.2 Add `bathroom` and `dragon` manifest entries (usd source, resolution, spp budget, pbrt-truth tolerances, validity hints: dragon `megakernel_ok:false` + `material_class:subsurface`, bathroom `material_class:flat`).
- [x] 6.3 Run the matrix once to MEASURE current pbrt-truth deltas for bathroom/dragon and RECORD them as `baseline` (this change does NOT fix the mismatch — fixes are follow-ups). Confirm self-consistency (megakernel≡wavefront, integrator≡integrator) passes for these scenes even while pbrt-truth is on baseline.

## 7. Pytest matrix, tiers & coverage meta-test (D5, extensibility)

- [x] 7.1 Replace the single-combo gate in `test_parity.py` with a parametrization over (scene × valid combo) for both the pbrt-truth gate and the self-consistency gate; mark GPU work `@pytest.mark.gpu`, optional high-spp confirmation `@pytest.mark.slow`.
- [x] 7.2 Add the `not gpu` import tier: construct the full matrix and import/load every corpus scene (incl. bathroom/dragon) with no render; assert no unsupported feature.
- [x] 7.3 Add the coverage meta-test: every combo the app exposes (`renderer.integrator_modes` × `renderer.execution_modes` × declared proposal/reuse axes) must have a validity-table entry; a new integrator/mode without a row fails the meta-test, naming the uncovered combos. Runs without a GPU.

## 8. Verification & images (project rule: show renders)

- [x] 8.1 Run the full matrix on the Metal backend (`select_backend()`/`--backend metal`); obey the one-Metal-megakernel-compile-at-a-time thermal rule. Capture a labelled side-by-side grid (scene × combo, shared tonemap) for bathroom and dragon and surface it via `SendUserFile` + clickable link.
- [x] 8.2 Confirm self-consistency green across all combos; confirm pbrt-truth green-or-baselined; record measured numbers (full battery) back into the manifest `measured`/`baseline` fields.
- [x] 8.3 Run `.venv/bin/ruff check src/` and the `not gpu` test subset (metrics + matrix construction + coverage meta-test) on a hostless invocation to prove the harness builds and scenes import everywhere.

## 9. Docs, changelog & close-out

- [x] 9.1 Add a parity-matrix section to `docs/Architecture.md` (or a dedicated doc) describing the validity table, dual gate, anchor, baselines, and the `ImageMetrics` battery; cross-reference the README/CLAUDE compatibility matrix.
- [x] 9.2 Update `CHANGELOG.md`; keep README compatibility matrix and CLAUDE.md in sync with any combo/validity wording.
- [x] 9.3 Sync the change dir back to the main checkout's `openspec/changes/`; `openspec validate integrator-parity-matrix --strict`; confirm harness-first scope held (no bathroom/dragon fix landed); note the follow-up fix changes. **Keep the worktree** (durable pytest harness).
