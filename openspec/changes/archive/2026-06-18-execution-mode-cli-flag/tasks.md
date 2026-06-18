## 1. Command-line selection

- [x] 1.1 Add `--execution-mode megakernel|wavefront` to `app.py` (+ `SKINNY_EXECUTION_MODE` env fallback, default `megakernel`), mirroring `--backend`/`SKINNY_BACKEND`. Pass the resolved mode to `Renderer(...)` as a constructor argument.
- [x] 1.2 `Renderer.__init__` takes `execution_mode=` and stores the fixed index (clamped against the available modes — Metal pin collapses to `megakernel`). Apply `effective_execution_mode` (the bdpt capability gate) to the fixed value.

## 2. Remove the runtime/GUI path

- [x] 2.1 Drop the `execution_mode_index` `_disc` entry from `STATIC_PARAMS` (params.py) so it leaves `ALL_PARAMS`, the data-driven UI Combo, and the settings snapshot/restore.
- [x] 2.2 Remove `set_execution_mode` / `cycle_execution_mode` and `execution_mode_index` from `_current_state_hash()` (mode is fixed, so it can't change mid-session). Keep `effective_execution_mode_index` + `execution_mode_fallback_active`. (Also removed the now-orphaned `next_mode_index` cycle helper from params.py.)
- [x] 2.3 Remove the `9.6` front-end-parity test for the toggle and the `test_execution_mode` persistence/snapshot tests; keep the pure capability-gate + clamp tests (re-pointed at the constructor value).

## 3. Decouple the wavefront from the megakernel pipeline

- [x] 3.1 Factor the scene plumbing the wavefront borrows out of the megakernel `ComputePipeline`: the set-0 descriptor-set layout, the per-frame descriptor sets, `generated_materials`/per-graph emission, and the `graph_bindings` map. House it in a backend-independent owner (the renderer or a small shared object) both backends use. (Done via `ComputePipeline(compile_pipeline=False)` / `ComputePipeline.scene_bindings_only(...)`, held on `renderer._scene_bindings` + accessed through `_scene_set0_layout` / `_scene_graph_bindings`; same emission + layout code as the full build, so it stays identical across backends.)
- [x] 3.2 Build the set-0 layout for `wavefront` mode without compiling `main_pass` — reflect it from the wavefront kernels or define it explicitly — and ensure it stays consistent across every wavefront stage pipeline and the descriptor writes. (`scene_bindings_only` builds the explicit set-0 layout + material emission with no `main_pass` slangc; the wavefront passes + `_create_descriptors` all read `_scene_set0_layout`.)

## 4. Mutually-exclusive compilation

- [x] 4.1 Gate `_build_pipeline_for_current_graphs` so it builds the megakernel `ComputePipeline` only in `megakernel` mode; in `wavefront` mode it emits the shared plumbing (3.1) but skips the `main_pass` compile + megakernel pipeline.
- [x] 4.2 Build the wavefront stage pipelines only in `wavefront` mode (eagerly at scene load, or lazily but never in megakernel mode). Subsumes deferred task 6.4 (no megakernel rebuild on a wavefront-mode add). (Lazy `_ensure_wavefront_*` in the render gate; the megakernel render path never builds them — verified by `test_megakernel_mode_builds_no_wavefront_pass_and_renders`.)
- [x] 4.3 The per-frame render gate dispatches the one compiled backend. Both `render()` (windowed) and `render_headless()` now gate on `_backend_render_ready` and branch to the staged wavefront passes in wavefront mode (previously only `render_headless` did, so the windowed app was megakernel-only).

## 5. Tests + verification

- [x] 5.1 `wavefront` mode builds no megakernel pipeline (`renderer.pipeline is None`) and renders the demo (A/B still holds, selected per `Renderer` instance). (`test_execution_mode_compile::test_wavefront_mode_builds_no_megakernel_and_renders` + the per-instance A/B in `test_wavefront_path_ab`.)
- [x] 5.2 `megakernel` mode builds no wavefront stage pipeline and renders identically to today (megakernel output byte-unchanged). (`test_megakernel_mode_builds_no_wavefront_pass_and_renders`; emission path untouched so checked-in `main_pass.spv` is unmodified.)
- [x] 5.3 Adding a model with a new material in `wavefront` mode compiles exactly one shade pipeline and never (re)builds a megakernel pipeline. (`test_wavefront_graph_rebuild_never_builds_megakernel` spies on `ComputePipeline` and asserts no compiled megakernel across a rebuild; the one-shade-pipeline-per-material compile-win is covered by the existing per-material cache + staged-shade tests.)
- [x] 5.4 CLI/env selection resolves correctly; Metal pins to `megakernel`; `--execution-mode wavefront` + `bdpt` follows the capability gate. (Constructor string→index covered by the compile tests; Metal pin by `test_clamp_pins_single_mode`; wavefront+bdpt by `test_wavefront_bdpt_passes_through_once_supported` + the bdpt A/B. The `SKINNY_EXECUTION_MODE` env fallback is the argparse default in `app.py`.)
- [x] 5.5 `ruff check src/` + `pytest`; confirm `main_pass.slang` (megakernel) output unchanged when that mode is selected. (`ruff` error count identical to `main` baseline — zero new; the 21-test wavefront+execution GPU suite + the CPU unit suite pass; `main_pass.spv` is byte-unchanged in git.)
