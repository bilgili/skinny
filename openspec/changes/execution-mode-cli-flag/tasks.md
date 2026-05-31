## 1. Command-line selection

- [ ] 1.1 Add `--execution-mode megakernel|wavefront` to `app.py` (+ `SKINNY_EXECUTION_MODE` env fallback, default `megakernel`), mirroring `--backend`/`SKINNY_BACKEND`. Pass the resolved mode to `Renderer(...)` as a constructor argument.
- [ ] 1.2 `Renderer.__init__` takes `execution_mode=` and stores the fixed index (clamped against the available modes — Metal pin collapses to `megakernel`). Apply `effective_execution_mode` (the bdpt capability gate) to the fixed value.

## 2. Remove the runtime/GUI path

- [ ] 2.1 Drop the `execution_mode_index` `_disc` entry from `STATIC_PARAMS` (params.py) so it leaves `ALL_PARAMS`, the data-driven UI Combo, and the settings snapshot/restore.
- [ ] 2.2 Remove `set_execution_mode` / `cycle_execution_mode` and `execution_mode_index` from `_current_state_hash()` (mode is fixed, so it can't change mid-session). Keep `effective_execution_mode_index` + `execution_mode_fallback_active`.
- [ ] 2.3 Remove the `9.6` front-end-parity test for the toggle and the `test_execution_mode` persistence/snapshot tests; keep the pure capability-gate + clamp tests (re-pointed at the constructor value).

## 3. Decouple the wavefront from the megakernel pipeline

- [ ] 3.1 Factor the scene plumbing the wavefront borrows out of the megakernel `ComputePipeline`: the set-0 descriptor-set layout, the per-frame descriptor sets, `generated_materials`/per-graph emission, and the `graph_bindings` map. House it in a backend-independent owner (the renderer or a small shared object) both backends use.
- [ ] 3.2 Build the set-0 layout for `wavefront` mode without compiling `main_pass` — reflect it from the wavefront kernels or define it explicitly — and ensure it stays consistent across every wavefront stage pipeline and the descriptor writes.

## 4. Mutually-exclusive compilation

- [ ] 4.1 Gate `_build_pipeline_for_current_graphs` so it builds the megakernel `ComputePipeline` only in `megakernel` mode; in `wavefront` mode it emits the shared plumbing (3.1) but skips the `main_pass` compile + megakernel pipeline.
- [ ] 4.2 Build the wavefront stage pipelines only in `wavefront` mode (eagerly at scene load, or lazily but never in megakernel mode). Subsumes deferred task 6.4 (no megakernel rebuild on a wavefront-mode add).
- [ ] 4.3 The per-frame render gate dispatches the one compiled backend (no `_wavefront_env_pass` fallback needed since the staged passes are the only wavefront path).

## 5. Tests + verification

- [ ] 5.1 `wavefront` mode builds no megakernel pipeline (`renderer.pipeline is None`) and renders the demo (A/B still holds, selected per `Renderer` instance).
- [ ] 5.2 `megakernel` mode builds no wavefront stage pipeline and renders identically to today (megakernel output byte-unchanged).
- [ ] 5.3 Adding a model with a new material in `wavefront` mode compiles exactly one shade pipeline and never (re)builds a megakernel pipeline.
- [ ] 5.4 CLI/env selection resolves correctly; Metal pins to `megakernel`; `--execution-mode wavefront` + `bdpt` follows the capability gate.
- [ ] 5.5 `ruff check src/` + `pytest`; confirm `main_pass.slang` (megakernel) output unchanged when that mode is selected.
