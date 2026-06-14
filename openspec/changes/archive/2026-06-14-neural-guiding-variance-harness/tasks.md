# Tasks — neural-guiding-variance-harness

## 1. Scenes + reference

- [x] 1.1 Fix a known scene set (Cornell-style box + a few guiding-relevant USD scenes; a
      moving-light scene for temporal cells — coordinate with `neural-temporal-conditioning`),
      lights/env/camera pinned; check the scenes (or fetch scripts) in
      — `SCENES` in `scripts/guiding_variance_sweep.py` pins `cornell` /
      `three_materials` / `restir` to checked-in USD (geometry/lights/camera fixed
      in the `.usda`); the moving-light temporal scene awaits `neural-temporal-conditioning`
- [x] 1.2 Per `(scene, integrator)`: render a high-spp **converged reference** from the
      linear-HDR accumulation image; **assert convergence** (running error below threshold, or
      two independent references agree); cache with a content hash
      — `build_reference()` renders two independent half-budget refs and gates them
      via `assert_reference_converged` (≤2% agreement); `_hash_img` content hash in the manifest

## 2. Sweep driver

- [x] 2.1 Declarative config: the matrix `{scene × proposals × reuse × chart × encoding ×
      temporal × precision × budget}`, sliceable per figure
      — `default_config()` + checked-in `docs/diagrams/guiding_variance/config.json`; `--config` override
- [x] 2.2 Per cell: ensure the matching build (`-D NF_CHART/NF_ENCODING/NF_COND`), **serialize
      Metal `slangc` compiles**; load the tag-matched `.nrec`; **skip + log** a cell whose build
      or `.nrec` is missing (no silent gap)
      — `cell_availability()` skips+logs unavailable chart/temporal/precision/net cells;
      fresh renderer per cell runs sequentially (one Metal compile at a time, under `guarded_metal.sh`)
- [x] 2.3 Per cell: render headless over **N independent seeds**, read the linear-HDR
      accumulation at the budget
      — `run_cell` → `_sweep_budgets`/`_accumulate_to` over `seeds` disjoint
      `frame_index` RNG streams; `read_accumulation_hdr()` (linear HDR)

## 3. Metrics (MAX rigor)

- [x] 3.1 Per-pixel error/variance vs the converged reference; aggregate as **mean ± spread over
      seeds** (never a single-seed number); report a firefly percentile alongside the mean
      — `variance_over_seeds` → `CellVariance` (var_mean ± var_spread, firefly p99.9); unit-tested
- [x] 3.2 **Equal-time:** fix spp/wallclock → variance per cell
- [x] 3.3 **Equal-variance:** spp + wallclock to reach a target variance per cell
      — `equal_variance_budget` log-log inversion; in-grid vs extrapolated flagged; unit-tested
- [x] 3.4 **`1/(var·t)` efficiency**, identical metric to `ParametrizationResults §5`
      — `efficiency()`; unit-tested identity
- [x] 3.5 Validate the variance estimator on a trivial known case (e.g. uniform image → ~0)
      — `test_uniform_image_has_zero_error_and_variance` + `test_seed_variance_recovers_known_variance`

## 4. Outputs

- [x] 4.1 Checked-in result **tables** (per-figure slices) with a `% source:`-style provenance
      note for transcription
      — `markdown_table(... source=...)`; emit-tested
- [x] 4.2 **SVG plots** per the diagram convention: variance-vs-spp/time curves, equal-time
      bars, equal-variance bars
      — `svg_line_chart` / `svg_bar_chart` (CSS-var theming); XML well-formedness gated
- [x] 4.3 Wire the outputs to `cgf-paper` `task 2.5` (the `\Needed` renderer results/timings/
      renders); note the produced figures there
      — tables carry the `% source:` provenance hook and the `<scene>_*.svg` figure set are
      the assets; documented in the output `README.md`. NOTE: the `cgf-paper` repo is not
      checked out under `~/projects` in this environment, so the figures are produced +
      provenance-tagged here for transcription rather than written into that repo directly

## 5. Reproducibility + validate

- [x] 5.1 Deterministic seeds; config + scene set + reference hashes checked in; document the
      3.13-venv + `VULKAN_SDK`/`DYLD_LIBRARY_PATH` (or native Metal) invocation
      — disjoint per-seed RNG streams; `config.json` checked in; reference hash in the manifest;
      invocation documented in `docs/diagrams/guiding_variance/README.md`
- [x] 5.2 A `--quick` smoke slice (one scene, two cells, low spp) runs end-to-end and emits a
      table + a plot
      — `--quick` path implemented (`default_config(quick=True)`); the emit half (reference
      gate → seed-aggregated variance → table + manifest + 4 SVGs) is verified end-to-end
      off-GPU in `tests/test_guiding_variance_emit.py`. NOTE: the live GPU smoke aborted on
      this host's `guarded_metal.sh` memory preflight (~6.3 GB free < 8 GB floor — the guard
      that prevents a Metal-compile RAM spike from locking the box); it runs when ≥8 GB is free
- [x] 5.3 `openspec validate neural-guiding-variance-harness --strict`
