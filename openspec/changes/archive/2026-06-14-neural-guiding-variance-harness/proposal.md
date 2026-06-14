# Proposal — neural-guiding-variance-harness

## Why

The paper's renderer-side claims are variance-vs-time results on known scenes —
**equal-time variance**, **equal-variance time**, and **`1/(var·t)` efficiency** — swept across
the parameterization axes (chart, encoding, temporal) and the existing guiding baselines
(BSDF-only, env proposal, ReSTIR DI). The measurement *primitives* already exist
(`render_headless`, the linear-HDR accumulation image, accumulation reset on state change,
`tests/test_headless.py`), but there is **no harness** that sweeps configurations, measures
variance against a converged reference, and emits the paper's tables and plots. Today those
numbers would be produced ad hoc and unreproducibly.

This change builds that harness: a headless, reproducible sweep that turns the CLI knobs
(`--chart` P3, `--encoding` `renderer-conditioner-encoding`, `--temporal` P4, `--proposals`,
`--reuse`) into the `cgf-paper` §integration figures (`task 2.5`'s `\Needed` renderer
results/timings/renders).

## What Changes

- **Headless sweep driver.** A checked-in driver + config that enumerates a matrix
  `{scene × proposal-set × chart × encoding × temporal × precision × budget}`, and for each
  cell: builds skinny with the matching `-D` defines, loads the tag-matched `.nrec`
  (parametrization tag validated), renders headless, and reads the **linear-HDR accumulation
  image** at the budget.
- **Converged reference + variance.** Per `(scene, integrator)` it renders a high-spp converged
  reference (ground truth), asserts its convergence, and computes per-pixel error/variance of
  each cell against it. Variance estimates use **multiple independent seeds** (not a single
  run) so every reported number carries a spread.
- **Metrics + plots.** Emits **equal-time** (fix spp/wallclock → variance), **equal-variance**
  (time to a target variance), and **`1/(var·t)`** efficiency — the same metric as
  `ParametrizationResults §5` — as checked-in result tables plus **SVG** plots
  (variance-vs-spp/time curves, equal-time and equal-variance bars) per the diagram convention.
- **Known scenes.** A small fixed scene set (Cornell-style box + a couple of guiding-relevant
  USD scenes) with lights/env/camera pinned for reproducibility.
- **Reproducibility.** Deterministic seeds, the config checked in, runnable headless on the
  repo-root **Python 3.13 venv** (the built MaterialX+Vulkan/Metal environment, per
  `CLAUDE.md`'s headless note), with Metal `slangc` compiles **serialized** (the one-guarded-
  compile thermal rule).

## Capabilities

### New Capabilities
- `neural-guiding-variance-harness`: a reproducible headless sweep that renders known scenes
  across the parameterization axes and guiding baselines, measures variance against a converged
  reference over multiple seeds, and emits equal-time / equal-variance / `1/(var·t)` tables and
  SVG plots for the paper.

## Impact

- **Code:** a harness driver + config under `scripts/` (or `tests/`), variance/efficiency
  computation, and SVG plotting. Reuses `render_headless` + the accumulation image; **no
  renderer-core change**.
- **Effort tier:** the harness is engineering, but its **outputs are the paper's
  variance/efficiency claims** → apply MAX measurement rigor: assert reference convergence, use
  multiple seeds with reported spread, validate the variance estimator, no number from a single
  run.
- **Feeds:** `cgf-paper` `task 2.5` (renderer integration results/timings/renders).
- **Dependencies (degrades gracefully):** sweeps whatever axes are implemented — `--proposals`
  and `--chart` (P3) at minimum; `--encoding` and `--temporal` cells are included once those
  changes land. A cell whose `.nrec`/build is unavailable is skipped and **logged** (no silent
  gap).
- **Caveats:** each `(chart, encoding, temporal)` cell is a `slangc` recompile (build dims) →
  the sweep serializes Metal compiles; headless needs the 3.13 venv (+ `VULKAN_SDK`/
  `DYLD_LIBRARY_PATH` for the Vulkan backend, or native Metal).
- **Out of scope:** training the nets (that is `spline_flow` + the online loop); new
  integrators or scenes beyond the fixed known set.
