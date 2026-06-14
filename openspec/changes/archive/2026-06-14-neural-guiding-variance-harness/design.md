# Design — neural-guiding-variance-harness

## What makes the numbers trustworthy (the MAX-rigor core)

Variance-vs-time claims are only as good as the reference and the estimator. The harness is
engineering, but its outputs are paper claims, so three things are non-negotiable:

1. **Converged reference, asserted.** Per `(scene, integrator)` render a high-spp accumulation
   as ground truth and *assert it has converged* (e.g. the reference's own running error below a
   threshold, or two independent high-spp references agree). A variance/error number against an
   unconverged reference is meaningless — gate on it.
2. **Multiple seeds, reported spread.** Each cell's variance/error is computed over **N
   independent seeds**, not one run. Report mean ± spread; a single-seed number never ships.
3. **Linear-HDR, not display pixels.** Read the **accumulation image** (linear HDR), not
   `render_headless`'s tonemapped sRGB output, for error/variance — per `CLAUDE.md`.

## Metrics (consistent with the spline_flow study)

- **Equal-time:** fix a spp or wallclock budget per cell → measure error/variance vs the
  reference. Lower = better.
- **Equal-variance:** measure time (spp and wallclock) for a cell to reach a target error/
  variance. Lower = better.
- **Efficiency `1/(var·t)`:** the headline metric, identical to `ParametrizationResults §5`, so
  renderer and synthetic-study numbers are directly comparable.

`var` is the per-pixel squared error vs the reference, aggregated (mean over pixels, robust to
fireflies — report both mean and a percentile to surface the firefly tail the paper cares
about).

## The sweep matrix

```
 scene        ∈ {cornell, + a few guiding-relevant USD scenes}      (lights/env/camera pinned)
 proposals    ∈ {bsdf, bsdf+env, bsdf+neural, …}  + reuse ∈ {off, restir-di}
 chart        ∈ {V0,V1,V2,V5}            (build dim, P3)
 encoding     ∈ {E0,E1,E3}               (build dim, renderer-conditioner-encoding)
 temporal     ∈ {off, on}               (build dim, P4)
 precision    ∈ {fp32, fp16, fp8}        (inference)
 budget       = spp/time grid
```

A cell needs a matching **build** (`-D NF_CHART/NF_ENCODING/NF_COND`) and a matching
**`.nrec`** (parametrization-tag validated). The matrix is large → it is **declarative and
sliceable**: the paper figures each select a sub-slice (e.g. fix scene+proposals, vary chart).

## Execution shape

```
 build reference(scene, integrator) → assert converged → cache
 for cell in matrix:
   ensure build(cell.defines)            ← slang recompile; SERIALIZE on Metal (thermal rule)
   load .nrec(cell)  [tag must match]    ← skip + LOG if missing (no silent gap)
   for seed in seeds:
     render headless → accumulation(linear HDR) at budget
   error/var vs reference (mean ± spread over seeds)
 aggregate → equal-time / equal-variance / 1/(var·t) tables + SVG plots
```

## Reproducibility

- Deterministic seeds; the sweep config + scene set + reference hashes checked in.
- Runs on the repo-root **Python 3.13 venv** (built MaterialX + Vulkan/Metal), with
  `VULKAN_SDK` / `DYLD_LIBRARY_PATH` exported for the Vulkan backend (per `CLAUDE.md`).
- Metal: one guarded `slangc` compile at a time; the GPU sweep is the deliberate workload (unlike
  the `-m 'not gpu'` unit-test rule).
- Every dropped/skipped cell is logged (a missing build or `.nrec` must read as "skipped", never
  as "covered").

## Decisions

- **No renderer-core change** — pure harness over existing headless + accumulation APIs.
- **Degrades gracefully** — runs with whatever axes exist (≥ `--proposals` + `--chart`); the
  `--encoding`/`--temporal` cells activate as those changes land.
- **Outputs are checked-in** tables + SVGs, wired to `cgf-paper` 2.5.

## Open questions (do not block)

1. Scene set: exactly which USD scenes beyond Cornell (need a moving-light scene for the temporal
   cells — coordinate with `neural-temporal-conditioning`).
2. Reference strategy per scene: analytic where available vs high-spp accumulation (default
   high-spp + convergence assert).
3. Firefly metric: which percentile to report alongside the mean.
