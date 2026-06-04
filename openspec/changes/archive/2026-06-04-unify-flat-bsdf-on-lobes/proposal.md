## Why

The directional-proposal mixture (Env / BSDF+Env presets) biases layered
coat+metal materials: **brass renders ~3.7% too dark** under BSDF+Env versus the
BSDF-only and BDPT references (IBL-only, brass column ≈ 0.219). Root cause:
`FlatMaterial.sample()` and `FlatMaterial.evaluate()` are **two different BSDF
models**. `sample()` draws from a 3-lobe analytic GGX (VNDF) model over
`FlatHitMat`; `evaluate()` returns the full MaterialX `std_surface` closure
(`evalStdSurfaceBSDF` over `StdSurfaceParams`) glued onto the 3-lobe pdf. The
proposal seam **draws** directions with `sample()` but **weights** them (and
couples NEE) with `evaluate()` — `weight = evaluate().response / mixPdf`,
`mixPdf = α_b·evaluate().pdf + α_e·envPdf` — so the two must agree. For layered
materials they don't: `sample().pdf ≠ evaluate().pdf`, and `response/pdf` is two
unrelated models divided (unbounded). `evaluate()` is the renderer's **canonical
BSDF** (NEE, BDPT connections + reverse pdfs, ReSTIR, the proposal seam all call
it), so the inconsistency taxes every IBL/proposal estimator, not just brass.

## What Changes

- Introduce **one composable lobe set** (`{coat, spec, diffuse}`) as the single
  source of truth for the flat/`std_surface` BSDF. Both `sample()` and
  `evaluate()` walk it, so `sample().pdf == evaluate().pdf` **structurally** and
  `evaluate().response / pdf` reduces to the bounded native per-lobe weight
  (`F·G₁` for GGX lobes, Lambert albedo for diffuse) **by construction** —
  firefly-free, not by a clamp.
- Each lobe carries a per-lobe **sampler id** with a native default; the
  dispatch is a runtime-pluggable seam that ships **unpopulated** (only the
  native strategies registered in this change). A future Change 2
  (`per-lobe-sampler-registry`) populates it with a host registry, GUI selector,
  and alternative samplers.
- **BREAKING (internal estimator behavior):** drop `evalStdSurfaceBSDF` from the
  path-traced / BDPT estimator path. The unified lobe model becomes the
  canonical BSDF for **Path Tracer and BDPT, in both megakernel and wavefront
  modes**. `evalStdSurfaceBSDF` (and `StdSurfaceParams`, binding 19) are kept
  only for the raster `preview_pass`. The now-dead `FlatMaterial.sp` member and
  the `loadStdSurfaceParams` call in `loadFlatMaterial` are removed (per-hit
  perf win; grep-confirmed sole consumer was the replaced `evaluate()`).
- Absolute radiance shifts for flat/`std_surface` materials, because the canonical
  BSDF changes from the rich closure to the 3-lobe model the bounce **already**
  sampled (the closure was the outlier NEE/BDPT saw but the bounce could never
  produce). The path tracer and BDPT stay **mutually consistent** (per-column
  convergence is the correctness invariant): brass lands on the BSDF/BDPT
  reference (the +4.6% BSDF+Env bias goes to −0.2%), while e.g. wood's diffuse
  shading shifts as it moves from Oren-Nayar to Lambert. The gitignored
  full-image parity goldens are regenerated.

## Capabilities

### New Capabilities

- `flat-bsdf-lobes`: the flat/`std_surface` material's BSDF is a single
  lobe-structured model consumed **identically** by `sample()` and `evaluate()`
  — one solid-angle pdf and a bounded per-lobe weight — serving as the canonical
  BSDF for path + bidirectional path tracing in both execution modes, with a
  per-lobe runtime-pluggable sampler seam (native strategies only in this
  change).

### Modified Capabilities

- `scene-sampling`: make explicit the contract that the BSDF proposal's
  **drawing** density (`sample()`) and the density used in the mixture pdf and
  NEE coupling (`evaluate()`) are **identical for all materials, including
  layered/coated ones** — the precondition that keeps the proposal mixture
  unbiased. Previously implicit; violated by layered materials (the brass bias).

## Impact

- **Shaders:** refactor `src/skinny/shaders/materials/flat/flat_material.slang`
  (sample/evaluate onto lobes); new `materials/flat/flat_lobes.slang` (lobe
  kinds, sampler dispatch, `flatBsdfResponse`); `loadFlatMaterial` drops
  `sp`/`loadStdSurfaceParams`. Recompile `main_pass.spv` with `slangc`.
- **Consumers (no API change; behavior now consistent):**
  `integrators/path.slang`, `integrators/bdpt.slang`, `wavefront/*`,
  `restir/*`, `nee.slang`, `sampling/proposal.slang`.
- **Retained for raster only:** `evalStdSurfaceBSDF` + `StdSurfaceParams`
  binding 19, used by `preview_pass.slang`.
- **Tests:** `tests/test_sampling_parity.py::test_env_proposal_unbiased_and_reduces_variance`
  is the gate; regenerate gitignored goldens
  `tests/_sampling_parity_golden_{megakernel,wavefront}.txt`; the ReSTIR suite
  (uses flat materials) must still pass.
- **Docs:** `docs/Architecture.md` (module map; narrow the binding-19 usage
  note), the flat-BSDF estimator section, `docs/Wavefront.md` if it describes
  wavefront BSDF eval; SVG lobe-flow diagram under `docs/diagrams/`.
- **Host:** `renderer.py` — no change expected (per-lobe sampler id defaults to
  native; params already packed).
