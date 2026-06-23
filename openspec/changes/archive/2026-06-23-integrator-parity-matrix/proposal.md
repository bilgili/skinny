## Why

Parity is verified today for exactly **one** renderer combo — the default `path`
integrator in `megakernel` mode — against six tiny (128²) synthetic corpus
scenes. There is no standing guarantee that BDPT, SPPM, ReSTIR DI, or the neural
proposal agree with the path tracer, that `megakernel` and `wavefront` produce
the same image, or that the real heavy scenes the user renders — `bathroom.usda`
(imported from pbrt `contemporary-bathroom`) and the dragon — match pbrt at all.
`assets/bathroom.usda` currently does **not** match its pbrt reference, and that
regression went unnoticed precisely because nothing gates it. As new features
land (a new integrator, a new BSDF lobe, a new sampler) there is no matrix that
re-tests every renderer combination against a reference.

## What Changes

- Generalize the single-combo parity gate into an **integrator × execution-mode
  parity matrix** that sweeps `{Path, BDPT, SPPM}` × `{megakernel, wavefront}`
  plus the **ReSTIR DI** and **neural directional proposal** axes, encoding which
  combinations are valid vs. skipped-by-design (SPPM is wavefront-only; the
  neural proposal is wavefront + flat-material only; BDPT ignores the neural
  proposal; ReSTIR DI is a direct-light reuse layer). The matrix is **data-driven
  and extensible**: a new scene, integrator, mode, or feature flag is one entry,
  and every valid combo is exercised automatically.
- Add a **dual gate** per scene: (a) **pbrt-truth** — exposure-aligned
  relMSE/FLIP vs the pbrt v4 reference EXR (looser tolerance on heavy scenes with
  known residuals); and (b) **self-consistency** — every valid integrator×mode
  combo must agree with a designated golden combo within a tight tolerance
  (`megakernel ≡ wavefront`, `BDPT ≈ Path`, etc.), so a shared bug and a
  one-combo divergence are both caught.
- **Standardize image metrics** into one canonical battery used everywhere the
  harness reports a number: error vs reference (`MSE`, `RMSE`, `MAE`, `relMSE`,
  `PSNR`, `FLIP`) plus single-image quality stats (`variance`, a noise-σ
  estimate, and a **firefly** outlier fraction). One `ImageMetrics` struct, one
  `compute_metrics()` entry point — gates assert on the relevant fields and log
  the full battery, so no two call-sites compute "error" differently.
- Add **`bathroom`** and **`dragon`** as first-class corpus scenes with pbrt v4
  reference EXRs (rendered from `~/projects/pbrt-v4-scenes/contemporary-bathroom`
  and the dragon scene via the pinned pbrt binary at low resolution for CI cost).
- **Record current mismatches as tracked baselines** rather than fixing them
  here: where bathroom / dragon do not yet meet pbrt-truth tolerance, the harness
  records the measured delta and marks that combo `xfail`/baseline (gate stays
  green, delta is visible and pinned against further regression). The
  bathroom-mismatch and dragon-brightness **fixes are explicit follow-up
  changes** unblocked by this harness.
- Provide a single `pytest` entry point that runs the whole matrix and a `not
  gpu` import-only half that runs everywhere, so "tests always work for all
  integrators" holds on any host.

## Capabilities

### New Capabilities
- `render-parity-matrix`: a data-driven integrator × execution-mode × proposal/
  reuse parity matrix with a validity table, dual gating (pbrt-truth +
  self-consistency), heavy reference scenes (bathroom, dragon), recorded-baseline
  handling for known mismatches, and an extensibility contract so new
  renderer features are automatically swept.

### Modified Capabilities
<!-- None. The existing `integrator-convergence` convergence REQUIREMENTS are
     unchanged; this change adds the harness that enforces them across every
     combo. No spec-level behavior of an existing capability changes. -->

## Impact

- **Code**: `src/skinny/pbrt/parity.py` (matrix specs, validity table, dual-gate
  evaluate, golden-combo self-consistency, baseline records), `src/skinny/pbrt/
  metrics.py` (standardized `ImageMetrics` battery: MSE/RMSE/MAE/relMSE/PSNR/FLIP
  + variance/noise-σ/firefly), `tests/pbrt/test_parity.py` (matrix parametrization
  over combos) + `tests/pbrt/test_metrics.py` (battery unit tests), `tests/pbrt/
  corpus/manifest.json` (bathroom/dragon scenes + per-combo tolerances + baselines).
- **Assets/refs**: new `tests/pbrt/corpus/refs/bathroom.exr`,
  `dragon.exr` (low-res pbrt v4 renders) and the corresponding corpus `.pbrt`
  source (or a reference to the imported `.usda`).
- **Dependencies**: no new runtime deps. Reference generation uses the existing
  local pbrt v4 binary (`~/projects/pbrt-v4/build/pbrt`) offline; the gate itself
  needs only the checked-in EXRs (no pbrt at test time).
- **Shaders**: none — this is verification infrastructure, no `.slang`/`.spv`
  change.
- **CI/test time**: grows with the combo count; mitigated by small resolutions,
  per-combo spp budgets, and the `not gpu` import-only tier for hostless runs.
- **Docs**: `docs/Architecture.md` (or a new parity section), `README.md`
  compatibility-matrix cross-reference, `CHANGELOG.md`.
