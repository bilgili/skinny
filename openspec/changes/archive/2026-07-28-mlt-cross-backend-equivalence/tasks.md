# Tasks — mlt-cross-backend-equivalence

## 1. Measure (done — recorded in `design.md`)

- [x] 1.1 Render `int_caustic` under `(mlt, wavefront)` on Metal and on Vulkan,
      one backend per process, at the manifest budget (128×128, 512 spp, 16384
      chains), RGB and spectral. Metrics from `metrics.compute_metrics`.
- [x] 1.2 Render the same scene at the budget the old claim was measured at
      (64×64, 8 spp, 512 chains). Result: not bit-identical there either, and the
      recorded mean is not reproducible, so the claim is restated from new
      numbers.
- [x] 1.3 Control — re-render on the same backend in a second process. Both
      backends are pixel-identical with themselves (`maxdiff 0.0`).
- [x] 1.4 Attribute the difference. Max abs diff divided by the splat quantum
      `q = b / (256 × mpp_actual × frames)` is an integer on both budgets
      (69.959 and 6.000), and the per-pixel histogram clusters on ±1q, ±2q, ±3q.
      Mechanism: the Q24.8 truncation in `mltFilmSplat`. Not chain divergence.

## 2. Give the claim one owner

- [x] 2.1 `openspec/specs/metropolis-light-transport/spec.md` — apply the
      modified "MLT runs on both backends under dispatch hygiene" requirement:
      define the equivalence class (per-backend bit-reproducible, cross-backend
      bounded by the splat quantum), forbid the bit-identity claim, and require a
      budget beside any cross-backend measurement.

## 3. Restate the claim everywhere it appears

- [x] 3.1 `CLAUDE.md` — compatibility matrix MLT row and the MLT "Backends" row.
- [x] 3.2 `README.md` — compatibility matrix MLT row and the integrator table
      MLT entry.
- [x] 3.3 `docs/Wavefront.md` — the MLT section's backend sentence.
- [x] 3.4 `docs/MetropolisLightTransport.md` — add a cross-backend paragraph next
      to the existing verification note, with the measured numbers, the budget,
      and the mechanism.
- [x] 3.5 `docs/Spectral.md` — the spectral-MLT note that lists a Metal render as
      bit-identical to Vulkan. The Vulkan before/after half of that claim is a
      same-backend claim and stays.
- [x] 3.6 `CHANGELOG.md` — add the entry for this change, and qualify the
      `spectral-mlt` entry that repeats the cross-backend over-claim with the
      budget it was measured at. The two other `bit-identical` hits in the file
      (`reflection-owned-byte-layouts`, `renderer-module-carveout`) are
      **same-backend** pre/post claims; they are true and stay.
- [x] 3.7 `src/skinny/shaders/wavefront/wavefront_mlt.slang` — note beside
      `MLT_SPLAT_SCALE` that the truncation is what makes a last-bit
      cross-backend difference visible, and that it bounds that difference.

## 4. Gates

- [x] 4.1 `openspec validate --strict mlt-cross-backend-equivalence`.
- [x] 4.2 Hostless suite unchanged versus the branch point — no code path
      changes, so no test may move. Measured as a stashed control: the change
      stashed gives `17 failed, 2476 passed, 34 skipped`, the change applied
      gives `17 failed, 2476 passed, 34 skipped`, and the two FAILED sets are
      **identical**. The 17 are pre-existing in this worktree (missing corpus
      assets under `tests/pbrt/`, plus one MCP schema test); none of them reads
      a file this change touches.
- [x] 4.3 No tolerance, baseline, or manifest entry edited. Confirm with
      `git diff --stat` that `tests/pbrt/corpus/manifest.json` is untouched.
