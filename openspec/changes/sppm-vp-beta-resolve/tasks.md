# Tasks

## 1. Implementation

- [x] 1.1 `wfSppmUpdate`: multiply decoded flux by `vp.beta` (spectral + RGB
      blocks) in `lIndirect` only — per-λ inside the spectral resolve;
      `sppmUpdate` τ fold untouched (inert, design-review finding); update the
      stale "byte-identical .spv" comment.
- [x] 1.2 Recompile shaders (spv cache) — RGB + spectral wavefront SPPM
      kernels build clean.

## 2. Validation

- [x] 2.1 Regression A/B: directly-viewed caustic scene unchanged within noise
      (glass_caustics — vp.beta ≈ 1 there).
- [x] 2.2 Demonstration A/B: scene where the caustic is seen THROUGH glass —
      SPPM moves toward bdpt reference (was under-counted).
- [x] 2.3 Parity matrix SPPM gates still pass (self-consistency vs path
      wavefront anchor on sppm combos).

## 3. Ship

- [ ] 3.1 Pre-merge review (codex or fallback review subagent).
- [ ] 3.2 Merge to main, archive change, update docs (`docs/Wavefront.md` /
      SPPM section if it states the resolve formula).
