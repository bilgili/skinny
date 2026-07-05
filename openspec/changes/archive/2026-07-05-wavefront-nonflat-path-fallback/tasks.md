# Tasks

## 1. Failing parity gate (TDD)
- [x] 1.1 Add a headless test that renders the subsurface dragon under wavefront
  BDPT and wavefront SPPM and asserts the dragon region is **non-black** and
  within tolerance of the megakernel/path anchor. Confirm it FAILS pre-fix.

## 2. Wavefront BDPT fallback
- [x] 2.1 `wavefront_bdpt.slang` `wfBdptGenEye`: replace the non-flat bail with
  `aux.escaped = PathTracer().estimateRadiance(ray, firstHit, rng)`; import the
  path integrator; keep the lane out of eye/light subpath building.
- [x] 2.2 Recompile `_wfbdpt_*` SPIR-V; confirm Slang compile is clean.

## 3. Wavefront SPPM fallback
- [x] 3.1 `wavefront_sppm.slang` eye pass: for a non-flat eye hit, add
  `PathTracer().estimateRadiance(...)` to the pixel radiance (alongside the direct
  term); do not store a visible point.
- [x] 3.2 Recompile `_wfsppm_*` SPIR-V.

## 4. Verify
- [x] 4.1 Re-render `sss_dragon_small` wavefront bdpt + sppm on Metal; dragon
  visible, ≈ megakernel. Parity gate green.
- [x] 4.2 Regression: wavefront path + megakernel path/bdpt unchanged; a
  flat-only scene (no non-flat lanes) is byte-identical.
- [x] 4.3 Metal dispatch hygiene: `tests/test_metal_cleanup.py` still passes
  (no new unbounded kernel; fallback is the existing wavefront path envelope).

## 5. Corpus + docs
- [x] 5.1 Re-measure the `dragon` corpus scene under the affected combos; record
  baselines; update its `known_divergent`/notes disposition (no longer black).
- [x] 5.2 `docs/Wavefront.md`, `CLAUDE.md` + `README.md` compatibility matrix.

## 6. Review + archive
- [ ] 6.1 codex review; fold findings.
- [ ] 6.2 `openspec validate`; merge; archive.
