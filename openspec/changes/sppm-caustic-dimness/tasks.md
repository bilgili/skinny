# Tasks — fix SPPM progressive normalization

## 1. Root-cause
- [ ] 1.1 Instrument per-pass `ld` for one flat-lit pixel (shader debug buffer);
  find the frame-dependent factor (~58× at frame 0). path uses the same
  `allLightsNEE` and is invariant, so the defect is SPPM-eye-specific.
- [ ] 1.2 Confirm whether the direct term should be film-averaged (constant mean)
  or resolved differently; compare against textbook SPPM (accumulated τ).

## 2. Fix + gate
- [ ] 2.1 Make SPPM radiance sample-count-invariant; converge to the
  `(path, wavefront)` anchor on `glass_caustics_test`.
- [ ] 2.2 Headless regression: SPPM linear-energy within tolerance of the path
  anchor AND invariant across {64, 256, 1024} spp (the discriminator this fails).
- [ ] 2.3 openspec validate --strict; codex/reviewer pre-merge.
