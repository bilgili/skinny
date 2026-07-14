# Tasks — fix SPPM env-direct under-count

## 1. Diagnose (done)
- [x] 1.1 Decompose the ~20% dimness: env is the entire deficit (env-off matches).
- [x] 1.2 Isolate on an env-only scene: deficit is env DIRECT (flat plane 0.735×),
  not indirect; full-weight-NEE probe → 0.998 confirms the MIS-companion gap.

## 2. Fix
- [x] 2.1 `wfSppmEye`: at the terminal VP (before store), trace one BSDF sample and
  add its env-miss MIS companion (RGB + spectral). Gated bs.pdf>0 && !transmitted
  && furnace==0.

## 3. Verify (GPU, Metal — one guarded process)
- [x] 3.1 env-only scene: flat ground 0.735 → 0.998; whole → 0.979.
- [x] 3.2 glass_caustics_test: all regions 0.75-0.87 → 0.95-1.04; tonemapped
  path|bdpt|sppm montage brightness-matched.
- [ ] 3.3 Sphere-only (no env) no-regression + `tests/pbrt/test_parity.py -k matrix`
  SPPM self-consistency toward the (path, wavefront) anchor (was a recorded skip).
- [ ] 3.4 `tests/test_metal_cleanup.py` hostless 13 green (eye gained one trace).

## 4. Docs + review
- [ ] 4.1 docs/PhotonMapping.md + docs/Wavefront.md: SPPM eye env-direct = NEE +
  terminal-VP BSDF-miss companion.
- [ ] 4.2 openspec validate --strict; reviewer/codex pre-merge.
