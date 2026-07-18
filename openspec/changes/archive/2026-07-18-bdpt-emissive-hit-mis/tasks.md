## 1. Shared helper

- [x] 1.1 Add `emitterHitMisWeightT0(BDPTVertex eye[BDPT_MAX_VERTS], int s)` in `bdpt.slang`, placed after `misWeight` / `convertSAtoArea`. It reads `z = eye[s-1]`, `prev = eye[s-2]`; computes `lum = Rec709-lum(z.emission)` (SKINNY_SPECTRAL-guarded rgb extract), `pdf_zsm1_rev = fc.emissiveTotalPower > 0 ? lum / fc.emissiveTotalPower : 0`, `pdf_zsm2_rev = convertSAtoArea(cosOut/π, z.position, prev)` with `cosOut = max(dot(z.N, normalize(prev.position - z.position)), 0)`; returns `misWeight(eye, s, eye, 0, pdf_zsm1_rev, pdf_zsm2_rev, 0, 0)`.

## 2. Patch the three t=0 branches

- [x] 2.1 `bdpt.slang` (`BDPTIntegrator`, ~line 1268): replace `if (noNeePartner) L += …;` with `if (noNeePartner) L += full; else L += throughput·emission · emitterHitMisWeightT0(eye, s);`, keeping the existing `#if SKINNY_SPECTRAL … .rgb` accumulation form in both branches.
- [x] 2.2 `wavefront/wavefront_bdpt.slang` (staged, ~line 2497): same replacement; `eye` is `BDPTVertex` and `L` is `Spectrum`/`float3` per build, so `throughput·emission·w` works for both with no `#if`.
- [x] 2.3 `bdpt_spectral.slang` (`SpectralBDPTIntegrator`, ~line 996): move `BDPTVertex eyeRgb[…]; mirrorRgb(eye, eyeRgb);` above the `t = 0` loop; else-branch calls `emitterHitMisWeightT0(eyeRgb, s)`; remove the now-duplicate `mirrorRgb(eye, eyeRgb)` before the generic connections (keep the `litRgb` mirror).

## 3. Compile

- [x] 3.1 Metal is validated in-process (no `.spv` needed). If `slangc` is available, recompile the affected wavefront BDPT `.spv` kernels; note the Vulkan `main_pass.spv` megakernel recompile as the follow-up boundary otherwise.
- [x] 3.2 `.venv/bin/ruff check src/` clean; no host Python changed (shader-only + manifest).

## 4. GPU validation + baselines (Metal, harness-first)

- [x] 4.1 Export `VULKAN_SDK` + `DYLD_LIBRARY_PATH`; run `PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest tests/pbrt/test_suite.py -k "test_suite_matrix_gate and mat_emissive" -m gpu -q`.
- [x] 4.2 Measure post-fix `bdpt|megakernel`, `bdpt|wavefront` (+`|spectral`) relMSE/FLIP on `mat_emissive` and `mat_emissive_mtlx`; confirm each moved DOWN toward the path anchor (0.0522) and `bdpt|megakernel ≡ bdpt|wavefront` (RGB self-consistency ≈ 0).
- [x] 4.3 Lower those baselines in `tests/pbrt/corpus/manifest.json` (raw-text edit to preserve the hybrid JSON format). Never raise any baseline; leave `sppm|*` untouched. Re-run 4.1 to confirm the gate passes on the lowered baselines.
- [x] 4.4 Sanity: `int_caustic` (specular→light, delta bounce ⇒ full-weight branch) is unchanged by the fix; run its gate to confirm no drift.

## 5. Verify + review + docs

- [x] 5.1 Hostless suite green: `.venv/bin/python -m pytest tests/pbrt/test_matrix.py tests/pbrt/test_metrics.py tests/pbrt/test_parity.py tests/pbrt/test_suite.py -m "not gpu"`.
- [x] 5.2 `openspec validate bdpt-emissive-hit-mis --strict` passes.
- [x] 5.3 Codex (or review-subagent fallback) over the diff; fold findings back in.
- [x] 5.4 Docs upkeep: update `docs/ReSTIR.md`? no — BDPT MIS lives in the integrator; update `CHANGELOG.md` (Fixed) and, if it documents the BDPT MIS partition, `docs/Wavefront.md` / architecture notes. Confirm no other doc references the dropped-emission behaviour.
