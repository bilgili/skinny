# Tasks: spectral-mlt

## 1. Shader composition (SKINNY_MLT × SKINNY_SPECTRAL)

- [x] 1.1 Compile the wavefront MLT kernel set (`wfMltBootstrap`, `wfMltInit`,
      `wfMltMutate`, `wfMltResolve`) with `-DSKINNY_MLT -DSKINNY_SPECTRAL`;
      resolve any variant-interaction compile errors without touching the
      spectral BDPT estimator math — done as a conditional import + type switch
      to `SpectralBDPTIntegrator`; all 4 kernels compile.
- [x] 1.2 Verify + pin the wavelength primary-sample dimension: the estimator's
      existing `sampleWavelengths(rng.next())` draw is served by the PSS
      override (no estimator edit); worst-case ~73 << MLT_MAX_DIMS 192 —
      recorded in `common.slang` + `sampling/mlt_sampler.py` budget comments.
- [x] 1.3 Switch the scalar contribution `c` to CIE-Y luminance of the
      **gamut-clamped** resolved spectral sample under `SKINNY_SPECTRAL` —
      satisfied by composition: `estimateRadiance` returns clamped float3
      (`spectrumResolveToLinearSRGB`), fed to the existing `mltLuminance`.
- [x] 1.4 Resolve captured contributions at record-capture time through the
      existing clamped film resolve (`spectrumResolveToLinearSRGB`, clamp
      included) — satisfied by composition: eye + light-splat contributions
      resolve to clamped sRGB inside the estimator, then hit the existing
      `max(·,0)` guards in `mltEvaluate`/`mltFilmSplat`. (GPU no-negative-splat
      assertion test → Group 4.)
- [x] 1.5 Byte-identity guard: RGB wavefront MLT `.spv` (all 4 kernels) verified
      byte-identical between the edited worktree shader and main's shader with
      the same flags. (Megakernel unchanged — no megakernel edit.)

## 2. Host wiring

- [x] 2.1 Vulkan `WavefrontMltPass`: `spectral` kwarg threaded from
      `renderer._spectral`, passed to `_compile_full_spv(spectral=…)` (distinct
      `_mlt_spectral` .spv). MLT chain buffers are scalar-16B (unchanged); the
      spectral scene buffers (bindings 45–51) are already bound in scene set 0
      for a spectral session, which the MLT pass shares.
- [x] 2.2 Metal `MetalWavefrontMltPass`: `spectral` kwarg threaded; session
      compiled with `SKINNY_SPECTRAL`; MSL uniform layout reflected from the
      actual program (D6). **RESOLVED — was NOT Slang→Metal codegen.** The hang
      was TWO independent per-thread live-state overflows, in two different
      kernels (which is why fixing the first only moved it):
      1. `wfMltBootstrap` / `estimateRadiance` — `eye[7]` + `lightPath[7]`
         (`SpectralBDPTVertex`) + the `eyeRgb[7]`/`litRgb[7]` RGB mirrors +
         `misWeight`'s two internal copies, all live across the t≥2 connection
         double loop. Fixed by `misWeightS` (bdpt_spectral.slang): MIS over the
         spectral arrays lifting only the 4 scalars MIS reads (pdfFwd/pdfRev/
         isDelta/kind), deleting both mirrors and both internal copies; plus
         `mltProposalRecords` (binding 57) so captured records go to device
         memory instead of a thread-local `MltRecord[7]` in every MLT kernel.
      2. `wfMltMutate` / `RNG.reject()` — the fixed 192-trip read-modify-write
         scan over `mltPrimarySamples`. Fixed by `RNG.maxDim`, the high-water
         primary-sample index touched this iteration: reject scans only the ~70
         dims actually visited. Restore semantics unchanged (every dim with
         `lastMod == currentIteration` lies below `maxDim`; the `lastMod` guard
         already skips untouched dims in range).
      Isolated by single-variable A/B on int_caustic (64², 64 chains, 1 mutate
      dispatch, reject-scan stubbed vs real — identical otherwise): stub OK in
      0.01s, real hangs. Ruled out with measurements, NOT argument: Slang→Metal
      codegen, loop bounds (`eyeLen` 2–4 / `lightLen` 1–4 / trips 0–3 vs a cap of
      44), MSL uniform layout (all 4 entries reflect byte-identical `fc`, 688 B /
      73 fields, RGB ≡ spectral), the Metal 31-buffer argument table (22/24/25
      distinct slots, max index 31 — same as the passing kernels), dispersion
      (`mat_diffuse` hung too), and `[loop]`/unroll on the reject scan (measured:
      no effect). NOTE: an earlier `MTL_SHADER_VALIDATION=1` run appeared to pin
      the hang to `create_compute_pipeline` — that was a validation artifact;
      without it pipelines build in 0.4s.
      **Verified three-way bit-identical** (int_caustic 64², 8 spp, 512 chains):
      HEAD/Vulkan (orig `misWeight`) ≡ fix/Vulkan ≡ fix/Metal, `maxdiff 0`,
      mean 0.27545003 — the whole refactor is a no-op on output. Full Metal
      render 1.7 s (was: never returned).
- [~] 2.3 Spectral-MLT MSL tail-offset round-trip: structurally satisfied (the
      tail is reflected from the compiled spectral program, D6). A dedicated
      hostless pin needs Metal reflection → folded into Group 4 GPU verify.

## 3. Gate flips

- [x] 3.1 `cli_common.reject_mlt_unsupported`: spectral refusal dropped (docstring
      + `del spectral`); `reject_spectral_unsupported` already integrator-agnostic
      (admits `mlt`). Front-end help/CLI wording flows from the shared functions.
- [x] 3.2 `parity.combo_is_valid` + `spectral_envelope`: admit
      `(mlt, wavefront, spectral, flat)`; MLT/SPPM both marked wavefront-only
      under the spectral envelope; every other MLT exclusion kept.
- [x] 3.3 Hostless gate tests updated + passing: `test_mlt_selection.py`
      (spectral accepted, megakernel/neural/ReSTIR/online-training still refused),
      `tests/pbrt/test_matrix.py` (spectral envelope includes mlt; flat-only,
      wavefront-only skips). 158 matrix/gate + 19 selection tests green.

## 4. Validation (GPU, harness-first)

- [x] 4.1 NOT APPLICABLE as written — closed deliberately, no refs regenerated.
      The harness carries ONE pbrt reference per scene (`SceneSpec.ref`, e.g.
      `refs/suite_int_caustic.exr`); `pbrt_truth_result` gates EVERY combo against
      it and records known mismatches as per-combo `baselines`. Re-rendering those
      refs with `Integrator "mlt"` would replace the shared truth image that
      path/bdpt/sppm and all spectral combos are gated against, silently
      invalidating every baseline already in the manifest. MLT converges to the
      same radiance as path/BDPT — which is exactly what the existing
      `int_caustic` `mlt|wavefront` baseline (relmse 0.1207, flip 0.0515,
      measured against that shared ref in change mlt-integrator) asserts. Spectral
      MLT is therefore the same shape as spectral SPPM: same ref, new per-combo
      baseline entries → task 4.4. (`regen_refs.py --integrator` exists to force
      pbrt to match skinny's ANCHOR on a scene authored with another integrator,
      not to mint a per-integrator truth; the harness has no per-integrator ref
      field.) The three gated suite refs already exist and are unchanged.
- [x] 4.2 Vulkan smoke DONE: spectral MLT bootstrap→mutate→resolve renders
      int_caustic (96², 2048 chains/bootstrap, 48 spp) in 23.9s — image
      structurally matches the spectral path anchor; **means agree** (mlt 0.400
      vs path 0.377), **no NaN/inf/negatives** (no-negative-splat invariant,
      task 1.4, holds). Composition functionally correct + unbiased.
- [x] 4.3 Metal parity DONE (unblocked by the task 2.2 fix — it was a live-state
      overflow, not a perf pathology; no chain-batch retuning was needed).
      int_caustic 64², 8 spp, 512 chains, `SKINNY_MLT_METAL_CHAIN_BATCH=512`:
      Metal renders in 1.7 s and is **bit-identical to Vulkan** (`maxdiff 0`,
      PSNR inf, FLIP 0, mean 0.27545003 both) — and identical to the pre-change
      HEAD Vulkan image, so `misWeightS` has zero MIS drift vs `misWeight`.
      Kill-harness re-run (`tests/test_metal_cleanup.py -m gpu`) still owed.
- [x] 4.4 DONE. Measured harness-first on Metal, NO tolerance loosened. Spectral MLT
      is unbiased but Markov-correlated: at the suite's 256 spp its self-consistency vs
      the spectral anchor exceeded the tighten-only 0.15 `mlt` row on four scenes
      (int_caustic 0.339, spec_prism 0.173, mat_pbr_glass 0.180, mat_pbr_plastic_pc
      0.248). Confirmed VARIANCE not bias — 4× spp collapses int_caustic 0.339→0.036
      and pbrt-truth 0.229→0.092 while the mean holds (1.060→1.058× pbrt, RGB MLT
      1.054×); axis decomposition reproduced the recorded rows exactly (MLT-vs-path RGB
      0.0986 ≈ 0.098; spectral-path 0.0806 ≈ 0.085) and the anchor choice accounts for
      only ~0.03. Resolution: those four scenes were raised 256⇒512 spp (per user
      decision) rather than relaxing the row — all now pass the stock 0.15/0.12.
      Recorded baselines: int_caustic `mlt|wavefront|spectral` 0.1295/0.0523;
      spec_prism `mlt|wavefront|spectral` 0.3004/0.0791 + `mlt|wavefront` 0.1594/0.0616
      (the latter a PRE-EXISTING coverage gap — RGB MLT was never baselined on
      spec_prism, and its image is bit-identical before/after this change). The stale
      int_caustic `mlt|wavefront` 0.1207 baseline was REMOVED, not loosened: at 512 spp
      it measures 0.0905, inside the scene's own 0.12 tol. Superseded text:
      ~~Measure and record the spectral MLT self-consistency tolerance vs the
      spectral `(path, wavefront)` anchor and per-combo pbrt-truth `baselines`
      — per-scene baseline on the prism (pre-committed; dispersion stacks on
      the recorded spectral estimator divergence); never loosen an RGB or
      existing spectral tolerance.~~
- [x] 4.5 Full suite matrix gate GREEN on Metal including the spectral-mlt combos:
      `32 passed, 6 skipped, 0 failed` (429 renders, 48 min,
      `tests/pbrt/test_suite.py::test_suite_matrix_gate`). Skips = the 5 furnace scenes
      (gated separately by design) + `mat_textured_mtlx`, whose Metal `wfBdptWalk`
      pipeline fails to build ("declares 41 globals" vs Metal's 31-buffer argument
      table — the documented metal-record-drain failure mode). That skip is
      PRE-EXISTING and unrelated: verified identical at c5fbe9a (pre-fix). NOTE it
      surfaces as a SILENT skip, so that scene currently has no Metal GPU coverage —
      follow-up, not spectral-mlt's. NOT run: the full non-suite corpus sweep
      (`tests/pbrt/test_parity.py -k matrix`).

## 4b. PARKED follow-up: PSS dimension budget (correctness, not blocking)

- [ ] 4b.1 Spectral MLT consumes ~**160 of MLT_MAX_DIMS=192** primary-sample
      dimensions (RGB peaks 88) — only 17% headroom. Any draw past the budget hits
      `RNG.ensureReady`'s escape hatch, which returns a FRESH HASH that is **not
      restored on reject**, so detailed balance breaks and the chain stops being a
      correct Metropolis sampler. Failure mode is SILENT BIAS, not a crash, and a
      deeper path or a scene with more lights tips it over.
      Measured configurations are inside the budget, so this does not block the
      change — but it has no margin.
      **Do NOT fix it with a pbrt-style stream split.** That was attempted and
      REVERTED: `rng.startStream(1u)/(2u)` in `estimateRadiance` hangs
      `wfMltBootstrap` on Metal (single-variable bounded A/B, 64 chains / 1
      dispatch: split present ⇒ never returns; split absent ⇒ 0.02 s, full probe
      2.85 s). Mechanism is NOT the reject scan — bootstrap never calls `reject()`
      — it is that stream 0 keeps `index = 3*sampleIndex`, so the eye walk alone
      can drive the index past 192 and push draws into the overflow branch at high
      frequency in the kernel already at the Metal live-state cliff. The attempt is
      archived verbatim as `stream_split_ATTEMPT.patch` beside this file.
      Viable directions instead: raise `MLT_MAX_DIMS` (size `mltPrimarySamples` to
      match; note 192 was the reject-scan length that originally hung, so pair it
      with the `RNG.maxDim` bound), or pack dimensions densely rather than the
      strided `streamIndex + MLT_NUM_STREAMS*sampleIndex` layout.

## 5. Docs + finish

- [x] 5.1 Docs updated: CLAUDE.md (MLT row RGB+spectral, scope, capability gate,
      spectral scope row, Metal live-state rule in the backends row), README
      compatibility matrix (2 rows), `docs/Spectral.md` (new `## Spectral MLT`
      section: target-function framing, convergence evidence, the two Metal
      live-state constraints), `docs/Wavefront.md` (MLT stages: RGB+spectral + the
      three per-kernel Metal rules), CHANGELOG Unreleased/Added.
- [~] 5.2 NOT yet satisfiable: `spectral-bdpt-megakernel` and `spectral-wavefront` are
      still ACTIVE changes (only `spectral-rendering` is archived, 2026-07-10), so
      spectral-mlt must NOT be archived before them. Blocks archiving only, not the
      commit. Original: (`spectral-rendering`,
      `spectral-bdpt-megakernel`, `spectral-wavefront` archived first;
      re-validate the MODIFIED delta bases against the then-current specs)
- [~] 5.3 `openspec validate spectral-mlt` clean; `ruff check src/` clean; full hostless
      sweep shows ZERO regressions (18 pre-existing failures, identical set before/after,
      diffed against HEAD by stashing). Metal kill harness green (13 hostless + 3 gpu).
      Codex pre-merge review: CLEAN (`codex review --base main`, gpt-5.5) — "no
      discrete, introduced correctness issues in the diff".
      REMAINING: merge from the worktree (blocked on 5.2 for ARCHIVING only).
