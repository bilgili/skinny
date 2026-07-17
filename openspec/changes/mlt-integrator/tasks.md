# Tasks — mlt-integrator

Reference: pbrt-v4 at `~/projects/pbrt-v4` (`src/pbrt/cpu/integrators.cpp:2478-2750`
`MLTIntegrator`, `src/pbrt/samplers.{h,cpp}` `MLTSampler`) is the algorithmic
ground truth for every transport/sampler task below. Work in a dedicated
worktree off `main`.

## 1. Capability gate + registration (hostless, lands first)

- [x] 1.1 Add `src/skinny/mlt_capability.py` with `MLT_IMPLEMENTED = False`
      (mirror `spectral_capability.py`; monkeypatchable single source of truth)
- [x] 1.2 Register the integrator: `INTEGRATOR_INDEX["mlt"] = 3`
      (`cli_common.py:23`), `integrator_modes += ["MLT"]`
      (`renderer.py:1604`), `DEFAULT_EXECUTION_FOR_INTEGRATOR["mlt"] =
      "wavefront"` (`cli_common.py:29`); add `mlt` to `--integrator` choices
      (`cli_common.py:333`) and the headless `_INTEGRATORS` map
      (`headless.py:66`)
- [x] 1.3 Add `reject_mlt_unsupported` to `cli_common.py`: wavefront-only
      refusal (SPPM template at `:179`) + mlt × spectral / neural / ReSTIR /
      online-training refusals; wire into `validate_render_flags` and into the
      persisted-integrator call sites in `app.py` (~:503), `headless.py`,
      `web_app.py` beside the existing SPPM/spectral calls; while
      `MLT_IMPLEMENTED` is False, `--integrator mlt` itself is refused with a
      clear "not yet wired" error (never silently renders path)
- [x] 1.4 Hostless tests (model: `tests/test_sppm_selection.py` +
      `tests/pbrt/test_cli.py`): registration, index/state-hash participation,
      auto→wavefront resolution, every refusal scenario from the render-cli
      spec delta (mlt+megakernel, mlt+spectral, mlt+neural, mlt+ReSTIR,
      mlt+online-training, persisted-mlt+explicit-megakernel, mlt-alone clean)

## 2. Parity-matrix wiring (hostless, gate-off skips recorded)

- [x] 2.1 `parity.py`: `INTEGRATORS += ("mlt",)`; `combo_is_valid` rules —
      wavefront-only, RGB-only, layer-free, **explicit `material_class ==
      "flat"` gate** (new: no general flat gate exists today), gated on
      `mlt_capability.MLT_IMPLEMENTED` ("not yet wired" recorded skip while
      off); `render_linear` force-wavefront shim (SPPM precedent `:507`)
- [x] 2.2 `parity.py`: new `"mlt"` class in `combo_axis_class` +
      `_DEFAULT_SELF_CONSISTENCY` placeholder row; deterministic gate budget
      mapping — manifest `spp` = total mutations/pixel run as `spp` frames ×
      1 mutation/pixel/frame
- [x] 2.3 `tests/pbrt/test_matrix.py`: `test_mlt_is_wavefront_only`, flat-gate
      test (MLT skipped on subsurface/skin/volume scenes with recorded
      reason), update enumeration-count assertions; confirm the coverage
      meta-test passes with the new integrator

## 3. pbrt import mapping

- [x] 3.1 `pbrt/metadata.py` (~:100): emit `{"integrator": "mlt", "maxdepth",
      "mutationsperpixel", "largestepprobability", "sigma", "chains",
      "bootstrapsamples"}` (pbrt defaults 5/100/0.3/0.01/1000/100000) into
      `customLayerData["pbrt"]["skinny"]`; generalize the SPPM-only selection
      reader in `pbrt/api.py:139` and report `mlt` as mapped (`api.py:110`)
- [x] 3.2 Hostless import tests: `Integrator "mlt"` with explicit and default
      params round-trips to the skinny selection; non-mlt scenes unaffected

## 4. MLT sampler shader (net-new PSS machinery)

- [x] 4.1 `shaders/integrators/mlt_sampler.slang`: `PrimarySample` (16 B:
      value/valueBackup/lastMod/modBackup u32 iterations), fixed X budget =
      3 streams × DIMS_PER_STREAM(maxDepth) (≥192 for maxDepth 5; worst-case
      index ≈72 — overflow is a debug-assert invariant, NOT a fallback),
      3-stream indexing (`streamIndex + 3·sampleIndex`), lazy Kelemen
      `ensureReady` (large-step reset via lastLargeStepIteration; aggregated
      small step `σ·√nSmall`), `accept`/`reject` backup-restore — all verbatim
      pbrt `MLTSampler` semantics; net-new `erfInv`-based `sampleNormal`
      (inverse-CDF, exactly 1 uniform per dimension)
- [x] 4.2 numpy mirror of the sampler (mutation math + `erfInv`) +
      hostless tests: accept/reject restore exactness, large-step lazy reset,
      aggregated-small-step distribution, stream interleaving, initial-state
      bookkeeping (iteration 0, largeStep=true, lastMod=0, no startIteration
      before the initial evaluation)

## 5. MLT transport shader + wavefront sequence

- [ ] 5.1 `shaders/wavefront/wavefront_mlt.slang`: fused per-chain `wfMltEval`
      implementing pbrt `MLTIntegrator::L` — strategy pick (`nStrategies =
      depth+2`, depth-0 special case s=0,t=2, result ×nStrategies), eye subpath
      of exactly `t` / light subpath of exactly `s` via `bdpt.slang`
      `randomWalk`/`sampleLightOrigin`, single-strategy connect with existing
      `misWeight`; chain-metadata buffer {bootstrapIndex, depth, cCurrent,
      LCurrent, pCurrent, rng state, iteration counters}
- [ ] 5.2 Bootstrap kernels: fresh-fill L evaluation over `nBootstrap ×
      (maxDepth+1)` seeds writing luminance weights (breadth-tiled); host
      readback → numpy CDF, `b = (maxDepth+1)/N·Σw`, loud black-image error,
      resample `nChains` seeds, depth `k = idx mod (maxDepth+1)`,
      depth-contiguous chain layout; chain state reconstruction pass (pinned
      initial bookkeeping per design D3)
- [ ] 5.3 `wfMltMutate` (startIteration → proposal L → acceptance `min(1,
      c_p/c_c)` → dual splat `a/c_p` + `(1−a)/c_c` uint fixed-point, MLT splat
      scale from the D4 overflow inequality, NO upper clamp → accept/reject)
      + `wfMltResolve` (fold × `b/mpp_actual`, film-average; `mpp_actual =
      iterations × nChains / pixels`); decide splat-buffer reuse vs dedicated
      against the Metal argument-table budget
- [ ] 5.4 `wavefront_driver.py` `record_mlt_loop`: [reset] bootstrap phases →
      per-frame mutation iterations (breadth-tiled, 64-aligned, 65535·64
      ceiling, `flush()` at phase boundaries) → resolve; recorder primitives
      as needed; `wavefront_layout.py` `mlt_buffer_sizes` (sized by `nChains`,
      MSL-stride-aware); hostless driver tests with the recording stub
      (phase order, tiling alignment, budget math)
- [ ] 5.5 Host wiring: `_ensure/destroy_wavefront_mlt_pass` (+`_metal`) in
      `renderer.py`, dispatch selection branches (Vulkan ~:2322, Metal
      ~:9371), FrameConstants MLT fields, interactive quick-bootstrap +
      settle-debounce (D3), flip `MLT_IMPLEMENTED = True`; verify all
      existing RGB kernel `.spv` byte-identical

## 6. GPU validation + gates (guarded Metal env, one process at a time)

- [ ] 6.1 Kill harness: `tests/test_metal_cleanup.py -m gpu` passes with the
      new dispatch shapes
- [ ] 6.2 Self-consistency: measure `(mlt, wavefront)` vs the Path anchor on
      the flat suite scenes; record the `"mlt"` tolerance row harness-first
      (tighten-only); verify mega-session fallback renders path (SPPM-wart
      scenario)
- [ ] 6.3 pbrt-truth: regen suite EXRs with the pinned pbrt binary running
      `Integrator "mlt"` (`regen_refs.py`; perspective-camera-only scenes,
      recorded skips otherwise); record per-combo baselines; caustic/SDS
      discriminator scenes must not regress vs path/BDPT at the gate budget
- [ ] 6.4 Splat-scale validation on the brightest caustic suite scene: no
      wraparound at the default per-frame budget (assert headroom), Vulkan ↔
      Metal parity render

## 7. Docs + finish

- [ ] 7.1 README (`--integrator mlt`, compatibility matrix, MCMC "swim"
      note), CLAUDE.md + `docs/Architecture.md` compatibility matrix +
      binding map if a new binding landed, `docs/Wavefront.md` MLT sequence
      section, CHANGELOG
- [ ] 7.2 `ruff check src/`, full hostless pytest, `openspec validate
      mlt-integrator`; codex pre-merge review (or review-subagent fallback);
      merge from worktree, archive change
