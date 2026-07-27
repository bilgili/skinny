# Tasks: gpu-backend-adapter

## 1. Baseline capture

- [ ] 1.1 Enumerate the public surface of `vk_compute` and `metal_compute`
      (classes, methods, parameter names, module functions, constants) and
      record the diff as a permanent fixture. This is the drift the
      conformance test will pin.
- [ ] 1.2 Enumerate every backend branch outside the adapter modules: the 40
      `is_metal` sites in `renderer.py`, 5 in `debug_viewport.py`, 4
      `compute_queue` probes in `vk_wavefront.py`, 12 `descriptor_sets is
      None` gates, and the `is_metal` parameter of `mlt_chain`. For each,
      record the *reason* it branches. Group into capabilities.
- [ ] 1.3 Capture Vulkan `.spv` bytes and `build/spv_cache` keys for all
      kernels — the byte-identity baseline for any define touched by 3.3.

## 2. Stage 1 — capability record, broken probes replaced

- [ ] 2.1 Add `src/skinny/gpu_backend.py`: capability record + one-sided
      member table, populated from 1.2.
- [ ] 2.2 Replace `hasattr(ctx, "compute_queue")` at all 7 sites. Record that
      this is a behaviour fix (the probe was always true) and measure: run the
      Metal wavefront suite before and after and diff images.
- [ ] 2.3 Replace the 12 `descriptor_sets is None` gates.
- [ ] 2.4 Hostless test: no attribute-presence backend probe remains.

## 3. Stage 2 — naming and argument domains

- [ ] 3.1 Unify `PreviewPipeline` / `PreviewPipelineMetal` under one name.
- [ ] 3.2 Unify argument domains: one address-mode and format vocabulary; one
      parameter name per shared method (`rgba_f32` vs `data`). Delete
      `_VKFORMAT_INTS` / `_VK_ADDRESS_INTS` once no caller passes ints.
- [ ] 3.3 Fold `BINDLESS_TEXTURE_CAPACITY` into the capability record; make
      the `bindings.slang` "MUST equal" comment a test. Verify 1.3 byte
      identity.
- [ ] 3.4 Move the private cross-module reaches (`_make_sampler`,
      `_rgba_f32_to_rgba8`) onto the interface or into their callers.

## 4. Stage 3 — recording adapter

- [ ] 4.1 Add the recording adapter: records allocations, bindings, dispatch
      group counts, readbacks; returns zero-filled data.
- [ ] 4.2 Conformance test over all three adapters, modulo the one-sided
      table; fails on any drift from the 1.1 fixture that is not declared.
- [ ] 4.3 Binding-coverage check: a recorded dispatch whose bindings do not
      cover the reflected shader globals is reported.

## 5. Stage 4 — renderer branch migration

- [ ] 5.1 Migrate branches by region, ordered by existing test coverage;
      each region green before the next.
- [ ] 5.2 For each of the 15 Metal-only renderer methods, decide and record:
      adapter implementation, or capability-gated path that stays.
- [ ] 5.3 Reassess `wavefront_driver.WavefrontRecorder.flush_heavy_eye` — a
      watchdog concept in a backend-neutral protocol; move to the capability
      record if it fits.
- [ ] 5.4 Source gate: remaining `is_metal` occurrences are adapter selection
      or genuine two-implementation splits only.

## 6. Gates

- [ ] 6.1 `ruff check src/`; full hostless `pytest`.
- [ ] 6.2 `.spv` bytes and `spv_cache` keys unchanged vs 1.3.
- [ ] 6.3 Metal: megakernel + wavefront + preview dock + debug viewport smoke.
- [ ] 6.4 Vulkan: megakernel + wavefront smoke.
- [ ] 6.5 Parity matrix dual gate unchanged; both structural and shaded
      Metal↔Vulkan parity tests pass on a dual-device host.
- [ ] 6.6 `tests/test_metal_cleanup.py` incl. the gpu-marked kill harness.
- [ ] 6.7 Docs: `docs/Architecture.md` backend seam section; CLAUDE.md
      compatibility matrix if any capability changes user-visible behaviour.
- [ ] 6.8 `openspec validate gpu-backend-adapter --strict`.
