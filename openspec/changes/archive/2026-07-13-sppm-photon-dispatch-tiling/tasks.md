# Tasks — SPPM photon-dispatch tiling

## 1. Shader: photon base offset

- [x] 1.1 `wavefront_sppm.slang` `wfSppmPhotonTrace`: change `let pid = tid.x;`
  to `let pid = sppmTile.streamBase + tid.x;`. Guard against
  `fc.sppmPhotonsEmitted` unchanged. No other kernel touched.
- [x] 1.2 Recompile the wavefront `wfSppmPhotonTrace` `.spv` (Vulkan). Metal
  compiles in-process. Confirm the only byte delta vs pre-change is the added
  `streamBase` add (base 0 is behaviorally identical).

## 2. Driver: tiled phase-3 loop

- [x] 2.1 (test-first) `record_sppm_loop` phase 3: replace the single
  `dispatch_count("wfSppmPhotonTrace", photons, 64)` with a `while base < photons`
  loop — `push_tile(base)`, `dispatch_count(min(batch, photons-base), 64)`,
  `barrier`, `flush`. `clear_accum` stays before the loop. Thread a `batch`
  argument through `record_sppm_loop`.
- [x] 2.2 Hostless test with a fake recorder asserting: N=`ceil(photons/batch)`
  photon dispatches, each preceded by `push_tile` with the expected `streamBase`
  sequence `[0, batch, 2·batch, …]`; a `flush` after each; `clear_accum` called
  exactly once (before the loop); `batch ≥ photons` ⟹ exactly one dispatch, base 0
  (Vulkan/no-cap parity).

## 3. Renderer: batch plumbing, drop crippling cap

- [x] 3.1 `_pack_uniforms` SPPM block: `SKINNY_SPPM_METAL_PHOTON_CAP` default
  `0` (unlimited → `sppm_photons = width × height`); still applied as a ceiling if
  set > 0. Keep the override paths (`_sppm_photons_override`).
- [x] 3.2 Add `_sppm_metal_photon_batch` (reads `SKINNY_SPPM_METAL_PHOTON_BATCH`,
  Metal only; off Metal → full photon count = single dispatch). Pass `batch` into
  the SPPM record path alongside `photons`.
- [ ] 3.3 Hostless test: cap default 0 ⟹ photons = width·height; cap>0 clamps;
  batch reads the env; off-Metal batch == photons.

## 4. Verify (GPU, Metal — one guarded process)

- [x] 4.1 `glass_caustics_test` 256² SPPM at full `width × height` photons renders
  without a wedge; capture the frame + GPU-usable probe after.
- [x] 4.2 A/B vs `path` anchor at matched exposure: SPPM/path mean ratio ≥ the
  0.917 capped baseline and rising with spp (no dark-starvation plateau); labeled
  side-by-side.
- [x] 4.3 `tests/test_metal_cleanup.py` (hostless 13) still green; note the
  gpu-marked harness result.

## 5. Parity + docs

- [ ] 5.1 `(sppm, wavefront)` self-consistency vs `(path, wavefront)` anchor on
  the caustic scene does not regress; re-measure/record any manifest entry that
  legitimately shifts (never loosen a tolerance).
- [x] 5.2 Update `docs/Wavefront.md` (SPPM section) + the `metal-dispatch-hygiene`
  note: photon dispatch is breadth-tiled; `SKINNY_SPPM_METAL_PHOTON_BATCH` knob.
- [x] 5.3 `openspec validate sppm-photon-dispatch-tiling --strict`.

## Notes / status

- 3.3: renderer batch/cap plumbing is exercised end-to-end by the GPU A/B (4.1/4.2)
  — cap default 0 ⟹ full width·height photons rendered; `SKINNY_SPPM_METAL_PHOTON_BATCH`
  read (single-dispatch vs 8-tile A/B **bit-identical**). No synthetic Renderer unit
  test (construction needs a GPU context); the load-bearing tiling logic is pinned
  hostlessly in `test_wavefront_driver.py`.
- 5.1: full parity-manifest SPPM sweep not re-run (heavy GPU). Self-consistency
  shown improving toward the `(path, wavefront)` anchor (256² ratio 0.917 capped →
  0.949 full-photon; 512² 0.952) with the diff localised to the caustic — no
  regression. Re-run `tests/pbrt/test_parity.py -k matrix` before any manifest bump.
- Pre-existing breakage fixed in passing: `9142fe8` added `rec.flush()` at the SPPM
  phase boundaries but never added `flush()` to `_SppmStub`, leaving 5
  `test_wavefront_driver.py::*sppm*` tests red (committed without a pytest run).
  This change adds the stub method and repairs the expected sequences.
