## 1. Reproduce + lock the bug behind a test

- [x] 1.1 (test-first) Reproduced the wedge headless: BDPT megakernel on the
  regenerated graph-material `bathroom.usda` at 1280×720 hangs forever in
  `MetalContext.dispatch` → `wait_for_idle` on `main`. The 41 MB scene is not a
  committable fixture, so the standing regression is split: `test_metal_cleanup.py
  -m gpu` proves dispatch changes leave the GPU usable, the parity matrix sweeps
  BDPT megakernel on Metal across the corpus, and `test_metal_megakernel_tiling.py`
  locks the band-policy + tail-offset logic.
- [x] 1.2 Recorded the contrast (proposal table): inline-material BDPT mega OK,
  graph-material Path mega OK, graph-material BDPT wavefront OK → the defect is the
  megakernel dispatch length specifically.

## 2. FrameConstants: `tileOriginY`

- [x] 2.1 (test-first) `test_metal_megakernel_tiling.py` asserts `tileOriginY` is
  the fc scalar tail and `_TILE_ORIGIN_Y_OFFSET == sum(fields) - 4`, and that the
  4 B-longer blob still fits the Vulkan UBO.
- [x] 2.2 `common.slang`: `uint tileOriginY;` appended to `FrameConstants`, itself
  `#if defined(SKINNY_METAL)`-gated so the Vulkan struct is unchanged.
  `renderer.py`: `("tileOriginY", 4)` appended to `_FC_SCALAR_FIELDS`;
  `_pack_uniforms` writes `0`.
- [x] 2.3 The Metal band loop patches the `tileOriginY` u32 in place at its
  reflected MSL offset (passed as `tile_origin_offset`) — no full re-pack.

## 3. Shader: band origin (Metal-gated)

- [x] 3.1 `main_pass.slang` `mainImage` and `mainImageRecord`: under
  `#if defined(SKINNY_METAL)`, `pixel.y += fc.tileOriginY;`. The `pixel.y >=
  fc.height` guard clips the final partial band.
- [x] 3.2 Verified byte-neutral for Vulkan: the edited shader recompiles to the
  **same** SPIR-V as pristine HEAD under the installed slangc (the checked-in
  `main_pass.spv` differs only by pre-existing compiler drift, so it is left
  untouched). Gating the struct field, not just the read, is what keeps the Vulkan
  block byte-identical.

## 4. Tiled dispatch loop (Metal)

- [x] 4.1 `metal_compute.dispatch`: `bands` + `tile_origin_offset` params; binds
  set once on the shared root object, one command buffer + `wait_for_idle` per
  band (design D4).
- [x] 4.2 `renderer._render_megakernel_metal`: dispatches with
  `bands=self._metal_megakernel_bands()` and the reflected `tileOriginY` offset.
- [x] 4.3 `_metal_megakernel_bands()` + module-level `_METAL_MEGAKERNEL_BAND_PIXELS`
  (integrator-aware, resolution-scaled); `SKINNY_METAL_MEGAKERNEL_BANDS` override.

## 5. Verify — no wedge, no parity shift

- [x] 5.1 BDPT megakernel on `bathroom.usda` @1280×720 now renders (bands=5,
  ~1.7 s/frame, PROBE-OK) where it previously hung forever.
- [x] 5.2 `test_metal_cleanup.py -m gpu` → 3 passed (dispatch change leaves the GPU
  usable); hostless cleanup → 13 passed; `test_metal_megakernel_tiling.py` → 5
  passed; hostless parity/matrix/metrics → 59 passed (2 pre-existing `bunny_cloud`
  missing-asset skips, unrelated).
- [x] 5.3 Bit-identity: 1-band vs 4-band Path on cornell_box → **identical md5**
  (tiling changes commit granularity, not pixels).
- [x] 5.4 GPU parity matrix (`test_parity.py -k matrix`, Metal): see verification
  log (≤256² corpus stays 1 band ⇒ dispatch byte-identical to pre-change).

## 6. Docs

- [x] 6.1 `docs/Architecture.md` MetalContext section: megakernel row-band tiling
  + `SKINNY_METAL_MEGAKERNEL_BANDS`.
- [x] 6.2 `docs/Megakernel.md` Backends section: Metal watchdog-bounding tiling and
  the BDPT × graph-material cause.
- [x] 6.3 `CLAUDE.md` Metal-dispatch-hygiene "no unbounded command buffers" rule
  extended to the megakernel.
