# Design — Metal megakernel watchdog tiling

## Context

`MetalContext.dispatch` (`metal_compute.py:795`) records one compute pass over
`[width, height, 1]`, commits it, and blocks on `wait_for_idle()`. That single
command buffer holds the entire frame's per-pixel work. `mainImage`
(`main_pass.slang:441`) reads `pixel = dispatchThreadID.xy` and early-outs on
`pixel.x >= fc.width || pixel.y >= fc.height`.

The wavefront path never wedges because it is already tiled (`wavefront_driver.py`:
`push_tile(streamBase)` advances a per-tile lane window, one recorded segment per
tile). The megakernel has no equivalent — it is all-or-nothing per frame.

macOS cannot cancel another process's GPU work, so an over-budget command buffer
wedges the GPU until reboot (`metal-dispatch-hygiene`). The fix must **bound each
committed command buffer**, not merely speed the kernel up.

## Decision: row-band tiling of the megakernel dispatch (Metal only)

Split the frame into `N` horizontal row bands. Dispatch each band as its own
command buffer covering `width × bandHeight` threads, with a Y offset so the
shader addresses the correct pixels. The accumulation buffer persists across the
bands (each pixel is written exactly once per frame), so N-band output is
bit-identical to one-dispatch output.

### Shader (`main_pass.slang`, `SKINNY_METAL`-gated)

Add one `FrameConstants` field `tileOriginY` (u32). In `mainImage` /
`mainImageRecord`:

```slang
uint2 pixel = dispatchThreadID.xy;
#if defined(SKINNY_METAL)
    pixel.y += fc.tileOriginY;   // band base; Vulkan dispatches full frame, origin 0
#endif
```

The existing `pixel.y >= fc.height` guard already clips the final partial band.
Vulkan SPIR-V is byte-unchanged (field is appended to the UBO tail, gate is
compile-time false). `main_pass.spv` does not need recompilation for Vulkan.

### FrameConstants plumbing

- `common.slang`: append `uint tileOriginY;` to `FrameConstants` (tail, after the
  current last field, preserving every existing offset).
- `renderer.py` `_FRAME_CONSTANTS_FIELDS`: append `("tileOriginY", 4)`; bump the
  Vulkan UBO size assert if needed. `_pack_uniforms` writes `0`; the Metal
  megakernel path overrides it per band via a cheap re-pack (the field is the last
  4 bytes, so the band loop can patch the blob in place rather than re-pack).

### Metal dispatch loop (`_render_megakernel_metal` / `metal_compute.dispatch`)

```python
bands = self._metal_megakernel_bands()      # integrator-aware, env-overridable
band_h = ceil(height / bands)
for y0 in range(0, height, band_h):
    h = min(band_h, height - y0)
    blob = patch_tile_origin_y(uniform_blob, y0)
    self.pipeline.dispatch(width, h, uniform_blob=blob, binds=..., bindless=...,
                           tile_origin_y=y0)   # one command buffer + wait per band
```

`dispatch` already commits one command buffer and drains per call, so looping it
per band gives one bounded command buffer per band with no new sync surface (still
design D4: no per-field cursor writes around an open encoder).

### Band count policy

`_metal_megakernel_bands()`:
- Base from a per-pixel cost estimate keyed on the active integrator: Path/SPPM
  eye-walk = few bands; **BDPT = more, smaller bands** (widest `s×t` work).
- Scale with `max(1, round(width*height / TARGET_PIXELS_PER_BAND))` so higher
  resolutions get more bands automatically.
- `SKINNY_METAL_MEGAKERNEL_BANDS` env override wins for tuning, mirroring the
  existing `SKINNY_METAL` volume-cap knobs.
- Non-`SKINNY_METAL` (Vulkan) → always 1 band (full-frame dispatch, unchanged).

The goal is *watchdog-safe*, not *optimal*: bands add a fixed per-band submit
overhead, so the estimate biases toward the fewest bands that keep the worst-case
band comfortably under budget. Because the tiling is invisible to the result,
mis-tuning only costs a little wall-clock, never correctness.

## Alternatives considered

- **Auto-route BDPT → wavefront on Metal** (mirror `effective_execution_mode`).
  Cheapest change and wavefront BDPT already renders the scene, but it silently
  swaps execution semantics, only rescues BDPT (a heavy Path/SPPM megakernel on a
  future scene would still wedge), and diverges the Metal megakernel from Vulkan.
  Rejected as the primary fix; tiling is general and on-spec. (Could still ship as
  a belt-and-suspenders default, but not required.)
- **Cap BDPT depth/connections under `SKINNY_METAL`.** Changes the image (parity
  shift) and only narrows, not bounds, the work. Rejected.
- **Continue a single dispatch across accumulation frames** (as volume marches do).
  The megakernel has no resumable per-pixel loop state to checkpoint; row-band
  tiling is the spatial analog and needs no new persistent state. Preferred.

## Risks

- **Per-band submit overhead** on light frames. Mitigated by the band estimate
  collapsing to 1 band when the per-pixel cost is low (Path on a simple scene ≈
  current behavior).
- **FrameConstants layout drift.** Appending a tail field must not move existing
  offsets; covered by the existing `_pack_uniforms` size assert plus a packer test.
- **Record path.** `mainImageRecord` shares the entry shape; it takes the same
  `tileOriginY` treatment so online-training record dumps also tile.
