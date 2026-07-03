## Why

`skinny assets/bathroom.usda --width 1280 --height 720` hangs on launch: the
window shows a busy cursor forever and no frame is ever drawn. The GLFW app is
wedged in `MetalContext.dispatch` → `device.wait_for_idle()`
(`metal_compute.py:833`) — the committed megakernel command buffer never
completes, so the main thread never returns to pump GLFW events (busy cursor),
and because it is blocked in native code the SIGTERM/SIGINT handlers never run
(only SIGKILL clears it, and per `metal-dispatch-hygiene` an abandoned over-long
kernel can wedge the GPU until reboot).

Root cause (diagnosed headless on unmodified code — the concurrent
`qt-render-threading` change is unrelated; `skinny` is the GLFW front-end it
never touched):

- `assets/bathroom.usda` was regenerated in `-mtlx` sidecar style. It now binds
  **22 image-textured graph materials** (`MaterialX per-scene-material gen: 879
  mtlx-targeted, 22 graphs`) instead of the previous all-inline `standard_surface`
  set. `~/.skinny/settings.json` had `integrator_index = 1` (**BDPT**) persisted,
  so the app launches straight into the BDPT megakernel.
- The **BDPT megakernel** builds an eye subpath and a light subpath and evaluates
  the full `s × t` connection matrix per pixel — every connection is a BSDF eval
  at both endpoints plus a shadow ray. With graph materials each BSDF eval is a
  large inlined graph shader, so the per-pixel cost is roughly
  `Path × verts × connections × graph-shader-cost`. Dispatched as a **single
  full-frame command buffer**, that tips the frame past the macOS GPU watchdog
  budget and wedges the device.

Confirmed by a 2×2 matrix (headless Metal, 320×180, same camera):

| scene | integrator | exec mode | result |
|-------|-----------|-----------|--------|
| inline materials (0 graphs) | BDPT | megakernel | **OK** (0.04 s/frame) |
| graph materials (22 graphs) | BDPT | **megakernel** | **WEDGE** (never returns) |
| graph materials (22 graphs) | BDPT | wavefront | **OK** (0.30 s/frame) |
| graph materials (22 graphs) | Path | megakernel | **OK** (2.9 s/frame @720p) |

So graph-material BSDF evaluation is **not** broken (wavefront BDPT renders the
identical scene; Path megakernel renders it too). The defect is specific to the
**Metal megakernel dispatch being unbounded**: it commits the whole frame in one
command buffer, and the BDPT × graph-material combination is heavy enough per
pixel to exceed the watchdog. This is a `metal-dispatch-hygiene` "no unbounded
command buffers" violation for the megakernel path — the existing spec only
covers per-pixel *loops* (volume marches, SSS walks), not per-pixel integrator
*breadth*.

## What Changes

- **Tile the Metal megakernel dispatch.** Under `SKINNY_METAL`, the megakernel
  frame SHALL be dispatched as a sequence of screen-space row bands, one committed
  command buffer per band, so each command buffer's total work is bounded to
  `width × bandHeight` pixels regardless of integrator breadth or material cost.
  The accumulation image already persists across command buffers within a frame,
  so the tiled result is bit-identical to the single-dispatch result. Vulkan
  keeps its single full-frame dispatch (byte-unchanged SPIR-V — the shader
  addition is `#if defined(SKINNY_METAL)` gated).
- **Bound the band height by an estimated per-pixel budget.** Band count SHALL be
  chosen so the worst-case band completes within a watchdog-safe budget; BDPT
  (widest per-pixel work) SHALL use more, smaller bands than Path. The cap is
  env-overridable (`SKINNY_METAL_MEGAKERNEL_BANDS`) for tuning, mirroring the
  existing `SKINNY_METAL` volume caps.
- **Regression coverage.** `tests/test_metal_cleanup.py` gains a gpu-marked probe
  that renders the graph-material BDPT megakernel combo and asserts the frame
  completes (no wedge) and the GPU is reusable afterward.
- **No parity shift.** Tiling only changes how the frame is committed, not what is
  computed; the parity-matrix gates and `megakernel ≡ wavefront` self-consistency
  SHALL be unchanged.

Out of scope: any change to graph-material codegen (it is correct), and the
wavefront path (already watchdog-safe by construction).
