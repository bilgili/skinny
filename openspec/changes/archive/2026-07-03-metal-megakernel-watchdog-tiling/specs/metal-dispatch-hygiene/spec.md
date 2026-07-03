## ADDED Requirements

### Requirement: Metal megakernel dispatch is watchdog-bounded by tiling

On the Metal backend, the megakernel frame SHALL be committed as a sequence of
screen-space row bands — one command buffer per band — so that each command
buffer's total work is bounded to `width × bandHeight` pixels, independent of the
active integrator's per-pixel breadth (path, BDPT connection matrix, SPPM eye
walk) or per-material shader cost (including inlined MaterialX graph materials).
No single committed megakernel command buffer SHALL cover the full frame when the
estimated per-pixel cost would risk the macOS GPU watchdog budget.

The band count SHALL be chosen from an integrator-aware per-pixel cost estimate
that scales with resolution (BDPT, the widest per-pixel work, using more and
smaller bands than the path tracer), overridable via
`SKINNY_METAL_MEGAKERNEL_BANDS` for tuning. Tiling SHALL be purely a commit-time
subdivision: the accumulation image persists across the bands of a frame and each
pixel is written exactly once, so the tiled output is bit-identical to a single
full-frame dispatch. The Vulkan backend SHALL be unaffected — it continues to
dispatch the full frame in one command buffer, and the shader addition SHALL be
`#if defined(SKINNY_METAL)`-gated so the Vulkan SPIR-V is byte-unchanged.

#### Scenario: BDPT megakernel on a graph-material scene completes without wedging

- **WHEN** the BDPT megakernel renders a scene bound to image-textured MaterialX
  graph materials on Metal (e.g. the regenerated `bathroom.usda` with 22 graph
  materials) at 1280×720
- **THEN** every per-band command buffer completes without a GPU fault or watchdog
  kill, the frame is produced, and a subsequent Metal dispatch on the same device
  still succeeds (the GPU is not wedged)

#### Scenario: tiling does not change the image

- **WHEN** the same megakernel frame is rendered with 1 band and with N bands on
  Metal
- **THEN** the accumulation images are bit-identical, and the parity-matrix gates
  and `megakernel ≡ wavefront` self-consistency anchor are unchanged
