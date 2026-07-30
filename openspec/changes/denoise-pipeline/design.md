## Context

skinny writes one linear high-dynamic-range running mean per pixel into the
accumulation image (binding 2). Two kernels do this: the megakernel entry
`mainImage` and the wavefront resolve kernels. Each of those kernels then runs a
**display tail** in the same dispatch — exposure, tonemap, sRGB encoding, HUD
overlay, gizmo overlay — and writes the display image (binding 1). The megakernel
tail also draws two editor overlays that need the primary ray and the scene hit:
the focus plane and the furnace over-energy tint.

Everything downstream reads one of those two images. `render_headless()` reads
the display image. The parity harness (`pbrt/parity.py`) reads the accumulation
image, on purpose, so tonemapping cannot move a gate. The EXR writer reads the
accumulation image.

The renderer has one resolution. `renderer.width` and `renderer.height` size the
swapchain, the accumulation image, the HUD, the gizmo overlay, and every
size-dependent buffer that `gpu_resources.DECLARATIONS` marks.

There are no auxiliary images. The renderer produces no albedo, no normal, no
depth, and no motion vector.

Two facts decide the Metal approach. First, `slangpy` 0.42 exposes
`NativeHandleType.MTLDevice`, `MTLTexture`, `MTLCommandQueue`, and
`MTLCommandBuffer`, with `Device.native_handles`, `Texture.native_handle`, and
`CommandEncoder.native_handle`. So the renderer can hand a slang-rhi texture to a
Metal framework. Second, MetalFX has no C interface, and PyObjC is not installed
in this repo. The Objective-C bridge is a new, optional dependency.

The user chose three things: the denoised image reaches file output as well as
the viewport; MetalFX runs as a denoised **upscaler**, not at 1:1; and the second
implementation is NRD, which is shader-based and needs motion vectors, view
depth, normal, roughness, and hit distance.

## Goals / Non-Goals

**Goals:**

- One denoiser seam that MetalFX and NRD both satisfy without changing the
  renderer.
- MetalFX denoising and upscaling on the native Metal backend.
- The denoised image reaches the viewport, `render_headless()`, screenshots, and
  the EXR writer.
- Zero change to the rendered image, to any `.spv` byte, and to any parity gate
  while no denoiser runs.
- The renderer never names a vendor.

**Non-Goals:**

- The NRD implementation. It needs demodulated diffuse and specular radiance,
  which the integrators do not produce today.
- A denoiser on the Vulkan backend. MetalFX is Metal-only, and the CLI refuses
  the combination.
- Denoising inside the transport. The denoiser never writes the accumulation
  image and never feeds a filtered value back into a sample.
- A denoiser axis in the parity matrix. The matrix sweeps transport, and the
  denoiser is post-processing.
- Motion vectors for animated or skinned geometry. Version 1 reprojects with the
  camera only.

## Decisions

### D1 — The seam sits after accumulation and before display

The denoiser reads the accumulation image and writes a **separate** linear image.
It never writes the accumulation image.

This one rule protects the whole regression estate. `pbrt/parity.py` reads the
accumulation image, so both parity gates keep their current inputs and need no
denoiser axis. Progressive accumulation stays a pure running mean, so a filtered
value never re-enters a sample.

*Alternative rejected:* write the denoised result back into the accumulation
image. Every consumer would then need no change, but the filtered image would
feed the next frame's running mean, and every recorded parity number would move.

### D2 — The existing display tails stay; a new display pass overwrites them

`main_pass.slang` and `wf_display.slang` keep their display tails, byte for byte.
When a denoiser runs, a new compute pass `display_resolve.slang` runs after the
denoiser and **overwrites** the display image at output resolution.

The cost is one redundant tonemap per pixel per frame. The benefit is that the
no-denoiser path is not merely equivalent, it is the same code — no new uniform
branch in two hot kernels, on two backends, across the whole shader-variant
matrix.

`display_resolve.slang` calls the existing `wfWriteDisplay` helper, so there is
still one definition of exposure, tonemap, sRGB encoding, HUD, and gizmo.

The megakernel's focus-plane and furnace overlays need the primary ray and the
scene hit, which the display pass does not have. Both are editor visualisers, and
the wavefront tail already omits them. They are **off while a denoiser runs**.
The design records this as a known limit, not as a defect.

*Alternative rejected:* gate the existing tails with a uniform. It saves one
cheap pass and costs an edit to the two hottest kernels on both backends.

### D3 — Auxiliary images come from one dedicated pass, not from the integrators

A new compute pass `gbuffer.slang` traces **one primary ray per pixel** and
writes the auxiliary images.

The integrators already compute the first hit, so writing the auxiliary images
there looks cheaper. It is not. That edit lands in the megakernel and in the
wavefront shade stages, for the path, BDPT, SPPM, and MLT integrators, on two
backends, across every entry in the shader-variant matrix — and it changes
`main_pass.spv`. The dedicated pass is integrator-independent, execution-mode
independent, and **cannot perturb an existing kernel**.

The pass runs at render resolution, once per frame, because a temporal denoiser
needs auxiliary images that match the jitter of the frame it denoises. One extra
primary-ray trace is a small fraction of one path-tracing sample.

### D4 — Auxiliary images are a named registry, and the denoiser declares what it needs

`denoise.py` declares each auxiliary image once: its name, its format, and its
semantics. A `Denoiser` implementation exposes `required_aovs`. The renderer
allocates and the pass writes **only** the declared images.

Version 1 registers four:

| Name | Format | Contents |
|------|--------|----------|
| `diffuse_albedo` | RGBA16F | first-hit diffuse albedo in rgb |
| `specular_albedo` | RGBA16F | first-hit specular albedo in rgb, roughness in a |
| `normal_depth` | RGBA32F | world normal in xyz, linear view depth in w |
| `motion` | RG16F | pixel-space motion vector |

This is the extension point the NRD change uses. NRD adds
`diffuse_radiance_hitdist` and `specular_radiance_hitdist` to the registry, and
the renderer code does not change — only the pass that produces them.

*Alternative rejected:* a fixed auxiliary-image set. The union of what MetalFX
and NRD want is already wider than either one needs, so a fixed set means every
host pays for images its denoiser ignores.

### D5 — The renderer gains a render resolution separate from its output resolution

`--denoise-scale` sets the ratio. `denoise.resolve_render_extent(output_w,
output_h, scale)` is the one pure function that derives the render extent. With
no denoiser the scale is 1.0 and the two extents are equal, which is today's
behaviour exactly.

Ownership follows the existing seam. `gpu_resources` already owns which resources
are size-dependent, so its sizes record gains `render_width` and
`render_height`. Each declaration states which extent it sizes on:

- **Render extent:** accumulation image, auxiliary images, light-splat buffer,
  every wavefront record buffer, ReSTIR reservoirs.
- **Output extent:** display image, denoised image, HUD overlay, swapchain.

`FrameConstants` already carries `width` and `height` as the render extent, which
is what every trace kernel indexes with. It gains `outputWidth` and
`outputHeight` for the display pass. `slang_layout.py` owns those fields, so the
packer and the pinned goldens move together.

Screen-space input keeps working because the HUD, the gizmo, and the swapchain
all stay at output extent. Tool picking maps an output pixel to a render pixel at
one site.

### D6 — Jitter is reported, not random, while a temporal denoiser runs

A temporal denoiser reconstructs detail from the sub-pixel offset of each frame.
It must be told the offset that was used. The primary ray takes its offset from a
Halton (2,3) sequence when `fc.jitterMode` selects it, and the renderer reports
the same offset to the denoiser.

The uniform gates it, so the image with no denoiser is bit-unchanged. The image
with a denoiser deliberately differs — a different sampling sequence is the
point.

*Alternative rejected:* keep the random offset and report zero jitter. The
denoiser then upscales without the sub-pixel information it needs, and upscaling
becomes interpolation.

### D7 — The `Denoiser` protocol is narrow and stateful

```python
class Denoiser(Protocol):
    name: str
    required_aovs: frozenset[str]
    def resize(self, render_w, render_h, output_w, output_h) -> None: ...
    def reset(self) -> None: ...
    def denoise(self, color, aovs, frame) -> object: ...
    def destroy(self) -> None: ...
```

`denoise()` takes the linear colour image, the auxiliary images the denoiser
declared, and a per-frame record (jitter offset, exposure, camera matrices,
frame index). It returns the denoised image. The implementation owns everything
else: its device interop, its history, and its command submission.

`reset()` exists because a temporal denoiser carries history that a camera cut
invalidates. The renderer calls it from the same place it resets accumulation, so
there is one reset owner.

`required_aovs` is what makes the protocol honest for two very different
implementations. MetalFX wants normal, roughness, depth, and motion. NRD wants
those plus demodulated radiance. Neither implementation asks the renderer to
produce what it will not read.

The registry is a plain dict from name to class, plus `create_denoiser(name,
ctx)`. There is no plugin discovery and no configuration file. Two entries do not
need a framework.

### D8 — MetalFX runs on its own command queue, built from the shared device

`denoise_metalfx.py` reads `device.native_handles` for the `MTLDevice` pointer,
wraps it with `objc.objc_object(c_void_p=…)`, and builds one
`MTLFXTemporalDenoisedScaler` plus one `MTLCommandQueue`. Each texture reaches
MetalFX through `Texture.native_handle`.

MetalFX encodes into its own command buffer on its own queue. The frame boundary
is `device.wait_for_idle()`, which the renderer already reaches, so the ordering
is: trace and auxiliary passes submit and drain, MetalFX submits and drains, the
display pass submits.

*Alternative rejected:* encode MetalFX into the slang-rhi command encoder through
`CommandEncoder.native_handle`. It removes one drain per frame. It also couples
the denoiser to slang-rhi encoder lifetime rules that are not documented, and an
unsignalled fence on the Metal path is the failure mode this repo has already
been burned by.

### D9 — PyObjC is an optional extra, and its absence is a startup refusal

The `[metalfx]` extra installs PyObjC. `--denoiser metalfx` without it fails at
startup with an install hint, the same shape `--mcp` already uses. A default
install gains no dependency.

If `pyobjc-framework-MetalFX` proves unavailable, the fallback is
`pyobjc-core` plus `objc.loadBundle` on `MetalFX.framework`, which reaches any
Objective-C class without a generated wrapper. The spike settles which.

### D10 — The bring-up guard splits around `select_backend`

`select_backend` stays last in `plan_bringup`, because it is the step that probes
the GPU. The denoiser needs guards on both sides of it:

- **Before** `select_backend`, next to `reject_mcp_unsupported`: the denoiser
  name is known, the scale is in range, `--denoise-scale` is not given without a
  denoiser, and the extra is installed.
- **After** `select_backend`, before the plan is returned: the resolved backend
  can run the named denoiser.

The plan gains `denoiser` and `denoise_scale`. Both are guard-vetted and
identical on every front-end, which is this module's rule for what a plan field
may be.

### D11 — The envelope owns the denoiser rules

`render_envelope.py` gains `DENOISER_BACKEND_UNSUPPORTED`,
`DENOISER_EXTRA_MISSING`, and `DENOISE_SCALE_WITHOUT_DENOISER`, and
`EnvelopeQuery` gains `denoiser` and `denoise_scale`. The CLI guards read the
predicate; they do not restate the rules. `parity.combo_is_valid` never sets
`denoiser`, so the matrix is unchanged and its coverage meta-test stays green.

### D12 — Two shader-variant families, with the illegal axes refused

`shader_variants.py` gains `GBUFFER` and `DISPLAY_RESOLVE`. Neither carries the
spectral, MLT, or neural axes. Following this repo's own rule, an axis a family
cannot carry is **refused**, never accepted and then dropped — otherwise
`cache_token()` names one variant on disk while `slangc_flags()` compiles
another.

### D13 — One accessor decides the source image for file output

`renderer.display_source()` returns the denoised image when a denoiser runs and
the accumulation image otherwise. `read_accumulation_hdr()`, `save_screenshot()`,
and the EXR writer read it. The raw accumulation image keeps its own accessor,
`read_accumulation()`, which the parity harness uses and which never returns the
denoised image.

### D14 — Metal dispatch hygiene binds the denoiser

MetalFX work is GPU work on this machine, so the standing rules apply without
exception. The scaler and its command queue are destroyed in the renderer's
teardown list, before the device closes. MetalFX runs one dispatch per frame over
one image, which is bounded by construction. `tests/test_metal_cleanup.py` runs
before merge, because this change adds a kernel and changes context lifecycle.

### D15 — Motion vectors are camera-only in version 1

The auxiliary pass reprojects the first-hit world position with the previous
frame's view-projection matrix. This is exact for static geometry under a moving
camera, which is the interactive case. Animated and skinned geometry ghosts,
because the pass does not track previous object transforms. The renderer forces
`reset()` on a scene edit, which bounds the artefact. Per-instance previous
transforms are a follow-up.

### D16 — A spike comes before any renderer edit

The MetalFX input contract is not verifiable from this repo. Task group 1 is a
standalone script that builds a `MetalContext`, bridges the device handle, builds
a scaler, denoises a synthetic frame, and **records** the true required inputs,
the accepted texture formats, and the accepted size ratios. Every later task
reads that record. No renderer code changes until it exists.

## Risks / Trade-offs

- **The MetalFX input contract differs from the assumed one** → D16 makes the
  spike the first task and every later task depends on its recorded result. The
  auxiliary-image registry (D4) is what absorbs a different input set without
  changing the renderer.

- **MetalFX rejects the RGBA32F accumulation format** → the auxiliary pass and
  the resource set already know each image's format. If the spike records a
  narrower requirement, the denoiser declares an RGBA16F input image and the
  renderer copies into it. The copy is one pass and does not touch transport.

- **`pyobjc-framework-MetalFX` may not exist on PyPI** → D9 records the
  `objc.loadBundle` fallback, which needs only `pyobjc-core`. The spike settles
  it before the extra is written.

- **A temporal denoiser over a converging running mean is not what MetalFX was
  built for** → this is a real semantic mismatch, not a bug. Each accumulation
  frame is less noisy than the last, so history reuse helps rather than hurts,
  and `reset()` on an accumulation reset keeps a stale history out. The visible
  result is measured, not assumed: the acceptance gate is a rendered A/B at
  matched sample counts.

- **Upscaling changes what a screen pixel means** → the HUD, the gizmo, the
  swapchain, and the display pass all stay at output extent, so only tool picking
  maps between extents. That mapping lives at one site, and a test pins it.

- **A wedged GPU** → D14 binds this change to the existing hygiene rules, and the
  kill harness runs before merge. MetalFX submitting on its own queue means an
  unsignalled slang-rhi fence cannot be caused by the denoiser.

- **The redundant tonemap** (D2) costs one full-resolution pass per frame while a
  denoiser runs. It buys an untouched no-denoiser path. Revisit only if it
  measures.

- **Scope** — this change adds two shader families, a resolution split, four
  auxiliary images, an optional native dependency, and CLI surface. It is large.
  The task order lands it in provable stages: the spike, then the seam with a
  null denoiser that proves the plumbing without MetalFX, then MetalFX, then
  upscaling. Each stage is independently green.

## Migration Plan

No migration. The denoiser is off by default and the no-denoiser path is
unchanged. Rollback is `--denoiser none`, which is the default.

## Open Questions

- Does `MTLFXTemporalDenoisedScaler` accept an input extent equal to its output
  extent? If not, `--denoise-scale 1.0` with MetalFX is refused rather than
  silently upscaled. The spike answers this.
- Which exposure does MetalFX expect — the pre-exposed image, or the linear image
  plus an exposure texture? The spike records it, and D1 keeps the renderer free
  either way, because the denoiser reads a linear image.
- Does the auxiliary pass fit the Metal 128-texture compute-argument limit with
  the 119-slot bindless pool bound? Counted at task 3.1, before the pass is
  written.
