# Denoise pipeline

## Why

skinny converges by brute force. The renderer averages samples into the
accumulation image and shows the running mean. A noisy interactive view stays
noisy until the sample count is high. Every other production path tracer removes
that wait with a denoiser.

The host has a Metal GPU. Apple ships MetalFX, which denoises and upscales
ray-traced frames on that GPU. A denoiser is also a per-platform component: the
Metal host wants MetalFX, an NVIDIA host wants NRD. So the renderer needs a
denoiser **seam** first, and MetalFX as its first implementation. Without the
seam, the second denoiser rewrites the first one's call sites.

## What Changes

- **New denoiser seam.** One `Denoiser` protocol declares the auxiliary images a
  denoiser needs, its reset behaviour, and one `denoise()` call per frame. One
  registry maps a name to an implementation. The renderer holds at most one
  denoiser and never names a vendor.
- **New MetalFX denoiser.** `MTLFXTemporalDenoisedScaler` denoises and upscales
  the linear accumulation image on the native Metal backend. It reads the Metal
  device and texture handles that slang-rhi already exposes.
- **New auxiliary image pass.** One compute pass traces one primary ray per
  pixel. It writes the auxiliary images the active denoiser declares: diffuse
  albedo, specular albedo, world normal, roughness, view depth, and motion
  vector. The pass writes only the declared images.
- **New display pass.** A standalone compute pass applies exposure, tonemap,
  sRGB encoding, and the HUD and gizmo overlays. It runs at output resolution
  over the denoised image. The existing display tails in `main_pass.slang` and
  `wf_display.slang` stay unchanged and keep their current owners.
- **New render resolution.** The renderer gains a render resolution that is
  separate from its output resolution. `--denoise-scale` sets the ratio. The two
  are equal when no denoiser runs, so today's behaviour is unchanged.
- **New camera jitter.** The primary ray takes its sub-pixel offset from a
  reported Halton sequence while a temporal denoiser runs. The denoiser needs the
  offset it was rendered with. The random offset stays in use otherwise.
- **New flags.** `--denoiser {none,metalfx}` and `--denoise-scale FLOAT` reach
  all four front-ends. New startup guards refuse a denoiser the host cannot run.
- **New runtime controls.** A denoise on/off toggle and a strength control join
  the parameter registry. Both are post-process controls, so neither resets
  accumulation.
- **Denoised file output.** `render_headless()`, `save_screenshot()`, and the
  EXR writer emit the denoised image while a denoiser runs.
- **The accumulation image is never overwritten.** The denoiser reads it and
  writes a separate image. The pbrt-truth and self-consistency parity gates read
  the accumulation image, so they stay unchanged.
- **New optional dependency.** The `[metalfx]` extra installs PyObjC. MetalFX has
  no C interface, so the Metal denoiser reaches it through the Objective-C
  bridge. A default install gains nothing.

## Capabilities

### New Capabilities

- `denoise-pipeline`: the backend-neutral denoiser seam — the `Denoiser`
  protocol, the auxiliary-image contract and its registry, the per-frame stage
  order, the render/output resolution split, the reset rule, and the routing of
  the denoised image to display and to file output.
- `metalfx-denoiser`: the MetalFX implementation — native-handle bridging from
  slang-rhi to Metal, scaler construction and lifetime, the MetalFX auxiliary
  images and jitter contract, and its refusal conditions.

### Modified Capabilities

- `render-cli`: adds the `--denoiser` and `--denoise-scale` flags, their
  environment variables, and their refusal messages.
- `frontend-bringup`: adds the denoiser guards to the canonical bring-up order —
  the flag-level guard before `select_backend`, the backend-compatibility guard
  after it.
- `render-envelope`: adds the denoiser refusal codes and their rules, so the CLI
  guards and the renderer read one predicate.
- `accumulation-reset-registry`: the set of parameters that opt out of the
  accumulation reset grows beyond `tonemap_index` and `exposure`.
- `shader-byte-layouts`: `FrameConstants` gains the output resolution, the jitter
  offset, and the denoiser state fields.
- `renderer-output-fidelity`: the offscreen and screenshot paths emit the
  denoised image while a denoiser runs.
- `metal-dispatch-hygiene`: MetalFX work is GPU work, so scaler teardown and
  command-buffer bounds obey the existing hygiene rules.

## Impact

**New modules.** `denoise.py` (protocol, auxiliary-image registry, name
registry, resolution resolver), `denoise_metalfx.py` (MetalFX implementation),
`shaders/gbuffer.slang` (auxiliary-image pass),
`shaders/display_resolve.slang` (display pass).

**Modified modules.** `renderer.py` (stage order, resolution split, output
routing), `gpu_resources.py` (denoised image, auxiliary images, render-resolution
sizing), `slang_layout.py` (`FrameConstants` fields), `cli_common.py` (flags and
guards), `bringup.py` (guard order), `render_envelope.py` (codes and rules),
`params.py` (two post-process controls), `film_io.py` and `headless.py` (output
routing), `shaders/cameras/pinhole.slang` (reported jitter),
`shader_variants.py` (the two new shader families).

**Dependencies.** PyObjC through the new `[metalfx]` extra. No change to the
default install.

**Risk.** MetalFX has no C interface and no Python binding in this repo today. A
spike proves the handle bridge and records the true MetalFX input contract before
any renderer code changes.

**Out of scope.** The NRD implementation. NRD needs demodulated diffuse and
specular radiance, which the integrators do not produce. The auxiliary-image
registry is the extension point that later change uses.
