## Why

P1 (`add-metal-backend-foundation`, merged to `main`) stood up a native Metal
device, a trivial compute dispatch, present, and the `--backend` selection seam —
but **nothing renders the head on Metal**. `auto` still resolves to Vulkan and an
explicit `--backend metal` refuses the full renderer with a foundation-phase
notice. The renderer (`renderer.py`, 8.6k lines) builds every GPU resource through
the `vk_compute.py` classes on a raw `VkDevice`, so a `MetalContext` cannot drive
it yet. P2 is the phase that makes the megakernel path actually image the head on
Metal at parity with Vulkan, and flips `auto`→Metal on Apple Silicon. It is the
prerequisite for every later Metal phase (ReSTIR P3, neural inference P4,
wavefront P5, the MLX↔Metal handoff P6) — all of them assume a working megakernel
render on the Metal context.

## What Changes

- **Backend-polymorphic GPU resource layer.** Expand `metal_compute.py` to full
  parity with the `vk_compute.py` classes the renderer consumes (`StorageBuffer`,
  `StorageImage`, `SampledImage`, `UniformBuffer`, `HostStorageBuffer`,
  `ComputePipeline`), matching their constructor signatures and upload helpers, so
  the renderer builds its megakernel resources on either backend without
  backend-specific branches at the construction sites. The renderer selects the
  resource module by `ctx.is_metal` (or an equivalent factory) instead of the
  hardwired `from skinny.vk_compute import …`.
- **MSL-correct uniform packing.** Add an MSL layout path for the uniform block
  (`FrameConstants` + `SkinParams` + `Light`): Slang's Metal target pads `float3`
  to 16 B, so the scalar/`std140` blob (`FrameConstants`/`Camera` 272 B scalar)
  does not match the MSL struct (288 B). Query the MSL field offsets from the
  compiled module and pack an MSL-correct blob (`_pack_uniforms_msl`), set via
  `set_data` byte blobs only — never per-field cursor writes (the P1 D4 fence-hang
  discipline). `SkinParameters.pack()` keeps the Vulkan `std140` layout; the Metal
  layout is a sibling.
- **Megakernel pipeline + descriptors on Metal.** Compile `main_pass.slang`
  (`mainImage`) to Metal in-process via SlangPy and bind the full descriptor set
  (bindings 0–24, 30–32 — flat-material params, bindless textures, MaterialX
  skin/std params, light buffers, gizmo, env importance-sampling CDFs) through the
  Metal `ComputePipeline`. Resource binding via `dispatch(vars=…)` / `ShaderCursor`
  resource binds (allowed; only scalar per-field writes are banned).
- **Textures + `commonSampler` on Metal.** Bindless `SampledImage` pool and the
  shared sampler bound through the Metal pipeline so head normal/roughness/
  displacement maps, tattoos, and the environment map sample correctly.
- **Flip `auto`→Metal; remove the foundation refusal.** `select_backend("auto")`
  resolves to **Metal** on Apple Silicon when the device constructs (else Vulkan);
  the real front-ends launch the full renderer on Metal instead of exiting with
  the `METAL_FOUNDATION_NOTICE`. `--backend vulkan` stays byte-identical on every
  platform.
- **Out of scope (later phases):** ReSTIR DI (P3), neural inference (P4),
  wavefront execution (P5), the MLX↔Metal zero-copy weight handoff (P6), and any
  fp16 storage/compute on Metal. The Metal capability flags (`supports_external_*`,
  `supports_fp16_*`) stay `false`; the megakernel needs none of them, and the
  existing capability gates already fold ReSTIR/wavefront back to the megakernel on
  the Metal backend.

## Capabilities

### New Capabilities
<!-- none — P2 extends the existing metal-backend capability with render-parity
     requirements rather than introducing a new capability. -->

### Modified Capabilities
- `metal-backend`: adds megakernel render-parity requirements (backend-polymorphic
  resource layer, MSL-correct uniform packing, megakernel pipeline + descriptor
  binding, textures + `commonSampler`, structural-parity acceptance vs the Vulkan
  megakernel) and changes the backend-selection requirement from the foundation
  phase (`auto`→Vulkan, explicit-`metal` refusal) to render-capable
  (`auto`→Metal on Apple Silicon, the full renderer runs on Metal).

## Impact

- **Code (modified):** `src/skinny/metal_compute.py` (full resource-wrapper
  parity with `vk_compute.py`); `src/skinny/renderer.py` (backend-select the
  resource module by `ctx.is_metal`; MSL uniform-pack path; Metal megakernel
  pipeline compile + descriptor binding; bindless texture + sampler binding);
  `src/skinny/backend_select.py` (`auto`→Metal on Apple Silicon; drop the
  `METAL_FOUNDATION_NOTICE` refusal); the four front-ends (`app.py`,
  `headless.py`, `ui/qt/app.py`, `web_app.py`) — launch the renderer on a resolved
  Metal backend instead of exiting.
- **Code (possibly new):** a small resource-module factory if the `is_metal`
  branch is cleaner as one helper than inline at each construction site.
- **Shaders:** `main_pass.slang` must compile clean on `-target metal` (the
  `mainImage` entry is not the reserved `main`; verify no Vulkan-only constructs).
  No SPIR-V change for the Vulkan path.
- **Dependencies:** none new — SlangPy (already a dependency) provides the Metal
  device, in-process Slang→Metal compile, and MSL reflection.
- **Tests:** headless megakernel render on Metal vs Vulkan — exact parity on the
  deterministic structural outputs (geometry/instance IDs, hit/miss, depth) and a
  perceptual tolerance on the converged shaded color; MSL uniform-offset
  correctness; `auto`→Metal resolution on Apple Silicon; the foundation-refusal
  removal. All Metal tests skip cleanly off Apple Silicon.
- **Docs:** `docs/Architecture.md` (Metal megakernel path + the resource-layer
  backend split; descriptor binding map note for Metal), `docs/Megakernel.md`
  (Metal parity + MSL uniform layout), `README.md` (`--backend auto`→Metal on
  Apple Silicon), `CHANGELOG.md`, `CLAUDE.md`/`openspec/config.yaml` (Metal now
  renders the megakernel).
- **Platform:** non-macOS and `--backend vulkan` stay byte-identical; Metal render
  is Apple-Silicon-only and guarded; `auto` degrades to Vulkan where the Metal
  device does not construct.
