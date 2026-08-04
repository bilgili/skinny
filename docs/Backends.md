# Skinny — GPU Backends

This document covers backend selection and the two GPU backends: the native
Metal path built on SlangPy and slang-rhi, and the Vulkan path. It also
covers the `gfx/` abstraction layer.

For the renderer overview see [Architecture.md](Architecture.md).

---

## Backend selection

The active GPU backend is resolved once per session by a single shared resolver
in `backend_select.py`, used by every front-end:

- `select_backend(prefer, *, persisted=None)` applies the precedence **explicit
  `--backend` flag > `SKINNY_BACKEND` env > persisted setting > `auto`**,
  returning `"vulkan"` or `"metal"`. `auto` resolves to native **Metal** on a
  Metal-capable Apple-Silicon host — the native backend is at full parity with
  Vulkan (geometry 6.1, shaded color 6.2, windowed present 6.5) — and falls back to
  **Vulkan** everywhere else. An explicit `--backend metal` returns `"metal"` only
  when the `DeviceType.metal` device constructs, otherwise raising a clear error
  naming the missing requirement.
- `make_context(backend, window, width, height, **kw)` constructs the matching
  context — a `VulkanContext` (`vk_context.py`) or a `MetalContext`
  (`metal_context.py`) — both exposing the same duck-typed surface the renderer
  reads (`width`/`height`, compute/present queues, `swapchain_info`, `gpu_info`,
  `allocate_command_buffers`, `recreate_swapchain`, `destroy`, the
  `backend_name`/`is_metal` predicate, and the capability flags). `gpu_info`
  carries `.name`, `.is_discrete`, and `.preferred_h264_encoder` on both
  backends, so the front-ends' status line and the video encoder stay
  backend-agnostic. No front-end constructs a context directly, and since change
  `frontend-bringup-builder` none of them calls `make_context` directly either:
  it is reached through
  [`BringupPlan.create`](HostModules.md#front-end-bring-up-bringuppy-change-frontend-bringup-builder).
  `select_backend` is called from the sibling `plan_bringup` step, which is
  where its `RuntimeError` becomes a `{prog}:`-prefixed `SystemExit`.
  `app.py` and `skinny-gui` persist/restore the
  selected backend like the other render flags — they are the two front-ends
  that hand their settings dict to `plan_bringup(persisted=…)`.

The renderer builds its GPU resources through whichever sibling module matches
the context, resolved once by `resource_module(ctx)` (keyed on `ctx.is_metal`):
`vk_compute` for a `VulkanContext`, `metal_compute` for a `MetalContext`. Both
expose the **same public API** (`StorageBuffer`, `StorageImage`, `SampledImage`,
`UniformBuffer`, `HostStorageBuffer`, `ComputePipeline`, …), so the construction
sites stay backend-agnostic (`self._gpu = resource_module(self.ctx)`); imports are
deferred so the Metal path never imports `vulkan`.

## The declared seam (`gpu_backend.py`, change `gpu-backend-adapter`)

"The same public API" is a **tested** claim now, not a comment. `gpu_backend.py`
declares what the adapters must agree on and what they may not:

- **`BackendCapabilities`** — one frozen record naming each reason a consumer
  used to branch on the vendor: `has_descriptor_sets`, `has_frame_sync_objects`,
  `has_external_memory`, `has_external_semaphore`,
  `has_shared_in_place_write`, `needs_shared_bindless_sampler`,
  `has_merged_record_header`, `has_indirect_dispatch`, `has_gpu_skinning`,
  `has_megakernel_record_source`, `has_reflected_record_layouts`,
  `needs_watchdog_tiling`, and `bindless_texture_capacity`. **Read it through `capabilities(ctx)`**, which
  folds the four runtime device probes (`supports_external_memory`,
  `supports_external_semaphore`, `supports_shared_memory`,
  `supports_indirect_dispatch`) in, so a consumer asks one named question rather
  than a backend test plus a probe. Every field replaces at least one live
  branch — a capability with no branch behind it does not belong in the record,
  and a field the two device backends agree on is caught by test, **excluding**
  `has_indirect_dispatch`, which is a device probe either backend may report
  either way. One field replaces no *pre-existing* branch:
  `has_external_semaphore` encodes the branch that should have existed, because
  the CUDA handoff needed both extensions all along and silently assumed one
  implied the other.

  **Name the reason, not the binding model.** `has_descriptor_sets` was briefly
  overloaded into "is bind-by-name", which is the same answer on the two device
  backends and the wrong one on the recording adapter — it binds by name, splits
  no samplers, and compiles no records. The shared bindless sampler (an
  argument-table fact) and the merged record header (a shader-build fact) each
  own a field, so a capability-gated reach at a one-sided member is true only
  where that member exists.
  On the renderer it is the memoised `self.caps` property, derived from
  `self.ctx`.
- **`ONE_SIDED_MEMBERS` / `DIVERGENT_SIGNATURES`** — members that genuinely
  exist on one adapter (`ExternalTimelineSemaphore`, `MetalFrameEncoder`,
  `DebugRasterMetal`, `make_sampler`, …) or whose signature genuinely differs
  (`PreviewPipeline.__init__`). Declared, never discovered: adding one is a
  deliberate edit, exactly like `METAL_ONLY_DEFINES` in `shader_variants`.
- **`recording_compute`** — a third adapter that records allocations, bindings
  and dispatches instead of executing them, and returns zero-filled data. It
  makes dispatch ordering and binding coverage assertable with **no device**
  (`Recorder.dispatch_entries()`, `Recorder.missing_bindings()`). It records, it
  does not simulate: image correctness stays the parity matrix's job. Its
  live-binding coverage path is the section below.

Two probes were removed and must not come back. `hasattr(ctx, "compute_queue")`
was used as "is this Vulkan?" at 7 sites, but `MetalContext.compute_queue` is
`None` rather than absent, so it was **unconditionally true**; and
`descriptor_sets is None` was an "is Metal" sentinel at 13 gates, 5 of them
compounded with `is_metal` in the same expression. `tests/test_gpu_backend.py`
fails if either returns, if the adapter surfaces drift from the pinned fixture
undeclared, if the bindless capacity disagrees with the array size compiled into
`shaders/bindings.slang` for that target, or if a new `is_metal` branch appears
in `renderer.py` that is neither the adapter selection nor a genuine second
implementation.

Four `is_metal` branches remain in the renderer, and all four are genuine
two-implementation splits, not questions the capability record can answer: the
wavefront pass factory (`vk_wavefront` vs `metal_wavefront`), and the windowed,
headless and material-preview dispatch paths. The rule that fell out of the
migration: **resource construction, binding, readback and dispatch belong on the
adapter; assembling a frame does not.**

### Live bindings on the recording adapter (change `recording-adapter-live-bindings`)

`Recorder.missing_bindings()` names every shader global a recorded dispatch left
unbound. On Metal an unbound global reads as **zero rather than raising**, so
catching one before the GPU is worth a lot — but only if neither side of the
comparison is written by the test. Originally both were: a caller handed
`reflect_globals()` a literal set and `dispatch()` a literal bind dict, so the
assertion was that two literals disagreed where their author expected. Three
pieces make it real.

**The declared side comes from the compiler.** A hand parser was tried first and
abandoned: a line/regex parser cannot tell a file-scope resource global from a
function parameter of resource type without full scope tracking, and pre-merge
review kept finding valid Slang spellings it under-reported — the exact fail-open
the gate exists to prevent. So the declared globals come from `slangc`'s own
reflection. `tests/fixtures/gen_recording_pass_globals.py` emits the generated
MaterialX Slang the megakernel imports, compiles each registered pass **to the
Metal target** under its `ShaderVariantKey` defines with `-reflection-json`,
takes the top-level `parameters` (uniform block included) and refuses on any
bindable entry-point parameter (a `uniform` that would lower to a push constant,
absent from `parameters`), and writes `tests/fixtures/recording_pass_globals.json`.
The hostless gate reads that golden — a checked-in generated artifact trusted the
way the parity harness trusts its reference EXRs; a `gpu`-marked freshness test
re-runs the compiler and diffs. Regenerate after any shader edit that changes a
registered pass's globals::

    PYTHONPATH=src .venv/bin/python -m tests.fixtures.gen_recording_pass_globals

**The bound side comes from the host.** `SceneResourceSet.metal_binds()` is the
single builder — the renderer's `_build_metal_binds` delegates to it, passing the
two globals the renderer owns rather than the set (`commonSampler`,
`graphParamsCombined`) as keyword arguments instead of adding them afterwards.
`recording_compute.scene_binds(ctx)` allocates a `SceneResourceSet` **against the
recording adapter** and calls that same method, so the gate compares the
compiler's truth against production code with no device and no `Renderer`.

**Registration is enforced.** `RECORDABLE_PASSES` lists each covered pass — today
the Metal megakernel; the denoise auxiliary and display passes join when
`denoise-pipeline` lands (coverage is scoped per registered
`(pass, ShaderVariantKey)`). `RECORDABLE_EXCLUSIONS` lists every other compute
entry with a reason. `tests/test_recording_pass_coverage.py` asserts each
registered pass leaves nothing unbound, that the golden names exactly the
registered passes, that every entry `compute_entry_points()` finds is registered
or excluded, and that no exclusion is stale. Entry keys are `(module,
entry_point)`, never the name alone: several modules declare `computeMain`.
`compute_entry_points` is the one scan that stays source-side — a far simpler
grammar than declarations, with a single-pass comment lexer (a `/*` inside a
`//` line, or either delimiter inside a string, is text) and a platform-neutral
module key.

Two independent legs guard the golden. It is **cross-checked** against
`gpu_resources.DECLARATIONS`: the megakernel golden must name exactly the
default-layout inventory resources it reaches (all but the neural buffers it
strips at build), so a drift on either side fails. And a **negative control**: a
fixture pass whose bind map omits one reflected global, asserted reported through
the same call the real gate uses — weakening the comparison fails it.

The hand-driven scenarios in `tests/test_gpu_backend.py` stay, relabelled: they
test the **recorder** (a `None` value is unbound, the uniform blob binds `fc`, an
undeclared pipeline reports nothing), which is what a literal set is right for.

### MetalContext (`metal_context.py`, `metal_compute.py`)

`MetalContext` stands up a **native** Metal device through SlangPy's
`DeviceType.metal` (slang-rhi — no MoltenVK, no raw PyObjC) and mirrors the
`VulkanContext` surface. The present path uses the slang-rhi `Surface`
(`configure` / `acquire_next_image` / `present`) bridged to a GLFW window via its
Cocoa `NSWindow` pointer (`WindowHandle(nswindow=…)` from `glfw.get_cocoa_window`)
— no manual `CAMetalLayer`. `metal_compute.py` provides the full resource layer at
API parity with `vk_compute`, including the megakernel `ComputePipeline`: it runs
`emit_megakernel_sources` then compiles `main_pass.slang` (`mainImage`) to Metal
**in-process** (no `slangc` shell-out, no `.metallib`) with
`SKINNY_COMPUTE_PIPELINE=1` + `SKINNY_METAL=1`, reflects the global binding map,
and dispatches by **binding resources by name** (the renderer's binding map drives
the same logical slots). Pipeline parameters are bound as whole resources or via
`set_data` byte blobs, **never per-field cursor writes** (a scalar cursor write
around an open Metal encoder can leave the GPU fence un-signalled). Megakernel
entry is `mainImage`; trivial/foundation kernels name their entry `computeMain`,
never `main` (Slang's Metal target reserves `main` and the rename breaks pipeline
creation).

**Metal-target shader adaptations** (gated `#if defined(SKINNY_METAL)`, Vulkan
SPIR-V byte-unchanged): the combined `Sampler2D` pool exceeds Apple's compute
argument limits and slang-rhi cannot bind a combined `Sampler2D` at all, so the
bindless `flatMaterialTextures` pool becomes `Texture2D[119]`
(`capabilities(ctx).bindless_texture_capacity`) sampled through a
shared `commonSampler` (binding 38), the five discrete maps (env/tattoo/normal/
roughness/displacement) split into `Texture2D` + a per-map `SamplerState`
(bindings 39–43), and `NonUniformResourceIndex` (unavailable in the compute stage
on the Metal target) collapses to identity via the `NRI(x)` macro.

**Spectral compile variant** (change `spectral-rendering`): hero-wavelength
spectral rendering is a **compile-time variant** of the megakernel selected at
startup, not a runtime branch. The spectral megakernel compiles with
`-DSKINNY_SPECTRAL` (Vulkan: appended to the `slangc` flags, with the flag
hashed into the `spv_cache` key so it lands in a distinct cache slot; Metal:
added to `SlangCompilerOptions.defines`), pulling in `spectrum.slang` and the
`SpectralPathTracer` (`integrators/path_spectral.slang`) that carries a `float4`
`Spectrum` throughput/radiance and reuses the RGB flat sampler for
λ-independent geometry. `common.slang` holds the gated `Spectrum` typealias
(`float4` spectral / `float3` RGB) so the carriers type-check in both builds;
the default RGB build never imports `spectrum.slang`, so its SPIR-V is
**byte-unchanged**. It compiles on demand on both backends (spectral bindings
45–47 in the [Descriptor Binding Map](GpuResources.md#descriptor-binding-map)). The estimator,
upsampling model, exact sources, and film resolve are documented in
[Spectral.md](Spectral.md).

**Wavefront on Metal** (change `metal-wavefront-parity`): the wavefront
execution mode — staged path + BDPT integrators, ReSTIR DI reuse, and the
neural directional proposal — runs on the native Metal backend at parity with
Vulkan. The stage orders live in the backend-neutral `wavefront_driver.py`;
pass *construction* lives in per-backend factories `vk_wavefront.ensure_pass`
/ `metal_wavefront.ensure_pass(renderer, integrator)` (change
`renderer-module-carveout`, Stage C) — the renderer holds one
`_ensure_wavefront_pass(integrator)` dispatcher as the single `is_metal` seam,
with the rebuild keys and None-fallback gates preserved verbatim inside the
factories. `metal_wavefront.py` supplies the Metal pass classes (per-entry in-process
pipelines, queue buffers sized from the **reflected MSL strides**, one
`MetalFrameEncoder` per frame with global barriers, and the CPU
slot-count-readback fallback while slang-rhi's Metal indirect dispatch is a
no-op — selected by the logged `supports_indirect_dispatch` probe). Metal caps
a kernel's argument table at **31 buffer slots** (also 128 textures, 16
samplers), assigned program-wide in declaration order. That limit has one owner,
`argument_budget.py`, and a **hostless per-variant census** is the primary gate:
it derives each variant's buffer/texture/sampler count from the compiler
(`slangc -reflection-json`, no GPU device) and fails before a device is built if
a count passes the limit or drifts from the checked-in baseline. The runtime
pipeline-creation error here stays as the backstop; the exact per-kernel count is
the gpu-marked cross-check against `program.layout.parameters`. See
[GpuResources.md § Argument-table budget](GpuResources.md#argument-table-budget-argument_budgetpy-change-spectral-table-fold).
Several builds compile globals out to fit the cap: the default wavefront build
stubs the record emitters
(`wf_records.slang`), and the neural-active build (`SKINNY_METAL_NEURAL=1`)
additionally compiles out `toolBuffer`/`recordBuf`/`recordCounter` (dead in
every wavefront kernel) to fit the un-stubbed `neuralWeights/Biases/Layers`.
The records build (`SKINNY_METAL_RECORDS=1`, change `metal-record-drain` —
armed only while online training runs) un-stubs the emitters and re-fits the
cap by compiling out `lightSplatBuffer`/`gizmoSegments` (inert on a training
render) and folding both record counters into their data buffers (the per-lane
count into a stack header element; `recordCounter` into a 64-byte header of a
byte-address `recordBuf`). See
[Wavefront.md → Metal wavefront backend](Wavefront.md#metal-wavefront-backend).

The capability flags `supports_external_memory` / `supports_external_semaphore`
report `false` on Metal — there are no exported memory or semaphore handles. The
Metal interop seam is **`supports_shared_memory`** instead (change
`metal-neural-interop`): `true` when an upload-heap buffer carrying full storage
usage constructs (UMA shared storage; Vulkan contexts don't define the flag, so
`getattr(ctx, "supports_shared_memory", False)` reads `false` there). It gates
`StorageBuffer(shared=True)` — host-visible buffers whose `write_in_place`
lands bytes the next dispatch reads with no staging upload — which the online
weight handoff writes at the frame boundary (`MetalSharedWeightPublisher`).
`supports_fp16_storage` / `supports_fp16_compute` come from a device probe —
`false` on current slang-rhi (0.42 under-reports `half` on Metal), so neural
weights load fp32 via `_effective_neural_config()`'s graceful downgrade.
`supports_indirect_dispatch` is probed **empirically** (a real indirect
dispatch + sentinel readback; a structural `hasattr` check would lie).

**Megakernel watchdog tiling** (change `metal-megakernel-watchdog-tiling`): the
megakernel dispatch (`ComputePipeline.dispatch`) commits one command buffer per
screen-space **row band** under `SKINNY_METAL`, so no single buffer covers the
full frame. This closes the same `metal-dispatch-hygiene` "no unbounded command
buffers" hole that the volume caps close for per-pixel loops — but for integrator
*breadth*: a full-frame **BDPT** megakernel over inlined graph materials (eye ×
light subpaths + `s × t` connections, each a graph-shader BSDF eval) exceeded the
watchdog and wedged the GPU. `renderer._metal_megakernel_bands()` picks the band
count from an integrator-aware per-pixel budget (`_METAL_MEGAKERNEL_BAND_PIXELS`)
scaled by resolution, overridable via `SKINNY_METAL_MEGAKERNEL_BANDS`; the band Y
origin rides a Metal-only `FrameConstants.tileOriginY` (`#if defined(SKINNY_METAL)`
gated ⇒ Vulkan SPIR-V byte-unchanged) that `mainImage` adds to the thread's `y`.
The accumulation image persists across a frame's bands, so N-band output is
bit-identical to one dispatch, and ≤256² scenes stay a single band (parity corpus
unaffected). See [Megakernel.md → Backends](Megakernel.md#backends-vulkan-and-metal).

**Tool-dock render paths** (change `metal-tool-dock-render`): the two View-menu
tool docks whose render paths were Vulkan-only now run on Metal via compute.
- **Material Graph preview** — the adapter's `PreviewPipeline` compiles `preview_pass.slang`
  (`previewMain`) in-process (same session config as the megakernel
  `ComputePipeline`, linking the emit-time `generated_materials` so it shades
  identically) and dispatches by binding the scene material resources + the output
  image **by name** — no Vulkan descriptor sets. `Renderer.render_material_preview`
  branches on `is_metal`, reuses `_build_metal_binds` + `_pack_uniforms_msl` (packed
  against the preview program's own reflected `fc` layout so it works in wavefront
  mode too), and reads the RGBA32F float image back directly. The preview `size` is
  clamped to `_METAL_PREVIEW_MAX_SIZE` (one bounded command buffer). The Metal-only
  `pc` push block is a plain `uniform` (slang-rhi rejects `set_data` on a
  `[[vk::push_constant]]` ConstantBuffer; `#if defined(SKINNY_METAL)` ⇒ Vulkan SPIR-V
  unchanged).
- **Camera Debug viewport** — the native backend has no graphics pipeline, so
  `DebugRasterMetal` (`debug_raster.slang`) is a **software line/triangle
  rasteriser** in compute: `clearImage`/`clearDepth` → `depthLines` (`InterlockedMin`
  into a uint depth UAV) → `colorLines` (opaque, depth-owned pixels) → `blendTris`
  (edge-function fill, src-alpha over, depth-tested no-write; one thread per
  triangle×screen-row so no unbounded per-thread loop). `DebugViewport` on Metal
  builds this instead of the Vulkan render pass; `render_embedded` runs the
  unchanged `_generate_streams` `_gen_*` generators, dispatches, and returns RGBA8
  through the same worker `DebugFrame` path. `debug_raster_ref.py` is the numpy
  mirror the kernel is diffed against (host-checkable, no GPU). The Vulkan graphics
  rasteriser is untouched.

## Backend Abstraction (`gfx/`)

> Note: the `gfx/` ABC below is **distinct** from the live Metal backend in
> [Backend selection](#backend-selection) above. The renderer drives
> `VulkanContext` / `MetalContext` duck-typed via `make_context`; the `gfx/`
> abstraction has no importers outside `gfx/` and remains unused scaffolding (a
> possible later cleanup, not on the path to the Metal backend).

A new abstraction layer lets the renderer talk to a `Backend` instance
(`gfx/backend.py`) instead of touching Vulkan directly:

```
Backend
  ├─ caps: BackendCaps        # bindless / scalar layout / push descriptors
  ├─ device: Device            # queues, allocators, command recording
  ├─ presenter: Presenter|None # surface/swapchain (None = headless)
  └─ shader_target() -> "spirv" | "metal"
```

| Backend | Status |
|---------|--------|
| `gfx/vulkan/` | Production — wraps `vk_context.py` + `vk_compute.py` |
| `gfx/metal/` | Unused stub (`MetalBackend.create()` raises). The live native-Metal path is `metal_context.py` (see [Backend selection](#backend-selection)), **not** this ABC |

`vk_context.py` and `vk_compute.py` keep their direct Vulkan API; the
abstraction is layered above them so existing code keeps working while
new code paths (preview pass, debug viewport line pipeline) are
incrementally moved over.

---
