## Context

The renderer is hardwired to Vulkan. `renderer.py` (8,639 lines) holds a
duck-typed `self.ctx` constructed as a `VulkanContext` (`vk_context.py`, 485
lines) and builds GPU resources with the `vk_compute.py` classes
(`StorageBuffer`, `StorageImage`, `SampledImage`, `UniformBuffer`,
`ComputePipeline`, `ExternalTimelineSemaphore`), which call raw `vulkan`. The
four front-ends each construct the context directly:

- `app.py` → `VulkanContext(window, W, H)` → `Renderer(vk_ctx=…)`
- `headless.py:144` → `VulkanContext(window=None, …)`
- `ui/qt/app.py:71` → `VulkanContext(…)`
- `web_app.py:89` → `VulkanContext(…)`

All four share the CLI surface via `cli_common.add_render_flags`. There is **no**
backend abstraction in use: the `gfx/` Backend ABC (`gfx/backend.py`,
`gfx/vulkan/*`, the `gfx/metal` stub) has zero importers outside `gfx/` and is
Step-2-era scaffolding that does not model the renderer's grown needs (external
memory, timeline semaphores, bindless, indirect dispatch). A complete native
Metal renderer once existed on the `metal` branch (tip `e1300e7`) but it rewrote
the renderer entirely onto that ABC (0 raw-`vk.` calls) and sits 264 commits
behind `main`, sharing almost no lines with the current renderer — unmergeable.

SlangPy `0.41.0` (already a dependency) exposes `DeviceType.metal`, so a native
Metal **device** can be stood up through slang-rhi without raw PyObjC. This is
the conduit `CLAUDE.md`'s `MetalContext` description and the `metal` branch both
used.

## Goals / Non-Goals

**Goals (P1):**
- A `MetalContext` that constructs (windowed + headless) on Apple Silicon and
  mirrors the duck-typed surface the renderer reads off `VulkanContext`.
- A trivial Slang compute dispatch on Metal that produces **bit-identical**
  output to the same kernel on Vulkan (the foundation correctness proof), plus a
  windowed clear+present that does not hang.
- Backend selection (`--backend {auto,metal,vulkan}` + `SKINNY_BACKEND`) resolved
  by one shared resolver, routed through all four front-ends, with `auto`→Vulkan
  fallback and a clear error on an explicit-but-unavailable backend.
- `vulkan` (and therefore every non-macOS host and MoltenVK-on-macOS) stays
  byte-identical to today.

**Non-Goals (deferred to later phases):**
- The megakernel head render on Metal, MSL uniform layout, textures /
  `commonSampler`, the descriptor binding map, ReSTIR, neural inference,
  wavefront, the MLX↔Metal handoff. P1 renders no head.
- Porting the renderer onto the `gfx/` ABC, or salvaging the `metal` branch's
  renderer (its device wrappers are reference only).
- A second context abstraction layer — P1 deliberately keeps the renderer's
  existing duck-typed `self.ctx` contract.

## Phased roadmap (where P1 sits)

| Phase | Change | Delivers | Depends on |
|------|--------|----------|-----------|
| **P1** | **this change** | Metal device + context foundation; `--backend` selection seam; trivial dispatch + present parity | — |
| P2 | (next) | Megakernel render parity on Metal: uniforms (MSL layout), descriptors, textures + `commonSampler`, buffers; image-match Vulkan megakernel | P1 |
| P3 | (later) | ReSTIR DI on Metal | P2 |
| P4 | (later) | Neural inference on Metal; neural-weights buffer in shared-storage `MTLBuffer` | P2 |
| P5 | (later) | Wavefront on Metal (external/indirect buffers, counting-sort, shade kernels, splat) | P2 |
| P6 | (later) | MLX↔Metal zero-copy handoff (finishes `add-mlx-training-backend` §3) | P4 |

The independent MLX *compute* backend (`add-mlx-training-backend` §1–§2) needs
none of this and can ship in parallel.

## Decisions

### D1: Parallel `MetalContext` sibling, not a Backend-ABC port
`MetalContext` (`metal_context.py`) + `metal_compute.py` mirror the duck-typed
surface of `vk_context.py` / `vk_compute.py`. The renderer keeps its
`self.ctx`; backend selection branches only at the four context-construction
sites. *Alternatives rejected:* porting all 8.6k renderer lines onto the unused
`gfx/` ABC first (enormous, high-risk refactor with no user-visible value during
the port, across five active worktrees); forward-porting the `metal` branch
(unmergeable — its renderer shares almost no lines with `main`). The ABC port may
be revisited later as cleanup; it is not on the critical path to a working Metal
backend or the MLX handoff.

### D2: Device layer via SlangPy `DeviceType.metal`, not raw PyObjC
The Metal device, queue, buffers, and pipelines come from slang-rhi through
SlangPy (already a dependency), matching `CLAUDE.md` and the `metal` branch. Raw
PyObjC/Metal would be a far larger surface to own.

### D3: Selection in `cli_common` + a `backend_select.py` resolver
`--backend {auto,metal,vulkan}` (+ `SKINNY_BACKEND`) is added once to
`add_render_flags`, exactly like `--execution-mode`. A single
`select_backend(prefer)` applies precedence (explicit > env > persisted > `auto`)
and `make_context(backend, window, …)` returns a `VulkanContext` or
`MetalContext`. `auto` → Metal on Apple-Silicon macOS when the device constructs,
else Vulkan; explicit-but-unavailable raises a clear error (mirroring the
`--neural-trainer cuda` gating style).

### D4: `set_data` byte-blob pipeline params only — never per-field cursor writes
The `metal` branch hit a hang where any SlangPy `ShaderCursor` write (even a
scalar), around an open encoder, leaves the GPU fence un-signalled. P1 sets all
pipeline parameters via `set_data` byte blobs from the start and establishes that
discipline; the MSL-offset-correct uniform packing (`_pack_uniforms_msl`) is a P2
concern (P1's trivial kernel has no large uniform block).

### D5: Conservative capability flags on Metal in P1
`MetalContext.supports_external_memory`, `supports_external_semaphore`,
`supports_fp16_storage`, `supports_fp16_compute` are **False** in P1. The Metal
shared-storage `MTLBuffer` + `MTLSharedEvent` equivalents (the MLX-handoff hooks)
arrive in P4; fp16 on Metal is a later study. Conservative flags keep the renderer
on its safe fp32/file paths when running on Metal.

### D6: `backend_name` / `is_metal` predicate on both contexts
Both `VulkanContext` and `MetalContext` expose `backend_name` ("vulkan"/"metal")
and `is_metal`, so the small Metal-vs-Vulkan guards the later phases need have a
clean predicate. The renderer ctor keeps its `vk_ctx=` parameter name in P1
(duck-typed; a `MetalContext` is passed in) — renaming to `ctx=` is cosmetic and
deferred to avoid blast radius.

## Risks / Trade-offs

- **SlangPy in-process Slang→Metal compile may not be exposed cleanly.** → Spike
  first (O1). Fallback: shell `slangc -target metal` to a `.metallib` and load
  that, mirroring how the Vulkan path pre-compiles `.spv`.
- **MSL uniform layout drift** (Slang pads `float3`→16B; `FrameConstants`/`Camera`
  288B MSL vs 272B scalar). → Not exercised by P1's trivial kernel; pre-noted for
  P2, which queries MSL field offsets and packs an MSL-correct blob.
- **Windowed Metal needs a Mac with a display; headless must work for CI.** → The
  correctness proof is the **headless** trivial-dispatch parity test; the windowed
  present is a separate, display-gated smoke test.
- **Two context implementations can drift.** → Mitigated by the shared duck-typed
  contract and the cross-backend parity test; the surface is small in P1 and grows
  test-first per phase.
- **Mac stability:** force-killing a Metal Python process can wedge
  `MTLCompilerService` (per the `metal` branch notes). → Document graceful-stop;
  not a code risk.

### D7: `auto` resolves to Vulkan in P1; the `auto`→Metal flip lands with render parity (P2)
The renderer (`renderer.py`, 8.6k lines) builds every GPU resource through the
`vk_compute.py` classes on `ctx.device` — a raw `VkDevice`. A `MetalContext.device`
is a SlangPy device, not a `VkDevice`, so `Renderer(vk_ctx=metal_ctx)` cannot
build resources in P1 (the megakernel port is P2). Making `auto` select Metal
now would therefore break the default `skinny` invocation on Apple-Silicon Macs.
So in P1 **`select_backend("auto")` resolves to Vulkan**, and the real front-ends
**refuse an explicit `--backend metal` with a clear "foundation built, full render
lands in P2 — use `--backend vulkan`" message** rather than crashing. The Metal
*device + dispatch + present* foundation is exercised by the tests / smoke path
(which build a `MetalContext` directly via `make_context`), not by the full
renderer. The `auto`→Metal flip moves into P2 with megakernel render parity. The
spec's "auto selects Metal on Apple Silicon" scenario is amended to reflect this
phasing. *(User decision, this change.)*

## Open Questions

- **O1 (RESOLVED — in-process, no `slangc` shell-out):** SlangPy compiles a Slang
  compute entry to Metal in-process via `create_device(type=DeviceType.metal)` →
  `load_module_from_source` → `link_program` → `create_compute_kernel` →
  `kernel.dispatch(...)`; the read-back buffer was bit-identical to the expected
  values. Caveat: the entry point must not be named `main` (Metal reserves it;
  Slang renames `main`→`main_0` and pipeline creation fails) — foundation kernels
  use `computeMain`.
- **O2 (RESOLVED — slang-rhi surface, no manual `CAMetalLayer`):**
  `device.create_surface(WindowHandle(nswindow=<int>))` plus
  `Surface.configure` / `acquire_next_image` / `present` drives the swapchain
  directly; the GLFW Cocoa `NSWindow` pointer comes from
  `glfw.get_cocoa_window(window)`. `MetalContext.swapchain_info` wraps the
  `Surface` + its current config.
- **O3:** Rename `Renderer(vk_ctx=)` → `ctx=` now (touches the four call sites +
  the ctor) or defer as cosmetic cleanup? P1 defers unless review prefers it now.
