## Why

The repo's docs (`CLAUDE.md`, `openspec/config.yaml`) describe a native Metal
backend — "Metal backend in `metal_backend.py` (SlangPy/RHI)", "Default backend:
Metal on macOS", "force backend `--backend metal|vulkan`". **None of that exists
in the code.** The live renderer (`renderer.py`, 8.6k lines) talks raw Vulkan
through `vk_context.py` / `vk_compute.py`; `--backend` / `SKINNY_BACKEND` /
`select_backend` are defined nowhere; the only Metal artifact is
`src/skinny/gfx/metal/__init__.py`, a stub whose `MetalBackend.create()` raises
`NotImplementedError`. On macOS the app renders through **MoltenVK under the
Vulkan path**, not native Metal. The `gfx/` Backend ABC the stub belongs to is
unused (zero importers outside `gfx/`); a full native-Metal renderer once lived
on the unmerged `metal` branch but is 264 commits behind `main` and unmergeable.

This change lays the **foundation** for a real native Metal backend: a Metal
device + context that mirrors the Vulkan context's surface, the backend-selection
seam (`--backend`, `auto`, fallback) wired through every front-end, and a proven
trivial compute dispatch + present on Metal. It is **P1 of a phased plan** (see
`design.md`); subsequent changes build megakernel render parity, ReSTIR, neural
inference, wavefront, and the MLX↔Metal zero-copy weight handoff on top of this
seam. The handoff (the `add-mlx-training-backend` change's §3) is blocked today
precisely because there is no native Metal `MTLBuffer` to alias; this foundation
is the prerequisite that unblocks it.

## What Changes

- **Add a native Metal device foundation.** New `MetalContext`
  (`metal_context.py`) wrapping SlangPy's `DeviceType.metal` device: surface from
  the GLFW Cocoa window, command queue, frame fences, swapchain-equivalent +
  present. It exposes the **same duck-typed surface the renderer already reads off
  `VulkanContext`** (`width`/`height`, compute/present queues, swapchain info,
  command allocation, capability flags, `recreate_swapchain`, `destroy`), so the
  renderer drives it without Metal-specific knowledge at the context layer. A
  minimal `metal_compute.py` (`StorageBuffer`, `StorageImage`, `ComputePipeline`)
  proves a trivial Slang compute dispatch + present.
- **Add backend selection.** `--backend {auto,metal,vulkan}` (+ `SKINNY_BACKEND`
  env) added once to the shared `add_render_flags`, plus a `backend_select.py`
  resolver (`select_backend` + `make_context`) routed through all four front-ends
  (`skinny`, `skinny-gui`, `skinny-web`, `skinny-render`). `auto` selects native
  Metal on Apple-Silicon macOS when the Metal device constructs, otherwise Vulkan.
  An explicitly requested backend that cannot be built fails with a clear message;
  `auto` falls back to Vulkan. `vulkan` stays byte-identical to today.
- **Out of scope for P1 (later phases):** the megakernel head render on Metal,
  MSL uniform layout, textures/`commonSampler`, the descriptor binding map,
  ReSTIR, neural inference, wavefront, and the MLX↔Metal handoff. P1 proves the
  device + dispatch + present + selection seam; nothing renders the head yet.

## Capabilities

### New Capabilities
- `metal-backend`: a native Metal device backend foundation built on SlangPy
  `DeviceType.metal`, plus backend-selection resolution (`auto`/`metal`/`vulkan`)
  with capability gating and an `auto`→Vulkan fallback. Scoped to device
  bring-up + a trivial compute dispatch + present; later changes extend it toward
  full render parity.

### Modified Capabilities
- `render-cli`: the unified render-selection flag surface gains
  `--backend {auto,metal,vulkan}` (+ `SKINNY_BACKEND` env), exposed identically by
  every front-end from the single shared definition.

## Impact

- **Code (new):** `src/skinny/metal_context.py` (`MetalContext`),
  `src/skinny/metal_compute.py` (minimal resource wrappers),
  `src/skinny/backend_select.py` (`select_backend` + `make_context`).
- **Code (modified):** `src/skinny/cli_common.py` (`--backend` in
  `add_render_flags`); `src/skinny/app.py`, `src/skinny/headless.py`,
  `src/skinny/web_app.py`, `src/skinny/ui/qt/app.py` (route ctx construction
  through `make_context`; persist/restore the selected backend);
  `src/skinny/vk_context.py` (add a `backend_name`/`is_metal` attribute for
  downstream guards — Vulkan side unchanged behaviorally).
- **Dependencies:** none new — SlangPy (already a dependency) provides
  `DeviceType.metal`. Metal paths are Apple-Silicon/macOS only and guarded.
- **Tests:** backend-selection resolution (auto/explicit/fallback/clear-error);
  a headless trivial-dispatch parity test (Metal vs Vulkan identical output,
  skipped when a backend is unavailable); a windowed Metal present smoke test
  (gated to a Mac display).
- **Docs:** `README.md` (`--backend` flag), `docs/Architecture.md` (the
  selection seam + `MetalContext` foundation), `CHANGELOG.md`. `CLAUDE.md` /
  `config.yaml` already describe Metal as present — this change starts making
  that real (the file name is `metal_context.py`, not the docs' `metal_backend.py`).
- **Platform:** non-macOS and `--backend vulkan` are byte-identical to today;
  Metal selection is guarded and degrades to Vulkan under `auto`.
