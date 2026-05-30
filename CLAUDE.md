# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow

Always work in a dedicated git worktree for new features and bug fixes. Create
a worktree off `main` for each piece of work instead of editing the primary
checkout directly, then open a PR / merge from there. This keeps the main
working directory clean and lets multiple changes proceed in isolation.

## Commands

**Setup (from repo root):**
```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

**MaterialX from source (required for the Slang backend):** The PyPI MaterialX
wheel does not include `PyMaterialXGenSlang`, which `materialx_runtime.py`
imports unconditionally. Without it the renderer fails at import. Build
upstream MaterialX (≥1.39) with the Slang generator and Python bindings, then
install the produced Python tree into the venv:

```bash
git clone --depth 1 https://github.com/AcademySoftwareFoundation/MaterialX.git
cd MaterialX
cmake -S . -B build \
  -DMATERIALX_BUILD_PYTHON=ON \
  -DMATERIALX_BUILD_GEN_SLANG=ON \
  -DMATERIALX_PYTHON_EXECUTABLE="$(pwd)/../.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$(pwd)/install"
cmake --build build --parallel
cmake --install build
../.venv/bin/pip uninstall -y MaterialX  # remove the wheel if pip pulled it in
../.venv/bin/pip install ./install/python
.venv/bin/python -c "from MaterialX import PyMaterialXGenSlang"
```

**Run:**
```bash
.venv/bin/skinny
# or
.venv/bin/python -m skinny.app
# force a backend
.venv/bin/python -m skinny.app --backend metal
.venv/bin/python -m skinny.app --backend vulkan
```

**Lint:**
```bash
.venv/bin/ruff check src/
```

**Tests:**
```bash
.venv/bin/pytest
```

**Verify Python syntax:**
```bash
.venv/bin/python -m py_compile src/skinny/app.py src/skinny/renderer.py
```

**Headless / offscreen rendering (for automated A/B image comparison):**
The fully-built environment (with `PyMaterialXGenSlang` + a working Vulkan/MoltenVK
runtime) lives in the **Python 3.13 venv at the repo root** (`./bin/python3.13`),
*not* `.venv` (which is 3.12 and lacks the Slang generator). Vulkan needs the SDK
on the dynamic-library path or `import vulkan` fails with
`Cannot find Vulkan SDK version`. Export these before any headless run:

```bash
export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
```

Render offscreen with no window via `VulkanContext(window=None, ...)` +
`Renderer(...)` + `update(dt)` + `render_headless()` (returns raw RGBA8 bytes,
`width*height*4`). Drive a USD scene by passing `usd_scene_path=` to `Renderer`;
USD load is async, so pump `update()` until `renderer._usd_scene` has the expected
instances before sampling frames. Accumulation resets automatically when any field
in `_current_state_hash()` changes (includes `integrator_index`, `env_intensity`,
`direct_light_index`), so switching integrators mid-script gives a clean A/B.
Set `renderer.direct_light_index = 1` to disable direct lights (isolate IBL).
`render_headless()` output is tonemapped/sRGB display pixels; for linear-HDR
comparison read the accumulation image instead. See `tests/test_headless.py`
(`TestMaterialXGraphDemoRender`) for a complete headless USD render example.

**Compile the Slang shader manually (requires `slangc` on PATH):**
```bash
slangc src/skinny/shaders/main_pass.slang -target spirv -entry mainImage -stage compute \
  -o src/skinny/shaders/main_pass.spv -I src/skinny/shaders
```

On macOS the app defaults to the native Metal backend (`--backend metal`) when
Apple's Metal compiler tools are installed. If `xcrun -find metal` or
`xcrun -find metallib` fails, the automatic backend falls back to Vulkan;
explicit `--backend metal` fails early with an Xcode setup hint. Other platforms
default to Vulkan. `SKINNY_BACKEND=metal|vulkan` also selects the backend.

## Architecture

### Layers

**Application layer (`app.py`)** — GLFW window, `InputHandler` (keyboard/mouse callbacks), and the `ALL_PARAMS` list that drives every adjustable parameter. Parameters have a `path` string (e.g. `"skin.melanin_fraction"`) that `_get_nested`/`_set_nested` resolve on the `Renderer` object. Settings (parameter snapshot + camera + window geometry) are persisted to `~/.skinny/settings.json` at exit and restored at startup.

**Renderer (`renderer.py`)** — owns shared render state and orchestrates per-frame work for the selected backend. Key objects:
- `SkinParameters` — the physically-based skin model. `pack()` serializes it to `std140`-aligned bytes that must match the `SkinParams` Slang struct exactly (offsets documented in the docstring).
- `OrbitCamera` / `FreeCamera` — both are always live; `camera_mode` selects which one's matrices feed the UBO. `toggle_camera_mode` transfers viewpoint between them.
- `_pack_uniforms()` — assembles the uniform buffer (FrameConstants + SkinParams + Light). The `_look_at` helper stores the matrix transposed relative to normal convention; numpy is row-major, GLSL reads row-major as column-major, so what numpy writes as rows appear to the GPU as columns — the net result is correct. Same for `_perspective`.
- Progressive accumulation — `accum_frame` increments while `_current_state_hash()` is unchanged. Any state change resets it to 0. The accumulation image is persistent across frames.
- Mesh rebaking — triggered by source/subdivision/displacement changes. Displacement-slider changes are debounced to 300 ms to avoid per-drag bakes. Auto-subdivision reads frequency stats from the displacement and normal maps via `head_textures.compute_texture_stats`.

**Metal plumbing (`metal_backend.py`)** — `MetalContext` uses SlangPy/RHI with `DeviceType.metal`, creates a surface from the GLFW Cocoa window handle, binds shader globals through `ShaderCursor`, and dispatches `main_pass.slang` directly to Metal.

**Vulkan plumbing (`vk_context.py`, `vk_compute.py`)** — `VulkanContext` owns instance, device, queues, swapchain, and command pool. `ComputePipeline` loads the pre-compiled SPIR-V and reflects the descriptor set layout. `StorageBuffer`, `StorageImage`, `SampledImage`, and `UniformBuffer` wrap GPU memory with synchronous upload helpers.

### Descriptor set bindings

The binding set has grown well past the original skin-only layout (now bindings
0–24 plus 30–32: flat-material params, bindless textures, MaterialX skin/std
params, multiple light buffers, BDPT splat, gizmo, lens/exit-pupil, tool
readback, and environment importance-sampling CDFs). The authoritative,
up-to-date map lives in **[Architecture.md](Architecture.md) → Descriptor
Binding Map**; do not duplicate it here.

### Shaders

All shaders are Slang (`.slang`) compiled to SPIR-V. `main_pass.slang` is the
primary compute entry point (`mainImage`); it includes the other modules. Key
ones:
- `materials/skin/*` — layered skin optics, BSSRDF, GGX specular, and the §1–§6
  estimator chain (see [SkinRendering.md](SkinRendering.md))
- `materials/flat/*` — flat/standard_surface/OpenPBR BSDF (`IMaterial`)
- `materials/debug_normal_material.slang` — normal visualisation material
- `integrators/{path,bdpt}.slang` — path tracer + bidirectional path tracer
- `samplers/*` — GGX (VNDF), Lambert, uniform-sphere, Henyey-Greenstein, MIS
- `lights/*` — sphere, emissive-triangle, directional light implementations
- `volume_render.slang` — delta-tracked volume transport, Henyey-Greenstein phase
- `mesh_head.slang` / `sdf_head.slang` — mesh BVH traversal / analytic SDF head
- `skin.slang` / `bvh_refit.slang` — UsdSkel GPU linear-blend skinning + BVH
  refit (standalone compute pipelines in `vk_skinning.py`, own descriptor sets;
  Vulkan only — CPU fallback elsewhere). Not included by `main_pass.slang`
- `environment.slang` — equirectangular lookup, furnace override, env
  importance-sampling distribution
- `cameras/{pinhole,thick_lens}.slang` — camera ray generation
- `common.slang` / `bindings.slang` / `interfaces.slang` — shared types, binding
  declarations, pluggable interfaces

The full module map lives in **[Architecture.md](Architecture.md)**.

A pre-compiled `main_pass.spv` is checked in. Any shader change requires a recompile with `slangc`.

### Assets

- `hdrs/` — Radiance `.hdr` files (Poly Haven HDRIs). `fetch_hdrs.py` documents which ones.
- `heads/` — OBJ head models. Each immediate subdirectory with an `.obj`, or a loose top-level `.obj`, becomes a selectable head. Texture maps are discovered by filename keywords (`normal`/`nrm`/`nor`, `rough`/`roughness`, `displacement`/`disp`/`height`/`bump`).
- `tattoos/` — PNG/JPG tattoo images. Alpha channel drives ink density.

### Presets and settings

User settings live in `~/.skinny/`. `settings.json` stores window geometry + parameter values + camera state. User-saved presets are individual JSON files under `~/.skinny/presets/`. The `preset_index` parameter is intentionally excluded from the snapshot because the user's preset list can change between sessions.
