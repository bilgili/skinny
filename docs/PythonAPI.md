# Skinny — Python API Reference

This document is the developer reference for Skinny's **public Python API**:
the headless render interface, the `Renderer` programmatic surface, scene
loading, parameters, and the persistence/plugin helpers. For architecture and
GPU internals see [Architecture.md](Architecture.md); for the two GPU execution
modes see [Megakernel.md](Megakernel.md) / [Wavefront.md](Wavefront.md).

- **Package:** `skinny` (`src/skinny/`), `__version__ = "0.1.0"`, `requires-python >= 3.11`.
- **Most users want** `skinny.headless` — render a USD scene to a NumPy array or
  image file with no window. Everything else is the live-app surface.

> **Build prerequisite.** The Slang generator `PyMaterialXGenSlang` is **not** in
> the PyPI MaterialX wheel — build MaterialX from source with
> `-DMATERIALX_BUILD_PYTHON=ON -DMATERIALX_BUILD_GEN_SLANG=ON`. See `README.md`.
> Vulkan also needs the SDK on `DYLD_LIBRARY_PATH` (`VULKAN_SDK/lib`).

---

## 1. Console entry points

Declared in `pyproject.toml` `[project.scripts]`:

| Script | Target | Purpose |
|--------|--------|---------|
| `skinny` | `skinny.app:main` (`app.py:450`) | GLFW shader-debug window |
| `skinny-gui` | `skinny.ui.qt.app:main` | Qt desktop app |
| `skinny-web` | `skinny.web_app:main` | Panel web app (per-session server render) |
| `skinny-render` | `skinny.headless:main` (`headless.py:329`) | offscreen CLI renderer |
| `skinny-import-pbrt` | `skinny.pbrt.cli:main` | convert a pbrt v4 scene to USD |

The top-level `skinny` package defines no `__all__`; import submodules directly
(`from skinny import headless`, `from skinny.renderer import Renderer`).

### `skinny.pbrt` — pbrt v4 importer

`import_pbrt(path, out=None, materialx=False) -> (Usd.Stage, Report)` parses a
pbrt v4 scene and emits a USD stage loadable by `usd_loader`; `out` also writes a
`.usda`/`.usd`. With `materialx=True` it additionally writes a `<out>.mtlx`
sidecar of `standard_surface` materials (referenced from the stage; the
`skinny-import-pbrt -mtlx` flag). The returned `Report` classifies each construct
as exact / approx / skipped. See [PbrtImport.md](PbrtImport.md) for the full
mapping and the parity matrix.

`skinny.pbrt.materials.map_material_mtlx(pbrt_material, *, emissive_rgb=None,
textures=None, base_dir=None) -> (inputs, tex_inputs, status, notes)` is the
sibling of `map_material` that targets Autodesk `standard_surface` input names
(filling `transmission`/`coat`/`subsurface`/`specular_anisotropy`/`thin_walled`
that UsdPreviewSurface drops). `skinny.pbrt.mtlx_emit.write_mtlx_document(...)` /
`author_mtlx_reference(...)` author the `.mtlx` document and the stage reference
the loader resolves.

`skinny.pbrt.spectral` is the numpy mirror of the hero-wavelength spectral
estimator (change `spectral-rendering`). `d65_normalized() -> np.ndarray` returns
the CIE D65 SPD scaled to unit luminance — pbrt's whitepoint illuminant, and the
exact curve the renderer uploads to the GPU (binding 47) so the shader's
`upsampleIlluminant` matches this mirror. `upsample_illuminant(rgb, lam)`
implements pbrt's `RGBIlluminantSpectrum` — `scale·sigmoid(rgb/scale)·D65_norm`
with `scale = 2·max(rgb)` — so HDR emitters (values > 1) stay in the sigmoid's
`[0,1]` gamut while keeping their magnitude, and a unit RGB illuminant resolves
to unit radiance (its reflectance sibling is `upsample_reflectance(rgb, lam)`).

---

## 2. Headless render API — `skinny.headless`

The public offscreen interface. Accepts a USD file path **or** an already-open
`Usd.Stage`, holds the GPU context across calls, and returns RGBA8 pixels.

![Headless API flow: a module wrapper or HeadlessRenderer context manager owns a windowless VulkanContext + Renderer; render_headless returns RGBA8 bytes that become a NumPy array or an image file.](diagrams/headless_api.svg)

```python
Source = Union[str, Path, "Usd.Stage", object]   # headless.py:24
```

### One-shot module wrappers

Each opens a `HeadlessRenderer`, renders, and tears the context down — convenient
for a single image:

```python
def render_to_array(source, *, width=1024, height=1024, gpu=None, **kw) -> np.ndarray   # :272
def render_scene(source, output, *, width=1024, height=1024, gpu=None, **kw) -> None    # :278
def render_animation(source, outdir, *, width=1024, height=1024, gpu=None, **kw) -> list # :284
```

```python
import skinny.headless as sk

# RGBA8 array, shape (height, width, 4), dtype uint8
img = sk.render_to_array("assets/three_materials_demo.usda",
                         width=1280, height=720, samples=128, integrator="bdpt")

# write straight to a file (format inferred from extension)
sk.render_scene("scene.usda", "out.png", samples=256, tonemap="aces", exposure=0.5)
sk.render_scene("scene.usda", "out.exr")   # linear HDR from the accum buffer
```

### `HeadlessRenderer` — persistent context

Reuse one GPU context for many renders (pipeline compiles once). Use as a
context manager.

```python
class HeadlessRenderer:                                                  # :125
    def __init__(self, width, height, *, gpu=None,
                 execution_mode="megakernel", bdpt_walk="fused",
                 proposals=None, reuse=None): ...                        # :136
    def __enter__(self) -> "HeadlessRenderer": ...                       # :163
    def __exit__(self, *exc) -> None: ...                               # :166
    def cleanup(self) -> None: ...                                       # :169
    def render_to_array(self, source, *, samples=64, time=None, **opts) -> np.ndarray   # :192
    def render_scene(self, source, output, *, samples=64, time=None,
                     format=None, **opts) -> None                        # :205
    def render_animation(self, source, outdir, *, samples=64,
                         frames=None, fps=None, ext="png", **opts) -> list # :221
```

| Method | Returns | Notes |
|--------|---------|-------|
| `render_to_array` | `np.ndarray` `(H, W, 4)` uint8 RGBA8 (a `.copy()`) | tonemapped/sRGB display pixels |
| `render_scene` | `None` | writes a file via `save_screenshot`; LDR `{png,jpeg,bmp}` or HDR `{exr,hdr}` |
| `render_animation` | `list[Path]` | one file per timecode, `frame_{i:0Nd}.{ext}`; `fps` accepted but unused |

```python
with sk.HeadlessRenderer(1024, 1024, execution_mode="wavefront") as r:
    a = r.render_to_array("scene.usda", samples=64)
    b = r.render_to_array("scene.usda", samples=64, integrator="bdpt")  # A/B, same context
    r.render_animation("anim.usda", "frames/", frames=(0, 48), samples=32)
```

`render_to_array` raises `RuntimeError` if the pipeline failed to compile
(`self.renderer.pipeline is None`). `render_animation` raises `ValueError` if
`time`/`format` are passed through `**opts`.

### `RenderOptions` — per-render knobs

`**opts` on every render call are resolved against this dataclass:

```python
@dataclass
class RenderOptions:                                  # :32
    samples: int = 64
    integrator: str = "path"        # "path" | "bdpt"
    exposure: float = 0.0           # EV stops, 2^EV
    tonemap: str = "aces"           # aces | reinhard | hable | linear
    env_intensity: Optional[float] = None  # fallback IBL only
    direct_light: bool = True       # fallback DistantLight only
    time: object = None             # None | int | float | Usd.TimeCode
```

`_INTEGRATORS = {"path": 0, "bdpt": 1}`, `_TONEMAPS = {"aces":0,"reinhard":1,"hable":2,"linear":3}`.

`env_intensity` and `direct_light` configure Skinny's fallback pair only. If the
active USD scene contains any authored light or emissive material, the values
are retained for a later fallback scene but cannot alter current authored
lighting. A light-less scene enables the default DistantLight and built-in IBL
together; `direct_light=False` may then disable the fallback DistantLight.

### `skinny-render` CLI

```
skinny-render SOURCE [-o OUT] [--animate --outdir DIR --frames S:E[:STEP] --fps F]
              [--width W] [--height H] [--samples N]
              [--tonemap {aces,reinhard,hable,linear}] [--exposure EV]
              [--env-intensity X] [--no-direct] [--format FMT] [--ext EXT] [--gpu G]
              [--integrator {path,bdpt}] [--execution-mode {megakernel,wavefront}]
              [--bdpt-walk {fused,eye,eye_light}] [--proposals ...] [--reuse ...]
```

`main(argv=None) -> int` returns `0` ok / `1` on error. `--frames S:E[:STEP]`
parsed by `_parse_frames`. Shared render flags injected by
`cli_common.add_render_flags`.

---

## 3. `Renderer` — programmatic surface (`skinny.renderer`)

The renderer owns all GPU state and per-frame dispatch. `skinny.headless` wraps
it; drive it directly for custom loops (the live front-ends do).

### Construction

```python
class Renderer:                                       # :908
    WAVEFRONT_BDPT_SUPPORTED = True
    def __init__(self, vk_ctx: VulkanContext, shader_dir: Path,
                 hdr_dir: Path | None = None, tattoo_dir: Path | None = None,
                 usd_scene_path: Path | None = None,
                 use_usd_mtlx_plugin: bool = False,
                 execution_mode: str = "megakernel",
                 bdpt_walk: str = "fused",
                 neural_config: NeuralBuildConfig | None = None) -> None: ... # :917
```

`execution_mode` (`"megakernel"` | `"wavefront"`) and `bdpt_walk`
(`"fused"` | `"eye"` | `"eye_light"`) are **fixed for the renderer's lifetime** —
they are excluded from the accumulation state hash.

`neural_config` (`skinny.sampling.neural_weights.NeuralBuildConfig`, default
`None` ⇒ the shipped `fp32 @ 6/24/96`) selects the neural proposal's **size +
precision** for the renderer's lifetime (the size×precision study builds a fresh
renderer per grid cell). It is also fixed for the lifetime; on a device lacking
fp16 it falls back to fp32 (logged). See [Neural build config](#neural-build-config).

#### Neural build config

`NeuralBuildConfig` + `NeuralPrecision` (`skinny.sampling.neural_weights`) are the
single source of truth for the neural proposal's build-time size + precision
(study `neural-precision-size-study`):

```python
from skinny.sampling.neural_weights import NeuralBuildConfig, NeuralPrecision

NeuralBuildConfig()                                   # shipped fp32 @ 6/24/96 (default)
NeuralBuildConfig(precision=NeuralPrecision.FP16_STORAGE)   # half weights, float GEMM
NeuralBuildConfig(layers=4, bins=16, hidden=48)             # a smaller net
```

| Member | Notes |
|--------|-------|
| `NeuralBuildConfig(layers=6, bins=24, hidden=96, precision=FP32)` | frozen dataclass; the default reproduces the shipped net |
| `.slang_defines() -> tuple[str, ...]` | the `slangc -D` tokens; **empty for the default** (⇒ byte-identical compiles) |
| `.cache_tag -> str` | slug folded into the wavefront `.spv` name so configs don't collide |
| `.arch -> (layers, bins, hidden, cond)` | the NFW1 architecture the loader validates |
| `NeuralPrecision.{FP32, FP16_STORAGE, FP16_COMPUTE, FP8_STORAGE}` | `.weight_half` / `.compute_half` / `.weight_fp8` + `.storage_bytes` (4/2/1) drive the upload dtype + the `NF_WT`/`NF_CT`/`NF_FP8` defines |
| `NeuralWeights.weight_bytes_for(precision)` / `.bias_bytes_for(...)` | fp32 NFW1 → the upload bytes (half for the fp16 modes; e4m3 packed in `uint` words for `FP8_STORAGE`, a quarter of fp32) |
| `f32_to_e4m3(arr)` / `e4m3_to_f32(bytes)` (`skinny.sampling.neural_weights`) | the e4m3 (OCP E4M3FN) codec mirrored bit-for-bit by `neural_flow.slang nf_decode_e4m3` (the fp8-storage decode) |

#### Training backends

`skinny.sampling.training_backends` is the pluggable per-cycle **training-compute**
seam behind the online trainer (change `neural-trainer-backends`). `NeuralTrainer`
stays the orchestrator; the backend owns only the gradient step.

```python
from skinny.sampling.training_backends import (
    make_training_backend, build_dataset_np,
    NumpyTrainingBackend, TorchTrainingBackend, MlxTrainingBackend)

make_training_backend("auto")     # precedence cuda > mlx > cpu (best available)
make_training_backend("cpu")      # NumpyTrainingBackend — torch-free, always available
make_training_backend("cuda")     # TorchTrainingBackend(device="cuda"); raises if absent
make_training_backend("mlx")      # MlxTrainingBackend — Apple MLX on Metal; raises off Apple Silicon
```

| Member | Notes |
|--------|-------|
| `TrainingBackend` (ABC) | `is_available` / `supports_precision(p, device)` / `warm_start(weights, cfg)` / `update(cond, z, w) -> float\|None` / `export() -> NeuralWeights`; stateful across cycles (warm model + optimizer) |
| `make_training_backend(kind="auto", *, device="auto", train_precision="fp32", spline_flow_path=None)` | token → backend (`cpu`→numpy, `cuda`→torch, `mlx`→MLX on Apple-Silicon Metal); `auto` precedence `cuda > mlx > cpu`; unavailable explicit token raises clearly |
| `TRAINING_BACKENDS` | name-keyed token table (`cpu` / `cuda` / `mlx`) |
| `NumpyTrainingBackend` | torch-free reference oracle: forward + backward of the contribution-weighted MLE via a tiny pure-numpy autodiff tape; fp32 only |
| `TorchTrainingBackend(device="cpu\|mps\|cuda")` | the torch loop; CUDA autocast-fp16 at `train_precision="fp16"`; in-memory bake |
| `MlxTrainingBackend` | Apple MLX GPU loop on Apple-Silicon Metal (optional `[mlx]` extra); mirrors the numpy oracle's flow math with MLX autodiff + hand-rolled bias-corrected Adam; `train_precision="fp16"` = float16 compute over fp32 masters with runtime fp32 fall-back; in-memory fp32 bake |
| `build_dataset_np(batch, bounds) -> (cond, z, w)` | the shared numpy dataset contract (contiguous float32), consumed by every backend |
| `TrainerConfig.backend` / `.train_precision` | select the backend (`cpu\|cuda\|mlx\|auto`) and the optimizer precision (`fp32\|fp16`); `arch.precision` is the independent inference precision |

#### Online training driver

`Renderer` exposes the online-training lifecycle the front-ends drive (change
`online-training-trigger`). `enable_online_training` builds the
`ReplayBuffer` + `NeuralTrainer` + `NeuralWeightPublisher` and **starts a daemon
trainer thread** that loops `online_train_and_publish` off the render thread;
`disable_online_training` signals it to stop and joins it. The render loop calls
`online_training_tick()` once per frame; the frame-end swap inside
`render()` / `render_headless()` promotes newly published weights.

| Method | Signature | Notes |
|--------|-----------|-------|
| `can_online_train()` | `(self) -> tuple[bool, str]` | prerequisite gate: `(True, "")` only when the execution mode is wavefront **and** a neural proposal is active; otherwise `(False, reason)` naming the missing prerequisite. The front-ends refuse loudly on `False`, never a silent no-op |
| `enable_online_training(*, handoff=None, trainer_backend=None, train_precision=None, replay=None, trainer=None, capacity=1_000_000, **publisher_kwargs)` | `-> NeuralWeightPublisher` | builds the replay buffer + trainer + publisher, sets `_online_training`, resolves the record source, and starts the background trainer thread. `handoff`/`trainer_backend`/`train_precision` override the renderer's `--neural-handoff`/`--neural-trainer`/`--train-precision`. Surfaces the existing `mlx`/`interop` errors |
| `online_training_tick()` | `(self) -> int` | per-frame driver: drains GPU path records into the replay buffer on the **render thread** and returns the count; a no-op returning `0` when training is off or the scene isn't built yet. Returns promptly — the per-cycle training runs on the trainer thread |
| `disable_online_training()` | `(self) -> None` | clears `_online_training` and stops + joins the trainer thread; safe when never enabled |
| `online_training_active` (property) | `-> bool` | whether the loop is on |

`sampling.neural_handoff.make_publisher(kind, **kwargs) -> NeuralWeightPublisher`
is the factory behind the `handoff` value: `"file"` → `FileWeightPublisher`,
`"shared"` → `SharedWeightPublisher` (in-process CPU double-buffer in RAM, no disk
and no GPU-interop, any platform; change `shared-neural-handoff`), `"interop"` →
the per-backend GPU publisher. `shared` takes only `initial` / `expect_arch` (no
buffer/semaphore kwargs). The in-memory copy uses the new
`sampling.neural_weights.serialize_neural_weights(nw) -> bytes` /
`deserialize_neural_weights(data, expect=None) -> NeuralWeights` pair, which
`write_neural_weights` / `load_neural_weights` now wrap (NFW1 on-disk format
unchanged).

### Frame loop & output

| Method | Signature | Notes |
|--------|-----------|-------|
| `update(dt)` | `(self, dt: float) -> None` (`:6985`) | advance animation/camera, detect dirty state, reset accumulation |
| `render()` | `(self) -> None` (`:7065`) | windowed: dispatch + present to swapchain |
| `render_headless()` | `(self) -> bytes` (`:7289`) | **returns raw RGBA8 `bytes`, length `width*height*4`** (tonemapped/sRGB) |
| `read_accumulation_hdr()` | `(self) -> tuple[np.ndarray, int]` (`:8363`) | linear-HDR readback: `(H, W, 4)` float32 accum buffer + sample count; divide array by count for mean radiance. **Vulkan + Metal** (Metal drains the rgba32_float texture directly — no transfer/fence) |
| `read_structural_aov()` | `(self) -> np.ndarray` | dispatch one `TOOL_MODE_STRUCTURAL` frame and read the per-primary-ray structural channel: `(H, W, 4)` float32 = `(hit_mask, instance_id, material_id, depth)`. Deterministic across backends — backs the Metal↔Vulkan structural-parity test (6.1). Requires a built megakernel pipeline; resolution must fit the tool buffer (raises otherwise). **Vulkan + Metal** |
| `resize(width, height)` | `(self, int, int) -> None` (`:8265`) | change render resolution at runtime (clamped ≥64, workgroup-aligned); recreates offscreen/accum/HUD images, resets accumulation. **Vulkan + Metal** |
| `save_screenshot(path_or_file, fmt)` | `-> None` (`:7631`) | `png`/`jpeg`/`bmp` → LDR; `exr`/`hdr` → linear HDR from accum buffer |
| `dump_path_records(out_path, *, num_frames=256, ...)` | `-> int` | offline neural training-record dump → a `.nrec` file (per-vertex `(pos, N, wo, wiLocal, contribution)` via the `mainImageRecord` megakernel entry); returns the record count. Feeds `spline_flow/render_records.py`. |
| `cleanup()` | `(self) -> None` (`:7692`) | release GPU resources |

> There is **no** `render_offscreen` method — the offscreen primitive is
> `render_headless()` returning `bytes`.

Progressive accumulation: `update()` resets `accum_frame` to 0 whenever
`_current_state_hash()` changes (camera, params, env, integrator, proposal seam,
playback time, …), otherwise increments it. So mutating a public attribute
mid-loop gives a clean A/B — pump `update(dt)` until converged, then read.

```python
from skinny.vk_context import VulkanContext
from skinny.renderer import Renderer
from pathlib import Path

ctx = VulkanContext(window=None, width=1024, height=1024)
r = Renderer(ctx, Path("src/skinny/shaders"), usd_scene_path=Path("scene.usda"))
for _ in range(64):
    r.update(1/60)
rgba = r.render_headless()          # bytes, 1024*1024*4
r.cleanup(); ctx.destroy()
```

### Scene swap & editing

```python
def set_usd_scene(self, scene: "Scene", stage=None) -> None             # :5827
def add_model(self, usd_path, parent_prim_path="/World",
              name=None, transform=None) -> str                         # :6012
def add_light(self, light_type, parent_prim_path="/World",
              name=None, transform=None) -> str                         # :6063
def remove_node(self, prim_path: str) -> None                           # :6145
def set_transform(self, prim_path: str, matrix) -> None                 # :6161
def save_edits(self, path: str | None = None) -> str                    # :6191
def list_nodes(self) -> list[dict]                                      # :6206
```

- `set_usd_scene` is the synchronous headless scene-swap. It does **not** build
  the scene-graph model, so the edit API below is unavailable after a bare swap.
- The edit API (`add_model` / `add_light` / `remove_node` / `set_transform` /
  `save_edits` / `list_nodes`) requires a **USD stage with an attached edit
  layer** (the interactive load path); each raises `RuntimeError` otherwise.
  `add_model` returns the new prim path (USD only — OBJ raises `ValueError`).
  `add_light` accepts `DistantLight`, `SphereLight`, `DomeLight`, `RectLight`,
  or `DiskLight`, returns a unique prim path, authors explicit defaults, and
  immediately resyncs the scene. `remove_node` deactivates non-destructively;
  `save_edits` defaults to `<scene>.edits.usda`; `list_nodes` returns
  `[{"path", "type", "active"}, …]`.

### Public attributes set programmatically

`integrator_index` (0 path / 1 bdpt), `tonemap_index` (0 ACES…3 linear),
`exposure` (EV), `direct_light_index` (0 on / 1 off), `env_intensity`,
`proposal_preset_index`, `reuse_index`. `film` is a `FilmParameters` dataclass
(`film.iso=100`, `film.exposure_time=1.0`, change `pbrt-radiometric-parity`): the
pbrt film exposure controls read from the authored camera (`skinny:film:*`); their
imaging ratio `exposure_time·iso/100` is a live linear output scale (multiplies the
linear-HDR read; folded into the display exposure). Retunable via the `film.iso` /
`film.exposure_time` params; a change resets accumulation. Mode lists:
`integrator_modes = ["Path","BDPT"]`, `tonemap_modes = ["ACES","Reinhard","Hable","Linear"]`,
`proposal_preset_modes`, `reuse_modes = ["None"]`.

`renderer.uses_default_lights` reports the active lighting authority.
`direct_light_index` and `env_intensity` affect only the synthesized fallback
pair; authored USD lighting remains controlled by its USD light/material state.

### `SkinParameters` dataclass (`renderer.py:504`)

The physically-based skin model; `pack()` serialises to the 80-byte std140
`SkinParams` Slang struct. Fields (defaults): `melanin_fraction=0.15`,
`epidermis_thickness_mm=0.1`, `hemoglobin_fraction=0.05`,
`blood_oxygenation=0.75`, `dermis_thickness_mm=1.0`, `subcut_thickness_mm=3.0`,
`scattering_coefficient=[3.7,4.4,5.05]`, `anisotropy_g=0.8`, `roughness=0.35`,
`ior=1.4`, `pore_density`, `pore_depth`, `hair_density`, `hair_tilt`.

### Cameras

`OrbitCamera` (`:708`) and `FreeCamera` (`:804`) are both always live;
`camera_mode` (`"orbit"` | `"free"` | `"usd"`) selects which feeds the UBO.
`OrbitCamera`: `orbit(dx,dy)`, `set_distance(v)`, `zoom(d)`, `pan(dx,dy)`,
`position` (property), `view_matrix()`. `Renderer.toggle_camera_mode()` (`:1324`)
transfers the viewpoint between orbit and free.

---

## 4. `VulkanContext` (`skinny.vk_context`)

```python
class VulkanContext:                                  # :27
    VALIDATION_LAYERS = ["VK_LAYER_KHRONOS_validation"]
    def __init__(self, window=None, width=1280, height=720, *,
                 enable_validation=True, gpu_preference=None) -> None    # :32
    def wait_idle(self) -> None                                         # :477
    def destroy(self) -> None                                           # :356
```

`window=None` → headless: no surface/swapchain, `present_queue = None`,
compute queue only. Exposes `.width`, `.height`, `.device`, `.physical_device`,
`.gpu_info`, `.compute_queue`, `.command_pool`. `destroy()` waits idle and tears
everything down. (The wavefront path keys off `hasattr(ctx, "compute_queue")`.)

`wait_idle()` is the backend-neutral device drain (`vkDeviceWaitIdle` on Vulkan,
`Device.wait_for_idle` on the Metal twin `MetalContext`); the renderer calls
`ctx.wait_idle()` before tearing down a pipeline so neither backend reaches for a
Vulkan-only symbol on a rebuild.

---

## 5. Parameters (`skinny.params`)

Adjustable parameters are addressed by a **`path` string** resolved on the
`Renderer` instance, so UIs and presets can set anything generically.

```python
@dataclass
class ParamSpec:                                      # :40
    name: str
    path: str                       # e.g. "mtlx.skin_bsdf_roughness"
    kind: str                       # "continuous" | "discrete"
    step: float = 0.0
    lo: float = 0.0
    hi: float = 0.0
    choice_source: str | None = None  # discrete: renderer attribute holding the choice list
```

### `path` resolution (`_get_nested` / `_set_nested`, `params.py:214` / `:255`)

> These two are **module-private functions** in `params.py` (not `Renderer`
> methods), but they are the load-bearing mechanism behind every slider/preset.

- `"mtlx.<field>"` → `renderer.mtlx_overrides[field]` (scalar), falling back to
  the active material's uniform-block default.
- `"mtlx.<field>.<idx>"` → one vector component of `mtlx_overrides[field]`.
- `"<a>.<b>"` → plain `getattr` chain (e.g. `"skin.melanin_fraction"`).

Legacy `skin.*` paths alias to `mtlx.*` (`_SKIN_TO_MTLX`); linked fields gang via
`_GANGED_MTLX_FIELDS`.

### Building the live list

```python
STATIC_PARAMS: list[ParamSpec]                # :101
ALL_PARAMS = STATIC_PARAMS                     # :155  (back-compat alias; re-imported by app.py)
def build_dynamic_params(renderer) -> list[ParamSpec]   # :181  (material-driven)
def build_all_params(renderer) -> list[ParamSpec]       # :209  (static + dynamic)
```

Representative paths: `"env_intensity"`, `"exposure"`, `"integrator_index"`,
`"tonemap_index"`, `"direct_light_index"`, `"model_index"`, `"tattoo_density"`,
`"mtlx.layer_top_melanin"`, `"mtlx.layer_middle_hemoglobin"`,
`"mtlx.skin_bsdf_roughness"`, `"light_elevation"`, `"light_intensity"`.

### Execution-mode & integrator constants

```python
EXECUTION_MEGAKERNEL = 0    # :64
EXECUTION_WAVEFRONT  = 1    # :65
def clamp_mode_index(index, n_modes) -> int                              # :68
def effective_execution_mode(selected_index, integrator_index,
                             wavefront_bdpt_supported) -> int            # :79
```

Integrator indices `0 = path`, `1 = bdpt` (consistent with
`cli_common.INTEGRATOR_INDEX`). Resolution presets:
`RESOLUTION_PRESETS: list[tuple[str,int,int]]` (`:18`) +
`find_resolution_preset_index(width, height)` (`:32`).

### Shared CLI helpers (`skinny.cli_common`)

```python
INTEGRATOR_INDEX = {"path": 0, "bdpt": 1}             # :21
WALK_CHOICES = ("fused", "eye", "eye_light")          # :25
def resolve_walk(value: str) -> str                   # :29
def add_render_flags(parser, *, integrator=True, execution=True,
                     walk=True, proposals=True, reuse=True) -> None      # :42
```

---

## 6. Scene loading (`skinny.usd_loader`)

```python
def load_scene_from_usd(stage_path, *, time=None, use_usd_mtlx_plugin=False) -> Scene    # :1970
def load_scene_from_stage(stage, *, time=None, use_usd_mtlx_plugin=False) -> Scene       # :1998
def prepare_usd_streaming(stage_path, *, time=None, use_usd_mtlx_plugin=False
    ) -> tuple[Scene, list[tuple[MeshSource, np.ndarray, int]]]                          # :2026
def build_animation_index(stage) -> AnimationIndex                                       # :1542
def build_playback_clock(stage, index)                                                   # :1591
def extract_skeletal_bindings(stage) -> SkeletalScene                                    # :1761
def extract_ui_controls(stage) -> list[ControlSpec]                                      # :1822
def resolve_control_binding(renderer, spec)                                              # :1873
def summarize(scene) -> str                                                              # :2044
```

`load_scene_from_usd` is the blocking path loader; `load_scene_from_stage` takes
a caller-owned (possibly mutated) stage — the entry headless callers use to swap
scenes. Public dataclasses: `AnimationIndex` (`.has_animation()`),
`SkinnedMeshBinding`, `SkeletalScene` (`.has_skinning()`), `ControlSpec`.

```python
from pxr import Usd
from skinny import usd_loader

stage = Usd.Stage.Open("scene.usda")
scene = usd_loader.load_scene_from_stage(stage)
renderer.set_usd_scene(scene, stage=stage)
```

---

## 7. Settings & presets

### `skinny.settings` — `~/.skinny/`

```python
def ensure_dirs() -> None                             # :41
def load_settings() -> dict[str, Any]                 # :49
def save_settings(data: dict[str, Any]) -> None       # :60   (atomic tmp + os.replace)
def get_last_dir(category) -> str                     # :86
def record_last_dir(category, directory) -> None      # :99
def load_user_presets() -> list[Preset]               # :129
def save_user_preset(name, values) -> Path            # :153
def delete_user_preset(name) -> bool                  # :168
```

Constants: `SETTINGS_DIR` (`~/.skinny`), `PRESETS_DIR`, `MESH_CACHE_DIR`,
`SETTINGS_FILE`. `settings.json` holds window geometry + parameter snapshot +
camera; user presets are one JSON per file.

### `skinny.presets`

```python
@dataclass(frozen=True)
class Preset:                                          # :27
    name: str
    values: dict[str, float] = field(default_factory=dict)
    is_builtin: bool = True

PRESETS: list[Preset]                                  # :67  (Fitzpatrick I–VI × Female/Male)
def apply_preset(renderer, preset: Preset) -> None     # :72  (writes values via _set_nested)
```

---

## 8. Plugin & backend sub-APIs (`__all__`-exporting modules)

These submodules expose a curated `__all__` (the top-level package does not):

| Module | Exports |
|--------|---------|
| `skinny.sampling` (`__init__.py:19`) | `AttachPoint`, `SamplingPlugin`, `ProposalPlugin`, `ReusePlugin`, `BsdfProposal`, `EnvImportanceProposal`, `NeuralProposal`, `IdentityReuse`, `PROPOSAL_PLUGINS`, `REUSE_PLUGINS`, `parse_proposals`, `parse_reuse`, `proposal_mask_and_alpha` |
| `skinny.sampling.path_records` | neural training-record (`.nrec`) format — `RECORD_DTYPE`, `RECORD_STRIDE`, `pack_header`, `read_records` (shared with the offline `spline_flow` trainer) |
| `skinny.sampling.training_backends` | pluggable online training-compute backends — `TrainingBackend`, `NumpyTrainingBackend`, `TorchTrainingBackend`, `MlxTrainingBackend`, `make_training_backend`, `TRAINING_BACKENDS`, `build_dataset_np` |
| `skinny.gfx` (`__init__.py:76`) | backend abstraction — `Backend`, `Device`, `Buffer`, `ComputePipeline`, `DescriptorLayout`, `Format`, `Extent2D/3D`, … |
| `skinny.gfx.vulkan` (`__init__.py:22`) | `VulkanBackend`, `VulkanDevice`, `VulkanBuffer`, `VulkanImage`, `VulkanCommandList`, `VulkanQueue`, `VulkanFence`, `VulkanSemaphore`, `VulkanSampler`, `VulkanShaderModule`, `VulkanPresenter` |
| `skinny.gfx.metal` (`__init__.py:43`) | `MetalBackend` (stub; MoltenVK still uses the Vulkan backend) |
| `skinny.slangpile` (`__init__.py:48`) | Python→Slang transpiler DSL — `shader`, `struct`, `compile_module`, `build_module`, `load_module`, scalar types |

The **scene-sampling seam** (`skinny.sampling`) is where ReSTIR / neural
proposal & reuse plug in; see the proposal-mixture discussion in
[Wavefront.md § Proposal / scene-sampling seam](Wavefront.md#7-proposal--scene-sampling-seam).

---

## 9. Quick reference — common tasks

| Task | Call |
|------|------|
| Render a USD file to a NumPy array | `skinny.headless.render_to_array(path, width=W, height=H)` |
| Render to PNG/EXR | `skinny.headless.render_scene(path, "out.png")` |
| A/B two integrators, one GPU context | `with HeadlessRenderer(W,H) as r: r.render_to_array(s, integrator=...)` |
| Render an animation sequence | `r.render_animation(path, "frames/", frames=(0,48))` |
| Custom frame loop | `VulkanContext(window=None)` → `Renderer(...)` → `update(dt)`×N → `render_headless()` |
| Isolate IBL (no direct lights) | `direct_light=False` / `renderer.direct_light_index = 1` |
| Linear-HDR pixels (not tonemapped) | read the accumulation buffer / `save_screenshot(..., "exr")` |
| Apply a skin preset | `presets.apply_preset(renderer, presets.PRESETS[i])` |
| Set any parameter generically | `params._set_nested(renderer, "mtlx.skin_bsdf_roughness", 0.4)` |

See `tests/test_headless.py` (`TestMaterialXGraphDemoRender`) for a complete
headless USD render example.
