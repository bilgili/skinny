# Headless Render Entry Point — Design

**Date:** 2026-05-26
**Status:** Approved (design); pending implementation plan

## Summary

Promote offscreen rendering from ad-hoc example scripts into a supported
feature: a reusable Python API (`skinny.headless`) plus a `skinny-render`
console command. The API renders a USD scene — supplied either as a file path
or as a **live `Usd.Stage` the caller holds and mutates** — to an image or a
numpy array, with progressive accumulation. It supports rendering at a specific
USD time and an auto-driven animation loop over a stage's timecodes.

Scope is **USD scenes only**. The analytic skin head, head-OBJ meshes, and
skin presets/params are explicitly out of scope (the renderer builds its
compute pipeline lazily from a loaded scene's materials, so a scene must be
loaded; the no-scene skin-head path and async OBJ bake are not addressed here).

## Goals

- A reusable, importable API for offscreen rendering — no window, no GLFW.
- Accept a USD source uniformly as `Path | str | Usd.Stage`.
- Let the caller own the render loop: mutate their stage between frames and
  re-render, with the GPU context (and the one-time pipeline compile) held
  across calls.
- Save to PNG/JPEG/BMP (tonemapped LDR) or EXR/HDR (linear), or return raw
  RGBA pixels.
- Render at an explicit USD time, and provide a convenience animation loop
  driven by the stage's start/end timecodes.
- A `skinny-render` CLI for file-driven use, registered alongside
  `skinny`/`skinny-gui`/`skinny-web`.

## Non-Goals

- Rendering without a USD scene (analytic skin head). Out of scope.
- Head-OBJ (`--model`) loading and its async-bake teardown race. Out of scope.
- Skin presets / parameter overrides from the CLI/API. Out of scope.
- Video/encoded output (MP4/H264). Out of scope — frame images only; the
  existing `video_encoder` path is unaffected.
- Live USD time *playback*; `fps` is sequence-pacing metadata only.

## Architecture

Three components plus one small change to the renderer.

### 1. `src/skinny/usd_loader.py` — refactor + new public function

- **Refactor:** extract the body of `_read_usd_stage` (everything after
  `Usd.Stage.Open(...)`) into `_read_open_stage(stage, *, time,
  use_usd_mtlx_plugin, keep_stage)`. `_read_usd_stage(path, ...)` opens the
  stage then delegates — existing behaviour and callers unchanged.
- **New:** `load_scene_from_stage(stage, *, time=None,
  use_usd_mtlx_plugin=False) -> Scene` — mirror of `load_scene_from_usd` but
  takes an already-open `Usd.Stage`. Reads metadata via `_read_open_stage`,
  bakes meshes (blocking, same `ThreadPoolExecutor` path), returns a fully
  populated `Scene`.
- **Known limitation:** relative `.mtlx` asset resolution uses
  `stage.GetRootLayer().realPath` for the base directory. For in-memory /
  anonymous stages that is empty; the loader falls back to the current working
  directory and logs a warning. Document this; do not block the feature on it.

### 2. `src/skinny/headless.py` — new module (reusable API)

No Vulkan/GPU code of its own; pure orchestration over `Renderer`.

```python
class HeadlessRenderer:
    def __init__(self, width: int, height: int, *, gpu: str | None = None): ...
    def render_scene(self, source, output, *, samples=64, time=None,
                     **opts) -> None: ...
    def render_to_array(self, source, *, samples=64, time=None,
                        **opts) -> "np.ndarray": ...   # (H, W, 4) uint8
    def render_animation(self, source, outdir, *, samples=64,
                         frames=None, fps=None, **opts) -> list[Path]: ...
    def cleanup(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *exc): ...   # → cleanup() + ctx.destroy()

# Module-level convenience wrappers that build a HeadlessRenderer, do one
# job, and tear down:
def render_scene(source, output, *, width=1024, height=1024, **kw) -> None: ...
def render_to_array(source, *, width=1024, height=1024, **kw) -> "np.ndarray": ...
def render_animation(source, outdir, *, width=1024, height=1024, **kw) -> list[Path]: ...
```

- `source` is `Path | str | Usd.Stage`. Path/str → `load_scene_from_usd(path,
  time=T)`; `Usd.Stage` → `load_scene_from_stage(stage, time=T)`.
- `HeadlessRenderer` builds `VulkanContext(window=None, width, height)` +
  `Renderer(...)` once and reuses them across calls, so the one-time Slang →
  SPIR-V + driver pipeline compile (tens of seconds) is paid once per session.
- Per render call: resolve source → `Scene` at `time` → `renderer.set_usd_scene(scene)`
  → loop `samples`× `renderer.update(dt)/render_headless()` → either
  `renderer.save_screenshot(output, fmt)` or read back the array.

### 3. `Renderer.set_usd_scene(scene)` — new public method

The one change to existing renderer code. Synchronous scene swap composing the
already-proven finalize sequence from `_poll_usd_streaming` (renderer.py:2977+):

1. `self._usd_scene = scene`
2. `self._gen_scene_materials()` — already guarded to rebuild the pipeline only
   when the graph-set signature changes (renderer.py:1808), so repeated calls
   for the same stage skip the rebuild.
3. `self._apply_usd_lights(scene)`
4. `self._frame_camera_to_scene(scene)` — first call only (so a caller-driven
   loop doesn't re-frame every frame); subsequent calls keep the camera unless
   the scene authored a camera, in which case the authored camera wins.
5. `self._upload_usd_scene()`

Animation correctness: moving a prim changes its transform (re-read, cheap) but
not its mesh points, so the content-hash BVH cache hits and no re-bake occurs;
deforming meshes change points and re-bake. Transforms, cameras, and lights are
re-evaluated every call.

### 4. `skinny-render` console entry

Registered in `pyproject.toml` `[project.scripts]`. Thin `argparse` wrapper
(`skinny.headless:main` or a dedicated `_cli` function) over the module API for
file-driven use. No new rendering logic.

## Render Options

Shared by all entry points; defaults match the live renderer.

| Option | API kwarg | CLI flag | Default | Maps to |
|---|---|---|---|---|
| Samples (accumulation frames) | `samples` | `--samples` | 64 | accumulation loop count |
| Resolution | ctor `width`, `height` | `--width`, `--height` | 1024×1024 | `VulkanContext` size |
| Integrator | `integrator` | `--integrator` | `"path"` | `renderer.integrator_index` (path=0, bdpt=1) |
| Exposure (EV stops) | `exposure` | `--exposure` | `0.0` | `renderer.exposure` |
| Tonemap | `tonemap` | `--tonemap` | `"aces"` | `renderer.tonemap_index` (aces/reinhard/hable/linear = 0..3) |
| Env intensity | `env_intensity` | `--env-intensity` | scene/renderer default (unset = leave as-is) | `renderer.env_intensity` |
| Direct light | `direct_light` | `--no-direct` | `True` (on) | `renderer.direct_light_index` (On=0, Off=1) |
| Output format | inferred from `output` suffix | `--format` | from extension | `save_screenshot` fmt |

Unknown enum values (`integrator`, `tonemap`, `format`) raise `ValueError`
listing the valid choices.

## Time & Frame-Range Semantics

- `time=None` → `Usd.TimeCode.Default()`. `time` may be `int`, `float`, or
  `Usd.TimeCode`; ints/floats are wrapped in `Usd.TimeCode`.
- `render_animation` frame range:
  - Default: `(stage.GetStartTimeCode(), stage.GetEndTimeCode())`, step 1.
  - `frames=(start, end)` or `(start, end, step)` overrides the range.
  - `fps` overrides the stage's `timeCodesPerSecond`; used only as
    sequence-pacing metadata (e.g. for a later encode), not for playback here.
  - Output files: `outdir/frame_{NNNN}.{ext}`, zero-padded to the frame count;
    returns the list of written paths.
- CLI animation: `--animate` selects animation mode; `--frames START:END[:STEP]`
  and `--fps` parse to the above; `--time T` sets a single-frame timecode.

## Error Handling

- Missing file / unopenable stage → `FileNotFoundError` (API) / stderr + exit 1
  (CLI).
- Empty scene (0 instances after load) → warn on stderr, still render
  (environment only); do not crash.
- Invalid `integrator`/`tonemap`/`format` enum → `ValueError` listing valid
  options.
- `render_to_array` when the pipeline failed to build (e.g. genuinely empty /
  material-less stage) → raise rather than silently return a black frame, so
  misuse is loud. The file-writing path remains lenient (matches
  `save_screenshot`).
- Cleanup guaranteed via the context manager / `try/finally`; no async bake
  threads are involved, avoiding the teardown races seen with the OBJ path.

## Testing

`tests/test_headless_api.py`, GPU tests marked `@pytest.mark.gpu`; the loader
test needs no GPU.

- **Loader (no GPU):** `load_scene_from_stage(Usd.Stage.Open(path))` yields the
  same instance/material counts as `load_scene_from_usd(path)` for
  `cornell_box_sphere.usda`.
- `render_to_array(path, samples=…)` → shape `(H, W, 4)`, dtype uint8,
  non-black for `cornell_box_sphere.usda`.
- **Stage passing:** open a stage, mutate a prim transform, render two frames →
  the two arrays differ.
- **Time swap:** rendering an animated stage at two different `time=` values
  yields different output.
- **Format round-trip:** PNG and EXR outputs are valid files (PNG magic
  `\x89PNG`, EXR magic `\x76\x2f\x31\x01`).
- **Error paths:** bad `format`/`integrator` → `ValueError`; missing file →
  `FileNotFoundError`.

## Migration of Existing Examples

The two scripts added earlier (`examples/render_image.py`,
`examples/render_turntable.py`) are rewritten as thin wrappers over the new API
(a few lines each) or retired, so there is one supported offscreen-render path
rather than two parallel implementations. `examples/README.md` is updated to
point at `skinny-render` and `skinny.headless`.

## Risks / Open Items

- **Per-frame reload cost** for animation: each frame re-reads the stage and
  re-bakes changed meshes. Static geometry hits the BVH cache; deforming
  geometry re-bakes (expected). Acceptable for an offline tool.
- **Anonymous-stage `.mtlx` resolution** (see loader limitation). Falls back to
  cwd + warning.
- **`set_usd_scene` re-entrancy:** repeated calls must not leak GPU resources or
  trigger needless pipeline rebuilds. The existing graph-signature guard covers
  the common case (same stage topology); verify no buffer growth leaks across
  many calls during implementation.
