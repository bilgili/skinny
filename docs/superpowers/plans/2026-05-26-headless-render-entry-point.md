# Headless Render Entry Point Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable `skinny.headless` API and a `skinny-render` CLI for offscreen USD rendering (file path or a live `Usd.Stage` the caller mutates per frame), saving images or returning numpy arrays.

**Architecture:** A new `skinny/headless.py` orchestrates the existing `Renderer` (no new GPU code). A small `usd_loader` refactor lets the loader read an already-open `Usd.Stage`. One new public `Renderer.set_usd_scene(scene)` synchronously swaps the active scene by composing the renderer's proven streaming-finalize calls. The caller owns the render loop; the GPU context (and one-time pipeline compile) is held across calls.

**Tech Stack:** Python 3.11+, OpenUSD (`pxr`), Vulkan (via `skinny.vk_context`/`vk_compute`), Pillow (image writing, already used by `save_screenshot`), pytest.

---

## Environment & conventions (read first)

- **GPU tests** require the fully-built env: run with the repo-root Python 3.13 venv and the Vulkan SDK on the dylib path:
  ```bash
  export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
  export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
  ./bin/python3.13 -m pytest tests/test_headless_api.py -v
  ```
  Non-GPU tests run under either interpreter. GPU tests are marked `@pytest.mark.gpu`.
- **`.gitignore` is a blanket `*`.** Existing tracked files commit normally, but every **new** file must be staged with `git add -f`. Commit steps below use `-f` for new files.
- Render options map to existing `Renderer` fields: `integrator_index` (path=0, bdpt=1), `tonemap_index` (aces/reinhard/hable/linear = 0..3), `exposure` (float EV), `env_intensity` (float), `direct_light_index` (On=0, Off=1).
- `Renderer.render_headless()` returns tonemapped sRGB RGBA8 bytes (`width*height*4`). `Renderer.save_screenshot(path, fmt)` writes png/jpeg/bmp (LDR) or exr/hdr (linear from the accumulation buffer). `Renderer.screenshot_format_options()` → `["PNG","JPEG","BMP","EXR","HDR"]`.
- Reference design spec: `docs/superpowers/specs/2026-05-26-headless-render-entry-point-design.md`.

## File structure

- **Modify** `src/skinny/usd_loader.py` — extract `_read_open_stage`, add `load_scene_from_stage`.
- **Modify** `src/skinny/renderer.py` — add `set_usd_scene`.
- **Create** `src/skinny/headless.py` — `HeadlessRenderer` + module-level convenience functions + `main()` CLI.
- **Modify** `pyproject.toml` — register `skinny-render` console script.
- **Create** `tests/test_headless_api.py` — loader + API + CLI-parsing tests.
- **Modify** `examples/render_image.py`, `examples/render_turntable.py`, `examples/README.md` — thin wrappers over the new API.

---

## Task 1: Loader — read an already-open stage

**Files:**
- Modify: `src/skinny/usd_loader.py` (refactor `_read_usd_stage` at lines 1356–1427; add `_read_open_stage` and `load_scene_from_stage`)
- Test: `tests/test_headless_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_headless_api.py`:

```python
"""Tests for the headless render API (skinny.headless) + loader stage support."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE = PROJECT_ROOT / "assets" / "cornell_box_sphere.usda"


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(not _have_usd(), reason="OpenUSD (pxr) not installed")


@needs_usd
class TestLoadSceneFromStage:
    def test_stage_matches_path(self):
        from pxr import Usd
        from skinny.usd_loader import load_scene_from_stage, load_scene_from_usd

        by_path = load_scene_from_usd(SCENE)
        stage = Usd.Stage.Open(str(SCENE))
        by_stage = load_scene_from_stage(stage)

        assert len(by_stage.instances) == len(by_path.instances)
        assert len(by_stage.materials) == len(by_path.materials)
        assert len(by_stage.lights_dir) == len(by_path.lights_dir)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./bin/python3.13 -m pytest tests/test_headless_api.py::TestLoadSceneFromStage -v`
Expected: FAIL with `ImportError: cannot import name 'load_scene_from_stage'`.

- [ ] **Step 3: Refactor `_read_usd_stage` to delegate to a new `_read_open_stage`**

In `src/skinny/usd_loader.py`, replace the current `_read_usd_stage` function (lines 1356–1427) with a thin path-opening wrapper plus the extracted body. The extracted body is the existing code verbatim except: (a) it takes `stage` instead of opening it, (b) the "no usable mesh" error message uses a `source_label`, (c) the mtlx base dir falls back to cwd for anonymous stages.

```python
def _read_open_stage(
    stage: "Usd.Stage",
    *,
    time: Optional[Usd.TimeCode] = None,
    use_usd_mtlx_plugin: bool = False,
    keep_stage: bool = False,
    source_label: Optional[str] = None,
) -> tuple[Scene, list[tuple[MeshSource, np.ndarray, int]], Optional["Usd.Stage"]]:
    """Serial read of an already-open USD stage. See `_read_usd_stage`."""
    label = source_label or (stage.GetRootLayer().identifier or "<anonymous stage>")
    eval_time = time if time is not None else Usd.TimeCode.Default()

    mtlx_materials: dict[str, Material] = {}
    if not use_usd_mtlx_plugin:
        real = stage.GetRootLayer().realPath
        stage_dir = Path(real).parent if real else Path.cwd()
        mtlx_materials = _load_mtlx_materials(stage, stage_dir)

    materials: list[Material] = [Material(name="default")]
    material_index: dict[str, int] = {}

    prim_data: list[tuple[MeshSource, np.ndarray, int]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not prim.IsActive() or prim.IsAbstract():
            continue

        source = _read_mesh_attrs(prim, eval_time)
        if source is None:
            continue

        source.content_hash = compute_source_hash(source)
        transform = _world_transform(prim, eval_time)
        material_id = _resolve_material_binding(
            prim, materials, material_index, mtlx_materials,
        )
        prim_data.append((source, transform, material_id))

    if not prim_data:
        raise ValueError(
            f"USD stage {label} contains no usable UsdGeom.Mesh prims"
        )

    lights_dir, lights_sphere, environment, emissive_instances = _extract_lights(
        stage, eval_time, materials, material_index,
    )
    camera_override = _extract_camera(stage, eval_time)

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    mm_per_unit = max(meters_per_unit * 1000.0, 1e-6)

    partial_scene = Scene(
        instances=list(emissive_instances),
        materials=materials,
        lights_dir=lights_dir,
        lights_sphere=lights_sphere,
        environment=environment,
        camera_override=camera_override,
        mm_per_unit=mm_per_unit,
    )
    return partial_scene, prim_data, (stage if keep_stage else None)


def _read_usd_stage(
    stage_path: Path,
    *,
    time: Optional[Usd.TimeCode] = None,
    use_usd_mtlx_plugin: bool = False,
    keep_stage: bool = False,
) -> tuple[Scene, list[tuple[MeshSource, np.ndarray, int]], Optional["Usd.Stage"]]:
    """Open a USD stage from disk, then read it. See `_read_open_stage`."""
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise FileNotFoundError(f"could not open USD stage: {stage_path}")
    return _read_open_stage(
        stage,
        time=time,
        use_usd_mtlx_plugin=use_usd_mtlx_plugin,
        keep_stage=keep_stage,
        source_label=str(stage_path),
    )
```

- [ ] **Step 4: Add `load_scene_from_stage` next to `load_scene_from_usd`**

Insert after `load_scene_from_usd` (currently ends at line 1483):

```python
def load_scene_from_stage(
    stage: "Usd.Stage",
    *,
    time: Optional[Usd.TimeCode] = None,
    use_usd_mtlx_plugin: bool = False,
) -> Scene:
    """Read an already-open USD stage and return a fully-baked `Scene`.

    Same as `load_scene_from_usd` but takes a `Usd.Stage` the caller owns
    (e.g. one they mutate between frames). Blocking; bakes meshes in parallel.
    """
    scene, prim_data, _ = _read_open_stage(
        stage, time=time, use_usd_mtlx_plugin=use_usd_mtlx_plugin,
    )
    cache_index = load_cache_index()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=_USD_POOL_SIZE) as pool:
        instances = list(pool.map(
            lambda pd: bake_usd_prim(pd[0], pd[1], pd[2], cache_index),
            prim_data,
        ))

    scene.instances.extend(instances)
    return scene
```

- [ ] **Step 5: Run test to verify it passes**

Run: `./bin/python3.13 -m pytest tests/test_headless_api.py::TestLoadSceneFromStage -v`
Expected: PASS.

- [ ] **Step 6: Verify existing loader/headless tests still pass (no regression)**

Run: `./bin/python3.13 -m pytest tests/test_headless.py -v`
Expected: PASS (or pre-existing skips), no new failures.

- [ ] **Step 7: Commit**

```bash
git add -f tests/test_headless_api.py
git add src/skinny/usd_loader.py
git commit -m "feat(usd): read an already-open Usd.Stage (load_scene_from_stage)"
```

---

## Task 2: `Renderer.set_usd_scene` — synchronous scene swap

**Files:**
- Modify: `src/skinny/renderer.py` (add method near the other USD methods, e.g. after `_upload_usd_scene` which ends around line 3092)
- Test: `tests/test_headless_api.py`

Context the implementer needs (already verified in the codebase):
- `_is_usd_active()` (renderer.py:3048) returns `self._usd_model_index >= 0 and self.model_index == self._usd_model_index`. The render path only treats the USD scene as active when this is true.
- `_apply_usd_lights(scene)` (renderer.py:1814) is "run once" — it *appends* an `Environment` and reseeds light sliders. Calling it per frame would grow `self.environments` unbounded and clobber options. Call it only on the first scene.
- `_upload_usd_scene()` (renderer.py:3062) uploads instances + materials + sphere/distant/emissive lights from `self._usd_scene` every call — this is what makes animated transforms/lights/deforming meshes update.
- `_frame_camera_to_scene(scene)` (renderer.py:1222) applies the scene's `camera_override` (or frames the orbit camera if none).
- In headless use `self._usd_bake_done` stays `None`, so `_poll_usd_streaming` is a no-op and won't interfere.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_headless_api.py`:

```python
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"


def _have_vulkan() -> bool:
    try:
        import vulkan  # noqa: F401
        return True
    except Exception:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")


@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestSetUsdScene:
    def test_set_usd_scene_renders_nonblack(self):
        from skinny.renderer import Renderer
        from skinny.usd_loader import load_scene_from_usd
        from skinny.vk_context import VulkanContext

        ctx = VulkanContext(window=None, width=128, height=128)
        renderer = Renderer(
            vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
            tattoo_dir=TATTOO_DIR,
        )
        try:
            scene = load_scene_from_usd(SCENE)
            renderer.set_usd_scene(scene)
            assert renderer._is_usd_active()
            raw = b""
            for _ in range(8):
                renderer.update(0.016)
                raw = renderer.render_headless()
            assert any(b != 0 for b in raw), "USD scene should not render all-black"
        finally:
            renderer.cleanup()
            ctx.destroy()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./bin/python3.13 -m pytest tests/test_headless_api.py::TestSetUsdScene -v`
Expected: FAIL with `AttributeError: 'Renderer' object has no attribute 'set_usd_scene'`.

- [ ] **Step 3: Implement `set_usd_scene`**

Add to `src/skinny/renderer.py` after `_upload_usd_scene`:

```python
    def set_usd_scene(self, scene: "Scene") -> None:
        """Make `scene` the active USD scene synchronously and upload it.

        Composes the same finalize steps the async streaming path runs, but
        blocking and re-callable. Safe to call every frame with a freshly
        loaded scene (e.g. from a caller-mutated Usd.Stage): geometry,
        materials, and lights are re-uploaded each call; light/env sliders
        and orbit framing are seeded once. An authored (possibly animated)
        camera is re-applied every call.

        Used by the headless render API; not part of the live UI path.
        """
        first = self._usd_scene is None

        # Enter the USD-active state so update()/render treat this scene as
        # the subject instead of the default analytic head.
        if self._usd_model_index < 0:
            self.models.append("USD: (headless)")
            self._usd_model_index = len(self.models) - 1
        self.model_index = self._usd_model_index

        self._usd_scene = scene
        self._gen_scene_materials()           # guarded: rebuilds pipeline only on graph-set change
        if first:
            self._apply_usd_lights(scene)     # once: appends env + seeds sliders
            self._frame_camera_to_scene(scene)
        elif scene.camera_override is not None:
            self._frame_camera_to_scene(scene)  # animated authored camera
        self._upload_usd_scene()              # every call: geometry + materials + lights
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./bin/python3.13 -m pytest tests/test_headless_api.py::TestSetUsdScene -v`
Expected: PASS (first run compiles the pipeline; may take ~30 s).

- [ ] **Step 5: Commit**

```bash
git add src/skinny/renderer.py
git commit -m "feat(renderer): set_usd_scene() for synchronous headless scene swap"
```

---

## Task 3: `skinny.headless` — options, source resolution, core render

**Files:**
- Create: `src/skinny/headless.py`
- Test: `tests/test_headless_api.py`

- [ ] **Step 1: Write the failing tests (options + error paths, no GPU needed)**

Append to `tests/test_headless_api.py`:

```python
class TestRenderOptions:
    def test_resolve_defaults(self):
        from skinny.headless import RenderOptions
        opts = RenderOptions()
        assert opts.samples == 64
        assert opts.integrator_index == 0   # path
        assert opts.tonemap_index == 0       # aces

    def test_integrator_bdpt(self):
        from skinny.headless import RenderOptions
        assert RenderOptions(integrator="bdpt").integrator_index == 1

    def test_tonemap_hable(self):
        from skinny.headless import RenderOptions
        assert RenderOptions(tonemap="hable").tonemap_index == 2

    def test_bad_integrator_raises(self):
        from skinny.headless import RenderOptions
        with pytest.raises(ValueError, match="integrator"):
            RenderOptions(integrator="nope")

    def test_bad_tonemap_raises(self):
        from skinny.headless import RenderOptions
        with pytest.raises(ValueError, match="tonemap"):
            RenderOptions(tonemap="nope")


@needs_usd
def test_fmt_for_output():
    from skinny.headless import _fmt_for_output
    assert _fmt_for_output(Path("a.png"), None) == "png"
    assert _fmt_for_output(Path("a.JPG"), None) == "jpeg"
    assert _fmt_for_output(Path("a.png"), "exr") == "exr"
    with pytest.raises(ValueError, match="format"):
        _fmt_for_output(Path("a.gif"), None)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./bin/python3.13 -m pytest tests/test_headless_api.py::TestRenderOptions -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'skinny.headless'`.

- [ ] **Step 3: Create `src/skinny/headless.py` with options + helpers + core renderer**

```python
"""Offscreen (headless) rendering API.

Drives the renderer with no window and saves images or returns pixel arrays.
Accepts a USD source as a file path or an already-open `Usd.Stage` the caller
mutates between frames. The GPU context (and one-time pipeline compile) is held
across calls via `HeadlessRenderer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from pxr import Usd

# Source may be a file path or an open Usd.Stage. `object` covers the stage to
# avoid importing pxr at module load (it is optional at import time).
Source = Union[str, Path, "Usd.Stage", object]

_INTEGRATORS = {"path": 0, "bdpt": 1}
_TONEMAPS = {"aces": 0, "reinhard": 1, "hable": 2, "linear": 3}
_LDR_FORMATS = {"png", "jpeg", "bmp"}
_HDR_FORMATS = {"exr", "hdr"}


@dataclass
class RenderOptions:
    """Per-render knobs, resolved to renderer indices on construction."""

    samples: int = 64
    integrator: str = "path"
    exposure: float = 0.0
    tonemap: str = "aces"
    env_intensity: Optional[float] = None
    direct_light: bool = True
    time: object = None  # None | int | float | Usd.TimeCode

    integrator_index: int = field(init=False)
    tonemap_index: int = field(init=False)

    def __post_init__(self) -> None:
        if self.integrator not in _INTEGRATORS:
            raise ValueError(
                f"unknown integrator {self.integrator!r}; "
                f"choose from {sorted(_INTEGRATORS)}"
            )
        if self.tonemap not in _TONEMAPS:
            raise ValueError(
                f"unknown tonemap {self.tonemap!r}; "
                f"choose from {sorted(_TONEMAPS)}"
            )
        self.integrator_index = _INTEGRATORS[self.integrator]
        self.tonemap_index = _TONEMAPS[self.tonemap]


def _fmt_for_output(output: Path, override: Optional[str]) -> str:
    fmt = (override or output.suffix.lstrip(".")).lower()
    if fmt in ("jpg", "jpeg"):
        fmt = "jpeg"
    if fmt not in _LDR_FORMATS and fmt not in _HDR_FORMATS:
        raise ValueError(
            f"unsupported output format {fmt!r}; "
            f"choose from {sorted(_LDR_FORMATS | _HDR_FORMATS)}"
        )
    return fmt


def _to_timecode(time: object):
    from pxr import Usd
    if time is None:
        return Usd.TimeCode.Default()
    if isinstance(time, Usd.TimeCode):
        return time
    return Usd.TimeCode(float(time))


def _load_scene(source: Source, time: object):
    """Resolve a path-or-stage source to a baked Scene at `time`."""
    from pxr import Usd
    from skinny import usd_loader

    tc = _to_timecode(time)
    if isinstance(source, Usd.Stage):
        return usd_loader.load_scene_from_stage(source, time=tc)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"scene not found: {path}")
    return usd_loader.load_scene_from_usd(path, time=tc)


class HeadlessRenderer:
    """Windowless renderer that persists across calls.

    Use as a context manager so the Vulkan context is always torn down:

        with HeadlessRenderer(1920, 1080) as r:
            for i in range(120):
                mutate(stage)
                r.render_scene(stage, f"out/{i:04d}.png", samples=64)
    """

    def __init__(self, width: int, height: int, *, gpu: Optional[str] = None) -> None:
        from skinny.renderer import Renderer
        from skinny.vk_context import VulkanContext

        self.ctx = VulkanContext(window=None, width=width, height=height)
        self.renderer = Renderer(
            vk_ctx=self.ctx,
            shader_dir=Path(__import__("skinny").__file__).resolve().parent / "shaders",
            hdr_dir=_repo_root() / "hdrs",
            tattoo_dir=_repo_root() / "tattoos",
        )

    def __enter__(self) -> "HeadlessRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        self.renderer.cleanup()
        self.ctx.destroy()

    def _prepare(self, source: Source, opts: RenderOptions) -> None:
        scene = _load_scene(source, opts.time)
        self.renderer.set_usd_scene(scene)
        # Apply post-process / integrator options AFTER the scene swap so they
        # win over anything _apply_usd_lights seeded on the first scene.
        self.renderer.integrator_index = opts.integrator_index
        self.renderer.tonemap_index = opts.tonemap_index
        self.renderer.exposure = float(opts.exposure)
        self.renderer.direct_light_index = 0 if opts.direct_light else 1
        if opts.env_intensity is not None:
            self.renderer.env_intensity = float(opts.env_intensity)

    def _accumulate(self, samples: int) -> bytes:
        raw = b""
        for _ in range(max(1, samples)):
            self.renderer.update(1.0 / 60.0)
            raw = self.renderer.render_headless()
        return raw

    def render_to_array(self, source: Source, *, samples: int = 64,
                        time: object = None, **opts) -> np.ndarray:
        ro = RenderOptions(samples=samples, time=time, **opts)
        self._prepare(source, ro)
        if self.renderer.pipeline is None:
            raise RuntimeError(
                "render pipeline failed to build — scene has no usable materials"
            )
        raw = self._accumulate(ro.samples)
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            self.ctx.height, self.ctx.width, 4,
        ).copy()

    def render_scene(self, source: Source, output, *, samples: int = 64,
                     time: object = None, format: Optional[str] = None,
                     **opts) -> None:
        out = Path(output)
        fmt = _fmt_for_output(out, format)
        ro = RenderOptions(samples=samples, time=time, **opts)
        self._prepare(source, ro)
        if self.renderer.pipeline is None:
            raise RuntimeError(
                "render pipeline failed to build — scene has no usable materials"
            )
        self._accumulate(ro.samples)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.renderer.save_screenshot(str(out), fmt)


def _repo_root() -> Path:
    # src/skinny/headless.py → repo root is three parents up (… /src/skinny/).
    return Path(__file__).resolve().parent.parent.parent


# ── Module-level one-shot convenience wrappers ──────────────────────────

def render_to_array(source: Source, *, width: int = 1024, height: int = 1024,
                    **kw) -> np.ndarray:
    with HeadlessRenderer(width, height) as r:
        return r.render_to_array(source, **kw)


def render_scene(source: Source, output, *, width: int = 1024,
                 height: int = 1024, **kw) -> None:
    with HeadlessRenderer(width, height) as r:
        r.render_scene(source, output, **kw)
```

- [ ] **Step 4: Run the option/helper tests to verify they pass**

Run: `./bin/python3.13 -m pytest tests/test_headless_api.py::TestRenderOptions tests/test_headless_api.py::test_fmt_for_output -v`
Expected: PASS.

- [ ] **Step 5: Write the failing GPU render tests**

Append to `tests/test_headless_api.py`:

```python
@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestHeadlessRender:
    def test_render_to_array_shape_nonblack(self):
        from skinny.headless import HeadlessRenderer
        with HeadlessRenderer(128, 128) as r:
            arr = r.render_to_array(SCENE, samples=8)
        assert arr.shape == (128, 128, 4)
        assert arr.dtype.name == "uint8"
        assert int(arr[..., :3].max()) > 0

    def test_render_scene_writes_png(self, tmp_path):
        from skinny.headless import HeadlessRenderer
        out = tmp_path / "frame.png"
        with HeadlessRenderer(128, 128) as r:
            r.render_scene(SCENE, out, samples=8)
        assert out.exists()
        assert out.read_bytes()[:4] == b"\x89PNG"

    def test_render_scene_writes_exr(self, tmp_path):
        from skinny.headless import HeadlessRenderer
        out = tmp_path / "frame.exr"
        with HeadlessRenderer(96, 96) as r:
            r.render_scene(SCENE, out, samples=4)
        assert out.exists()
        assert out.read_bytes()[:4] == b"\x76\x2f\x31\x01"  # OpenEXR magic

    def test_stage_mutation_changes_output(self):
        from pxr import Gf, Usd, UsdGeom
        from skinny.headless import HeadlessRenderer
        stage = Usd.Stage.Open(str(SCENE))
        sphere = UsdGeom.Xformable(
            stage.GetPrimAtPath("/Cornell/GlassSphere/Sphere")
        )
        op = sphere.AddTranslateOp()
        with HeadlessRenderer(96, 96) as r:
            op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
            a = r.render_to_array(stage, samples=8)
            op.Set(Gf.Vec3d(0.4, 0.0, 0.0))
            b = r.render_to_array(stage, samples=8)
        assert np.abs(a.astype(int) - b.astype(int)).mean() > 1.0
```

- [ ] **Step 6: Run the GPU render tests**

Run: `./bin/python3.13 -m pytest tests/test_headless_api.py::TestHeadlessRender -v`
Expected: PASS. (If `/Cornell/GlassSphere/Sphere` is not the correct prim path for the bundled scene, run `./bin/python3.13 -c "from pxr import Usd; [print(p.GetPath()) for p in Usd.Stage.Open('assets/cornell_box_sphere.usda').Traverse()]"` and use an existing Xformable mesh prim path instead.)

- [ ] **Step 7: Commit**

```bash
git add -f src/skinny/headless.py
git add tests/test_headless_api.py
git commit -m "feat(headless): skinny.headless API (render_scene / render_to_array)"
```

---

## Task 4: Animation loop + frame-range parsing

**Files:**
- Modify: `src/skinny/headless.py` (add `_parse_frames`, `render_animation` method + module wrapper)
- Test: `tests/test_headless_api.py`

- [ ] **Step 1: Write the failing test (frame parsing, no GPU)**

Append to `tests/test_headless_api.py`:

```python
class TestFrameRange:
    def test_parse_start_end(self):
        from skinny.headless import _parse_frames
        assert _parse_frames("1:10") == (1.0, 10.0, 1.0)

    def test_parse_start_end_step(self):
        from skinny.headless import _parse_frames
        assert _parse_frames("0:48:2") == (0.0, 48.0, 2.0)

    def test_frame_times_inclusive(self):
        from skinny.headless import _frame_times
        assert _frame_times((1.0, 4.0, 1.0)) == [1.0, 2.0, 3.0, 4.0]

    def test_frame_times_step(self):
        from skinny.headless import _frame_times
        assert _frame_times((0.0, 10.0, 5.0)) == [0.0, 5.0, 10.0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `./bin/python3.13 -m pytest tests/test_headless_api.py::TestFrameRange -v`
Expected: FAIL with `ImportError: cannot import name '_parse_frames'`.

- [ ] **Step 3: Add frame helpers + `render_animation`**

Add to `src/skinny/headless.py` (helpers near `_fmt_for_output`; method on `HeadlessRenderer`; wrapper at module level):

```python
def _parse_frames(spec: str) -> tuple[float, float, float]:
    """Parse 'START:END' or 'START:END:STEP' into (start, end, step)."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"invalid --frames {spec!r}; expected START:END[:STEP]")
    start, end = float(parts[0]), float(parts[1])
    step = float(parts[2]) if len(parts) == 3 else 1.0
    if step <= 0:
        raise ValueError("--frames STEP must be > 0")
    return start, end, step


def _frame_times(rng: tuple[float, float, float]) -> list[float]:
    """Inclusive list of timecodes from (start, end, step)."""
    start, end, step = rng
    out: list[float] = []
    t = start
    # Guard float drift; stop once past end by less than half a step.
    while t <= end + step * 1e-6:
        out.append(round(t, 6))
        t += step
    return out
```

Add this method to `HeadlessRenderer` (after `render_scene`):

```python
    def render_animation(self, source: Source, outdir, *, samples: int = 64,
                         frames: Optional[tuple] = None, fps: Optional[float] = None,
                         ext: str = "png", **opts) -> list[Path]:
        """Render a frame sequence over a stage's timecodes.

        `frames` is (start, end[, step]); defaults to the stage's
        start/end timecode with step 1. `fps`/`ext` control naming/pacing
        metadata only. Returns the list of written paths.
        """
        from pxr import Usd
        from skinny import usd_loader

        stage = source if isinstance(source, Usd.Stage) else Usd.Stage.Open(str(source))
        if stage is None:
            raise FileNotFoundError(f"could not open USD stage: {source}")

        if frames is None:
            start = stage.GetStartTimeCode()
            end = stage.GetEndTimeCode()
            rng = (float(start), float(end), 1.0)
        elif len(frames) == 2:
            rng = (float(frames[0]), float(frames[1]), 1.0)
        else:
            rng = (float(frames[0]), float(frames[1]), float(frames[2]))

        times = _frame_times(rng)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        pad = max(4, len(str(len(times) - 1)))
        fmt = _fmt_for_output(Path("x." + ext), None)

        written: list[Path] = []
        for i, t in enumerate(times):
            out = outdir / f"frame_{i:0{pad}d}.{ext}"
            self.render_scene(stage, out, samples=samples, time=t,
                              format=fmt, **opts)
            written.append(out)
        return written
```

Add the module-level wrapper near the other wrappers:

```python
def render_animation(source: Source, outdir, *, width: int = 1024,
                     height: int = 1024, **kw) -> list[Path]:
    with HeadlessRenderer(width, height) as r:
        return r.render_animation(source, outdir, **kw)
```

- [ ] **Step 4: Run frame-parsing tests**

Run: `./bin/python3.13 -m pytest tests/test_headless_api.py::TestFrameRange -v`
Expected: PASS.

- [ ] **Step 5: Add + run a GPU time-swap test**

Append to `tests/test_headless_api.py`:

```python
@needs_vulkan
@needs_usd
@pytest.mark.gpu
class TestAnimation:
    def test_animation_writes_frames(self, tmp_path):
        from skinny.headless import HeadlessRenderer
        with HeadlessRenderer(64, 64) as r:
            paths = r.render_animation(
                SCENE, tmp_path, samples=4, frames=(0, 2, 1),
            )
        assert len(paths) == 3
        assert all(p.exists() for p in paths)
```

Run: `./bin/python3.13 -m pytest tests/test_headless_api.py::TestAnimation -v`
Expected: PASS (3 frames written; the static scene renders the same image 3×, which is fine).

- [ ] **Step 6: Commit**

```bash
git add src/skinny/headless.py tests/test_headless_api.py
git commit -m "feat(headless): render_animation over USD timecodes"
```

---

## Task 5: `skinny-render` CLI

**Files:**
- Modify: `src/skinny/headless.py` (add `main()` + `_build_parser()`)
- Modify: `pyproject.toml:56-59` (add console script)
- Test: `tests/test_headless_api.py`

- [ ] **Step 1: Write the failing test (parser only, no GPU)**

Append to `tests/test_headless_api.py`:

```python
class TestCli:
    def test_parser_single(self):
        from skinny.headless import _build_parser
        ns = _build_parser().parse_args(
            ["scene.usda", "-o", "out.png", "--samples", "32"]
        )
        assert ns.source == "scene.usda"
        assert ns.output == "out.png"
        assert ns.samples == 32
        assert not ns.animate

    def test_parser_animate(self):
        from skinny.headless import _build_parser
        ns = _build_parser().parse_args(
            ["shot.usda", "--outdir", "frames", "--animate",
             "--frames", "1:48:2", "--fps", "24"]
        )
        assert ns.animate
        assert ns.outdir == "frames"
        assert ns.frames == "1:48:2"
        assert ns.fps == 24.0

    def test_parser_render_opts(self):
        from skinny.headless import _build_parser
        ns = _build_parser().parse_args(
            ["s.usda", "-o", "o.exr", "--integrator", "bdpt",
             "--tonemap", "hable", "--exposure", "0.5", "--no-direct",
             "--width", "800", "--height", "600"]
        )
        assert ns.integrator == "bdpt"
        assert ns.tonemap == "hable"
        assert ns.exposure == 0.5
        assert ns.no_direct is True
        assert ns.width == 800 and ns.height == 600
```

- [ ] **Step 2: Run to verify it fails**

Run: `./bin/python3.13 -m pytest tests/test_headless_api.py::TestCli -v`
Expected: FAIL with `ImportError: cannot import name '_build_parser'`.

- [ ] **Step 3: Add the CLI to `src/skinny/headless.py`**

Append at the end of `src/skinny/headless.py`:

```python
import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skinny-render",
        description="Render a USD scene offscreen to an image (or a frame "
                    "sequence with --animate).",
    )
    p.add_argument("source", help="USD scene (.usd/.usda/.usdc/.usdz)")
    p.add_argument("-o", "--output", default="render.png",
                   help="output image path (single-frame mode)")
    p.add_argument("--outdir", default="frames",
                   help="output directory (animation mode)")
    p.add_argument("--animate", action="store_true",
                   help="render a frame sequence over the stage's timecodes")
    p.add_argument("--frames", default=None,
                   help="frame range START:END[:STEP] (animation mode)")
    p.add_argument("--fps", type=float, default=None,
                   help="frames per second (sequence pacing metadata)")
    p.add_argument("--time", type=float, default=None,
                   help="single-frame USD timecode")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--integrator", choices=["path", "bdpt"], default="path")
    p.add_argument("--tonemap", choices=["aces", "reinhard", "hable", "linear"],
                   default="aces")
    p.add_argument("--exposure", type=float, default=0.0)
    p.add_argument("--env-intensity", type=float, default=None, dest="env_intensity")
    p.add_argument("--no-direct", action="store_true",
                   help="disable the analytic direct light (IBL only)")
    p.add_argument("--format", default=None, dest="fmt",
                   help="override output format (png/jpeg/bmp/exr/hdr)")
    p.add_argument("--ext", default="png",
                   help="frame image extension (animation mode)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    ns = _build_parser().parse_args(argv)
    opts = dict(
        samples=ns.samples, integrator=ns.integrator, tonemap=ns.tonemap,
        exposure=ns.exposure, env_intensity=ns.env_intensity,
        direct_light=not ns.no_direct,
    )
    try:
        with HeadlessRenderer(ns.width, ns.height) as r:
            if ns.animate:
                frames = _parse_frames(ns.frames) if ns.frames else None
                paths = r.render_animation(
                    ns.source, ns.outdir, frames=frames, fps=ns.fps,
                    ext=ns.ext, **opts,
                )
                print(f"[skinny-render] wrote {len(paths)} frame(s) to {ns.outdir}/")
            else:
                r.render_scene(ns.source, ns.output, time=ns.time,
                               format=ns.fmt, **opts)
                print(f"[skinny-render] wrote {ns.output}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[skinny-render] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Register the console script in `pyproject.toml`**

Change lines 56–59 from:

```toml
[project.scripts]
skinny = "skinny.app:main"
skinny-gui = "skinny.ui.qt.app:main"
skinny-web = "skinny.web_app:main"
```

to:

```toml
[project.scripts]
skinny = "skinny.app:main"
skinny-gui = "skinny.ui.qt.app:main"
skinny-web = "skinny.web_app:main"
skinny-render = "skinny.headless:main"
```

- [ ] **Step 5: Run CLI parser tests**

Run: `./bin/python3.13 -m pytest tests/test_headless_api.py::TestCli -v`
Expected: PASS.

- [ ] **Step 6: GPU smoke of the CLI end-to-end (manual)**

Run:
```bash
export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
./bin/python3.13 -m skinny.headless assets/cornell_box_sphere.usda \
    -o /tmp/render.png --width 160 --height 160 --samples 16
```
Expected: prints `[skinny-render] wrote /tmp/render.png`; the file exists and is a non-trivial PNG. Clean up `/tmp/render.png`.

- [ ] **Step 7: Commit**

```bash
git add src/skinny/headless.py pyproject.toml tests/test_headless_api.py
git commit -m "feat(headless): skinny-render CLI"
```

---

## Task 6: Migrate examples to the API + update docs

**Files:**
- Modify: `examples/render_image.py`, `examples/render_turntable.py`, `examples/README.md`
- Modify: `README.md` (add a short "Headless / scripted rendering" subsection), `Architecture.md` (one line under the headless section), `CHANGELOG.md`

- [ ] **Step 1: Rewrite `examples/render_image.py` as a thin wrapper**

Replace its body so it calls the API (keep its argparse surface, drop the duplicated render loop):

```python
#!/usr/bin/env python3
"""Render a single offscreen image. Thin wrapper over skinny.headless."""
from __future__ import annotations
import argparse
from pathlib import Path
from skinny.headless import render_scene


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default="render.png")
    ap.add_argument("--scene", default="assets/cornell_box_sphere.usda")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--integrator", choices=["path", "bdpt"], default="path")
    ap.add_argument("--no-direct", action="store_true")
    args = ap.parse_args()
    render_scene(
        args.scene, args.output, width=args.width, height=args.height,
        samples=args.samples, integrator=args.integrator,
        direct_light=not args.no_direct,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Rewrite `examples/render_turntable.py` to use `render_to_array` + a stage**

```python
#!/usr/bin/env python3
"""Orbit-camera turntable. Mutates a camera xform on a Usd.Stage per frame."""
from __future__ import annotations
import argparse
import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom
from PIL import Image
from skinny.headless import HeadlessRenderer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="assets/cornell_box_sphere.usda")
    ap.add_argument("--outdir", default="turntable")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--samples", type=int, default=64)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(args.scene)
    cam = UsdGeom.Camera.Define(stage, "/TurntableCam")
    rot = cam.AddRotateYOp()
    pad = len(str(args.frames - 1))

    with HeadlessRenderer(args.width, args.height) as r:
        for i in range(args.frames):
            rot.Set(360.0 * i / args.frames)
            arr = r.render_to_array(stage, samples=args.samples)
            out = outdir / f"frame_{i:0{pad}d}.png"
            Image.fromarray(arr, "RGBA").save(out)
            print(f"  {out} ({i + 1}/{args.frames})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

(Note: the authored `/TurntableCam` becomes the scene camera; `set_usd_scene` re-applies an authored camera every frame, so the rotation animates. If the bundled scene already has a camera that overrides this, instead rotate that camera's existing prim — verify which camera wins with a 2-frame run and adjust the prim path.)

- [ ] **Step 3: Verify the rewritten examples run**

Run:
```bash
export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
./bin/python3.13 examples/render_image.py --width 160 --height 160 --samples 12 -o /tmp/ex.png
./bin/python3.13 examples/render_turntable.py --width 96 --height 96 --frames 3 --samples 6 --outdir /tmp/tt
```
Expected: `/tmp/ex.png` exists and is non-black; `/tmp/tt/frame_00.png..frame_02.png` exist and differ. Clean up afterward.

- [ ] **Step 4: Update `examples/README.md`**

Replace the script table + usage so it documents `skinny-render` and `skinny.headless` as the primary path, with the two scripts shown as minimal API-usage demos. Add an example invoking `skinny-render` and a 5-line Python snippet using `HeadlessRenderer` with a mutated stage. (Write real commands, not placeholders.)

- [ ] **Step 5: Update top-level docs**

In `README.md`, add a short "Headless / scripted rendering" subsection under Running:

```markdown
### Headless rendering (`skinny-render`)

Render a USD scene offscreen to an image without opening a window:

\`\`\`bash
skinny-render assets/cornell_box_sphere.usda -o out.png --samples 128
skinny-render shot.usda --animate --outdir frames/ --fps 24
\`\`\`

Programmatically, `skinny.headless.HeadlessRenderer` holds the GPU context
across calls so you can mutate a `Usd.Stage` and re-render per frame. See
`examples/` and `Architecture.md`.
```

In `Architecture.md`, add one line under the headless/web section pointing at `skinny.headless` + `Renderer.set_usd_scene`. In `CHANGELOG.md` under `## [Unreleased]`, add a `### Tooling` entry: "Headless render API (`skinny.headless`) and `skinny-render` CLI for offscreen USD rendering; accepts a file path or a live `Usd.Stage` mutated per frame."

- [ ] **Step 6: Run the full test module + lint**

Run:
```bash
./bin/python3.13 -m pytest tests/test_headless_api.py -v
.venv/bin/ruff check src/skinny/headless.py examples/render_image.py examples/render_turntable.py
```
Expected: tests PASS (GPU ones may be skipped if no Vulkan in the chosen interpreter — run GPU ones with the 3.13 venv + SDK env); ruff clean.

- [ ] **Step 7: Commit**

```bash
git add examples/render_image.py examples/render_turntable.py
git add -f examples/README.md
git add README.md Architecture.md CHANGELOG.md
git commit -m "docs: headless render API + CLI; migrate examples onto it"
```

(Note: `examples/` is gitignored by the blanket `*`; the `.py` files were added earlier this session only locally. Use `git add -f` for any example file not already tracked — check with `git ls-files examples/` first.)

---

## Self-review notes (already applied)

- **Spec coverage:** loader stage support (Task 1), `set_usd_scene` (Task 2), API render_scene/render_to_array + options + errors (Task 3), time + animation + frame range (Task 4), CLI (Task 5), example migration + docs + CHANGELOG (Task 6). All spec sections map to a task.
- **Type consistency:** `RenderOptions`, `_fmt_for_output`, `_load_scene`, `_parse_frames`, `_frame_times`, `HeadlessRenderer.{render_scene,render_to_array,render_animation}`, and `main`/`_build_parser` names are used identically across tasks and tests.
- **Known runtime caveats called out inline:** prim path for the mutation test (Task 3 Step 6), authored-camera-wins for the turntable (Task 6 Step 2), `git add -f` for gitignored new files.
```
