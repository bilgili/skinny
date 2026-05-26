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
    while t <= end + step * 1e-6:
        out.append(round(t, 6))
        t += step
    return out


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


def _repo_root() -> Path:
    # src/skinny/headless.py → repo root is three parents up.
    return Path(__file__).resolve().parent.parent.parent


class HeadlessRenderer:
    """Windowless renderer that persists across calls.

    Use as a context manager so the Vulkan context is always torn down:

        with HeadlessRenderer(1920, 1080) as r:
            for i in range(120):
                mutate(stage)
                r.render_scene(stage, f"out/{i:04d}.png", samples=64)
    """

    def __init__(self, width: int, height: int, *, gpu: Optional[str] = None) -> None:
        import skinny
        from skinny.renderer import Renderer
        from skinny.vk_context import VulkanContext

        self.ctx = VulkanContext(window=None, width=width, height=height, gpu_preference=gpu)
        try:
            self.renderer = Renderer(
                vk_ctx=self.ctx,
                shader_dir=Path(skinny.__file__).resolve().parent / "shaders",
                hdr_dir=_repo_root() / "hdrs",
                tattoo_dir=_repo_root() / "tattoos",
            )
        except Exception:
            self.ctx.destroy()
            raise

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
        # Apply options AFTER the scene swap so they win over anything
        # _apply_usd_lights seeded on the first scene.
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


    def render_animation(self, source: Source, outdir, *, samples: int = 64,
                         frames: Optional[tuple] = None, fps: Optional[float] = None,
                         ext: str = "png", **opts) -> list:
        """Render a frame sequence over a stage's timecodes.

        `frames` is (start, end[, step]); defaults to the stage's
        start/end timecode with step 1. `ext` selects the image format
        for all frames. `fps` is accepted for CLI/forward-compat symmetry
        but is not currently used (no playback or metadata is written).
        Returns the list of written Paths.
        """
        if "time" in opts or "format" in opts:
            raise ValueError(
                "render_animation controls 'time' (per frame) and 'format' "
                "(via 'ext'); pass 'frames' and 'ext' instead"
            )

        from pxr import Usd

        if isinstance(source, Usd.Stage):
            stage = source
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"scene not found: {path}")
            stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise FileNotFoundError(f"could not open USD stage: {source}")

        if frames is None:
            rng = (float(stage.GetStartTimeCode()), float(stage.GetEndTimeCode()), 1.0)
        elif len(frames) == 2:
            rng = (float(frames[0]), float(frames[1]), 1.0)
        else:
            rng = (float(frames[0]), float(frames[1]), float(frames[2]))

        times = _frame_times(rng)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        pad = max(4, len(str(len(times) - 1)))
        fmt = _fmt_for_output(Path("x." + ext), None)

        written = []
        for i, t in enumerate(times):
            out = outdir / f"frame_{i:0{pad}d}.{ext}"
            self.render_scene(stage, out, samples=samples, time=t,
                              format=fmt, **opts)
            written.append(out)
        return written


def render_to_array(source: Source, *, width: int = 1024, height: int = 1024,
                    gpu: Optional[str] = None, **kw) -> np.ndarray:
    with HeadlessRenderer(width, height, gpu=gpu) as r:
        return r.render_to_array(source, **kw)


def render_scene(source: Source, output, *, width: int = 1024,
                 height: int = 1024, gpu: Optional[str] = None, **kw) -> None:
    with HeadlessRenderer(width, height, gpu=gpu) as r:
        r.render_scene(source, output, **kw)


def render_animation(source: Source, outdir, *, width: int = 1024,
                     height: int = 1024, gpu: Optional[str] = None,
                     **kw) -> list:
    with HeadlessRenderer(width, height, gpu=gpu) as r:
        return r.render_animation(source, outdir, **kw)
