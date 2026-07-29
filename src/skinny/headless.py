"""Offscreen (headless) rendering API.

Drives the renderer with no window and saves images or returns pixel arrays.
Accepts a USD source as a file path or an already-open `Usd.Stage` the caller
mutates between frames. The GPU context (and one-time pipeline compile) is held
across calls via `HeadlessRenderer`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from skinny.bringup import BringupPlan, plan_bringup
from skinny.cli_common import add_render_flags, neural_config_from_args, resolve_walk

if TYPE_CHECKING:
    from pxr import Usd

Source = Union[str, Path, "Usd.Stage", object]

_INTEGRATORS = {"path": 0, "bdpt": 1, "sppm": 2, "mlt": 3}
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
    # SPPM glossy-continue threshold override; None → renderer's built-in default.
    sppm_glossy_roughness: Optional[float] = None

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
        # Preserve "unspecified" rather than resolving it here. The loader turns
        # None into the stage's START time code; collapsing it to `Default()`
        # here would hand the loader an explicit time code, skip that branch,
        # and read every time-sampled-only light at its schema fallback.
        return None
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

    def __init__(self, width: int, height: int, *, gpu: Optional[str] = None,
                 plan: Optional[BringupPlan] = None,
                 backend: str = "vulkan",
                 execution_mode: str = "megakernel", bdpt_walk: str = "fused",
                 proposals: Optional[str] = None, reuse: Optional[str] = None,
                 lobe_samplers: Optional[str] = None,
                 encoding: Optional[str] = None,
                 spectral: bool = False) -> None:
        import skinny

        if plan is None:
            # Direct API use (tests, the parity harness, Python callers): the
            # guards already ran — or there were no flags to guard — so the
            # kwargs become a plan directly. `skinny-render`'s main() passes the
            # guarded plan from plan_bringup instead. Conditioner encoding
            # (axis 2) is a build dim: E0/None keeps neural_config=None → the
            # renderer's default → byte-identical SPIR-V.
            plan = BringupPlan(
                prog="skinny-render", backend=backend,
                execution_mode=execution_mode, startup_integrator="path",
                spectral=spectral, bdpt_walk=resolve_walk(bdpt_walk),
                encoding=encoding,
                neural_config=neural_config_from_args(
                    argparse.Namespace(encoding=encoding)),
            )
        self.ctx, self.renderer = plan.create(
            window=None, width=width, height=height, gpu_preference=gpu,
            shader_dir=Path(skinny.__file__).resolve().parent / "shaders",
            hdr_dir=_repo_root() / "hdrs",
            tattoo_dir=_repo_root() / "tattoos",
        )
        try:
            # Scene-sampling seam selection (mirrors the interactive front-ends).
            if proposals is not None:
                self.renderer.proposal_preset_index = \
                    self.renderer.proposal_preset_from_token(proposals)
            if reuse is not None:
                self.renderer.reuse_index = self.renderer._REUSE_TOKENS.index(reuse)
            if lobe_samplers is not None:
                from skinny.sampling import parse_lobe_samplers

                c, s, d = parse_lobe_samplers(lobe_samplers)
                self.renderer.coat_sampler_index = c
                self.renderer.spec_sampler_index = s
                self.renderer.diff_sampler_index = d
        except Exception:
            self.ctx.destroy()
            raise

    def __enter__(self) -> "HeadlessRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        try:
            self.renderer.cleanup()
        finally:
            # Context teardown must run even if renderer cleanup raises (change
            # nanovdb-volume-rendering, D5.2 — teardown on every exit path).
            self.ctx.destroy()

    def _prepare(self, source: Source, opts: RenderOptions) -> None:
        scene = _load_scene(source, opts.time)
        self.renderer.set_usd_scene(scene)
        # Apply options AFTER the scene swap so they win over anything
        # _apply_usd_lights seeded on the first scene.
        self.renderer.integrator_index = opts.integrator_index
        self.renderer.tonemap_index = opts.tonemap_index
        self.renderer.exposure = float(opts.exposure)
        # These options belong to Skinny's fallback pair. Retain them even
        # while authored USD lighting owns the current scene; the renderer's
        # authority gates keep them from affecting authored contributions.
        self.renderer.direct_light_index = 0 if opts.direct_light else 1
        if opts.env_intensity is not None:
            self.renderer.env_intensity = float(opts.env_intensity)
        # SPPM glossy-continue threshold override (only read under SPPM). None
        # leaves the renderer's built-in default in place.
        if opts.sppm_glossy_roughness is not None:
            self.renderer._sppm_glossy_roughness_override = \
                float(opts.sppm_glossy_roughness)

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
        # Backend/execution-mode-aware readiness: in wavefront mode the
        # megakernel pipeline is never built (`scene_bindings_only`), so a
        # `pipeline is None` check would reject every valid wavefront render.
        if not self.renderer._backend_render_ready:
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
        # Same execution-mode-aware gate as render_to_array (wavefront builds
        # no megakernel pipeline by design).
        if not self.renderer._backend_render_ready:
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
    # headless owns its own --width/--height (offline output size, 1024^2
    # default), so it opts out of the shared render-area flags to avoid an
    # argparse conflict.
    add_render_flags(p, resolution=False, mcp=False)
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
    p.add_argument("--gpu", default=None,
                   help="GPU preference (e.g. intel/nvidia/amd/discrete/auto)")
    return p


def main(argv: Optional[list] = None) -> int:
    ns = _build_parser().parse_args(argv)
    # Shared bring-up: resolve the execution mode + backend and run every
    # refusal guard, in the one canonical order (skinny.bringup). skinny-render
    # is non-interactive — no persisted settings participate (persisted=None),
    # so resolution is flags + environment + auto only.
    plan = plan_bringup(ns, prog="skinny-render")
    opts = dict(
        samples=ns.samples, integrator=ns.integrator or "path", tonemap=ns.tonemap,
        exposure=ns.exposure, env_intensity=ns.env_intensity,
        direct_light=not ns.no_direct,
        sppm_glossy_roughness=ns.sppm_glossy_roughness,
    )
    try:
        with HeadlessRenderer(ns.width, ns.height, gpu=ns.gpu, plan=plan,
                              proposals=ns.proposals, reuse=ns.reuse,
                              lobe_samplers=ns.lobe_samplers) as r:
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
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"[skinny-render] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
