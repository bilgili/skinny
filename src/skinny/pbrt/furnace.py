"""White-furnace energy-closure gate (change confirming-test-scenes /
furnace-closure).

A lossless material lit by a constant-white environment must reflect exactly the
incident radiance — the accumulated image closes to 1.0 everywhere. Any energy
gained or lost by the BSDF shows up as a deviation from 1.0, localized to the
material under test. This module renders a suite furnace scene with furnace mode
armed and gates the linear-HDR accumulation against its recorded closure.

Heavy imports (renderer/GPU) are lazy via :mod:`skinny.pbrt.parity`, so this
module imports without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics, parity


@dataclass
class FurnaceResult:
    name: str
    combo_label: str
    mean: float
    target: float
    deviation: float          # |mean - target|
    passed: bool
    baseline_used: bool = False
    metrics: "metrics.ImageMetrics | None" = None


def _foreground_mask(img: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Pixels that received any radiance — the object + lit background.

    A furnace scene renders the constant-white environment behind the object, so
    (almost) every pixel is lit; the mask guards against a stray fully-black
    border and keeps the closure measurement on real samples.
    """
    lum = img.mean(axis=-1)
    return lum > eps


def closure_target(spec: parity.SceneSpec) -> tuple[float, float, bool]:
    """(target, tol, baseline_used) for *spec*'s furnace closure.

    A recorded ``baseline`` (legitimate energy loss, e.g. rough conductor without
    multiple-scattering compensation) replaces the ideal 1.0 target; the gate
    then asserts the render sits at the baseline, and the baseline may only be
    tightened toward the ideal by a later change — never loosened.
    """
    cfg = spec.furnace or {}
    ideal = float(cfg.get("closure", 1.0))
    tol = float(cfg.get("tol", 0.02))
    if "baseline" in cfg:
        return float(cfg["baseline"]), tol, True
    return ideal, tol, False


def furnace_closure_result(spec: parity.SceneSpec, img: np.ndarray,
                           combo: parity.RenderCombo) -> FurnaceResult:
    """Gate a furnace render *img* (linear-HDR, H×W×3) against its closure target.

    The measured statistic is the mean luminance over lit pixels; a lossless
    material closes to the ideal (or recorded-baseline) target within tolerance.
    """
    target, tol, baseline_used = closure_target(spec)
    mask = _foreground_mask(img)
    lum = img.mean(axis=-1)
    mean = float(lum[mask].mean()) if mask.any() else float(lum.mean())
    dev = abs(mean - target)
    return FurnaceResult(
        name=spec.name, combo_label=combo.label, mean=mean, target=target,
        deviation=dev, passed=dev <= tol, baseline_used=baseline_used,
    )


def render_furnace(spec: parity.SceneSpec, combo: parity.RenderCombo,
                   corpus_dir: str, gpu: str | None = None) -> np.ndarray:
    """Render *spec* with *combo* under furnace mode; return linear-HDR (H,W,3)."""
    cfg = spec.furnace or {}
    src = parity._scene_source(spec, corpus_dir)
    per_material = cfg.get("furnace_material") if cfg.get("per_material") else None
    return parity.render_linear(
        src["scene_pbrt"], spec.width, spec.height, spp=spec.spp,
        gpu=gpu, env_off=False,
        integrator=combo.integrator, execution_mode=combo.execution_mode,
        usd_path=src["usd_path"],
        furnace=True, furnace_material=per_material,
    )
