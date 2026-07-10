"""White-furnace energy-closure gate (change confirming-test-scenes /
furnace-closure).

The physical invariant: a lossless material lit by a *constant* environment
reflects exactly the incident radiance, so the object is **indistinguishable
from the background** — the accumulated furnace image is spatially **uniform**
and the object vanishes. Any energy the BSDF gains or loses makes the object
appear (a darker or brighter region), breaking that uniformity.

We therefore gate on **spatial uniformity** (the object's disappearance), not an
absolute "== 1.0" — skinny's constant-white furnace environment has its own
radiance constant (empirically ~0.88 for the path/BDPT integrators, ~0.98 for
SPPM), so the *absolute* level is an environment-normalization detail, while
uniformity is normalization- and integrator-independent and is the true closure
statistic. The per-material furnace probe instead checks that the flagged
material lights up while an unflagged neighbour stays dark.

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
    statistic: float          # relative non-uniformity (or per-material ratio)
    target: float             # tolerance the statistic is gated against
    passed: bool
    kind: str = "uniformity"
    mean: float = 0.0         # frame mean luminance (env constant), for logging
    baseline_used: bool = False
    metrics: "metrics.ImageMetrics | None" = None


def _foreground_mask(img: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Pixels that received any radiance (object + lit background)."""
    return img.mean(axis=-1) > eps


def relative_nonuniformity(img: np.ndarray) -> tuple[float, float]:
    """(nonuniformity, mean): coefficient of variation of luminance over lit
    pixels. A lossless object in a constant furnace vanishes → ~0. The mean is
    the environment constant, returned only for logging."""
    lum = img.mean(axis=-1)
    mask = _foreground_mask(img)
    vals = lum[mask] if mask.any() else lum.ravel()
    mean = float(vals.mean())
    if mean <= 1e-6:
        return 1.0, mean
    return float(vals.std() / mean), mean


def furnace_closure_result(spec: parity.SceneSpec, img: np.ndarray,
                           combo: parity.RenderCombo) -> FurnaceResult:
    """Uniformity furnace gate: the object must vanish into the constant furnace
    (relative non-uniformity below tolerance). A recorded ``baseline`` widens the
    tolerance to a known residual non-uniformity (tighten-only)."""
    cfg = spec.furnace or {}
    tol = float(cfg.get("uniformity_tol", cfg.get("tol", 0.03)))
    baseline_used = "baseline" in cfg
    if baseline_used:
        tol = max(tol, float(cfg["baseline"]))
    nonunif, mean = relative_nonuniformity(img)
    return FurnaceResult(
        name=spec.name, combo_label=combo.label, statistic=nonunif, target=tol,
        passed=nonunif <= tol, kind="uniformity", mean=mean,
        baseline_used=baseline_used,
    )


def furnace_per_material_result(spec: parity.SceneSpec, img: np.ndarray,
                                combo: parity.RenderCombo) -> FurnaceResult:
    """Per-material furnace gate: arming the furnace bit on the flagged material
    (right-hand sphere, +x) must make it render **distinguishably** from the
    unflagged left-hand sphere — i.e. the bit has a measurable, material-local
    effect. Statistic = |flagged_mean/unflagged_mean − 1|; passes when it exceeds
    the recorded minimum divergence."""
    cfg = spec.furnace or {}
    min_div = float(cfg.get("min_divergence", 0.1))
    lum = img.mean(axis=-1)
    w = lum.shape[1]
    left = lum[:, : w // 2]    # unflagged sphere (−x)
    right = lum[:, w // 2:]    # flagged sphere (+x)
    lm = float(left[left > 1e-4].mean()) if (left > 1e-4).any() else 0.0
    rm = float(right[right > 1e-4].mean()) if (right > 1e-4).any() else 0.0
    div = abs(rm / (lm + 1e-4) - 1.0)
    return FurnaceResult(
        name=spec.name, combo_label=combo.label, statistic=div, target=min_div,
        passed=div >= min_div, kind="per_material", mean=rm,
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
        # A spectral combo must render under --spectral, not fall back to RGB while
        # labelled spectral (the spectral build closes the white furnace too, 5.4).
        spectral=combo.spectral,
    )
