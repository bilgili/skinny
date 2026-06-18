"""BDPT absolute-energy gate vs the pbrt v4 reference (change ``path-bdpt-convergence``).

The bidirectional path tracer is an unbiased estimator of the same rendering
equation as the unidirectional path tracer, so at convergence its **absolute**
(pre-exposure-alignment) energy must match the pbrt reference just as the path
tracer's does — it may differ only in Monte-Carlo noise.

This gate is the counterpart to ``test_convergence.py``. That file's docstring
records the bug this guards: BDPT read **~1.7× too bright** vs pbrt on *both* the
glass scene and a purely diffuse scene (no delta bounces). The cause was the
``t = 0`` emissive eye-vertex strategy — the camera/BSDF subpath landing on an
emissive triangle — being added at *full weight* while the ``t = 1`` NEE
(``connectT1``) already counted the same area light power-heuristic-weighted.
That double-counts direct area-light transport. The fix gates the emissive eye
hit exactly like the path tracer (``path.slang``): full weight only at the
primary/first hit, a delta bounce into the light, or a scene with no
emissive-triangle NEE; otherwise NEE owns it (``bdpt.slang`` /
``wavefront_bdpt.slang``).

Why a *separate* gate from ``test_convergence.py``: the corpus parity gate and
the convergence gate both **exposure-align** before comparing (``align_exposure``
divides out a global scalar), so a uniform 1.7× brightness error is invisible to
them — relMSE stays ~0.02. The error only shows in absolute energy. This gate
therefore compares un-aligned mean energy.

Two assertions per case:
  * ``energy_ratio`` = mean(bdpt) / mean(ref) — the headline absolute-energy
    check the bug violated (pre-fix ≈ 1.76, post-fix ≈ 0.88).
  * ``bdpt_over_path`` = mean(bdpt) / mean(path) — the sharp regression signal:
    BDPT must track the (already-pbrt-converged) unidirectional tracer. Pre-fix
    ≈ 2.0; post-fix ≈ 1.00. This isolates a BDPT-specific bias from the shared
    path-tracer-vs-pbrt residual (RGB-vs-spectral, ~0.87×) that both integrators
    inherit, so the tolerance can be tight.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from skinny.pbrt import metrics
from skinny.pbrt.parity import render_linear

pytest.importorskip("skinny.usd_loader")

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
WIDTH = HEIGHT = 128
SPP = 256

# Absolute mean(bdpt)/mean(ref). Both integrators inherit the path-tracer's ~0.87×
# RGB-vs-spectral offset from the spectral pbrt reference, so a correct BDPT lands
# near 0.88, not 1.0. Tol 0.20 passes that comfortably and fails the pre-fix 1.76
# (|1 - 1.76| = 0.76) hard.
ABS_ENERGY_TOL = 0.20
# mean(bdpt)/mean(path): BDPT must track the unidirectional tracer to within
# Monte-Carlo noise + the small t≥2 indirect it adds. Post-fix ≈ 1.00; the pre-fix
# double-count sat at ≈ 2.0, so a 0.08 band fails the bug decisively.
BDPT_OVER_PATH_TOL = 0.08


def _mean_energy(img: np.ndarray) -> float:
    """Mean luminance over the frame (absolute, no exposure alignment)."""
    a = np.asarray(img, dtype=np.float64)[..., :3]
    return float(np.mean(0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]))


def _energies(name: str, execution_mode: str, spp: int) -> tuple[float, float, float]:
    """Return (ref, path, bdpt) mean energy for *name* at *execution_mode*."""
    scene = os.path.join(CORPUS_DIR, name + ".pbrt")
    ref = metrics.read_exr(os.path.join(CORPUS_DIR, "refs", name + ".exr"))
    path = render_linear(scene, WIDTH, HEIGHT, spp, env_off=True,
                         integrator="path", execution_mode=execution_mode)
    bdpt = render_linear(scene, WIDTH, HEIGHT, spp, env_off=True,
                         integrator="bdpt", execution_mode=execution_mode)
    return _mean_energy(ref), _mean_energy(path), _mean_energy(bdpt)


def _assert_bdpt_energy(name: str, execution_mode: str = "megakernel",
                        spp: int = SPP) -> None:
    try:
        e_ref, e_path, e_bdpt = _energies(name, execution_mode, spp)
    except Exception as exc:  # noqa: BLE001 - GPU/backend unavailable in this env
        pytest.skip(f"render backend unavailable: {exc}")

    energy_ratio = e_bdpt / e_ref
    bdpt_over_path = e_bdpt / e_path
    tag = f"{name}/{execution_mode}"

    assert abs(energy_ratio - 1.0) <= ABS_ENERGY_TOL, (
        f"{tag}: BDPT absolute energy mean(bdpt)/mean(ref) = {energy_ratio:.3f} "
        f"(|1 - r| > {ABS_ENERGY_TOL}). BDPT is double-counting area-light "
        f"transport (t=0 emissive eye hit not MIS-gated vs the t=1 NEE?)."
    )
    assert abs(bdpt_over_path - 1.0) <= BDPT_OVER_PATH_TOL, (
        f"{tag}: BDPT diverges from the path tracer — mean(bdpt)/mean(path) = "
        f"{bdpt_over_path:.3f} (|1 - r| > {BDPT_OVER_PATH_TOL}). Path tracer "
        f"matches pbrt, so BDPT must too; an energy gap is a BDPT-specific bias."
    )


@pytest.mark.gpu
def test_diffuse_arealight_bdpt_energy_megakernel():
    """Purely diffuse area-light scene: the cleanest signal — no specular
    transport, so any over-brightness is a pure direct-lighting double-count.
    Pre-fix mean(bdpt)/mean(ref) ≈ 1.76; post-fix ≈ 0.88 (≈ path tracer)."""
    _assert_bdpt_energy("diffuse_arealight")


@pytest.mark.gpu
def test_glass_arealight_bdpt_energy_megakernel():
    """Smooth-dielectric area-light scene: BDPT must match pbrt absolute energy
    with specular transport present too. Pre-fix ≈ 1.5×; post-fix ≈ 0.87×."""
    _assert_bdpt_energy("glass_arealight")


@pytest.mark.gpu
def test_diffuse_arealight_bdpt_energy_wavefront():
    """The wavefront BDPT shade stage carried the identical unweighted t=0
    emissive hit (wavefront_bdpt.slang); gate it too so both execution modes
    stay energy-correct."""
    _assert_bdpt_energy("diffuse_arealight", execution_mode="wavefront")
