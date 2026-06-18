"""Path-tracer convergence gate (change ``path-bdpt-convergence``).

The unidirectional path tracer is an unbiased estimator of the rendering
equation, so at convergence it must match the **pbrt v4 reference** (differing
only in Monte-Carlo noise). Before the specular→area-light fix the path tracer
dropped the reflection of the area light in a smooth dielectric and the specular
leg of the caustic, so on ``glass_arealight`` it was biased *dark* (FLIP vs pbrt
≈ 0.058); the fix restores both (FLIP ≈ 0.025).

The reference is the checked-in pbrt corpus EXR — NOT skinny's BDPT. During apply
BDPT was measured ~1.7× too bright versus pbrt on *both* the glass scene and a
purely diffuse scene (no delta bounces), i.e. a separate BDPT normalization bug,
so it cannot anchor this gate. See ``design.md`` D4.

The gate reuses the pbrt corpus scene + importer + the linear-HDR render path
(``skinny.pbrt.parity.render_linear``) and the corpus FLIP/relMSE metrics. A
future change that re-introduces a specular-emitter bias pushes FLIP/relMSE past
tolerance and fails. relMSE is kept loose-ish because the path tracer stays
noisier than the reference on caustics (a non-goal of the fix — same expected
image, not equal noise).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from skinny.pbrt import metrics
from skinny.pbrt.parity import render_linear

pytest.importorskip("skinny.usd_loader")

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")

# Match the reference EXR resolution (128²) so FLIP compares like for like; 512
# spp tames (not removes) the path tracer's caustic variance.
WIDTH = HEIGHT = 128
SPP = 512

# Convergence tolerances vs the pbrt reference. Post-fix FLIP ≈ 0.025 / relMSE ≈
# 0.018; the pre-fix bias sits at FLIP ≈ 0.058, so a 0.045 FLIP gate fails hard
# before the fix and passes comfortably after. Tighter than the corpus parity
# tol (0.09) on purpose — this gate exists to catch the specular bias.
FLIP_TOL = 0.045
RELMSE_TOL = 0.035


def _ref(name: str) -> np.ndarray:
    return metrics.read_exr(os.path.join(CORPUS_DIR, "refs", name + ".exr"))


def _path_vs_ref(name: str, execution_mode: str = "megakernel",
                 spp: int = SPP) -> tuple[float, float]:
    """Render *name* with the path tracer; return (relMSE, FLIP) vs pbrt ref."""
    scene = os.path.join(CORPUS_DIR, name + ".pbrt")
    img = render_linear(scene, WIDTH, HEIGHT, spp, env_off=True,
                        integrator="path", execution_mode=execution_mode)
    ref = _ref(name)
    aligned = metrics.align_exposure(img, ref)
    return metrics.relmse(aligned, ref), metrics.flip(aligned, ref)


@pytest.mark.gpu
def test_glass_arealight_path_converges_to_pbrt():
    """Smooth-dielectric area-light scene: path tracer matches pbrt v4.

    Captures the specular→area-light bias — the path tracer dropped the
    area-light reflection and the specular caustic leg, biasing the image dark
    (FLIP ≈ 0.058). After the delta-gated emission fix the caustic + reflection
    are present and FLIP/relMSE fall within tolerance.
    """
    try:
        relmse, flip = _path_vs_ref("glass_arealight")
    except Exception as exc:  # noqa: BLE001 - GPU/backend unavailable in this env
        pytest.skip(f"render backend unavailable: {exc}")

    assert flip <= FLIP_TOL, (
        f"glass_arealight path FLIP vs pbrt = {flip:.4f} > {FLIP_TOL} — path "
        f"tracer is biased (specular→emitter transport missing?)"
    )
    assert relmse <= RELMSE_TOL, (
        f"glass_arealight path relMSE vs pbrt = {relmse:.4f} > {RELMSE_TOL}"
    )


@pytest.mark.gpu
def test_diffuse_scene_unchanged_and_unbiased():
    """A purely diffuse (non-delta) scene must stay unbiased — no double-count.

    Diffuse bounces have an NEE partner, so the area light is counted via NEE and
    the delta-gated BSDF emission stays skipped. This guards against the fix
    double-counting on non-delta surfaces (which would push the image bright,
    away from pbrt). The fix is a no-op here, so the path tracer matches pbrt.
    """
    try:
        relmse, flip = _path_vs_ref("diffuse_arealight")
    except Exception as exc:  # noqa: BLE001 - GPU/backend unavailable in this env
        pytest.skip(f"render backend unavailable: {exc}")

    assert flip <= FLIP_TOL, (
        f"diffuse_arealight path FLIP vs pbrt = {flip:.4f} > {FLIP_TOL} — diffuse "
        f"transport must stay unbiased (the specular fix must not double-count)"
    )
    assert relmse <= RELMSE_TOL, (
        f"diffuse_arealight path relMSE vs pbrt = {relmse:.4f} > {RELMSE_TOL}"
    )


@pytest.mark.gpu
def test_glass_arealight_wavefront_converges_to_pbrt():
    """The wavefront path integrator must also be unbiased (not just megakernel).

    The wavefront shade stage carried the identical specular→emitter bias (mean
    0.147, FLIP ≈ 0.061 pre-fix); the delta-gated emission rule is applied there
    too via the PATH_FLAG_SPECULAR path-state bit, so the wavefront path also
    converges to the pbrt reference (mean 0.210, FLIP ≈ 0.032). 256 spp keeps the
    Metal CPU-readback wavefront fallback bounded while leaving FLIP margin.
    """
    try:
        relmse, flip = _path_vs_ref("glass_arealight",
                                    execution_mode="wavefront", spp=256)
    except Exception as exc:  # noqa: BLE001 - GPU/backend unavailable in this env
        pytest.skip(f"render backend unavailable: {exc}")

    assert flip <= FLIP_TOL, (
        f"glass_arealight WAVEFRONT path FLIP vs pbrt = {flip:.4f} > {FLIP_TOL} — "
        f"the wavefront path is left biased (specular→emitter transport missing?)"
    )
    assert relmse <= RELMSE_TOL, (
        f"glass_arealight WAVEFRONT path relMSE vs pbrt = {relmse:.4f} > {RELMSE_TOL}"
    )
