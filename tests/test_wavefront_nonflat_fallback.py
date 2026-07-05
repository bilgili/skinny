"""Wavefront BDPT/SPPM must shade a non-flat (subsurface) first hit via the path
tracer, not black it out — parity with the megakernel, which routes every
non-flat first hit to ``PathTracer.estimateRadiance`` (``main_pass.slang`` gates
BDPT on ``MATERIAL_TYPE_FLAT``).

Regression for change ``wavefront-nonflat-path-fallback``: before the fix the
subsurface object rendered as a solid black silhouette under
``--execution-mode wavefront --integrator {bdpt,sppm}`` while it shaded correctly
under wavefront ``path`` and every megakernel combo.

GPU-marked (opt-in): the default ``pytest`` sweep never runs it, honouring the
one-Metal-process-at-a-time thermal rule (each render builds/destroys its own
context sequentially via the ``HeadlessRenderer`` context manager).
"""

import os

import numpy as np
import pytest

SCENE = os.path.join(
    os.path.dirname(__file__), "assets", "suite", "mat_subsurface", "mat_subsurface.usda"
)


def _central_patch_mean(execution_mode: str, integrator: str,
                        samples: int = 24, size: int = 96) -> float:
    """Render the mat_subsurface scene and return the mean linear radiance of a
    central patch that lies fully inside the subsurface object."""
    from skinny.backend_select import select_backend
    from skinny.headless import HeadlessRenderer, RenderOptions

    backend = select_backend()
    with HeadlessRenderer(size, size, backend=backend,
                          execution_mode=execution_mode) as r:
        r._prepare(SCENE, RenderOptions(samples=samples, integrator=integrator))
        if not r.renderer._backend_render_ready:
            pytest.skip(f"{execution_mode}/{integrator}: render pipeline not ready")
        for _ in range(samples):
            r.renderer.update(1.0 / 60.0)
            r.renderer.render_headless()
        hdr, n = r.renderer.read_accumulation_hdr()
    img = hdr[..., :3] / max(1, n)
    assert np.isfinite(img).all(), f"{execution_mode}/{integrator}: non-finite pixels"
    H, W, _ = img.shape
    patch = img[int(H * 0.44):int(H * 0.58), int(W * 0.44):int(W * 0.58)]
    return float(patch.mean())


@pytest.mark.gpu
def test_wavefront_bdpt_sppm_shade_subsurface_first_hit():
    if not os.path.isfile(SCENE):
        pytest.skip(f"mat_subsurface scene absent: {SCENE}")

    # Anchor: wavefront PATH shades the subsurface object correctly today. It is
    # also the exact integrator the fix routes non-flat BDPT/SPPM lanes to, so
    # the fixed BDPT/SPPM central patch should track it closely.
    try:
        anchor = _central_patch_mean("wavefront", "path")
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # noqa: BLE001 - GPU/backend unavailable in this env
        pytest.skip(f"render backend unavailable: {exc}")

    assert anchor > 1e-3, (
        f"anchor central patch is not on the (bright) subsurface object "
        f"(got {anchor:.5f}); the scene may have changed")

    bdpt = _central_patch_mean("wavefront", "bdpt")
    sppm = _central_patch_mean("wavefront", "sppm")

    # Pre-fix these are ~0 (black silhouette); post-fix they equal the wavefront
    # path anchor within noise/incoming-splat slack.
    assert bdpt >= 0.6 * anchor, (
        f"wavefront BDPT blacks out the subsurface first hit: "
        f"patch={bdpt:.5f} vs wavefront-path anchor {anchor:.5f}")
    assert sppm >= 0.6 * anchor, (
        f"wavefront SPPM blacks out the subsurface first hit: "
        f"patch={sppm:.5f} vs wavefront-path anchor {anchor:.5f}")
