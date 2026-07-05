"""Wavefront BDPT/SPPM must shade a NON-TERMINAL non-flat (python / volume) first
hit via the path tracer too — not only the terminal subsurface/skin types — with
the heavy multi-bounce fallback bounded per tile so it stays within the Metal GPU
watchdog.

Regression for change ``wavefront-nonflat-tiled-fallback`` (follow-up to
``wavefront-nonflat-path-fallback``). Before the fix a ``MATERIAL_TYPE_PYTHON``
object rendered black under ``--execution-mode wavefront --integrator {bdpt,sppm}``
(the terminal-only gate skipped it) while wavefront ``path`` shaded it.

GPU-marked (opt-in); one Metal process at a time. The watchdog-safety of the
unbounded-loop concern is covered by ``tests/test_metal_cleanup.py -m gpu``; this
test covers the render (black → shaded) parity.
"""

import os

import numpy as np
import pytest

SCENE = os.path.join(
    os.path.dirname(__file__), "..", "assets", "cornell_box_python_material.usda"
)


def _central_patch_mean(execution_mode: str, integrator: str,
                        samples: int = 24, size: int = 96) -> float:
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
def test_wavefront_bdpt_sppm_shade_python_material_first_hit():
    if not os.path.isfile(SCENE):
        pytest.skip(f"python-material scene absent: {SCENE}")

    try:
        anchor = _central_patch_mean("wavefront", "path")
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # noqa: BLE001 - GPU/backend unavailable in this env
        pytest.skip(f"render backend unavailable: {exc}")

    assert anchor > 1e-3, (
        f"anchor central patch is not on the python-material object "
        f"(got {anchor:.5f}); the scene may have changed")

    bdpt = _central_patch_mean("wavefront", "bdpt")
    sppm = _central_patch_mean("wavefront", "sppm")

    assert bdpt >= 0.5 * anchor, (
        f"wavefront BDPT blacks out the python-material first hit: "
        f"patch={bdpt:.5f} vs wavefront-path anchor {anchor:.5f}")
    assert sppm >= 0.5 * anchor, (
        f"wavefront SPPM blacks out the python-material first hit: "
        f"patch={sppm:.5f} vs wavefront-path anchor {anchor:.5f}")
