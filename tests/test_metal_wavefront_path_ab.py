"""Headless A/B parity: staged wavefront path tracer on Metal vs Vulkan
(change metal-wavefront-parity, task 3.4).

Renders ``assets/three_materials_demo.usda`` with the wavefront path integrator
on a ``MetalContext`` and a ``VulkanContext`` and asserts the converged
accumulation images agree:

* **Noise-floor criterion** (the ``test_wavefront_path_ab`` methodology): two
  independent Vulkan wavefront renders of the same scene differ by a measurable
  Monte-Carlo + GPU-nondeterminism floor; the Metal render (the same estimator
  and the same per-(pixel, frameIndex) RNG sequence on another backend) must
  match the Vulkan reference no worse than that floor — i.e. it computes the
  same integral, not a similar-looking image.
* **Perceptual metrics** (the megakernel-parity precedent,
  ``test_metal_procedural_graph_parity``): whole-frame rel-MSE < 0.02 and
  correlation > 0.98 on converged color.

Structural note: the structural AOV channel (``read_structural_aov``) requires
the compiled megakernel tool dispatch, which wavefront mode intentionally does
not build — exact geometric/ID parity across backends is pinned by the archived
megakernel structural test (same ``traceScene``/BVH on both); this A/B pins the
wavefront estimator on top of it via the noise-floor criterion.

DANGER — compiles the wavefront stage pipelines in-process on Metal
(MTLCompilerService) and constructs a Vulkan device. Run ONLY through
``scripts/guarded_metal.sh`` with the compile gate + the Vulkan SDK on the
dylib path:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_WAVEFRONT_COMPILE=1 PYTHONPATH=$PWD/src TIMEOUT_S=600 \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest tests/test_metal_wavefront_path_ab.py -q -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from skinny.backend_select import metal_available

pytest.importorskip("slangpy")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SHADER_DIR = _PROJECT_ROOT / "src" / "skinny" / "shaders"
_HDR_DIR = _PROJECT_ROOT / "hdrs"
_TATTOO_DIR = _PROJECT_ROOT / "tattoos"
_SCENE = _PROJECT_ROOT / "assets" / "three_materials_demo.usda"
_COMPILE_GATE = "RUN_METAL_WAVEFRONT_COMPILE"

_RES = 64
_SAMPLES = 96
_REL_MSE_MAX = 0.02
_CORR_MIN = 0.98


def _match(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels whose absolute channel error is inside the
    noise-floor tolerance band (mirrors tests/test_wavefront_path_ab.py)."""
    d = np.abs(a - b).max(axis=2)
    tol = 8e-3 + 0.04 * np.abs(b).max(axis=2)
    return float((d <= tol).mean())


def _pump_until_ready(r, *, budget_s: float = 120.0) -> bool:
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        r.update(0.016)
        if r._backend_render_ready and r._num_instances >= 3:
            return True
        time.sleep(0.02)
    return False


def _converge_wavefront(ctx) -> np.ndarray:
    """Build a wavefront-mode renderer on ``ctx``, converge ``_SAMPLES``
    accumulation frames from a clean start, return mean linear radiance."""
    from skinny.renderer import Renderer

    r = Renderer(
        vk_ctx=ctx, shader_dir=_SHADER_DIR, execution_mode="wavefront",
        hdr_dir=_HDR_DIR if _HDR_DIR.is_dir() else None,
        tattoo_dir=_TATTOO_DIR if _TATTOO_DIR.is_dir() else None,
        usd_scene_path=_SCENE,
    )
    r.integrator_index = 0  # path tracer
    assert _pump_until_ready(r), "three-material scene not ready (bake/build timed out)"
    assert r.effective_execution_mode_index == 1, "wavefront mode not active"
    r.accum_frame = 0
    for _ in range(_SAMPLES):
        r.update(0.016)
        r.render_headless()
    arr, n = r.read_accumulation_hdr()
    return (arr[..., :3] / max(1, n)).astype(np.float64)


@pytest.mark.skipif(
    os.environ.get(_COMPILE_GATE) != "1",
    reason=(
        f"compiles the wavefront stage pipelines on Metal (MTLCompilerService); "
        f"set {_COMPILE_GATE}=1 and run under scripts/guarded_metal.sh"
    ),
)
def test_metal_wavefront_path_matches_vulkan():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    if not _SCENE.exists():
        pytest.skip(f"scene asset not found: {_SCENE}")
    try:
        from skinny.metal_context import MetalContext
        from skinny.vk_context import VulkanContext
    except OSError as exc:  # libvulkan not on the dylib path
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")

    m_ctx = MetalContext(window=None, width=_RES, height=_RES)
    try:
        m_img = _converge_wavefront(m_ctx)
    finally:
        m_ctx.destroy()

    # Vulkan wavefront reference + an independent re-render for the noise floor.
    v_ctx = VulkanContext(window=None, width=_RES, height=_RES)
    try:
        v_img = _converge_wavefront(v_ctx)
    finally:
        v_ctx.destroy()
    v_ctx2 = VulkanContext(window=None, width=_RES, height=_RES)
    try:
        v_img2 = _converge_wavefront(v_ctx2)
    finally:
        v_ctx2.destroy()

    assert np.isfinite(m_img).all(), "Metal wavefront produced non-finite radiance"
    assert np.isfinite(v_img).all(), "Vulkan wavefront produced non-finite radiance"
    assert m_img.max() > 0 and v_img.max() > 0

    floor = _match(v_img2, v_img)        # Vulkan-vs-Vulkan re-render noise floor
    cross = _match(m_img, v_img)         # Metal-vs-Vulkan
    rel_mse = float(np.mean((m_img - v_img) ** 2) / (np.mean(v_img ** 2) + 1e-8))
    corr = float(np.corrcoef(m_img.ravel(), v_img.ravel())[0, 1])
    print(f"[metal-wf-ab] match: cross={cross:.4f} floor={floor:.4f} "
          f"rel_mse={rel_mse:.5f} corr={corr:.5f}")

    # Same integral: Metal matches the Vulkan reference no worse than Vulkan
    # matches itself (small slack for the different-backend FP noise).
    assert cross >= floor - 0.02, (
        f"Metal wavefront diverges beyond the Vulkan re-render noise floor: "
        f"cross={cross:.4f} < floor={floor:.4f} - 0.02"
    )
    assert rel_mse < _REL_MSE_MAX, f"wavefront parity rel-MSE too high: {rel_mse}"
    assert corr > _CORR_MIN, f"wavefront parity correlation too low: {corr}"
