"""Headless A/B parity: ReSTIR DI on Metal vs Vulkan (change
metal-wavefront-parity, task 5.3).

Three pins, mirroring the wavefront path/bdpt A/B methodology
(``test_metal_wavefront_path_ab.py``) plus the Vulkan ReSTIR test suite
(``test_restir_render.py`` / ``test_restir_variance.py``):

* **Cross-backend parity** — ReSTIR DI (spatial regime, the shipped default)
  on the Metal wavefront path matches the Vulkan wavefront reference no worse
  than the Vulkan-vs-Vulkan re-render noise floor, and within the megakernel
  perceptual tolerances (rel-MSE < 0.02, correlation > 0.98).
* **Unbiasedness on Metal** — the ReSTIR estimate converges to the stock-NEE
  reference (identity reuse) on the same Metal device.
* **Variance win on Metal** — on the high-contrast many-light scene, ReSTIR
  has lower error than stock NEE at equal low spp (the variance-reduction
  requirement, now on Metal).

DANGER — compiles the wavefront + ReSTIR stage pipelines in-process on Metal
(MTLCompilerService) and constructs Vulkan devices. Run ONLY through
``scripts/guarded_metal.sh`` with the compile gate + the Vulkan SDK on the
dylib path:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_WAVEFRONT_COMPILE=1 PYTHONPATH=$PWD/src TIMEOUT_S=1800 \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest tests/test_metal_restir_ab.py -q -s
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
_VARIANCE_SCENE = _PROJECT_ROOT / "assets" / "restir_variance_demo.usda"
_COMPILE_GATE = "RUN_METAL_WAVEFRONT_COMPILE"

_RES = 64
_SAMPLES = 96
_REL_MSE_MAX = 0.02
_CORR_MIN = 0.98
REUSE_NONE = 0
REUSE_RESTIR = 1

needs_gate = pytest.mark.skipif(
    os.environ.get(_COMPILE_GATE) != "1",
    reason=(
        f"compiles the wavefront + ReSTIR pipelines on Metal "
        f"(MTLCompilerService); set {_COMPILE_GATE}=1 and run under "
        f"scripts/guarded_metal.sh"
    ),
)


def _lum(a: np.ndarray) -> np.ndarray:
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def _match(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels whose absolute channel error is inside the
    noise-floor tolerance band (mirrors tests/test_wavefront_path_ab.py)."""
    d = np.abs(a - b).max(axis=2)
    tol = 8e-3 + 0.04 * np.abs(b).max(axis=2)
    return float((d <= tol).mean())


def _build_renderer(ctx, scene: Path, *, min_instances: int):
    from skinny.renderer import Renderer

    r = Renderer(
        vk_ctx=ctx, shader_dir=_SHADER_DIR, execution_mode="wavefront",
        hdr_dir=_HDR_DIR if _HDR_DIR.is_dir() else None,
        tattoo_dir=_TATTOO_DIR if _TATTOO_DIR.is_dir() else None,
        usd_scene_path=scene,
    )
    r.integrator_index = 0  # path tracer
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        r.update(0.016)
        if r._backend_render_ready and r._num_instances >= min_instances:
            break
        time.sleep(0.02)
    else:
        raise AssertionError(f"scene not ready (bake/build timed out): {scene}")
    for _ in range(16):
        r.update(0.04)
    assert r.effective_execution_mode_index == 1, "wavefront mode not active"
    return r


def _accumulate(renderer, reuse_index: int, n_samples: int, base_seed: int,
                cfg: dict | None = None) -> np.ndarray:
    """Deterministically accumulate n_samples wavefront frames with the given
    reuse mode; return the raw running-average accumulation image (H, W, 3) —
    the mean linear radiance (wfPathResolve keeps a running mean, so do NOT
    divide; mirrors tests/test_restir_variance.py). Switching reuse rebuilds
    the wavefront pass set (keyed on reuse_mode)."""
    renderer._restir_config = cfg
    renderer.reuse_index = int(reuse_index)
    for i in range(n_samples):
        renderer.frame_index = base_seed + i
        renderer.accum_frame = i
        renderer.render_headless()
    arr, _ = renderer.read_accumulation_hdr()
    return np.ascontiguousarray(arr, dtype=np.float64)[..., :3]


def _skip_unless_ready(scene: Path):
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    if not scene.exists():
        pytest.skip(f"scene asset not found: {scene}")
    try:
        from skinny.metal_context import MetalContext  # noqa: F401
        from skinny.vk_context import VulkanContext  # noqa: F401
    except OSError as exc:  # libvulkan not on the dylib path
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")


@needs_gate
def test_metal_restir_matches_vulkan_and_converges_to_nee():
    """Cross-backend parity + unbiasedness: Metal ReSTIR matches the Vulkan
    ReSTIR reference within the noise floor / perceptual tolerances, and on
    Metal the ReSTIR estimate converges to the stock-NEE reference."""
    _skip_unless_ready(_SCENE)
    from skinny.metal_context import MetalContext
    from skinny.vk_context import VulkanContext

    m_ctx = MetalContext(window=None, width=_RES, height=_RES)
    try:
        r = _build_renderer(m_ctx, _SCENE, min_instances=3)
        try:
            m_none = _accumulate(r, REUSE_NONE, _SAMPLES, base_seed=1000)
            m_restir = _accumulate(r, REUSE_RESTIR, _SAMPLES, base_seed=2000)
            assert r._restir_pass is not None, \
                "ReSTIR pass was not built on the Metal wavefront path"
        finally:
            r.cleanup()
    finally:
        m_ctx.destroy()

    # Vulkan ReSTIR reference + an independent re-render for the noise floor.
    def vulkan_restir(base_seed: int) -> np.ndarray:
        ctx = VulkanContext(window=None, width=_RES, height=_RES)
        try:
            r = _build_renderer(ctx, _SCENE, min_instances=3)
            try:
                return _accumulate(r, REUSE_RESTIR, _SAMPLES, base_seed=base_seed)
            finally:
                r.cleanup()
        finally:
            ctx.destroy()

    v_restir = vulkan_restir(2000)   # same seeds as the Metal render
    v_restir2 = vulkan_restir(7000)  # independent seeds — the noise floor

    for name, img in (("metal-none", m_none), ("metal-restir", m_restir),
                      ("vulkan-restir", v_restir)):
        assert np.isfinite(img).all(), f"{name} has non-finite radiance"
        assert img.max() > 0, f"{name} is black"

    # Unbiasedness on Metal: ReSTIR converges to the stock-NEE reference.
    mean_none = float(_lum(m_none).mean())
    mean_restir = float(_lum(m_restir).mean())
    assert mean_none > 1e-3, f"NEE reference ~black (mean={mean_none})"
    rel = abs(mean_restir - mean_none) / mean_none
    print(f"[metal-restir-ab] metal restir-vs-nee rel mean diff: {rel:.4f}")
    assert rel < 0.02, (
        f"Metal ReSTIR did not converge to NEE: mean {mean_restir:.6g} vs "
        f"{mean_none:.6g} (rel {rel:.4f} ≥ 0.02)"
    )

    # Cross-backend parity: no worse than the Vulkan re-render noise floor
    # (small slack for the different-backend FP noise) + perceptual metrics.
    floor = _match(v_restir2, v_restir)
    cross = _match(m_restir, v_restir)
    rel_mse = float(np.mean((m_restir - v_restir) ** 2)
                    / (np.mean(v_restir ** 2) + 1e-8))
    corr = float(np.corrcoef(m_restir.ravel(), v_restir.ravel())[0, 1])
    print(f"[metal-restir-ab] match: cross={cross:.4f} floor={floor:.4f} "
          f"rel_mse={rel_mse:.5f} corr={corr:.5f}")
    assert cross >= floor - 0.02, (
        f"Metal ReSTIR diverges beyond the Vulkan re-render noise floor: "
        f"cross={cross:.4f} < floor={floor:.4f} - 0.02"
    )
    assert rel_mse < _REL_MSE_MAX, f"ReSTIR parity rel-MSE too high: {rel_mse}"
    assert corr > _CORR_MIN, f"ReSTIR parity correlation too low: {corr}"


# Spatial-only config (the shipped default regime), pinned explicitly so the
# variance comparison is regime-stable (mirrors test_restir_variance.py).
_SPATIAL_CFG = dict(mLight=8, spatialK=5, spatialRadius=16.0, normalThresh=0.9,
                    depthThresh=0.1, mCap=20, mBsdf=1, flags=0x1)
_VAR_RES = 128


@needs_gate
def test_metal_restir_reduces_variance_vs_nee():
    """ReSTIR DI (spatial) beats stock NEE at equal low spp on the
    high-contrast many-light scene — on the Metal wavefront path."""
    _skip_unless_ready(_VARIANCE_SCENE)
    from skinny.metal_context import MetalContext

    ctx = MetalContext(window=None, width=_VAR_RES, height=_VAR_RES)
    try:
        r = _build_renderer(ctx, _VARIANCE_SCENE, min_instances=5)
        try:
            assert r._num_emissive_tris > 100, \
                "scene should have many emissive triangles"
            # Method-neutral converged reference (both unbiased ⇒ same mean).
            ref = 0.5 * (_accumulate(r, REUSE_NONE, 224, 9000)
                         + _accumulate(r, REUSE_RESTIR, 224, 17000, _SPATIAL_CFG))
            mask = _lum(ref) > 0.02
            assert int(mask.sum()) > _VAR_RES * _VAR_RES // 4

            def rmse(img: np.ndarray) -> float:
                return float(np.sqrt(np.mean(
                    (_lum(img)[mask] - _lum(ref)[mask]) ** 2)))

            spp = 24
            err_none = rmse(_accumulate(r, REUSE_NONE, spp, 2000))
            err_restir = rmse(_accumulate(r, REUSE_RESTIR, spp, 4000, _SPATIAL_CFG))
            print(f"[metal-restir-var] rmse restir={err_restir:.4f} "
                  f"nee={err_none:.4f} ratio={err_restir / err_none:.3f}")
            assert err_none > 0 and np.isfinite(err_restir)
            assert err_restir < 0.85 * err_none, (
                f"ReSTIR did not reduce variance on Metal: rmse "
                f"{err_restir:.4f} vs NEE {err_none:.4f} "
                f"(ratio {err_restir / err_none:.3f} ≥ 0.85)"
            )
        finally:
            r.cleanup()
    finally:
        ctx.destroy()
