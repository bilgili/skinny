"""Metal↔Vulkan parity for MaterialX **procedural-graph** flat materials.

Regression guard for the bug where the per-graph parameter SSBOs
(``graphParams_<name>``) were (a) never bound on the Metal megakernel dispatch
and (b) packed in scalar layout while the Metal shader reads them with MSL
``float3``→16 B padding. Together those made any graph that reads its params
rather than a bindless texture collapse — the purely-procedural marble
(``fractal3d`` noise → ``color_mix_*`` colour params, no texture) rendered
**black** on Metal while Vulkan showed it correctly. Pre-fix this scene's
converged Metal vs Vulkan rel-MSE was ≈0.27 (corr ≈0.27); post-fix ≈0.0004
(corr ≈0.999).

Unlike the head-only structural/shaded parity tests, this drives
``assets/three_materials_demo.usda`` (brass / wood / marble standard_surface
graphs) so the graph-param path is exercised at all.

DANGER — builds the megakernel on Metal (MTLCompilerService RAM spike) and also
constructs a Vulkan device. Run ONLY through ``scripts/guarded_metal.sh`` with
the compile gate + the Vulkan SDK on the dylib path:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_MEGAKERNEL_COMPILE=1 PYTHONPATH=$PWD/src TIMEOUT_S=420 \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest \
        tests/test_metal_procedural_graph_parity.py -q -s
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
_COMPILE_GATE = "RUN_METAL_MEGAKERNEL_COMPILE"

_RES = 64
_SAMPLES = 96
_REL_MSE_MAX = 0.02   # whole-frame; pre-fix ≈0.27
_CORR_MIN = 0.98      # pre-fix ≈0.27


def _pump_until_ready(r, *, budget_s: float = 120.0) -> bool:
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        r.update(0.016)
        if r._backend_render_ready and r._num_instances >= 3:
            return True
        time.sleep(0.02)
    return False


def _converge(ctx) -> tuple[np.ndarray, np.ndarray]:
    from skinny.renderer import Renderer

    r = Renderer(
        vk_ctx=ctx, shader_dir=_SHADER_DIR, execution_mode="megakernel",
        hdr_dir=_HDR_DIR if _HDR_DIR.is_dir() else None,
        tattoo_dir=_TATTOO_DIR if _TATTOO_DIR.is_dir() else None,
        usd_scene_path=_SCENE,
    )
    r.integrator_index = 0  # path tracer
    assert _pump_until_ready(r), "three-material scene not ready (bake/build timed out)"
    r.accum_frame = 0
    for _ in range(_SAMPLES):
        r.update(0.016)
        r.render_headless()
    arr, n = r.read_accumulation_hdr()
    img = (arr[..., :3] / max(1, n)).astype(np.float64)
    aov = r.read_structural_aov()  # (H, W, 4): hit, instanceId, materialId, depth
    return img, aov


@pytest.mark.skipif(
    os.environ.get(_COMPILE_GATE) != "1",
    reason=(
        f"renders the megakernel on Metal (MTLCompilerService RAM spike); set "
        f"{_COMPILE_GATE}=1 and run under scripts/guarded_metal.sh"
    ),
)
def test_metal_vulkan_procedural_graph_parity():
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
        m_img, _ = _converge(m_ctx)
    finally:
        m_ctx.destroy()

    v_ctx = VulkanContext(window=None, width=_RES, height=_RES)
    try:
        v_img, v_aov = _converge(v_ctx)
    finally:
        v_ctx.destroy()

    assert np.isfinite(m_img).all() and np.isfinite(v_img).all()

    rel_mse = float(np.mean((m_img - v_img) ** 2) / (np.mean(v_img ** 2) + 1e-8))
    corr = float(np.corrcoef(m_img.ravel(), v_img.ravel())[0, 1])

    # Per-material guard: no flat/graph material that is lit on Vulkan may be
    # near-black on Metal (the exact marble failure — graph params unbound /
    # misaligned zeroed its colour). Keyed off the Vulkan structural AOV's
    # material-id channel so it is independent of streaming/instance order.
    v_hit = v_aov[..., 0] > 0.5
    v_mat = v_aov[..., 2]
    worst = ""
    for mid in sorted({int(x) for x in np.unique(v_mat[v_hit])}):
        mask = v_hit & (v_mat == mid)
        vm = float(v_img[mask].mean())
        mm = float(m_img[mask].mean())
        if vm > 1e-4:  # only materials that actually carry radiance on Vulkan
            ratio = mm / vm
            if ratio < 0.5 or ratio > 2.0:
                worst += f" mat{mid}:M/V={ratio:.3f}(V={vm:.4f},M={mm:.4f})"

    print(
        f"\n[graph-parity] rel_mse={rel_mse:.4f} (<{_REL_MSE_MAX}) "
        f"corr={corr:.4f} (>{_CORR_MIN}) per-mat-outliers:{worst or ' none'}"
    )

    assert not worst, f"material(s) diverge >2x between backends:{worst}"
    assert corr > _CORR_MIN, f"procedural-graph parity correlation too low: {corr}"
    assert rel_mse < _REL_MSE_MAX, f"procedural-graph parity rel-MSE too high: {rel_mse}"
