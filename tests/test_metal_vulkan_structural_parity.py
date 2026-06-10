"""Metal↔Vulkan **structural-parity** test (task 6.1).

Renders the *same* fixed scene (``heads/head.obj`` under an HDR environment)
through the megakernel on **both** backends in ``TOOL_MODE_STRUCTURAL`` and
compares the deterministic structural channel — per primary ray: hit/miss mask,
instance id, material id, and depth.

Design D5 splits parity into two tiers: **exact** on the structural outputs
(integer/position data with no transcendental divergence) and a perceptual
tolerance on the converged shaded color (the separate 6.2 test). This is the
exact tier:

* hit/miss mask, instance id, material id — asserted **bit-exact** (discrete
  integer-valued floats).
* depth — asserted within a tiny relative tolerance. The same ray math runs on
  both backends (``column_major`` matrices + float4-packed BVH/vertex structs
  make the layouts agree), so depth *should* match to the ULP; the small
  tolerance only absorbs FMA-contraction differences between the two shader
  compilers. The actual max deviation is printed.

Determinism: the structural write reuses the real ``runFrame`` primary ray, so
both backends use ``createRNG(pixel, frameIndex)`` — identical AA jitter at the
same ``frameIndex`` ⇒ identical rays ⇒ identical hits. Both legs dispatch a
single frame from ``accum_frame == 0``.

DANGER — builds the megakernel on Metal (MTLCompilerService RAM spike) and also
constructs a Vulkan device. Run ONLY through ``scripts/guarded_metal.sh`` with
the compile gate + the Vulkan SDK on the dylib path:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_MEGAKERNEL_COMPILE=1 PYTHONPATH=$PWD/src TIMEOUT_S=420 \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest \
        tests/test_metal_vulkan_structural_parity.py -q -s
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
_HEAD_OBJ = _PROJECT_ROOT / "heads" / "head.obj"
_COMPILE_GATE = "RUN_METAL_MEGAKERNEL_COMPILE"

_RES = 64               # square render; small fits the tool-buffer structural slots
_DEPTH_RTOL = 1e-4      # FMA-contraction slack between the two shader compilers
_DEPTH_ATOL = 1e-4


def _pump_until_ready(renderer, *, budget_s: float = 90.0) -> bool:
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        renderer.update(0.016)
        if (
            renderer._backend_render_ready
            and renderer._num_instances >= 1
            and renderer._baked_source_idx >= 0
        ):
            return True
        time.sleep(0.02)
    return False


def _build(ctx):
    from skinny.renderer import Renderer

    r = Renderer(
        vk_ctx=ctx, shader_dir=_SHADER_DIR, execution_mode="megakernel",
        hdr_dir=_HDR_DIR if _HDR_DIR.is_dir() else None,
        tattoo_dir=_TATTOO_DIR if _TATTOO_DIR.is_dir() else None,
    )
    r.integrator_index = 0  # path tracer (deterministic seed sequence per frame)
    r.load_model_from_path(_HEAD_OBJ)
    assert _pump_until_ready(r), "scene not ready (bake/pipeline build timed out)"
    return r


@pytest.mark.skipif(
    os.environ.get(_COMPILE_GATE) != "1",
    reason=(
        f"renders the megakernel on Metal (MTLCompilerService RAM spike); set "
        f"{_COMPILE_GATE}=1 and run under scripts/guarded_metal.sh"
    ),
)
def test_metal_vulkan_structural_parity():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    if not _HEAD_OBJ.exists():
        pytest.skip(f"head asset not found: {_HEAD_OBJ}")
    try:
        from skinny.metal_context import MetalContext
        from skinny.vk_context import VulkanContext
    except OSError as exc:  # libvulkan not on the dylib path
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")

    # ── Metal leg ───────────────────────────────────────────────────
    m_ctx = MetalContext(window=None, width=_RES, height=_RES)
    try:
        m_aov = _build(m_ctx).read_structural_aov()
    finally:
        m_ctx.destroy()

    # ── Vulkan leg (reference) ──────────────────────────────────────
    v_ctx = VulkanContext(window=None, width=_RES, height=_RES)
    try:
        v_aov = _build(v_ctx).read_structural_aov()
    finally:
        v_ctx.destroy()

    assert m_aov.shape == v_aov.shape == (_RES, _RES, 4)

    m_hit = m_aov[..., 0] > 0.5
    v_hit = v_aov[..., 0] > 0.5
    n_hit = int(v_hit.sum())
    edge_disagree = int(np.count_nonzero(m_hit != v_hit))

    # Depth deviation on the commonly-hit pixels (both backends agree it's a hit).
    both = m_hit & v_hit
    if both.any():
        md, vd = m_aov[..., 3][both], v_aov[..., 3][both]
        depth_max_abs = float(np.max(np.abs(md - vd)))
        depth_max_rel = float(np.max(np.abs(md - vd) / (np.abs(vd) + 1e-8)))
    else:
        depth_max_abs = depth_max_rel = float("nan")

    print(
        f"\n[6.1 structural] res={_RES} vulkan_hits={n_hit} "
        f"hitmask_disagree={edge_disagree} px "
        f"depth_max_abs={depth_max_abs:.3e} depth_max_rel={depth_max_rel:.3e}"
    )

    # The head must be visible — otherwise the channel carries no signal.
    assert n_hit > 0, "no primary-ray hits on the reference (Vulkan) backend"

    # Hit/miss mask exact (D5: deterministic, no transcendental divergence).
    assert edge_disagree == 0, (
        f"hit/miss mask differs on {edge_disagree} pixel(s) between backends"
    )

    # Instance id + material id exact on hit pixels (discrete integer-valued).
    assert np.array_equal(m_aov[..., 1][both], v_aov[..., 1][both]), "instance id mismatch"
    assert np.array_equal(m_aov[..., 2][both], v_aov[..., 2][both]), "material id mismatch"

    # Depth within tolerance (exact-or-ULP per D5; tolerance absorbs FMA only).
    assert np.allclose(
        m_aov[..., 3][both], v_aov[..., 3][both],
        rtol=_DEPTH_RTOL, atol=_DEPTH_ATOL,
    ), f"depth parity exceeded tolerance (max_abs={depth_max_abs:.3e})"
