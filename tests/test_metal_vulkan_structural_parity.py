"""Metal↔Vulkan **structural-parity** test (task 6.1).

Renders the *same* fixed scene (``heads/head.obj`` under an HDR environment)
through the megakernel on **both** backends in ``TOOL_MODE_STRUCTURAL`` and
compares the deterministic structural channel — per primary ray: hit/miss mask,
instance id, material id, and depth.

Design D5 splits parity into two tiers: the **structural** outputs (this test)
and a perceptual tolerance on the converged shaded color (the separate 6.2 test).

The rays are identical across backends — the structural write reuses the real
``runFrame`` primary ray, both legs dispatch a single frame from
``accum_frame == 0``, and ``_pack_uniforms`` packs ``frameIndex = accum_frame``,
so ``createRNG(pixel, 0)`` produces the same AA jitter ⇒ the same rays. Any
divergence is therefore *pure FP cross-compiler codegen* (Slang→Metal vs
Slang→SPIRV: FMA contraction, rounding). That is irreducible and — exactly as D5
already concedes for color — not required to be bit-exact. It surfaces only at
silhouette / grazing pixels: a sub-ULP ray difference flips a near-tangent
hit/miss, and depth along a grazing ray is hypersensitive. The structural bar:

* geometry is traced on **both** backends (the core claim — Metal traced nothing
  before the read-before-disarm fix in ``read_structural_aov``);
* **hit-count parity** (``_COUNT_REL_MAX``) and **hit-mask agreement**
  (``_AGREE_FRAC_MIN``) — disagreements bounded to the thin edge band;
* instance id + material id **exact** on commonly-hit pixels (discrete, no FP);
* depth **robust-tolerant**: median tight (same surface), 95th pct loose
  (grazing outliers). All metrics are printed.

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

_RES = 64                  # square render; small fits the tool-buffer structural slots
# Structural-parity bar. The rays are identical across backends (same
# createRNG(pixel, frameIndex=0) AA jitter), so any divergence is pure FP cross-
# compiler codegen (Slang→Metal vs Slang→SPIRV — FMA contraction, rounding). That
# is irreducible and, per design D5, not required to be bit-exact (the same reason
# shaded color uses a perceptual bar in 6.2). It shows up only at silhouette /
# grazing pixels: a sub-ULP ray difference flips a near-tangent hit/miss, and depth
# along a grazing ray is hypersensitive. So: geometry traced on both, hit-count
# parity, IDs exact on commonly-hit pixels, depth robust-tolerant.
_COUNT_REL_MAX = 0.10      # |metal_hits - vulkan_hits| / vulkan_hits
_AGREE_FRAC_MIN = 0.90     # common hits / larger hit set (bounds the edge band)
_DEPTH_MEDIAN_REL_MAX = 1e-2   # robust (median) depth agreement on common hits
_DEPTH_P95_REL_MAX = 0.06      # 95th pct; absorbs grazing-pixel outliers


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
    m_n = int(m_hit.sum())
    edge_disagree = int(np.count_nonzero(m_hit != v_hit))

    both = m_hit & v_hit
    n_both = int(both.sum())
    larger = max(m_n, n_hit)
    count_rel = abs(m_n - n_hit) / max(1, n_hit)
    agree_frac = n_both / max(1, larger)

    # Depth deviation on the commonly-hit pixels (both backends agree it's a hit).
    if both.any():
        md, vd = m_aov[..., 3][both], v_aov[..., 3][both]
        rel = np.abs(md - vd) / (np.abs(vd) + 1e-8)
        depth_med_rel = float(np.median(rel))
        depth_p95_rel = float(np.percentile(rel, 95))
        depth_max_abs = float(np.max(np.abs(md - vd)))
    else:
        depth_med_rel = depth_p95_rel = depth_max_abs = float("nan")

    print(
        f"\n[6.1 structural] res={_RES} vulkan_hits={n_hit} metal_hits={m_n} "
        f"both={n_both} count_rel={count_rel:.3f} agree_frac={agree_frac:.3f} "
        f"hitmask_disagree={edge_disagree} px depth_med_rel={depth_med_rel:.2e} "
        f"depth_p95_rel={depth_p95_rel:.2e} depth_max_abs={depth_max_abs:.3e}"
    )

    # Geometry must be traced on BOTH backends (the fix's core claim — Metal
    # traced nothing before the read-before-disarm fix).
    assert n_hit > 0, "no primary-ray hits on the reference (Vulkan) backend"
    assert m_n > 0, "Metal traced no geometry (structural channel empty)"

    # Hit-count parity: the two backends agree on how many pixels see the head.
    assert count_rel < _COUNT_REL_MAX, (
        f"hit-count parity off: metal={m_n} vulkan={n_hit} (rel {count_rel:.3f})"
    )
    # Hit-mask agreement: the bulk of hits are common; disagreements are the thin
    # silhouette edge band where FP cross-compiler codegen flips near-tangent rays.
    assert agree_frac >= _AGREE_FRAC_MIN, (
        f"hit-mask agreement too low: {n_both}/{larger} = {agree_frac:.3f} "
        f"({edge_disagree} px disagree)"
    )

    # Instance id + material id exact on commonly-hit pixels (discrete integers,
    # no FP path — these must match exactly).
    assert np.array_equal(m_aov[..., 1][both], v_aov[..., 1][both]), "instance id mismatch"
    assert np.array_equal(m_aov[..., 2][both], v_aov[..., 2][both]), "material id mismatch"

    # Depth: robust agreement on common hits — median tight (same surface), 95th
    # pct loose (grazing-pixel outliers along near-tangent rays).
    assert depth_med_rel < _DEPTH_MEDIAN_REL_MAX, (
        f"median depth divergence too high: {depth_med_rel:.2e}"
    )
    assert depth_p95_rel < _DEPTH_P95_REL_MAX, (
        f"95th-pct depth divergence too high: {depth_p95_rel:.2e}"
    )
