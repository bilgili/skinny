"""Headless gate for the neural training-record dump (task 5.1).

Drives the flat Cornell box (the canonical directional-proposal scene — strong
indirect / colour bleed), runs `Renderer.dump_path_records`, and validates the
`.nrec` file the offline `spline_flow` trainer consumes: records are finite,
upper-hemisphere, non-negative, inside the scene AABB, and carry real radiance.

GPU: needs the headless Vulkan/MoltenVK runtime + the build venv. Run with:
  VULKAN_SDK=.../macOS DYLD_LIBRARY_PATH=$VULKAN_SDK/lib \
  PYTHONPATH=src <py3.13> -m pytest tests/test_record_dump.py -q -s
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
SCENE = PROJECT_ROOT / "assets" / "cornell_box_emissive.usda"

pytestmark = pytest.mark.gpu

WIDTH = HEIGHT = 64
WARMUP = 200


def _load(execution_mode: str, proposals_token: str):
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
        tattoo_dir=TATTOO_DIR, usd_scene_path=SCENE,
        execution_mode=execution_mode,
    )
    renderer.proposal_preset_index = renderer.proposal_preset_from_token(proposals_token)
    deadline = WARMUP
    while deadline > 0 and (
        renderer._usd_scene is None
        or len(renderer._usd_scene.instances) < 1
        or renderer._scene_bindings is None
    ):
        renderer.update(0.025)
        deadline -= 1
    assert renderer._scene_bindings is not None, "scene bindings never built"
    return ctx, renderer


def _check_records(out, total):
    from skinny.sampling.path_records import read_records

    recs, bmin, bext = read_records(out)
    assert len(recs) == total, (len(recs), total)
    assert total > 0, "no records emitted"
    assert np.all(np.isfinite(recs["contrib"])), "non-finite contribution"
    assert np.all(np.isfinite(recs["wi_local"])), "non-finite wiLocal"
    # The flow's domain is the upper hemisphere (y = N·wi > 0).
    assert np.all(recs["wi_local"][:, 1] > 0.0), "wiLocal below the hemisphere"
    # wiLocal is a unit direction.
    norms = np.linalg.norm(recs["wi_local"].astype(np.float64), axis=1)
    assert np.allclose(norms, 1.0, atol=2e-3), f"wiLocal not unit: {norms.min()}..{norms.max()}"
    assert np.all(recs["contrib"] >= 0.0), "negative training weight"
    assert float(recs["contrib"].sum()) > 0.0, "all-zero radiance (light never reached)"
    # Positions must sit inside the scene AABB the header advertises.
    p = recs["pos"].astype(np.float64)
    assert np.all(p >= bmin - 1e-2) and np.all(p <= bmin + bext + 1e-2), "record outside AABB"
    return recs


def test_record_dump_megakernel(tmp_path):
    """5.1: the dump writes a valid `.nrec` on the megakernel backend."""
    ctx, r = _load("megakernel", "bsdf")
    try:
        out = tmp_path / "cornell_mega.nrec"
        total = r.dump_path_records(str(out), num_frames=8)
        recs = _check_records(out, total)
        nz = float((recs["contrib"].sum(axis=1) > 0).mean())
        print(f"\n[5.1 mega] {total} records, {nz*100:.1f}% carry radiance, "
              f"depths {recs['depth'].min()}..{recs['depth'].max()}")
    finally:
        r.cleanup()
        ctx.destroy()


def test_record_dump_wavefront(tmp_path):
    """5.1: the dump is backend-independent — it builds its own megakernel record
    pipeline and reuses the scene set even when the renderer is in wavefront
    mode (the inference backend)."""
    ctx, r = _load("wavefront", "bsdf")
    try:
        out = tmp_path / "cornell_wave.nrec"
        total = r.dump_path_records(str(out), num_frames=8)
        _check_records(out, total)
        print(f"\n[5.1 wave] {total} records from a wavefront-mode renderer")
    finally:
        r.cleanup()
        ctx.destroy()


if __name__ == "__main__":  # pragma: no cover - manual harness
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
