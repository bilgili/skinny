"""Loaded-scene headless gates for the neural directional proposal.

Unblocks the 1a/1c verification that needs a REAL scene (the wavefront path pass
+ neural pre-pass only build once scene bindings exist — an empty-scene smoke
can't exercise them). Drives a flat Cornell box (strong indirect / colour bleed —
where a directional proposal matters), converges to the linear-HDR accumulation
image, and compares means:

  6.1  default {bsdf} megakernel ≡ wavefront   — my seam changes didn't regress
       either backend ({bsdf} takes the fast path; the wavefront flat shade reads
       a valid==0 neural record → no behavioural change).
  4.3/6.2  {bsdf,neural} (dummy net) converges to the SAME image as {bsdf} — the
       mixture-MIS estimator is UNBIASED regardless of the proposal's quality,
       which is exactly what the 1a plumbing milestone proves (the dummy net is a
       valid-but-poor proposal). Also asserts the neural pre-pass actually builds.

GPU: needs the headless Vulkan/MoltenVK runtime + the build venv. Run with:
  VULKAN_SDK=.../macOS DYLD_LIBRARY_PATH=$VULKAN_SDK/lib \
  PYTHONPATH=src <py3.13> -m pytest tests/test_neural_headless.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
# Flat Cornell box + emissive area light: indirect-dominated, all-flat materials
# (so the wavefront flat shade kernel — the only path that runs neural — covers
# the whole frame), the canonical directional-proposal stress scene.
SCENE = PROJECT_ROOT / "assets" / "cornell_box_emissive.usda"

pytestmark = pytest.mark.gpu

WIDTH = HEIGHT = 96
WARMUP = 200          # update() pumps allowed for async USD load + bindings build
CONVERGE_FRAMES = 64  # accumulation frames for the A/B means


def _load(execution_mode: str, proposals_token: str):
    """Build a renderer on `SCENE`, select the proposal set, and pump update()
    until the scene bindings exist (async USD load). Returns (ctx, renderer)."""
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


def _converge(renderer, frames: int = CONVERGE_FRAMES) -> np.ndarray:
    """Accumulate `frames` samples and return the mean linear-HDR RGB (H, W, 3)."""
    for _ in range(frames):
        renderer.update(0.04)
        renderer.render_headless()
    img, samples = renderer.read_accumulation_hdr()
    assert samples > 0, "no samples accumulated"
    return img[..., :3].astype(np.float64) / float(samples)


def _mean_luma(rgb: np.ndarray) -> float:
    return float(rgb.mean())


def _rel_mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Global energy difference |mean(a) - mean(b)| / mean(b) — noise-robust:
    Monte-Carlo noise is zero-mean so it cancels in the spatial mean, leaving any
    systematic BIAS. The discriminating unbiasedness statistic."""
    mb = _mean_luma(b)
    return abs(_mean_luma(a) - mb) / max(mb, 1e-8)


def test_neural_pass_builds_on_loaded_scene():
    """4.3: with {bsdf,neural} on the wavefront backend the neural pre-pass is
    built and hooked into the path pass."""
    ctx, r = _load("wavefront", "bsdf,neural")
    try:
        assert r._neural_active(), "neural should be active (wavefront + bit2)"
        r.update(0.04)
        r.render_headless()  # records the bounce loop → builds the pass
        assert r._wavefront_path_pass is not None, "wavefront path pass not built"
        assert r._neural_pass is not None, "neural pre-pass not built"
        assert r._wavefront_path_pass._neural is r._neural_pass, "pass not hooked"
        frame = np.frombuffer(r.render_headless(), dtype=np.uint8)
        assert int(frame.max()) > 0, "{bsdf,neural} produced an all-black frame"
    finally:
        r.cleanup()
        ctx.destroy()


def test_default_bsdf_megakernel_wavefront_parity():
    """6.1: default {bsdf} converges to the same image on both backends — my seam
    changes regressed neither (the {bsdf} fast path is untouched)."""
    ctx_m, r_m = _load("megakernel", "bsdf")
    try:
        mega = _converge(r_m)
    finally:
        r_m.cleanup()
        ctx_m.destroy()
    ctx_w, r_w = _load("wavefront", "bsdf")
    try:
        wave = _converge(r_w)
    finally:
        r_w.cleanup()
        ctx_w.destroy()
    rel = _rel_mean_diff(wave, mega)
    print(f"\n[6.1] {{bsdf}} mega-vs-wavefront rel-mean-diff = {rel:.4f} "
          f"(mega={_mean_luma(mega):.4f} wave={_mean_luma(wave):.4f})")
    assert rel < 0.03, f"megakernel/wavefront {{bsdf}} energy mismatch: rel={rel:.4f}"


def test_neural_unbiased_matches_bsdf():
    """4.3/6.2: {bsdf,neural} (dummy net) converges to the same image as {bsdf} —
    the mixture-MIS estimator is unbiased regardless of proposal quality."""
    ctx_b, r_b = _load("wavefront", "bsdf")
    try:
        ref = _converge(r_b)
    finally:
        r_b.cleanup()
        ctx_b.destroy()
    ctx_n, r_n = _load("wavefront", "bsdf,neural")
    try:
        neural = _converge(r_n)
    finally:
        r_n.cleanup()
        ctx_n.destroy()
    rel = _rel_mean_diff(neural, ref)
    print(f"\n[4.3/6.2] {{bsdf,neural}} vs {{bsdf}} rel-mean-diff = {rel:.4f} "
          f"(ref={_mean_luma(ref):.4f} neural={_mean_luma(neural):.4f})")
    # Global-energy (bias) tolerance: a biased mixture would shift the mean; MC
    # noise cancels in the spatial mean. Looser than 6.1 — the dummy net's high
    # per-pixel variance leaves a larger finite-sample mean wobble.
    assert rel < 0.05, f"{{bsdf,neural}} biased vs {{bsdf}} reference: rel={rel:.4f}"


if __name__ == "__main__":  # pragma: no cover - manual harness
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
