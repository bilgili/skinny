"""Headless A/B parity: megakernel vs staged wavefront path tracer (task 9.1).

Renders the demo scene with the `path` integrator in both execution modes and
asserts the wavefront accumulation matches the megakernel. The path tracer is
Monte Carlo and MoltenVK is not bit-reproducible, so two independent renders of
the *same* mode differ by a measurable noise floor; the wavefront (a different
sample sequence of the same estimator) must match the megakernel no worse than
that floor — i.e. it is the same integral, not a different image.

The execution mode is fixed per renderer instance (CLI `--execution-mode`), so
each A/B builds a megakernel renderer (rendered twice → noise floor) and a
separate wavefront renderer, rather than toggling the mode at runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
DEMO_SCENE = PROJECT_ROOT / "assets" / "three_materials_demo.usda"

pytestmark = pytest.mark.gpu

WIDTH = HEIGHT = 96
WARMUP = 48


def _match(a, b):
    d = np.abs(a - b).max(axis=2)
    tol = 8e-3 + 0.04 * np.abs(b).max(axis=2)
    return float((d <= tol).mean())


def _load(execution_mode, *, integrator_index=0, stream_cap=None, bdpt_walk="fused"):
    """Build a headless renderer in the given (fixed) execution mode, pump the
    async USD load, and apply the integrator + optional stream cap. Returns
    (ctx, renderer); the caller owns cleanup of both."""
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
        tattoo_dir=TATTOO_DIR, usd_scene_path=DEMO_SCENE,
        execution_mode=execution_mode, bdpt_walk=bdpt_walk,
    )
    deadline = 200
    while deadline > 0 and (
        renderer._usd_scene is None or len(renderer._usd_scene.instances) < 3
    ):
        renderer.update(0.025)
        deadline -= 1
    assert renderer._scene_bindings is not None, "scene bindings not built"
    renderer.integrator_index = integrator_index
    if stream_cap is not None:
        renderer._wf_stream_cap = stream_cap
    return ctx, renderer


def _accumulate(renderer, frames=WARMUP):
    """Render `frames` accumulation frames from a clean start, return the
    linear-HDR accumulation image."""
    renderer._material_version += 1  # force a clean accumulation
    for _ in range(frames):
        renderer.update(0.04)
        renderer.render_headless()
    return renderer.read_accumulation()[:, :, :3].copy()


def test_wavefront_path_matches_megakernel():
    mega_ctx, mega = _load("megakernel", integrator_index=0)
    try:
        assert mega.pipeline is not None
        mega1 = _accumulate(mega)
        mega2 = _accumulate(mega)        # independent megakernel render → noise floor
    finally:
        mega.cleanup()
        mega_ctx.destroy()

    wave_ctx, wave = _load("wavefront", integrator_index=0)
    try:
        assert wave.pipeline is None, "wavefront must not build the megakernel"
        wavef = _accumulate(wave)        # staged wavefront path tracer
        assert wave.effective_execution_mode_index == 1, "wavefront not active"
        assert np.all(np.isfinite(wavef))
        assert float(wavef.max()) > 1e-2, "wavefront path render is black"
        # The demo is flat/graph materials only ⇒ the heavy catch-all shade
        # kernel (skin/python) is never compiled — flat scenes shade through the
        # small per-material flat kernel (the MoltenVK compile fix).
        assert wave._wavefront_path_pass.build_catchall is False, (
            "flat-only scene should not compile the catch-all shade kernel"
        )
    finally:
        wave.cleanup()
        wave_ctx.destroy()

    noise_floor = _match(mega1, mega2)
    match_wave = _match(wavef, mega1)
    assert noise_floor > 0.5, f"renderer too unstable to test ({noise_floor:.2f})"
    # The wavefront is the same estimator relocated across dispatches: it
    # must agree with the megakernel as well as two megakernel renders agree
    # with each other (both differ only by independent MC noise).
    assert match_wave >= noise_floor - 0.06, (
        f"wavefront path tracer diverges from the megakernel "
        f"(match {match_wave:.3f} vs megakernel-vs-megakernel floor "
        f"{noise_floor:.3f})"
    )


def test_wavefront_path_tiled_streaming():
    """Tiled streaming (task 6.2): with a small stream cap the frame is
    processed in many fixed-size tiles (incl. a partial tail), the path-state
    buffer is sized to the cap — not the pixel count — and the image still
    matches the megakernel."""
    from skinny.wavefront_layout import PATH_STATE_STRIDE

    cap = 1000  # WIDTH*HEIGHT = 9216 ⇒ 10 tiles, last partial

    mega_ctx, mega = _load("megakernel", integrator_index=0)
    try:
        assert mega.pipeline is not None
        mega1 = _accumulate(mega)
        mega2 = _accumulate(mega)
    finally:
        mega.cleanup()
        mega_ctx.destroy()

    wave_ctx, wave = _load("wavefront", integrator_index=0, stream_cap=cap)
    try:
        wavef = _accumulate(wave)
        # Path-state buffer is bounded by the stream cap, not the pixel count.
        assert wave._wf_path_state_buf.size == cap * PATH_STATE_STRIDE, (
            f"path-state buffer {wave._wf_path_state_buf.size} B != "
            f"cap {cap * PATH_STATE_STRIDE} B (should not scale with pixels)"
        )
        assert wave._wavefront_path_pass.stream_size == cap
        assert wave._wavefront_path_pass.num_pixels == WIDTH * HEIGHT
    finally:
        wave.cleanup()
        wave_ctx.destroy()

    noise_floor = _match(mega1, mega2)
    match_wave = _match(wavef, mega1)
    assert noise_floor > 0.5, f"renderer too unstable ({noise_floor:.2f})"
    assert match_wave >= noise_floor - 0.06, (
        f"tiled wavefront diverges from the megakernel "
        f"(match {match_wave:.3f} vs floor {noise_floor:.3f})"
    )


@pytest.mark.parametrize("walk_mode", ["fused", "eye", "eye_light"])
def test_wavefront_bdpt_matches_megakernel(walk_mode):
    """A/B parity for the bidirectional integrator (task 9.2): megakernel bdpt
    vs the staged wavefront bdpt, for each `--bdpt-walk` mode (fused = one
    walk kernel + connect compaction; eye / eye_light stage the eye / eye+light
    walks). All three are the same estimator, so each must match the megakernel
    no worse than the megakernel-vs-megakernel noise floor."""
    cap = 1000  # 10 tiles over the 96² frame — exercises tiled streaming for bdpt

    mega_ctx, mega = _load("megakernel", integrator_index=1)
    try:
        assert mega.pipeline is not None
        mega1 = _accumulate(mega)
        mega2 = _accumulate(mega)
    finally:
        mega.cleanup()
        mega_ctx.destroy()

    wave_ctx, wave = _load("wavefront", integrator_index=1, stream_cap=cap,
                           bdpt_walk=walk_mode)
    try:
        assert wave.WAVEFRONT_BDPT_SUPPORTED, "wavefront bdpt gated off"
        assert wave.pipeline is None, "wavefront must not build the megakernel"
        wavef = _accumulate(wave)
        assert wave.effective_execution_mode_index == 1, (
            "wavefront not active for bdpt (capability gate fell back)"
        )
        assert wave._wavefront_bdpt_pass.walk_mode == walk_mode
        assert np.all(np.isfinite(wavef))
        assert float(wavef.max()) > 1e-2, "wavefront bdpt render is black"
        # bdpt subpath buffers are bounded by the stream cap, not the pixel count.
        assert wave._wavefront_bdpt_pass.stream_size == cap
        assert wave._wavefront_bdpt_pass.num_pixels == WIDTH * HEIGHT
    finally:
        wave.cleanup()
        wave_ctx.destroy()

    noise_floor = _match(mega1, mega2)
    match_wave = _match(wavef, mega1)
    assert noise_floor > 0.5, f"renderer too unstable to test ({noise_floor:.2f})"
    assert match_wave >= noise_floor - 0.06, (
        f"wavefront bdpt ({walk_mode}) diverges from the megakernel "
        f"(match {match_wave:.3f} vs floor {noise_floor:.3f})"
    )
