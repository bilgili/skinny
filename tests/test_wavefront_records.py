"""Wavefront-native path-record emission (change wavefront-native-path-records).

The wavefront path integrator emits the shipped ``PathRecord`` training stream
directly (per-lane vertex stack + terminate-time backward attribution), so the
live online-training drain reads records off the wavefront backend with NO
``mainImageRecord`` megakernel dispatch — removing the 2 s-TDR / ~400 s-compile
megakernel seam that loses the device on NVIDIA/Windows.

Host tests (no GPU): the RecVertex layout mirror.

GPU tests (``pytest.mark.gpu``; need the headless Vulkan runtime + build venv):
  4.1  drain real wavefront records end to end (the test the megakernel drain
       could not run on this box) + the record stream is well-formed and
       byte-compatible with the shipped reader / offline ``.nrec`` dump.
  4.3  default-render invariance — recordMode off vs on is bit-identical
       (recording observes the path, never alters transport).
       Plus: the record-source resolver picks wavefront vs megakernel correctly.

  4.2  parity against the megakernel ``.nrec`` dump runs only where the
       megakernel record pipeline is usable (it ~400 s-compiles / device-losts
       under the 2 s Windows TDR), so it is gated behind
       ``SKINNY_RUN_MEGAKERNEL_PARITY=1``.

Run:  VULKAN_SDK=... PYTHONPATH=src <py3.13> -m pytest tests/test_wavefront_records.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
SCENE = PROJECT_ROOT / "assets" / "cornell_box_emissive.usda"

WIDTH = HEIGHT = 96
WARMUP = 300


# ── Host: RecVertex layout mirror (locked to wf_records.slang) ─────────────

def test_rec_vertex_layout_mirror():
    """The host RecVertex mirror must match the scalar-layout Slang struct: six
    float3 (pos/normal/wo/wiLocal/L_k/beta_in) + one uint depth = 76 B tight.
    The path pass sizes the record-stack buffer against this stride."""
    from skinny import wavefront_layout as wl

    assert wl.REC_MAX_BOUNCES == 6
    assert wl.REC_VERTEX_STRIDE == 76
    assert wl.rec_vertex_size() == 76
    names = [n for n, _ in wl.REC_VERTEX_FIELDS]
    assert names == ["pos", "normal", "wo", "wiLocal", "L_k", "beta_in", "depth"]
    # Lockstep with the path pass's class constant.
    from skinny.vk_wavefront import WavefrontPathPass
    assert WavefrontPathPass.REC_VERTEX_STRIDE == wl.REC_VERTEX_STRIDE


# ── GPU helpers ────────────────────────────────────────────────────────────

def _load_wavefront(width=WIDTH, height=HEIGHT):
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=width, height=height)
    r = Renderer(vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
                 tattoo_dir=TATTOO_DIR, usd_scene_path=SCENE,
                 execution_mode="wavefront")
    deadline = WARMUP
    while deadline > 0 and (
        r._usd_scene is None or len(r._usd_scene.instances) < 1
        or r._scene_bindings is None
    ):
        r.update(0.025)
        deadline -= 1
    assert r._scene_bindings is not None, "scene bindings never built"
    return ctx, r


# ── GPU 4.1: drain real wavefront records, no megakernel ───────────────────

@pytest.mark.gpu
def test_wavefront_drain_end_to_end():
    from skinny.sampling.neural_replay import ReplayBuffer
    from skinny.sampling.path_records import RECORD_STRIDE, records_from_buffer

    ctx, r = _load_wavefront()
    try:
        # auto resolves to the wavefront source (wavefront execution + path).
        assert r._resolve_record_source() == "wavefront"

        r._wf_record_active = True
        cap = r._ensure_wf_record_drain()
        assert cap > 0

        r.update(0.04)
        r.render_headless()                 # wavefront emits records → bindings 36/37

        replay = ReplayBuffer(capacity=1_000_000)
        n = r.drain_path_records_to_replay(replay)
        assert n > 0, "no records drained — wavefront emission is dead"

        raw = r._drain_buffer.download_sync(min(n, cap) * RECORD_STRIDE)
        arr = records_from_buffer(raw, min(n, cap))
        # Guideable guard: sampled dir in the upper hemisphere of the HitInfo frame.
        assert (arr["wi_local"][:, 1] > 0).all()
        # Backward attribution drops non-finite weights.
        assert np.isfinite(arr["contrib"]).all()
        # Bounded per-lane stack.
        assert int(arr["depth"].max()) < 6
        # Positive training weight (clamped ≥ 0; some exactly 0 are fine).
        assert (arr["contrib"] >= 0).all()

        # Counter resets per frame: a second render+drain produces a fresh batch.
        r.update(0.04)
        r.render_headless()
        n2 = r.drain_path_records_to_replay(replay)
        assert n2 > 0
    finally:
        r.cleanup()
        ctx.destroy()


@pytest.mark.gpu
def test_record_stream_format_roundtrip(tmp_path):
    """The wavefront record stream is byte-for-byte the shipped ``.nrec`` layout:
    drained records survive a ``.nrec`` write→read roundtrip unchanged."""
    from skinny.sampling.neural_replay import ReplayBuffer
    from skinny.sampling.path_records import (
        RECORD_STRIDE,
        pack_header,
        read_records,
        records_from_buffer,
    )

    ctx, r = _load_wavefront()
    try:
        r._wf_record_active = True
        cap = r._ensure_wf_record_drain()
        r.update(0.04)
        r.render_headless()
        n = r.drain_path_records_to_replay(ReplayBuffer(capacity=1_000_000))
        assert n > 0
        raw = r._drain_buffer.download_sync(min(n, cap) * RECORD_STRIDE)
        arr = records_from_buffer(raw, min(n, cap))

        bmin, bext = r._neural_scene_bounds()
        out = tmp_path / "wf.nrec"
        out.write_bytes(pack_header(bmin, bext) + arr.tobytes())
        back, rmin, rext = read_records(out)
        assert back.shape == arr.shape
        assert (back["wi_local"] == arr["wi_local"]).all()
        assert (back["contrib"] == arr["contrib"]).all()
        assert (back["depth"] == arr["depth"]).all()
    finally:
        r.cleanup()
        ctx.destroy()


# ── GPU 4.3: default-render invariance + source resolution ─────────────────

@pytest.mark.gpu
def test_record_mode_invariance():
    """recordMode off vs on must produce a bit-identical image: recording reads
    the path but never consumes RNG or mutates transport. Same seed, accum 0."""
    ctx, r = _load_wavefront()
    try:
        r.update(0.04)
        fi = int(r.frame_index)

        r._wf_record_active = False
        r.frame_index = fi
        r.accum_frame = 0
        off = r.render_headless()

        r._wf_record_active = True
        r._ensure_wf_record_drain()
        r.frame_index = fi
        r.accum_frame = 0
        on = r.render_headless()

        assert off == on, "recordMode changed the rendered image (biased the render)"
    finally:
        r.cleanup()
        ctx.destroy()


@pytest.mark.gpu
def test_record_source_resolution():
    ctx, r = _load_wavefront()
    try:
        # auto on wavefront + path → wavefront.
        r._record_source = "auto"
        r.integrator_index = 0
        assert r._resolve_record_source() == "wavefront"
        # auto on bdpt → megakernel (wavefront bdpt records are out of scope).
        r.integrator_index = 1
        assert r._resolve_record_source() == "megakernel"
        r.integrator_index = 0
        # explicit overrides win.
        r._record_source = "megakernel"
        assert r._resolve_record_source() == "megakernel"
        r._record_source = "wavefront"
        assert r._resolve_record_source() == "wavefront"
    finally:
        r.cleanup()
        ctx.destroy()


# ── GPU 4.2: parity vs the megakernel .nrec dump (gated; TDR-unsafe box) ────

@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("SKINNY_RUN_MEGAKERNEL_PARITY") != "1",
    reason="megakernel record pipeline ~400 s-compiles / device-losts under the "
           "Windows TDR; set SKINNY_RUN_MEGAKERNEL_PARITY=1 on a non-TDR box",
)
def test_wavefront_megakernel_parity(tmp_path):
    """The wavefront record stream and the megakernel ``.nrec`` dump describe the
    same training signal: same scene + generator → statistically equivalent
    contribution means (the wavefront path uses Russian roulette, so the match
    is in expectation, not byte-exact)."""
    from skinny.sampling.neural_replay import ReplayBuffer
    from skinny.sampling.path_records import RECORD_STRIDE, read_records, records_from_buffer

    ctx, r = _load_wavefront()
    try:
        # Megakernel dump (the reference).
        dump = tmp_path / "mega.nrec"
        r.dump_path_records(str(dump), num_frames=32)
        mega, _, _ = read_records(dump)
        assert len(mega) > 0

        # Wavefront stream over the same number of frames.
        r._record_source = "wavefront"
        r._wf_record_active = True
        cap = r._ensure_wf_record_drain()
        replay = ReplayBuffer(capacity=4_000_000)
        wf_chunks = []
        for _ in range(32):
            r.update(0.04)
            r.render_headless()
            n = r.drain_path_records_to_replay(replay)
            if n:
                raw = r._drain_buffer.download_sync(min(n, cap) * RECORD_STRIDE)
                wf_chunks.append(records_from_buffer(raw, min(n, cap)))
        wf = np.concatenate(wf_chunks)
        assert len(wf) > 0

        # Aggregate contribution means agree within a loose statistical tolerance.
        mega_mean = float(np.mean(mega["contrib"]))
        wf_mean = float(np.mean(wf["contrib"]))
        assert abs(mega_mean - wf_mean) <= 0.25 * max(mega_mean, 1e-3)
    finally:
        r.cleanup()
        ctx.destroy()
