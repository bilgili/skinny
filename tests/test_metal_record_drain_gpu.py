"""Metal wavefront record emission + live drain on a real device
(change metal-record-drain, tasks 1.5, 3.1–3.5).

Drives the flat emissive Cornell box headless on the native Metal backend and
asserts the delta-spec scenarios:

* the records build (SKINNY_METAL_RECORDS) compiles under the 31-slot cap and
  a records-enabled render drains valid records (count > 0, finite
  contributions, depth < REC_MAX_BOUNCES, parsed by the shipped reader);
* the Metal record stream is equivalent to the Vulkan wavefront stream for
  the same scene/config (same vertices + contributions, order-independent);
* with records off the render is bit-identical through an arm → disarm
  round-trip (the records build leaves no residue);
* the fully-on-Metal online loop runs: records drain → numpy trainer →
  `interop` weight handoff → `networkVersion` increments — no Vulkan device,
  no NFW1 file.

DANGER — compiles the wavefront stage pipelines in-process on Metal
(MTLCompilerService) and constructs a Vulkan (MoltenVK) device for the A/B.
Run ONLY through ``scripts/guarded_metal.sh`` with the compile gate + the
Vulkan SDK on the dylib path:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_WAVEFRONT_COMPILE=1 scripts/guarded_metal.sh -- \
        ./bin/python3.13 -m pytest tests/test_metal_record_drain_gpu.py -v -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from skinny.backend_select import metal_available

pytest.importorskip("slangpy")

pytestmark = pytest.mark.gpu

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SHADER_DIR = _PROJECT_ROOT / "src" / "skinny" / "shaders"
_SCENE = _PROJECT_ROOT / "assets" / "cornell_box_emissive.usda"
_NET_FIXTURE = _PROJECT_ROOT / "tests" / "data" / "cornell_neural.nfw1"
_COMPILE_GATE = "RUN_METAL_WAVEFRONT_COMPILE"

_RES = 64
_REC_MAX_BOUNCES = 6

requires_gate = pytest.mark.skipif(
    os.environ.get(_COMPILE_GATE) != "1",
    reason=(
        f"compiles the wavefront stage pipelines on Metal (MTLCompilerService); "
        f"set {_COMPILE_GATE}=1 and run under scripts/guarded_metal.sh"
    ),
)


def _require_metal():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")


def _make_renderer(ctx, proposals_token: str = "bsdf", neural_net: Path | None = None,
                   neural_handoff: str = "file", neural_trainer: str = "auto"):
    from skinny.renderer import Renderer

    r = Renderer(
        vk_ctx=ctx, shader_dir=_SHADER_DIR, execution_mode="wavefront",
        usd_scene_path=_SCENE, neural_handoff=neural_handoff,
        neural_trainer=neural_trainer,
    )
    r.integrator_index = 0  # path tracer (records are path-only)
    r.proposal_preset_index = r.proposal_preset_from_token(proposals_token)
    if neural_net is not None:
        r._neural_weights_path = str(neural_net)
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        r.update(0.016)
        if r._scene_bindings is not None and r._usd_scene is not None \
                and len(r._usd_scene.instances) >= 1:
            break
        time.sleep(0.02)
    assert r._scene_bindings is not None, "scene bindings never built"
    assert r.effective_execution_mode_index == 1, "wavefront mode not active"
    return r


def _render_pinned(r, frames: int) -> None:
    r._last_state_hash = None
    for i in range(frames):
        r.update(0.016)
        r.frame_index = i
        r.time_elapsed = 0.0
        r.render_headless()


def _drain_all(r, replay, frames: int) -> int:
    """Render `frames` pinned frames, draining after each (per-frame contract)."""
    total = 0
    r._last_state_hash = None
    for i in range(frames):
        r.update(0.016)
        r.frame_index = i
        r.time_elapsed = 0.0
        r.render_headless()
        total += r.drain_path_records_to_replay(replay, max_records_per_frame=1 << 18)
    return total


def _record_set(replay) -> np.ndarray:
    """All drained records as one array sorted order-independently."""
    recs = np.concatenate([np.asarray(a) for a in replay._chunks()]) \
        if hasattr(replay, "_chunks") else replay.all_records()
    flat = np.stack([
        recs["pos"][:, 0], recs["pos"][:, 1], recs["pos"][:, 2],
        recs["wi_local"][:, 0], recs["wi_local"][:, 1], recs["wi_local"][:, 2],
        recs["contrib"][:, 0], recs["contrib"][:, 1], recs["contrib"][:, 2],
        recs["depth"].astype(np.float64),
    ], axis=1)
    order = np.lexsort(flat.T[::-1])
    return flat[order]


class _ListReplay:
    """Minimal replay sink — keeps the raw record arrays."""

    def __init__(self):
        self.arrays = []

    def add(self, records):
        self.arrays.append(np.array(records, copy=True))

    def all_records(self) -> np.ndarray:
        import numpy as _np
        from skinny.sampling.path_records import RECORD_DTYPE
        if not self.arrays:
            return _np.zeros(0, dtype=RECORD_DTYPE)
        return _np.concatenate(self.arrays)

    def __len__(self):
        return int(sum(len(a) for a in self.arrays))


@requires_gate
def test_metal_records_drain_valid_and_match_vulkan():
    """Spec: Metal bounces produce parseable records; Metal ≡ Vulkan streams."""
    from skinny.metal_context import MetalContext
    from skinny.vk_context import VulkanContext

    _require_metal()
    frames = 4
    streams = {}
    for name, ctx_cls in (("metal", MetalContext), ("vulkan", VulkanContext)):
        ctx = ctx_cls(window=None, width=_RES, height=_RES)
        try:
            r = _make_renderer(ctx)
            r._wf_record_active = True   # arm the records build (no trainer needed)
            replay = _ListReplay()
            total = _drain_all(r, replay, frames)
            recs = replay.all_records()
            assert total > 0 and len(recs) == total, f"{name}: no records drained"
            assert np.isfinite(recs["contrib"]).all(), f"{name}: non-finite contrib"
            assert (recs["depth"] < _REC_MAX_BOUNCES).all(), f"{name}: depth overflow"
            assert np.isfinite(recs["pos"]).all() and np.isfinite(recs["wi_local"]).all()
            # upper-hemisphere flow domain
            assert (recs["wi_local"][:, 1] > 0).all(), f"{name}: wiLocal below hemisphere"
            streams[name] = _record_set(replay)
            if name == "metal":
                pass_obj = r._wavefront_path_pass
                assert getattr(pass_obj, "records_active", False) is True
                n_globals = len(list(
                    pass_obj._entries["wfPathShadeFlat"].program.layout.parameters))
                print(f"\n[metal records build] shadeFlat globals: {n_globals}; "
                      f"records drained: {total}")
        finally:
            ctx.destroy()

    m, v = streams["metal"], streams["vulkan"]
    assert len(m) == len(v), f"record counts differ: metal {len(m)} vs vulkan {len(v)}"
    # Cross-backend float contraction (MSL vs SPIR-V) drifts values in the
    # ~1e-4 range, which scrambles a row-wise lexsort pairing — the same reason
    # the wavefront shaded parity is perceptual, not bit-exact. Equivalence is
    # therefore asserted order-robustly: identical per-depth counts and
    # matching marginal distributions (per-column sorted quantiles) plus the
    # aggregate training weight.
    np.testing.assert_array_equal(
        np.bincount(m[:, 9].astype(int), minlength=_REC_MAX_BOUNCES),
        np.bincount(v[:, 9].astype(int), minlength=_REC_MAX_BOUNCES))
    for col in range(9):
        mc, vc = np.sort(m[:, col]), np.sort(v[:, col])
        np.testing.assert_allclose(mc, vc, rtol=5e-3, atol=5e-3,
                                   err_msg=f"marginal distribution col {col}")
    m_contrib, v_contrib = m[:, 6:9].mean(), v[:, 6:9].mean()
    assert abs(m_contrib - v_contrib) / max(abs(v_contrib), 1e-8) < 0.01, (
        f"mean contribution differs: metal {m_contrib} vs vulkan {v_contrib}")


@requires_gate
def test_metal_records_off_render_bit_identical_roundtrip():
    """Spec: records-off render unchanged — arm → disarm reproduces the
    accumulation bit-identically and releases the records build."""
    from skinny.metal_context import MetalContext

    _require_metal()
    frames = 8
    ctx = MetalContext(window=None, width=_RES, height=_RES)
    try:
        r = _make_renderer(ctx)
        _render_pinned(r, frames)
        before, n0 = r.read_accumulation_hdr()
        before = np.array(before, copy=True)
        assert getattr(r._wavefront_path_pass, "records_active", False) is False

        r._wf_record_active = True            # arm: rebuild with records
        replay = _ListReplay()
        assert _drain_all(r, replay, frames) > 0
        assert r._wavefront_path_pass.records_active is True

        r._wf_record_active = False           # disarm: rebuild without records
        _render_pinned(r, frames)
        after, n1 = r.read_accumulation_hdr()
        assert n0 == n1
        assert r._wavefront_path_pass.records_active is False
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    finally:
        ctx.destroy()


@requires_gate
def test_fully_on_metal_online_loop():
    """Spec: records drain → numpy trainer → interop weight handoff →
    networkVersion increments — no Vulkan device, no NFW1 file."""
    from skinny.metal_context import MetalContext
    from skinny.sampling.neural_handoff_interop_metal import MetalSharedWeightPublisher

    if not _NET_FIXTURE.is_file():
        pytest.skip("trained cornell net fixture missing")
    _require_metal()
    ctx = MetalContext(window=None, width=_RES, height=_RES)
    try:
        r = _make_renderer(ctx, proposals_token="bsdf,neural", neural_net=_NET_FIXTURE,
                           neural_handoff="interop", neural_trainer="cpu")
        pub = r.enable_online_training(capacity=200_000)
        assert isinstance(pub, MetalSharedWeightPublisher)
        # This loop drives training itself (manual `online_train_and_publish` per
        # frame), so stop the auto-started daemon trainer to keep the loop the
        # sole, deterministic driver.
        r._stop_trainer_thread()
        try:
            rng = np.random.default_rng(0)
            drained = 0
            versions = []
            r._last_state_hash = None
            for i in range(6):
                r.update(0.016)
                r.frame_index = i
                r.time_elapsed = 0.0
                r.render_headless()           # frame-end swap runs inside
                drained += r.online_training_tick()
                if len(r._neural_replay) > 0:
                    r.online_train_and_publish(rng)
                versions.append(r._neural_publisher.current_version())
            assert drained > 0, "online tick drained no records on Metal"
            assert versions[-1] >= 1, f"network version never advanced: {versions}"
            assert r._wf_record_active and r._wavefront_path_pass.records_active
            # interop publisher → no NFW1 files anywhere under the default dir
            assert not list(Path(".skinny_neural").glob("*.nfw1")) \
                if Path(".skinny_neural").is_dir() else True
            print(f"\n[online loop] drained={drained} versions={versions}")
        finally:
            r.disable_online_training()
    finally:
        ctx.destroy()
