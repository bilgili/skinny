"""Headless A/B parity: neural directional proposal on the Metal wavefront
path vs Vulkan (change metal-wavefront-parity, task 6.3).

Drives the flat emissive Cornell box (the canonical directional-proposal
stress scene from ``tests/test_neural_headless.py`` — all-flat so the
wavefront flat shade, the only kernel that consumes the neural sample, covers
the whole frame) with the trained per-scene net fixture
(``tests/data/cornell_neural.nfw1``), and asserts:

* **Unbiasedness on Metal** — ``{bsdf,neural}`` converges to the same global
  energy as ``{bsdf}`` (the mixture-MIS estimator is unbiased regardless of
  proposal quality; the spatial mean cancels MC noise, exposing bias).
* **Metal ≡ Vulkan with neural on** — the converged ``{bsdf,neural}`` images
  agree across backends within the established wavefront perceptual
  tolerances (fp32 weights on this host: ``_effective_neural_config()``
  degrades to fp32 on both backends, so the structural criterion applies).
  GPU-side pdf parity on fixed inputs is pinned transitively: the CPU mirror
  vs PyTorch goldens (``tests/test_neural_parity.py``) fixes the math, and
  this render-level A/B fixes Metal's GPU evaluation against Vulkan's.
* **Default selection unchanged + release on deselection** — converging the
  default ``{bsdf}`` preset, round-tripping through the neural preset (pass
  rebuild with ``SKINNY_METAL_NEURAL=1``), and back, reproduces the original
  accumulation **bit-identically**, and the neural pass object is released.

DANGER — compiles the wavefront stage pipelines in-process on Metal
(MTLCompilerService) and constructs a Vulkan device. Run ONLY through
``scripts/guarded_metal.sh`` with the compile gate + the Vulkan SDK on the
dylib path:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_WAVEFRONT_COMPILE=1 PYTHONPATH=$PWD/src TIMEOUT_S=900 \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest tests/test_metal_neural_ab.py -q -s
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
_SCENE = _PROJECT_ROOT / "assets" / "cornell_box_emissive.usda"
_NET_FIXTURE = _PROJECT_ROOT / "tests" / "data" / "cornell_neural.nfw1"
_COMPILE_GATE = "RUN_METAL_WAVEFRONT_COMPILE"

_RES = 64
_SAMPLES = 96
_REL_MSE_MAX = 0.02
_CORR_MIN = 0.98
_BIAS_REL_MAX = 0.05  # global-energy tolerance from test_neural_unbiased_matches_bsdf

requires_gate = pytest.mark.skipif(
    os.environ.get(_COMPILE_GATE) != "1",
    reason=(
        f"compiles the wavefront stage pipelines on Metal (MTLCompilerService); "
        f"set {_COMPILE_GATE}=1 and run under scripts/guarded_metal.sh"
    ),
)


def _make_renderer(ctx, proposals_token: str, neural_net: Path | None = None):
    from skinny.renderer import Renderer

    r = Renderer(
        vk_ctx=ctx, shader_dir=_SHADER_DIR, execution_mode="wavefront",
        hdr_dir=_HDR_DIR if _HDR_DIR.is_dir() else None,
        tattoo_dir=_TATTOO_DIR if _TATTOO_DIR.is_dir() else None,
        usd_scene_path=_SCENE,
    )
    r.integrator_index = 0  # path tracer
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


def _converge(r, frames: int = _SAMPLES, pin_frames: bool = False) -> np.ndarray:
    """Accumulate ``frames`` samples and return mean linear radiance.

    ``pin_frames`` fixes ``frame_index``/``time_elapsed`` per accumulation
    step so the render becomes a pure function of the step index — required
    for the bit-identity round-trip (the global frame counter otherwise
    advances the per-sample RNG stream between two converge windows)."""
    # Force a clean accumulation restart through the state-hash path: a bare
    # `accum_frame = 0` is bumped back to 1 by the next update()'s unchanged
    # hash, so the first frame would BLEND into stale accumulation content
    # instead of overwriting it.
    r._last_state_hash = None
    for i in range(frames):
        r.update(0.016)
        if pin_frames:
            r.frame_index = i
            r.time_elapsed = 0.0
        r.render_headless()
    arr, n = r.read_accumulation_hdr()
    assert n > 0, "no samples accumulated"
    return (arr[..., :3] / max(1, n)).astype(np.float64)


def _rel_mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    mb = float(b.mean())
    return abs(float(a.mean()) - mb) / max(mb, 1e-8)


def _render_pair(ctx_cls, w, h):
    """Converge {bsdf} and {bsdf,neural} (trained fixture) on one backend.
    Returns (ref, neural, neural_pass_seen)."""
    ctx = ctx_cls(window=None, width=w, height=h)
    try:
        r = _make_renderer(ctx, "bsdf")
        ref = _converge(r)
        r.cleanup()
        r = _make_renderer(ctx, "bsdf,neural", neural_net=_NET_FIXTURE)
        assert r._neural_active(), "bsdf,neural did not activate the neural pass"
        neural = _converge(r)
        npass = r._neural_pass
        assert npass is not None, "neural pre-pass was not constructed"
        assert npass.network_version == r._neural_network_version, \
            "neural pass lost the host network-version stamp"
        r.cleanup()
    finally:
        ctx.destroy()
    return ref, neural


@requires_gate
def test_metal_neural_unbiased_and_matches_vulkan():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    if not _SCENE.exists():
        pytest.skip(f"scene asset not found: {_SCENE}")
    if not _NET_FIXTURE.exists():
        pytest.skip(f"trained net fixture not found: {_NET_FIXTURE}")
    try:
        from skinny.metal_context import MetalContext
        from skinny.vk_context import VulkanContext
    except OSError as exc:
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")

    m_ref, m_neural = _render_pair(MetalContext, _RES, _RES)
    v_ref, v_neural = _render_pair(VulkanContext, _RES, _RES)

    for name, img in (("metal-bsdf", m_ref), ("metal-neural", m_neural),
                      ("vulkan-bsdf", v_ref), ("vulkan-neural", v_neural)):
        assert np.isfinite(img).all(), f"{name} produced non-finite radiance"
        assert img.max() > 0, f"{name} produced an empty frame"

    # Unbiasedness on Metal (and Vulkan, pinning the reference is still valid).
    m_bias = _rel_mean_diff(m_neural, m_ref)
    v_bias = _rel_mean_diff(v_neural, v_ref)
    print(f"[metal-neural-ab] bias: metal={m_bias:.4f} vulkan={v_bias:.4f}")
    assert m_bias < _BIAS_REL_MAX, \
        f"Metal {{bsdf,neural}} biased vs {{bsdf}}: rel={m_bias:.4f}"
    assert v_bias < _BIAS_REL_MAX, \
        f"Vulkan {{bsdf,neural}} biased vs {{bsdf}}: rel={v_bias:.4f}"

    # Cross-backend structural agreement with neural on (fp32 on this host).
    rel_mse = float(np.mean((m_neural - v_neural) ** 2)
                    / (np.mean(v_neural ** 2) + 1e-8))
    corr = float(np.corrcoef(m_neural.ravel(), v_neural.ravel())[0, 1])
    exact = float(np.mean(m_neural == v_neural))
    print(f"[metal-neural-ab] cross: rel_mse={rel_mse:.5f} corr={corr:.5f} "
          f"exact-px={exact:.4f}")
    assert rel_mse < _REL_MSE_MAX, f"neural parity rel-MSE too high: {rel_mse}"
    assert corr > _CORR_MIN, f"neural parity correlation too low: {corr}"


@requires_gate
def test_metal_neural_default_selection_unchanged():
    """Round-tripping the neural preset must not perturb the default {bsdf}
    render (byte-identical accumulation) and must release the neural pass on
    deselection (spec: 'Deselection releases')."""
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    if not _SCENE.exists():
        pytest.skip(f"scene asset not found: {_SCENE}")
    from skinny.metal_context import MetalContext

    ctx = MetalContext(window=None, width=_RES, height=_RES)
    try:
        r = _make_renderer(ctx, "bsdf")
        before = _converge(r, frames=16, pin_frames=True)

        # Select neural (rebuild with SKINNY_METAL_NEURAL=1) and render a bit.
        r.proposal_preset_index = r.proposal_preset_from_token("bsdf,neural")
        for _ in range(4):
            r.update(0.016)
            r.render_headless()
        assert r._neural_pass is not None, "neural pass did not build on select"

        # Deselect → rebuild without neural; the pass object must be released.
        r.proposal_preset_index = r.proposal_preset_from_token("bsdf")
        for _ in range(2):
            r.update(0.016)
            r.render_headless()
        assert r._neural_pass is None, "neural pass not released on deselection"
        assert r._wavefront_path_pass._neural is None, \
            "wavefront pass still holds the neural hook after deselection"

        after = _converge(r, frames=16, pin_frames=True)
        r.cleanup()
    finally:
        ctx.destroy()

    assert np.array_equal(before, after), (
        "default {bsdf} accumulation changed after a neural select/deselect "
        "round-trip — the rebuild is not bit-identical"
    )
