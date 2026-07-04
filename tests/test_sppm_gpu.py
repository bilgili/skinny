"""GPU correctness tests for the SPPM integrator (change photon-mapping-sppm).

Renders the cornell-box-sphere scene under the SPPM integrator and the path
tracer (both wavefront) and asserts the SPPM estimate is energy-consistent with
the path tracer — the end-to-end guard that the eye/grid/photon/update pipeline
deposits the indirect/caustic term once and disjoint from the NEE direct term
(no double-counting, no missing indirect). Implicitly exercises the counting-sort
grid + atomic deposit: a broken grid would mis-deposit and skew the energy.

Marked ``gpu`` (needs a Vulkan runtime + VULKAN_SDK on the dylib path) and skips
if the scene asset is absent (e.g. a bare worktree without the asset tree).

History: these two ``VulkanContext`` tests silently failed to *build* on
MoltenVK for months — the megakernel referenced ``volumeDensity`` (set 0,
binding 26) which was missing from the Vulkan descriptor-set layout, so every
raw-Vulkan render died at pipeline create with ``SPIR-V to MSL conversion
error: nullptr`` (VUID-07988). Fixed in change fix-vulkan-volume-density-binding
(the failure was NOT a big-kernel SPIRV-Cross limit in the ``wfSppm*`` entries —
all eight convert to MSL cleanly; it was the megakernel's undeclared binding).
With the gate running again, the energy test immediately caught a genuine,
cross-backend SPPM diffuse-indirect energy deficit — zero photon flux from
undefined Stage-2 rich inputs at deposit time — fixed in change
fix-sppm-bathroom-black-walls (VisiblePoint stores the rich inputs; hostless
FlatHitMat⊆VisiblePoint locks in tests/test_sppm_state.py keep it fixed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
SCENE = PROJECT_ROOT / "assets" / "cornell_box_sphere.usda"


def _have_vulkan() -> bool:
    try:
        import vulkan as vk  # noqa: F401
        return True
    except Exception:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")
needs_scene = pytest.mark.skipif(not SCENE.exists(), reason="cornell_box_sphere asset absent")

_W = _H = 128


def _render(integrator_index: int, frames: int = 16):
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=_W, height=_H)
    try:
        r = Renderer(vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
                     tattoo_dir=TATTOO_DIR, usd_scene_path=SCENE)
        for _ in range(200):
            if r._usd_scene is not None and len(r._usd_scene.instances) >= 1:
                break
            r.update(0.025)
        r.execution_mode_index = 1  # wavefront
        r.integrator_index = integrator_index
        r.accum_frame = 0
        raw = None
        for _ in range(frames):
            r.update(0.04)
            raw = r.render_headless()
        sppm_built = r._wavefront_sppm_pass is not None
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(_H, _W, 4)[:, :, :3].astype(np.float32)
        return arr, sppm_built
    finally:
        try:
            r.cleanup()
        except Exception:
            pass
        ctx.destroy()


@needs_vulkan
@needs_scene
def test_sppm_builds_and_renders_finite():
    arr, built = _render(2)  # INTEGRATOR_SPPM
    assert built, "SPPM pass was not constructed (fell back?)"
    assert np.isfinite(arr).all(), "SPPM produced non-finite pixels"
    assert (arr.sum(2) > 0).mean() > 0.5, "SPPM frame is mostly black"


@needs_vulkan
@needs_scene
def test_sppm_energy_matches_path_tracer():
    # (Was xfail(strict): the SPPM diffuse-indirect energy deficit — photon
    # deposits evaluated with undefined Stage-2 rich inputs, zeroing all flux —
    # was fixed by change fix-sppm-bathroom-black-walls: the VisiblePoint now
    # stores transmissionColor/specularColor/diffuseRoughness and
    # sppmLoadMaterial rebuilds them, restoring the indirect term.)
    # SPPM (NEE direct + photon indirect) must be energy-consistent with the path
    # tracer's full GI: a double-counted direct term would push the ratio toward
    # ~2x; a missing indirect term would push it well below 1.
    path, _ = _render(0)
    sppm, built = _render(2)
    assert built
    ratio = float(sppm.mean()) / max(float(path.mean()), 1e-6)
    assert 0.85 <= ratio <= 1.15, f"SPPM/path energy ratio {ratio:.3f} out of band"


# ── caustic parity vs the pbrt reference (task 7) ──────────────────

_CORPUS = PROJECT_ROOT / "tests" / "pbrt" / "corpus"
_GLASS = _CORPUS / "glass_arealight.pbrt"
_GLASS_REF = _CORPUS / "refs" / "glass_arealight.exr"
needs_corpus = pytest.mark.skipif(
    not (_GLASS.exists() and _GLASS_REF.exists()), reason="glass_arealight corpus absent")


@needs_vulkan
@needs_corpus
def test_sppm_caustic_parity_vs_pbrt_reference():
    # The glass-sphere-over-diffuse-floor area-light scene is a caustic: SPPM
    # must converge to the same pbrt ground truth as the (converged) path
    # reference. Compares skinny SPPM's linear-HDR output to the checked-in pbrt
    # reference EXR via the parity harness (relMSE + FLIP), exposure-aligned.
    from skinny.pbrt import metrics, parity
    ref = metrics.read_exr(str(_GLASS_REF))
    img = parity.render_linear(
        str(_GLASS), 128, 128, spp=128, integrator="sppm",
        env_off=not parity.scene_has_environment(str(_GLASS)))
    aligned = metrics.align_exposure(img, ref)
    rm = metrics.relmse(aligned, ref)
    fl = metrics.flip(aligned, ref)
    # SPPM matched the reference at relMSE 0.025 / FLIP ~0.03 on M5 Pro; gate
    # with headroom for sampling noise across hosts.
    assert rm <= 0.06, f"SPPM caustic relMSE {rm:.4f} vs pbrt reference too high"
    assert fl <= 0.08, f"SPPM caustic FLIP {fl:.4f} vs pbrt reference too high"
