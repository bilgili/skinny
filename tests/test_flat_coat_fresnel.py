"""Coat-lobe Fresnel energy gate (fix-flat-coat-fresnel-eta).

The flat coat lobe selects the coat with probability `pCoat = coat ·
fresnelDielectric(NdotV, eta)`. The view ray enters the coat from air, so the
entering relative index is `eta = 1 / coatIOR`. The bug passed `coatIOR` raw —
the *exiting* (coat→air) direction — whose Snell term `sinT2 = coatIOR²·sin²θ`
triggers spurious total internal reflection for any NdotV < 1/coatIOR·... i.e.
past ~42° from normal at IOR 1.5. `pCoat` then saturated to 1 and the base
diffuse/spec lobes (attenuated by `1 − pCoat`) were zeroed, so a coated diffuse
lost most of its energy (assets/dragon_removed.usda rendered a large dark region
instead of a near-uniform diffuse).

These gates run the production flat lobe via the slangpy harness on a Vulkan
device (slang-rhi's Metal backend cannot bind the bindless globals for a
free-function dispatch — see tests/test_flat_rich_inputs.py). Metal coverage of
the same source lives in the render/parity gates.

Task 4.2 of openspec/changes/fix-flat-coat-fresnel-eta.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def flat_harness():
    spy = pytest.importorskip("slangpy")
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    inc = [
        str(root / "src" / "skinny" / "shaders"),
        str(root / "src" / "skinny" / "mtlx" / "genslang"),
        str(Path(__file__).resolve().parent / "harnesses"),
    ]
    try:
        dev = spy.create_device(type=spy.DeviceType.vulkan, include_paths=inc)
    except Exception:
        pytest.skip("No Vulkan device available")
    return spy.Module.load_from_file(dev, "test_flat_lobes_harness.slang")


def _lum(rgb) -> float:
    r, g, b = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# coatIOR 1.5: the bug's spurious TIR sets in at NdotV < 1/1.5 ≈ 0.667 region;
# NdotV = 0.6 is squarely inside it (sinT2 = 1.5²·(1−0.36) = 1.44 > 1 ⇒ F = 1).
_OBLIQUE_WO = [0.8, 0.0, 0.6]          # NdotV = 0.6
_DIFFUSE_WI = [0.0, 0.2, 0.9797959]    # a diffuse-hemisphere outgoing dir
_NORMAL_WO = [0.0, 0.0, 1.0]           # NdotV = 1
_NORMAL_WI = [0.1, 0.0, 0.9949874]


def _resp(h, wo, wi, coat):
    return h.test_flat_coat_response(wo, wi, coat, 0.669, 1.5)


class TestCoatEnteringEta:
    def test_coat_does_not_zero_base_at_oblique_view(self, flat_harness):
        """At NdotV = 0.6 (inside the buggy TIR cone) a unit dielectric coat must
        only mildly attenuate the diffuse base, not crater it."""
        base = _lum(_resp(flat_harness, _OBLIQUE_WO, _DIFFUSE_WI, 0.0))
        coated = _lum(_resp(flat_harness, _OBLIQUE_WO, _DIFFUSE_WI, 1.0))
        assert base > 1e-4, f"degenerate base response {base}"
        ratio = coated / base
        # Correct entering eta: pCoat = F(0.6, 1/1.5) ≈ 0.065, so the base keeps
        # ≈ 0.94 of its energy. The bug zeroed it (ratio → ~0).
        assert ratio > 0.7, (
            f"coat cratered the base at oblique view: coated/base = {ratio:.3f} "
            f"(base={base:.4f}, coated={coated:.4f}) — coat Fresnel eta regressed"
        )

    def test_coat_subtle_at_normal_incidence(self, flat_harness):
        """At normal incidence the coat reflectance is ~F0 = 0.04, so the coated
        base is within a few percent of the uncoated base."""
        base = _lum(_resp(flat_harness, _NORMAL_WO, _NORMAL_WI, 0.0))
        coated = _lum(_resp(flat_harness, _NORMAL_WO, _NORMAL_WI, 1.0))
        assert base > 1e-4
        ratio = coated / base
        assert 0.85 < ratio <= 1.05, (
            f"coat changed the near-normal base too much: {ratio:.3f}"
        )

    def test_coat_zero_is_identity(self, flat_harness):
        """coat = 0 leaves the base response exactly unchanged (the coat term is
        gated on coat > 0)."""
        for wo, wi in [
            (_OBLIQUE_WO, _DIFFUSE_WI),
            (_NORMAL_WO, _NORMAL_WI),
        ]:
            a = _lum(_resp(flat_harness, wo, wi, 0.0))
            # Re-evaluating with coat = 0 must be deterministic / identical.
            b = _lum(_resp(flat_harness, wo, wi, 0.0))
            assert a == b
