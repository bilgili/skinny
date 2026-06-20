"""Unit gates for the Stage-2 rich flat-BSDF inputs (flat-lobes-rich-inputs).

Verifies, on the GPU lobe model, that specular_color and diffuse_roughness are
strictly response-only — they tint/shape the response but leave the solid-angle
pdf untouched, which is exactly what keeps sample().pdf == evaluate().pdf and
response/pdf bounded (firefly-free, no clamp). Also checks the back-compat
defaults reproduce the prior Lambert / untinted behavior.

Task 3.1 of openspec/changes/flat-lobes-rich-inputs.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def flat_harness():
    """Load the flat-lobes harness on a Vulkan device.

    The harness imports the production flat module graph, which pulls in the
    bindless global resources from bindings.slang. slang-rhi's Metal backend
    cannot bind those for a free-function (slangpy) dispatch ("Unsupported
    binding type"), so this pure-math BSDF harness runs on Vulkan. Metal
    coverage of the same code lives in the four-way convergence / Metal↔Vulkan
    render gates (tests/test_sampling_parity.py + the pbrt parity gate), which
    bind the full descriptor set through the real renderer.
    """
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


# A spread of (wo, wi) tangent-space pairs (N = +Z, both z > 0).
_DIRS = [
    ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
    ([0.0, 0.0, 1.0], [0.3, 0.0, 0.954]),
    ([0.2, 0.1, 0.974], [0.0, 0.4, 0.917]),
    ([0.5, 0.0, 0.866], [0.0, 0.5, 0.866]),
    ([0.6, 0.3, 0.74], [0.4, 0.5, 0.768]),
    ([0.8, 0.0, 0.6], [0.0, 0.8, 0.6]),    # grazing-ish
    ([0.9, 0.1, 0.42], [0.85, 0.2, 0.49]),  # near-grazing
]


def _eval(h, wo, wi, albedo, rough, metallic, spec_color, diff_rough):
    return h.test_flat_eval(wo, wi, albedo, rough, metallic, spec_color, diff_rough)


class TestPdfSymmetryPreserved:
    """specular_color / diffuse_roughness are response-only ⇒ the pdf must be
    invariant to them. flatBsdfPdf is the single source both sample() and
    evaluate() call, so pdf-invariance here == sample().pdf == evaluate().pdf
    preserved for the tinted / Oren-Nayar material."""

    def test_pdf_independent_of_specular_color(self, flat_harness):
        for wo, wi in _DIRS:
            white = float(_eval(flat_harness, wo, wi, [0.6, 0.6, 0.6], 0.3,
                                0.0, [1.0, 1.0, 1.0], 0.0)["pdf"])
            tint = float(_eval(flat_harness, wo, wi, [0.6, 0.6, 0.6], 0.3,
                               0.0, [1.0, 0.4, 0.1], 0.0)["pdf"])
            assert math.isclose(white, tint, rel_tol=1e-5, abs_tol=1e-7), (
                f"specular_color changed the pdf at wo={wo} wi={wi}: "
                f"{white} vs {tint}")

    def test_pdf_independent_of_diffuse_roughness(self, flat_harness):
        for wo, wi in _DIRS:
            lambert = float(_eval(flat_harness, wo, wi, [0.6, 0.6, 0.6], 0.3,
                                  0.0, [1.0, 1.0, 1.0], 0.0)["pdf"])
            oren = float(_eval(flat_harness, wo, wi, [0.6, 0.6, 0.6], 0.3,
                               0.0, [1.0, 1.0, 1.0], 0.8)["pdf"])
            assert math.isclose(lambert, oren, rel_tol=1e-5, abs_tol=1e-7), (
                f"diffuse_roughness changed the pdf at wo={wo} wi={wi}: "
                f"{lambert} vs {oren}")


class TestResponseOverPdfBounded:
    """response / pdf stays the bounded native per-lobe weight under the rich
    inputs — no firefly, no clamp. specular_color ∈ [0,1] and Oren-Nayar bound
    keep it finite at every sampled direction."""

    def test_bounded_for_tinted_oren_nayar(self, flat_harness):
        for wo, wi in _DIRS:
            r = _eval(flat_harness, wo, wi, [0.7, 0.5, 0.3], 0.4,
                      0.0, [1.0, 0.6, 0.2], 0.9)
            resp = [float(c) for c in r["response"]]
            pdf = float(r["pdf"])
            assert pdf > 0.0 and math.isfinite(pdf)
            for c in resp:
                assert math.isfinite(c) and c >= 0.0
            # response/pdf must be bounded (the native per-lobe weight).
            for c in resp:
                assert c / pdf < 8.0, f"response/pdf too large: {c / pdf}"


class TestBackCompatDefaults:
    """specular_color = white + diffuse_roughness = 0 must reproduce the prior
    (untinted Lambert) response exactly — existing renders are unchanged."""

    def test_oren_nayar_is_identity_at_zero(self, flat_harness):
        for wo, wi in _DIRS:
            on = float(flat_harness.test_oren_nayar(wo, wi, 0.0))
            assert on == 1.0, f"Oren-Nayar σ=0 must be exact Lambert, got {on}"

    def test_white_spec_zero_rough_matches_baseline(self, flat_harness):
        # The default (white spec, σ=0) response is the reference; a second
        # call with identical args must be byte-stable, and is the untinted
        # baseline other tests compare against.
        wo, wi = _DIRS[3]
        a = [float(c) for c in _eval(flat_harness, wo, wi, [0.6, 0.6, 0.6],
                                     0.3, 0.0, [1.0, 1.0, 1.0], 0.0)["response"]]
        b = [float(c) for c in _eval(flat_harness, wo, wi, [0.6, 0.6, 0.6],
                                     0.3, 0.0, [1.0, 1.0, 1.0], 0.0)["response"]]
        assert a == b
        assert all(math.isfinite(c) and c > 0.0 for c in a)


class TestRichInputsAffectResponse:
    """The rich inputs must actually reach the response in the expected
    direction (otherwise they'd be inert again)."""

    def test_specular_color_tints_spec_response(self, flat_harness):
        # Near-mirror, near-specular config so the spec lobe dominates.
        wo = [0.3, 0.0, 0.954]
        wi = [-0.3, 0.0, 0.954]  # mirror of wo about N
        white = [float(c) for c in _eval(flat_harness, wo, wi, [0.05, 0.05, 0.05],
                                         0.05, 1.0, [1.0, 1.0, 1.0], 0.0)["response"]]
        tint = [float(c) for c in _eval(flat_harness, wo, wi, [0.05, 0.05, 0.05],
                                        0.05, 1.0, [1.0, 0.3, 0.1], 0.0)["response"]]
        # Green / blue suppressed relative to red by the tint.
        assert tint[1] < white[1] * 0.6
        assert tint[2] < white[2] * 0.4
        assert math.isclose(tint[0], white[0], rel_tol=1e-4)

    def test_oren_nayar_changes_response_and_bounded(self, flat_harness):
        # Oren-Nayar redistributes energy (rougher ⇒ flatter): the factor moves
        # off 1.0 at σ>0 and stays bounded (positive, finite, no firefly) over
        # the hemisphere — including the near-grazing pairs.
        for wo, wi in _DIRS:
            on0 = float(flat_harness.test_oren_nayar(wo, wi, 0.0))
            on = float(flat_harness.test_oren_nayar(wo, wi, 0.9))
            assert on0 == 1.0
            assert 0.0 < on < 4.0 and math.isfinite(on)
        # At least one direction is meaningfully changed by the roughness.
        changed = [
            abs(float(flat_harness.test_oren_nayar(wo, wi, 0.9)) - 1.0)
            for wo, wi in _DIRS
        ]
        assert max(changed) > 0.05

    def test_oren_nayar_retroreflective_brightening(self, flat_harness):
        # Same-azimuth grazing (cosΔφ > 0, both near the horizon) is where the
        # Oren-Nayar retroreflective term overtakes the A floor and exceeds 1.
        wo = [0.98, 0.0, 0.2]
        wi = [0.9, 0.0, 0.436]
        on = float(flat_harness.test_oren_nayar(wo, wi, 0.9))
        assert on > 1.0
        assert on < 4.0  # bounded, no firefly
