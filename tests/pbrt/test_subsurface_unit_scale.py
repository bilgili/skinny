"""pbrt subsurface medium optical-density unit scale (change pbrt-subsurface-unit-scale).

The renderer's volumetric walk computes optical depth as
``τ = σ_packed · L_world · mm_per_unit``. pbrt applies no such ``mm_per_unit``
factor to its media (``τ = σ · L`` per scene unit). The importer declares
``metersPerUnit = 1.0`` so the loader derives ``mm_per_unit = 1000``; therefore
``media.subsurface_overrides`` must pre-divide the packed σ by 1000 so the walk
recovers pbrt's per-scene-unit optical depth instead of a 1000×-inflated one.
"""

from __future__ import annotations

import pytest

from skinny.pbrt import media as Mmedia
from skinny.pbrt import emit as Memit
from skinny.pbrt.parser import parse_directives
from skinny.pbrt.tokenizer import tokenize


def _params(text):
    (d,) = parse_directives(tokenize(text))
    return d.params


# The loader derives mm_per_unit = metersPerUnit * 1000 (usd_loader._read_*).
_MM_PER_UNIT = Memit.PBRT_STAGE_METERS_PER_UNIT * 1000.0


def test_stage_unit_constant_matches_emit_call():
    # The divisor must be tied to the actual SetStageMetersPerUnit value.
    assert Memit.PBRT_STAGE_METERS_PER_UNIT == 1.0


def test_named_skin1_scaled_by_mm_per_unit():
    p = _params('Material "subsurface" "string name" "Skin1" "float scale" 10')
    ov = Mmedia.subsurface_overrides(p)
    # Skin1 σ' = (0.74,0.88,1.01), σ_a = (0.032,0.17,0.48); × scale 10; ÷ 1000.
    assert ov["subsurface_sigma_s"] == pytest.approx(
        [0.0074, 0.0088, 0.0101], rel=1e-6)
    assert ov["subsurface_sigma_a"] == pytest.approx(
        [0.00032, 0.0017, 0.0048], rel=1e-6)


def test_explicit_sigma_scaled_by_mm_per_unit():
    p = _params('Material "subsurface" "rgb sigma_a" [1 2 3] "rgb sigma_s" [4 5 6]')
    ov = Mmedia.subsurface_overrides(p)
    assert ov["subsurface_sigma_a"] == pytest.approx(
        [1.0 / _MM_PER_UNIT, 2.0 / _MM_PER_UNIT, 3.0 / _MM_PER_UNIT])
    assert ov["subsurface_sigma_s"] == pytest.approx(
        [4.0 / _MM_PER_UNIT, 5.0 / _MM_PER_UNIT, 6.0 / _MM_PER_UNIT])


def test_packed_sigma_roundtrips_to_pbrt_mm_coefficients():
    # σ_packed · mm_per_unit must recover the original pbrt mm⁻¹ coefficients,
    # so the walk's optical depth equals pbrt's σ · L (no spurious factor).
    p = _params('Material "subsurface" "string name" "Skin1" "float scale" 10')
    ov = Mmedia.subsurface_overrides(p)
    recovered_a = [c * _MM_PER_UNIT for c in ov["subsurface_sigma_a"]]
    recovered_s = [c * _MM_PER_UNIT for c in ov["subsurface_sigma_s"]]
    assert recovered_a == pytest.approx([0.32, 1.7, 4.8])
    assert recovered_s == pytest.approx([7.4, 8.8, 10.1])


def test_g_eta_ior_not_rescaled():
    p = _params('Material "subsurface" "string name" "Skin1" "float eta" 1.4')
    ov = Mmedia.subsurface_overrides(p)
    assert ov["subsurface_g"] == 0.0
    assert ov["subsurface_eta"] == pytest.approx(1.4)
    assert ov["ior"] == pytest.approx(1.4)
