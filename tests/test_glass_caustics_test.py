"""Scene-level regression for `assets/glass_caustics_test.usda`.

The scene holds two glass spheres that must BOTH render transparent:

* ``GlassSphere``     → ``/Scene/GlassMat``            (UsdPreviewSurface,
  ``opacity = 0``)
* ``MtlxGlassSphere`` → ``/MaterialX/Materials/Glass`` (MaterialX
  ``standard_surface``, ``transmission = 1``)

The MaterialX sphere previously rendered OPAQUE: skinny's flat path only refracts
through surfaces whose ``opacity < 1`` (``flat_material.slang``), and the loader's
``transmission → opacity`` bridge skipped because the ``standard_surface`` shader
authors a default-opaque ``opacity = (1, 1, 1)``. This test loads the real asset
and asserts every glass material reaches the flat refraction gate (``opacity < 1``).
"""

from pathlib import Path

import pytest

_ASSET = Path(__file__).resolve().parents[1] / "assets" / "glass_caustics_test.usda"


def _min_opacity(overrides: dict) -> float:
    """Smallest opacity channel as a float (scalar or color3 are both valid)."""
    o = overrides.get("opacity")
    assert o is not None, "glass material derived no opacity (refraction gate shut)"
    if isinstance(o, (int, float)):
        return float(o)
    return min(float(c) for c in o)


def test_glass_caustics_both_spheres_transparent():
    """Both glass spheres in glass_caustics_test.usda reach the flat refraction
    gate (opacity < 1) — the MaterialX standard_surface glass must not stay opaque.
    """
    pytest.importorskip("pxr")
    from pxr import Usd, UsdShade

    from skinny.usd_loader import _extract_material

    assert _ASSET.is_file(), f"missing test asset: {_ASSET}"
    stage = Usd.Stage.Open(str(_ASSET))

    # Control: the UsdPreviewSurface glass authors opacity = 0 directly.
    prev = stage.GetPrimAtPath("/Scene/GlassMat")
    assert prev and prev.IsValid()
    prev_mat = _extract_material(UsdShade.Material(prev))
    assert _min_opacity(prev_mat.parameter_overrides) < 1.0

    # Regression: the MaterialX standard_surface glass (transmission = 1, default
    # opacity = (1, 1, 1)) must be bridged to opacity < 1 so it refracts.
    mtlx = stage.GetPrimAtPath("/MaterialX/Materials/Glass")
    if not (mtlx and mtlx.IsValid()):
        pytest.skip("usdMtlx plugin absent: MaterialX material did not resolve")
    mtlx_mat = _extract_material(UsdShade.Material(mtlx))
    assert mtlx_mat.parameter_overrides.get("transmission") == pytest.approx(1.0)
    assert _min_opacity(mtlx_mat.parameter_overrides) < 1.0, (
        "MaterialX standard_surface glass stayed opaque — the transmission→opacity "
        "bridge was blocked by the default-opaque (1,1,1) opacity"
    )
