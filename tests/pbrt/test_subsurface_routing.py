"""Phase-4 integration: a pbrt ``subsurface`` material routes to
MATERIAL_TYPE_SUBSURFACE and its volumetric medium coefficients survive the
import all the way into ``Material.parameter_overrides`` (via the
``skinnyOverrides`` customData channel the loader merges), using the phase-1
pbrt-precedence coefficients — NOT the cruder ``volume_*`` stand-in.
"""

from __future__ import annotations

import pytest

from skinny.pbrt.api import import_pbrt
from skinny.pbrt.subsurface import subsurface_coefficients

usd_loader = pytest.importorskip("skinny.usd_loader")

SKIN1 = (
    'WorldBegin\n'
    'Material "subsurface" "string name" "Skin1"\n'
    'Shape "sphere" "float radius" 1\n'
)


def _skinny_overrides(stage):
    """Return the first material prim's skinnyOverrides dict (or {}).

    Uses TraverseAll so the -mtlx path's typeless ``over`` material prims (which
    the default predicate skips) are visited too.
    """
    for prim in stage.TraverseAll():
        cd = prim.GetCustomDataByKey("skinnyOverrides")
        if cd and "subsurface_sigma_a" in cd:
            return dict(cd)
    return {}


def test_native_subsurface_import_does_not_crash_and_carries_coeffs(tmp_path):
    # The non-mtlx (UsdPreviewSurface) path must not crash authoring the
    # list-valued medium coefficients, and must carry them on skinnyOverrides.
    src = tmp_path / "sss.pbrt"
    src.write_text(SKIN1)
    stage, _report = import_pbrt(str(src))
    cd = _skinny_overrides(stage)
    exp = subsurface_coefficients(name="Skin1")
    assert list(cd["subsurface_sigma_a"]) == pytest.approx(list(exp["sigma_a"]))
    assert list(cd["subsurface_sigma_s"]) == pytest.approx(list(exp["sigma_s"]))
    assert cd["subsurface_g"] == pytest.approx(exp["g"])
    # Boundary IOR reaches the renderer through `ior` (resolveMedium reads it).
    assert cd["ior"] == pytest.approx(exp["eta"])


def test_mtlx_subsurface_import_carries_same_coeffs(tmp_path):
    src = tmp_path / "sss.pbrt"
    src.write_text(SKIN1)
    out = tmp_path / "sss.usda"
    stage, _report = import_pbrt(str(src), out=str(out), materialx=True)
    cd = _skinny_overrides(stage)
    exp = subsurface_coefficients(name="Skin1")
    assert list(cd["subsurface_sigma_a"]) == pytest.approx(list(exp["sigma_a"]))
    assert list(cd["subsurface_sigma_s"]) == pytest.approx(list(exp["sigma_s"]))


def test_renderer_detects_and_tags_subsurface():
    from types import SimpleNamespace
    try:
        from skinny.renderer import (
            _material_is_subsurface,
            MATERIAL_TYPE_SUBSURFACE,
            MATERIAL_TYPE_FLAT,
        )
    except OSError as exc:  # renderer imports vulkan unconditionally
        pytest.skip(f"renderer import needs the Vulkan SDK on the dylib path: {exc}")

    assert MATERIAL_TYPE_SUBSURFACE == 4
    assert MATERIAL_TYPE_SUBSURFACE != MATERIAL_TYPE_FLAT
    sss = SimpleNamespace(parameter_overrides={
        "subsurface_sigma_a": (0.032, 0.17, 0.48),
        "subsurface_sigma_s": (0.74, 0.88, 1.01),
    })
    flat = SimpleNamespace(parameter_overrides={"diffuseColor": (0.5, 0.5, 0.5)})
    # A fog MediumInterface carries volume_* (not subsurface_*) — must NOT
    # be misdetected as a subsurface boundary material.
    fog = SimpleNamespace(parameter_overrides={"volume_sigma_s": (0.5, 0.6, 0.7)})
    assert _material_is_subsurface(sss)
    assert not _material_is_subsurface(flat)
    assert not _material_is_subsurface(fog)
