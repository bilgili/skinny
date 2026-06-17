"""Tests for participating media + subsurface best-effort (task 9.1)."""

from __future__ import annotations

import pytest

from skinny.pbrt.api import import_pbrt

usd_loader = pytest.importorskip("skinny.usd_loader")

HOMOGENEOUS = """
WorldBegin
MakeNamedMedium "fog" "string type" "homogeneous" "rgb sigma_s" [0.5 0.6 0.7] "float g" 0.3
AttributeBegin
  MediumInterface "fog" ""
  Material "dielectric" "float eta" 1.33
  Shape "sphere" "float radius" 1
AttributeEnd
"""

HETEROGENEOUS = """
WorldBegin
MakeNamedMedium "cloud" "string type" "uniformgrid" "rgb sigma_s" [1 1 1]
AttributeBegin
  MediumInterface "cloud" ""
  Material "dielectric"
  Shape "sphere" "float radius" 1
AttributeEnd
"""


def test_homogeneous_medium_carried_via_customdata(tmp_path):
    src = tmp_path / "m.pbrt"
    src.write_text(HOMOGENEOUS)
    stage, report = import_pbrt(str(src))
    # the material prim carries skinnyOverrides with volume coefficients
    found = False
    for prim in stage.Traverse():
        cd = prim.GetCustomDataByKey("skinnyOverrides")
        if cd and "volume_sigma_s" in cd:
            found = True
            assert cd["volume_g"] == pytest.approx(0.3)
    assert found
    assert any("medium:homogeneous" in e.construct for e in report.entries)


def test_heterogeneous_medium_flagged(tmp_path):
    src = tmp_path / "h.pbrt"
    src.write_text(HETEROGENEOUS)
    _stage, report = import_pbrt(str(src))
    assert report.count("skipped") >= 1
    assert any("heterogeneous" in e.detail for e in report.entries)


def test_subsurface_material_flagged(tmp_path):
    src = tmp_path / "s.pbrt"
    src.write_text(
        'WorldBegin\nMaterial "subsurface" "float eta" 1.33\nShape "sphere" "float radius" 1\n'
    )
    stage, report = import_pbrt(str(src))
    assert any(e.status == "approx" and "subsurface" in e.construct for e in report.entries) or any(
        "subsurface" in (e.detail or "") for e in report.entries
    )
