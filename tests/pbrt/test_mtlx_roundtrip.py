"""End-to-end -mtlx round-trip (tasks 4.3): import a corpus pbrt scene with
``materialx=True`` to a temp dir, then LOAD the exported ``.usda`` + ``.mtlx``
back through ``skinny.usd_loader`` and assert the bound meshes carry the rich
``standard_surface`` overrides (transmission/IOR/metalness/...) that the
UsdPreviewSurface path drops.

Both loader intake paths are covered:

- ``_load_mtlx_materials`` (the usdMtlx-plugin-absent fallback) — always
  exercised, since this interpreter has no ``.mtlx`` file-format plugin.
- ``_extract_material('mtlx')`` (the usdMtlx-plugin-present path) — exercised
  only when ``Sdf.FileFormat.FindByExtension('mtlx')`` resolves; otherwise the
  reference cannot compose and the path is skipped (the fallback already proves
  the overrides round-trip).

The two corpus scenes deliberately hit the two richest standard_surface slots
that UsdPreviewSurface cannot represent:

- ``glass_arealight`` — a ``dielectric`` → ``transmission`` + ``specular_IOR``
  (the loader bridges ``transmission`` → ``opacity`` and ``specular_IOR`` →
  ``ior``).
- ``conductor_infinite`` — a ``conductor`` → ``metalness`` + ``specular_color``
  (the loader bridges ``metalness`` → ``metallic``).
"""

from __future__ import annotations

import os

import pytest

mx = pytest.importorskip("MaterialX")
pytest.importorskip("pxr")
usd_loader = pytest.importorskip("skinny.usd_loader")

from pxr import Sdf, Usd  # noqa: E402

from skinny.pbrt.api import import_pbrt  # noqa: E402

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")

# usdMtlx file-format plugin presence: when absent (the norm in this
# interpreter), the .mtlx reference cannot compose and ComputeBoundMaterial
# falls back to _load_mtlx_materials. When present, _extract_material('mtlx')
# reads the composed standard_surface directly.
_USDMTLX_PLUGIN = Sdf.FileFormat.FindByExtension("mtlx") is not None


def _export_mtlx(tmp_path, scene_file: str):
    """Import a corpus scene with -mtlx into *tmp_path*; return the .usda path."""
    out = os.path.join(str(tmp_path), "out.usda")
    _stage, report = import_pbrt(
        os.path.join(CORPUS_DIR, scene_file), out=out, materialx=True
    )
    assert os.path.exists(out), "no .usda written"
    assert os.path.exists(os.path.splitext(out)[0] + ".mtlx"), "no .mtlx sidecar"
    assert report.count("skipped") == 0, str(report)
    return out


# ── exported artifacts ─────────────────────────────────────────────────


def test_export_writes_usda_and_mtlx_sidecar(tmp_path):
    out = _export_mtlx(tmp_path, "glass_arealight.pbrt")
    stage = Usd.Stage.Open(out)
    # The .mtlx sidecar is referenced from the stage (collectible by the loader).
    assert usd_loader._collect_mtlx_asset_paths(stage) == {"out.mtlx"}


def test_no_shadowing_preview_surface_authored(tmp_path):
    """The -mtlx export must not author a UsdPreviewSurface on the prims (else
    ComputeBoundMaterial bypasses the .mtlx sidecar)."""
    from pxr import UsdShade

    out = _export_mtlx(tmp_path, "glass_arealight.pbrt")
    stage = Usd.Stage.Open(out)
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Shader":
            assert (
                UsdShade.Shader(prim).GetShaderId() != "UsdPreviewSurface"
            ), "shadowing UsdPreviewSurface authored under -mtlx"


# ── intake path 1: _load_mtlx_materials (plugin-absent fallback) ────────


def test_glass_rich_overrides_via_fallback(tmp_path):
    out = _export_mtlx(tmp_path, "glass_arealight.pbrt")
    stage = Usd.Stage.Open(out)
    mats = usd_loader._load_mtlx_materials(stage, tmp_path)
    # The dielectric sphere material carries the transmission/IOR round-trip.
    glass = next(
        (m for m in mats.values() if m.parameter_overrides.get("ior") == 1.5),
        None,
    )
    assert glass is not None, f"no glass material in {list(mats)}"
    ovr = glass.parameter_overrides
    # transmission -> opacity bridge (delta-dielectric refraction gate)
    assert ovr["opacity"] == pytest.approx(0.0)
    # specular_IOR survives under both the canonical and flat-aliased keys
    assert ovr["specular_IOR"] == pytest.approx(1.5)
    assert ovr["ior"] == pytest.approx(1.5)
    # transmission_color (white) migrates to diffuseColor for the flat path
    assert ovr["diffuseColor"] == pytest.approx((1.0, 1.0, 1.0))


def test_arealight_emission_survives_roundtrip(tmp_path):
    """Regression: an area-light material's radiance must survive the .mtlx
    round-trip. standard_surface emission is weight(scalar) x emission_color and
    _load_mtlx_materials recovers ``emissiveColor`` ONLY when ``emission`` > 0 —
    so the exporter must author the unit emission weight, not emission_color
    alone. Without it the light is dropped and the scene renders black.

    glass_arealight's emitter is a quad with ``L = (15, 15, 15)``.
    """
    out = _export_mtlx(tmp_path, "glass_arealight.pbrt")
    stage = Usd.Stage.Open(out)
    mats = usd_loader._load_mtlx_materials(stage, tmp_path)
    emissive = [
        m for m in mats.values()
        if any(c > 0 for c in (m.parameter_overrides.get("emissiveColor") or (0, 0, 0)))
    ]
    assert emissive, (
        "no emissive material recovered from the .mtlx round-trip — the area "
        "light was dropped (scene would render black)"
    )
    ec = emissive[0].parameter_overrides["emissiveColor"]
    # emission(1.0) x emission_color(15,15,15) == (15,15,15)
    assert ec[0] == pytest.approx(15.0, rel=1e-3)


def test_conductor_rich_overrides_via_fallback(tmp_path):
    out = _export_mtlx(tmp_path, "conductor_infinite.pbrt")
    stage = Usd.Stage.Open(out)
    mats = usd_loader._load_mtlx_materials(stage, tmp_path)
    metal = next(
        (m for m in mats.values() if m.parameter_overrides.get("metalness") == 1.0),
        None,
    )
    assert metal is not None, f"no conductor material in {list(mats)}"
    ovr = metal.parameter_overrides
    # metalness -> metallic bridge (so the flat path renders a metal, not plastic)
    assert ovr["metallic"] == pytest.approx(1.0)
    assert ovr["metalness"] == pytest.approx(1.0)
    # GGX roughness remap survives (pbrt roughness 0.1 -> standard_surface 0.562)
    assert ovr["specular_roughness"] == pytest.approx(0.562341, abs=1e-5)
    assert ovr["roughness"] == pytest.approx(0.562341, abs=1e-5)
    # named-spectrum gold reflectance reaches base_color / specular_color
    assert ovr["base_color"][0] == pytest.approx(0.966688, abs=1e-4)
    assert ovr["specular_color"][0] == pytest.approx(0.966688, abs=1e-4)


# ── intake path 1, full scene loader: bound meshes carry rich overrides ─


def test_glass_bound_mesh_carries_overrides_full_loader(tmp_path):
    """load_scene_from_stage routes the bound mesh's material through the
    _resolve_material_binding -> _load_mtlx_materials fallback; assert each
    bound instance resolves a Material with the rich overrides."""
    out = _export_mtlx(tmp_path, "glass_arealight.pbrt")
    stage = Usd.Stage.Open(out)
    scene = usd_loader.load_scene_from_stage(stage)
    # At least one bound mesh resolves the glass material (ior=1.5, opacity=0).
    bound_overrides = [
        scene.materials[inst.material_id].parameter_overrides
        for inst in scene.instances
    ]
    glass = [o for o in bound_overrides if o.get("ior") == 1.5]
    assert glass, "no bound mesh carried the glass ior override"
    assert glass[0]["opacity"] == pytest.approx(0.0)


def test_conductor_bound_mesh_carries_overrides_full_loader(tmp_path):
    out = _export_mtlx(tmp_path, "conductor_infinite.pbrt")
    stage = Usd.Stage.Open(out)
    scene = usd_loader.load_scene_from_stage(stage)
    bound_overrides = [
        scene.materials[inst.material_id].parameter_overrides
        for inst in scene.instances
    ]
    metal = [o for o in bound_overrides if o.get("metallic") == 1.0]
    assert metal, "no bound mesh carried the conductor metallic override"
    assert metal[0]["specular_roughness"] == pytest.approx(0.562341, abs=1e-5)


# ── intake path 2: usdMtlx plugin present (_extract_material) ───────────


@pytest.mark.skipif(
    not _USDMTLX_PLUGIN,
    reason="usdMtlx file-format plugin absent; .mtlx reference cannot compose "
    "(fallback path covers the round-trip)",
)
def test_glass_overrides_equivalent_via_usdmtlx_plugin(tmp_path):
    """When the usdMtlx plugin is present, the composed standard_surface read by
    _extract_material('mtlx') must surface overrides equivalent to the fallback.

    Both loaders post-process transmission -> opacity identically
    (_derive_opacity_from_transmission mirrors _load_mtlx_materials), so the
    glass material's opacity/ior must match across both intake paths."""
    out = _export_mtlx(tmp_path, "glass_arealight.pbrt")

    # Path 1 (fallback): force the fallback regardless of plugin presence.
    stage_a = Usd.Stage.Open(out)
    scene_a = usd_loader.load_scene_from_stage(stage_a)
    glass_a = [
        scene_a.materials[i.material_id].parameter_overrides
        for i in scene_a.instances
        if scene_a.materials[i.material_id].parameter_overrides.get("ior") == 1.5
    ]
    assert glass_a, "fallback path resolved no glass material"

    # Path 2 (usdMtlx plugin): the loader's use_usd_mtlx_plugin flag routes the
    # bound material through ComputeBoundMaterial -> _extract_material('mtlx').
    stage_b = Usd.Stage.Open(out)
    scene_b = usd_loader.load_scene_from_stage(stage_b, use_usd_mtlx_plugin=True)
    glass_b = [
        scene_b.materials[i.material_id].parameter_overrides
        for i in scene_b.instances
        if scene_b.materials[i.material_id].parameter_overrides.get("ior") == 1.5
    ]
    assert glass_b, "usdMtlx plugin path resolved no glass material"

    # Equivalent opacity (transmission bridge) and IOR across both paths.
    assert glass_a[0]["opacity"] == pytest.approx(glass_b[0]["opacity"])
    assert glass_a[0]["ior"] == pytest.approx(glass_b[0]["ior"])
