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


# ── subsurface + coated: -mtlx vs UsdPreviewSurface round-trip equivalence ──
#
# These two material types are NOT in the parity corpus, and they diverge
# between the two export paths (the -mtlx dragon rendered opaque white). The
# round-trip must yield equivalent FlatMaterial parameters whether the scene is
# exported as UsdPreviewSurface or as -mtlx:
#
#   - subsurface  -> opacity == 0 (transmissive boundary) + the homogeneous
#                    interior (volume_*) recovered from skinnyOverrides.
#   - coateddiffuse -> coat weight + coat roughness land in the same FlatMaterial
#                    slot regardless of clearcoat-vs-coat authoring.
#
# Pure loader/exporter test — no renderer/GPU import (pack_flat_material lives
# in skinny.renderer, which eagerly imports vulkan). The projection below
# mirrors the FlatMaterial fields pack_flat_material reads, with its defaults.

_SUBSURFACE_COAT_SCENE = """\
LookAt 0 1 5  0 0 0  0 1 0
Camera "perspective" "float fov" 35
Sampler "independent" "integer pixelsamples" 8
Integrator "path" "integer maxdepth" 8
Film "rgb" "integer xresolution" 16 "integer yresolution" 16 "string filename" "ss.exr"
WorldBegin

AttributeBegin
  Material "subsurface" "float scale" 10 "float eta" 1.5
  Shape "sphere" "float radius" 1
AttributeEnd

AttributeBegin
  Material "coateddiffuse" "float roughness" 0.2
  Translate 0 -1 0
  Shape "trianglemesh" "point3 P" [ -6 0 -6  6 0 -6  6 0 6  -6 0 6 ] "integer indices" [ 0 1 2 0 2 3 ]
AttributeEnd
"""


def _flat_view(ov: dict) -> dict:
    """Project parameter_overrides onto the FlatMaterial fields pack_flat_material
    reads, applying the same defaults. Reads CANONICAL keys (``coat``/
    ``coat_roughness``) — so a UsdPreviewSurface material that only authored
    ``clearcoat`` shows coat == 0 unless the loader canonicalized it."""
    def f(key, default):
        v = ov.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return default
        return float(v)

    def c3(key, default):
        v = ov.get(key)
        try:
            return (float(v[0]), float(v[1]), float(v[2]))
        except (TypeError, IndexError, ValueError):
            return default

    return {
        "diffuseColor": c3("diffuseColor", (0.5, 0.5, 0.5)),
        "roughness": f("roughness", 0.5),
        "metallic": f("metallic", 0.0),
        "specular": f("specular", 0.5),
        "opacity": f("opacity", 1.0),
        "ior": f("ior", 1.5),
        "coat": f("coat", 0.0),
        "coat_roughness": f("coat_roughness", 0.0),
        "coat_IOR": f("coat_IOR", 1.5),
        "coat_color": c3("coat_color", (1.0, 1.0, 1.0)),
        "emissiveColor": c3("emissiveColor", (0.0, 0.0, 0.0)),
    }


def _assert_flat_equiv(a: dict, b: dict):
    """Assert two FlatMaterial projections agree (float32 tolerance — USD stores
    shader inputs as float32 while the .mtlx value strings parse as float64)."""
    va, vb = _flat_view(a), _flat_view(b)
    assert set(va) == set(vb)
    for key in va:
        assert va[key] == pytest.approx(vb[key], abs=1e-5), key


def _bound_materials(out: str):
    """Load *out* and return the materials bound to instances (fallback excluded)."""
    stage = Usd.Stage.Open(out)
    scene = usd_loader.load_scene_from_stage(stage)
    return [scene.materials[i.material_id] for i in scene.instances]


def _export_both(tmp_path):
    """Write the subsurface+coat scene; export it UsdPreviewSurface and -mtlx.

    Returns (preview_materials, mtlx_materials) — each a list of bound Materials,
    classified below by base/diffuse colour (subsurface ~1.0, coat ~0.5)."""
    scene_file = os.path.join(str(tmp_path), "ss_coat.pbrt")
    with open(scene_file, "w") as fh:
        fh.write(_SUBSURFACE_COAT_SCENE)

    preview_out = os.path.join(str(tmp_path), "preview", "out.usda")
    os.makedirs(os.path.dirname(preview_out), exist_ok=True)
    import_pbrt(scene_file, out=preview_out, materialx=False)

    mtlx_out = os.path.join(str(tmp_path), "mtlx", "out.usda")
    os.makedirs(os.path.dirname(mtlx_out), exist_ok=True)
    import_pbrt(scene_file, out=mtlx_out, materialx=True)

    return _bound_materials(preview_out), _bound_materials(mtlx_out)


def _classify(materials):
    """Split bound materials into (subsurface, coat) by diffuse colour."""
    subsurface = next(
        m for m in materials
        if _flat_view(m.parameter_overrides)["diffuseColor"][0] > 0.9
    )
    coat = next(
        m for m in materials
        if _flat_view(m.parameter_overrides)["diffuseColor"][0] < 0.7
    )
    return subsurface, coat


def test_subsurface_roundtrip_equivalent(tmp_path):
    """The -mtlx subsurface material must load with the same FlatMaterial params
    as the UsdPreviewSurface export: opacity == 0 and the homogeneous interior."""
    preview_mats, mtlx_mats = _export_both(tmp_path)
    ss_p, _ = _classify(preview_mats)
    ss_m, _ = _classify(mtlx_mats)

    # FlatMaterial projection equivalent across both export paths.
    _assert_flat_equiv(ss_m.parameter_overrides, ss_p.parameter_overrides)

    # subsurface -> opacity 0 on the -mtlx path (the bug: it stayed opaque).
    assert ss_m.parameter_overrides.get("opacity") == pytest.approx(0.0)

    # SSS interior recovered from skinnyOverrides on the -mtlx path (the bug: it
    # was dropped). Carried under the renderer's medium keys (subsurface_sigma_*,
    # mm⁻¹, via the pbrt-v4 precedence) so the renderer packs the inline medium
    # and routes to MATERIAL_TYPE_SUBSURFACE; equal across both export paths.
    ovr_m = ss_m.parameter_overrides
    assert "subsurface_sigma_s" in ovr_m, "SSS interior lost on -mtlx path"
    assert "subsurface_sigma_a" in ovr_m
    assert ovr_m["subsurface_sigma_s"] == pytest.approx(
        ss_p.parameter_overrides["subsurface_sigma_s"]
    )
    assert ovr_m["ior"] == pytest.approx(ss_p.parameter_overrides["ior"])


def test_coateddiffuse_roundtrip_equivalent(tmp_path):
    """The coateddiffuse coat lobe must reach the same FlatMaterial slot on both
    export paths, with the same coat roughness (from pbrt `roughness`)."""
    preview_mats, mtlx_mats = _export_both(tmp_path)
    _, coat_p = _classify(preview_mats)
    _, coat_m = _classify(mtlx_mats)

    # FlatMaterial projection equivalent across both export paths.
    _assert_flat_equiv(coat_m.parameter_overrides, coat_p.parameter_overrides)

    # coat weight present (> 0) on both — UsdPreviewSurface authored clearcoat,
    # -mtlx authored coat; both must canonicalize to the coat slot.
    assert _flat_view(coat_p.parameter_overrides)["coat"] > 0.0
    assert _flat_view(coat_m.parameter_overrides)["coat"] > 0.0

    # coat roughness derived from pbrt roughness 0.2 (remapped) agrees and is
    # non-trivial (the -mtlx bug read a non-existent interface.roughness -> 0).
    assert _flat_view(coat_m.parameter_overrides)["coat_roughness"] == pytest.approx(
        _flat_view(coat_p.parameter_overrides)["coat_roughness"]
    )
    assert _flat_view(coat_m.parameter_overrides)["coat_roughness"] > 0.0
