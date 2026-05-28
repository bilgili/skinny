"""USD-declared control (skinny:ui:*) discovery + parsing."""

from __future__ import annotations

import pytest


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_usd(), reason="pxr/USD not installed")


def _control(stage, path, **attrs):
    from pxr import Sdf, UsdGeom
    prim = UsdGeom.Scope.Define(stage, path).GetPrim()
    type_map = {
        "type": (Sdf.ValueTypeNames.Token, "skinny:ui:type"),
        "target": (Sdf.ValueTypeNames.String, "skinny:ui:target"),
        "label": (Sdf.ValueTypeNames.String, "skinny:ui:label"),
        "min": (Sdf.ValueTypeNames.Float, "skinny:ui:min"),
        "max": (Sdf.ValueTypeNames.Float, "skinny:ui:max"),
        "step": (Sdf.ValueTypeNames.Float, "skinny:ui:step"),
        "default": (Sdf.ValueTypeNames.Float, "skinny:ui:default"),
        "order": (Sdf.ValueTypeNames.Int, "skinny:ui:order"),
        "choices": (Sdf.ValueTypeNames.TokenArray, "skinny:ui:choices"),
    }
    for key, val in attrs.items():
        vt, name = type_map[key]
        prim.CreateAttribute(name, vt).Set(val)
    return prim


def _stage_with_controls():
    from pxr import Usd
    stage = Usd.Stage.CreateInMemory()
    _control(stage, "/SkinnyControls/IBL", type="slider",
             target="renderer:env_intensity", label="IBL Intensity",
             min=0.0, max=3.0, default=1.5, order=1)
    _control(stage, "/SkinnyControls/Env", type="combo",
             target="renderer:env_index", choices=["studio", "sunset"], order=2)
    _control(stage, "/SkinnyControls/Toggle", type="toggle",
             target="usd:/Light.inputs:enable", order=3)
    _control(stage, "/SkinnyControls/Tint", type="color",
             target="mtlx:base_color", order=0)
    # no label → falls back to prim name
    _control(stage, "/SkinnyControls/Roughness", type="slider",
             target="material:Skin:roughness", min=0.0, max=1.0, order=4)
    # malformed: unknown type → skipped
    _control(stage, "/SkinnyControls/Bad", type="frobnicate",
             target="renderer:whatever", order=5)
    return stage


class TestExtractControls:
    def test_valid_controls_parsed(self):
        from skinny.usd_loader import extract_ui_controls
        ctrls = extract_ui_controls(_stage_with_controls())
        # 5 valid (Bad skipped)
        assert len(ctrls) == 5
        assert [c.type for c in ctrls[:1]] == ["color"]  # order=0 first

    def test_ordering(self):
        from skinny.usd_loader import extract_ui_controls
        ctrls = extract_ui_controls(_stage_with_controls())
        orders = [c.order for c in ctrls]
        assert orders == sorted(orders)
        assert ctrls[0].target == "mtlx:base_color"   # order 0

    def test_fields_parsed(self):
        from skinny.usd_loader import extract_ui_controls
        by_target = {c.target: c for c in extract_ui_controls(_stage_with_controls())}
        ibl = by_target["renderer:env_intensity"]
        assert ibl.type == "slider"
        assert ibl.label == "IBL Intensity"
        assert ibl.lo == pytest.approx(0.0)
        assert ibl.hi == pytest.approx(3.0)
        assert ibl.default == pytest.approx(1.5)

    def test_combo_choices(self):
        from skinny.usd_loader import extract_ui_controls
        by_target = {c.target: c for c in extract_ui_controls(_stage_with_controls())}
        env = by_target["renderer:env_index"]
        assert env.type == "combo"
        assert list(env.choices) == ["studio", "sunset"]

    def test_label_fallback_to_prim_name(self):
        from skinny.usd_loader import extract_ui_controls
        by_target = {c.target: c for c in extract_ui_controls(_stage_with_controls())}
        assert by_target["material:Skin:roughness"].label == "Roughness"

    def test_malformed_skipped(self):
        from skinny.usd_loader import extract_ui_controls
        targets = {c.target for c in extract_ui_controls(_stage_with_controls())}
        assert "renderer:whatever" not in targets


class _StubRenderer:
    def __init__(self, stage=None):
        import types
        self.env_intensity = 1.0
        self.mtlx_overrides = {}
        self._mtlx_skin_material = None
        self._usd_scene = types.SimpleNamespace(materials=[
            types.SimpleNamespace(name="default", mtlx_target_name=None,
                                  parameter_overrides={}),
            types.SimpleNamespace(name="Skin", mtlx_target_name=None,
                                  parameter_overrides={}),
        ])
        self._usd_stage = stage
        self._usd_live_dirty = False

    def apply_material_override(self, mid, k, v):
        self._usd_scene.materials[mid].parameter_overrides[k] = v


def _spec(target, type="slider"):
    from skinny.usd_loader import ControlSpec
    return ControlSpec(name="c", label="c", type=type, target=target)


class TestResolveBinding:
    def test_renderer_target(self):
        from skinny.usd_loader import resolve_control_binding
        r = _StubRenderer()
        get, set_ = resolve_control_binding(r, _spec("renderer:env_intensity"))
        assert get() == pytest.approx(1.0)
        set_(2.5)
        assert r.env_intensity == pytest.approx(2.5)

    def test_mtlx_target(self):
        from skinny.usd_loader import resolve_control_binding
        r = _StubRenderer()
        _get, set_ = resolve_control_binding(r, _spec("mtlx:skin_bsdf_roughness"))
        set_(0.4)
        assert r.mtlx_overrides["skin_bsdf_roughness"] == pytest.approx(0.4)

    def test_material_target(self):
        from skinny.usd_loader import resolve_control_binding
        r = _StubRenderer()
        get, set_ = resolve_control_binding(r, _spec("material:Skin:roughness"))
        set_(0.7)
        assert r._usd_scene.materials[1].parameter_overrides["roughness"] == pytest.approx(0.7)
        assert get() == pytest.approx(0.7)

    def test_material_missing_is_inert(self):
        from skinny.usd_loader import resolve_control_binding
        r = _StubRenderer()
        get, set_ = resolve_control_binding(r, _spec("material:Nope:roughness"))
        set_(0.5)  # no crash
        assert get() is None

    def test_usd_target_writes_and_marks_dirty(self):
        from pxr import Sdf, Usd, UsdLux
        from skinny.usd_loader import resolve_control_binding
        stage = Usd.Stage.CreateInMemory()
        light = UsdLux.SphereLight.Define(stage, "/Light")
        light.GetIntensityAttr().Set(100.0)
        r = _StubRenderer(stage)
        get, set_ = resolve_control_binding(r, _spec("usd:/Light.inputs:intensity"))
        assert get() == pytest.approx(100.0)
        set_(250.0)
        assert light.GetIntensityAttr().Get() == pytest.approx(250.0)
        assert r._usd_live_dirty

    def test_unknown_prefix_is_inert(self):
        from skinny.usd_loader import resolve_control_binding
        r = _StubRenderer()
        get, set_ = resolve_control_binding(r, _spec("bogus:whatever"))
        set_(1.0)  # no crash
        assert get() is None


def test_empty_stage_has_no_controls():
    from pxr import Usd, UsdGeom
    from skinny.usd_loader import extract_ui_controls
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, "/M")
    assert extract_ui_controls(stage) == []
