"""Render-parity matrix: validity table, enumeration, tolerances, coverage.

All no-GPU: the matrix is data-driven, so its construction and the
compatibility rules are tested without rendering.
"""

from __future__ import annotations

import pytest

from skinny.pbrt import parity
from skinny.pbrt.parity import RenderCombo, SceneSpec


def _flat_scene(**kw) -> SceneSpec:
    base = dict(name="flat", file="flat.pbrt", ref="refs/flat.exr", width=64,
                height=64, spp=8, relmse_tol=0.1, flip_tol=0.1, material_class="flat",
                megakernel_ok=True)
    base.update(kw)
    return SceneSpec(**base)


def _sss_scene(**kw) -> SceneSpec:
    return _flat_scene(name="dragon", material_class="subsurface",
                       megakernel_ok=False, **kw)


# ─── validity table ────────────────────────────────────────────────────────

def test_sppm_is_wavefront_only():
    ok, reason = parity.combo_is_valid(RenderCombo("sppm", "megakernel"), _flat_scene())
    assert not ok and "wavefront-only" in reason
    assert parity.combo_is_valid(RenderCombo("sppm", "wavefront"), _flat_scene())[0]


def test_neural_requires_wavefront_path_flat():
    flat = _flat_scene()
    assert parity.combo_is_valid(RenderCombo("path", "wavefront", ("neural",)), flat)[0]
    # megakernel
    ok, reason = parity.combo_is_valid(RenderCombo("path", "megakernel", ("neural",)), flat)
    assert not ok and "wavefront-only" in reason
    # bdpt ignores neural
    ok, reason = parity.combo_is_valid(RenderCombo("bdpt", "wavefront", ("neural",)), flat)
    assert not ok and "path integrator" in reason
    # subsurface scene → neural skipped (flat-only)
    ok, reason = parity.combo_is_valid(RenderCombo("path", "wavefront", ("neural",)), _sss_scene())
    assert not ok and "flat-material" in reason


def test_restir_is_wavefront_only():
    ok, reason = parity.combo_is_valid(RenderCombo("path", "megakernel", (), "restir-di"), _flat_scene())
    assert not ok and "wavefront-only" in reason
    assert parity.combo_is_valid(RenderCombo("path", "wavefront", (), "restir-di"), _flat_scene())[0]


def test_heavy_scene_is_wavefront_only():
    sss = _sss_scene()
    ok, reason = parity.combo_is_valid(RenderCombo("path", "megakernel"), sss)
    assert not ok and "megakernel budget" in reason
    assert parity.combo_is_valid(RenderCombo("path", "wavefront"), sss)[0]


def _volume_scene(**kw) -> SceneSpec:
    return _flat_scene(name="disney_cloud", material_class="volume", **kw)


def test_volume_scene_is_path_only():
    vol = _volume_scene()
    # Path runs in both execution modes.
    assert parity.combo_is_valid(RenderCombo("path", "megakernel"), vol)[0]
    assert parity.combo_is_valid(RenderCombo("path", "wavefront"), vol)[0]
    # BDPT/SPPM have no volume transport — recorded exclusions.
    ok, reason = parity.combo_is_valid(RenderCombo("bdpt", "wavefront"), vol)
    assert not ok and "volume transport" in reason
    ok, reason = parity.combo_is_valid(RenderCombo("sppm", "wavefront"), vol)
    assert not ok and "volume transport" in reason
    # ReSTIR reuse untested with media; neural is flat-only (existing rule).
    ok, reason = parity.combo_is_valid(RenderCombo("path", "wavefront", (), "restir-di"), vol)
    assert not ok and "volume" in reason
    ok, reason = parity.combo_is_valid(RenderCombo("path", "wavefront", ("neural",)), vol)
    assert not ok and "flat-material" in reason


def test_anchor_is_valid_everywhere():
    assert parity.combo_is_valid(parity.ANCHOR, _flat_scene())[0]
    assert parity.combo_is_valid(parity.ANCHOR, _sss_scene())[0]


# ─── enumeration ────────────────────────────────────────────────────────────

def test_enumerate_flat_scene():
    combos = parity.enumerate_combos(_flat_scene())
    labels = {c.label for c in combos}
    assert combos[0] == parity.ANCHOR  # anchor first
    assert "path|megakernel" in labels
    assert "path|wavefront" in labels
    assert "bdpt|megakernel" in labels
    assert "bdpt|wavefront" in labels
    assert "sppm|wavefront" in labels
    assert "path|wavefront|neural" in labels
    assert "path|wavefront|restir-di" in labels
    # excluded by design
    assert "sppm|megakernel" not in labels
    assert "bdpt|wavefront|neural" not in labels
    assert "path|megakernel|neural" not in labels
    assert "bdpt|wavefront|restir-di" not in labels  # ReSTIR DI is path-only


def test_enumerate_subsurface_scene_is_wavefront_only_no_neural():
    combos = parity.enumerate_combos(_sss_scene())
    labels = {c.label for c in combos}
    assert all(c.execution_mode == "wavefront" for c in combos)
    assert not any(c.has_neural for c in combos)  # flat-only axis
    assert "sppm|wavefront" in labels
    assert "path|wavefront|restir-di" in labels


# ─── self-consistency tolerance classes ────────────────────────────────────

def test_axis_class_mapping():
    assert parity.combo_axis_class(RenderCombo("path", "megakernel")) == "mode"
    assert parity.combo_axis_class(RenderCombo("bdpt", "wavefront")) == "integrator"
    assert parity.combo_axis_class(RenderCombo("sppm", "wavefront")) == "sppm"
    assert parity.combo_axis_class(RenderCombo("path", "wavefront", ("neural",))) == "unbiased"
    assert parity.combo_axis_class(RenderCombo("path", "wavefront", (), "restir-di")) == "unbiased"


def test_self_consistency_tol_scene_override():
    scene = _flat_scene(self_consistency={"mode": {"relmse": 0.001, "flip": 0.002}})
    rel, flip = parity.self_consistency_tol(RenderCombo("path", "megakernel"), scene)
    assert (rel, flip) == (0.001, 0.002)
    # unspecified class falls back to default
    rel, flip = parity.self_consistency_tol(RenderCombo("sppm", "wavefront"), scene)
    assert rel == parity._DEFAULT_SELF_CONSISTENCY["sppm"]["relmse"]


# ─── coverage meta-test: app-exposed combos all have a validity entry ───────

def test_coverage_meta_app_integrators_covered():
    """Every integrator the headless app exposes must be in the validity table.

    Guards: a new integrator added to the app without a matrix entry fails here.
    """
    try:
        from skinny.headless import _INTEGRATORS
    except Exception as exc:  # noqa: BLE001 - GPU/renderer libs unavailable in this env
        pytest.skip(f"headless import unavailable: {exc}")
    app = set(_INTEGRATORS)
    table = set(parity.INTEGRATORS)
    assert app == table, (
        f"integrator coverage drift: app={sorted(app)} table={sorted(table)} — "
        f"add the missing integrator(s) to parity.INTEGRATORS + the validity table"
    )


def test_coverage_meta_execution_modes_pinned():
    assert parity.EXECUTION_MODES == ("megakernel", "wavefront")
    # every baseline (integrator × mode) is represented in the unfiltered space
    space = {(c.integrator, c.execution_mode) for c in parity.all_combos()}
    for integ in parity.INTEGRATORS:
        for mode in parity.EXECUTION_MODES:
            assert (integ, mode) in space
