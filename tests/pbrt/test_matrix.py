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


# ─── spectral axis (change spectral-rendering) ──────────────────────────────


def test_spectral_envelope_accepts_path_megakernel_flat():
    # The intended v1 envelope accepts (path, megakernel) on a flat scene,
    # independent of whether the transport is wired yet.
    ok, _ = parity.spectral_envelope(
        RenderCombo("path", "megakernel", spectral=True), _flat_scene())
    assert ok


def test_spectral_not_yet_wired_is_skipped_not_rendered_as_rgb():
    # Until SPECTRAL_IMPLEMENTED flips (Group 5), an in-envelope spectral combo is
    # a recorded skip — never rendered (which would silently produce RGB).
    from skinny import spectral_capability
    assert spectral_capability.SPECTRAL_IMPLEMENTED is False
    ok, reason = parity.combo_is_valid(
        RenderCombo("path", "megakernel", spectral=True), _flat_scene())
    assert not ok and "not yet wired" in reason
    # And it is absent from the rendered (enumerated) set.
    assert not any(c.spectral for c in parity.enumerate_combos(_flat_scene()))


def test_spectral_becomes_valid_when_wired(monkeypatch):
    # Flipping the capability gate admits the in-envelope combo into the sweep.
    from skinny import spectral_capability
    monkeypatch.setattr(spectral_capability, "SPECTRAL_IMPLEMENTED", True)
    ok, _ = parity.combo_is_valid(
        RenderCombo("path", "megakernel", spectral=True), _flat_scene())
    assert ok
    rendered = parity.enumerate_combos(_flat_scene())
    assert RenderCombo("path", "megakernel", spectral=True) in rendered
    # Out-of-envelope combos stay skipped even when wired.
    for bad in [RenderCombo("bdpt", "megakernel", spectral=True),
                RenderCombo("path", "wavefront", spectral=True)]:
        assert not parity.combo_is_valid(bad, _flat_scene())[0]


def test_spectral_bdpt_skipped_path_only():
    ok, reason = parity.combo_is_valid(
        RenderCombo("bdpt", "megakernel", spectral=True), _flat_scene())
    assert not ok and "path-only" in reason


@pytest.mark.parametrize("integrator", ["bdpt", "sppm"])
def test_spectral_non_path_skipped(integrator):
    # bdpt/sppm spectral combos are always skipped with a reason (sppm trips the
    # earlier SPPM-wavefront-only rule; bdpt trips the spectral path-only rule).
    ok, reason = parity.combo_is_valid(
        RenderCombo(integrator, "megakernel", spectral=True), _flat_scene())
    assert not ok and reason


def test_spectral_wavefront_skipped():
    ok, reason = parity.combo_is_valid(
        RenderCombo("path", "wavefront", spectral=True), _flat_scene())
    assert not ok and "megakernel-only" in reason


def test_spectral_with_neural_skipped():
    ok, reason = parity.combo_is_valid(
        RenderCombo("path", "megakernel", ("neural",), spectral=True), _flat_scene())
    assert not ok and "neural" in reason


def test_spectral_with_reuse_skipped():
    ok, reason = parity.combo_is_valid(
        RenderCombo("path", "megakernel", (), "restir-di", spectral=True), _flat_scene())
    assert not ok and "ReSTIR" in reason


@pytest.mark.parametrize("scene", [_sss_scene(), _volume_scene()])
def test_spectral_non_flat_scene_all_skipped(scene):
    # Every spectral combo is skipped on a skin/subsurface or volume scene.
    spectral_combos = [c for c in parity.all_combos() if c.spectral]
    assert spectral_combos  # the axis is enumerated
    for c in spectral_combos:
        ok, reason = parity.combo_is_valid(c, scene)
        assert not ok and reason


def test_spectral_absent_from_enumeration_until_wired():
    # While unwired, no spectral combo is in the rendered set (see the
    # not-yet-wired test); only (path, megakernel) is even envelope-eligible.
    eligible = [c for c in parity.all_combos()
                if c.spectral and parity.spectral_envelope(c, _flat_scene())[0]]
    assert eligible == [RenderCombo("path", "megakernel", spectral=True)]
    assert "spectral" in eligible[0].label
    assert not any(c.spectral for c in parity.enumerate_combos(_flat_scene()))


def test_coverage_meta_spectral_axis_covered():
    """Every integrator × spectral has a validity verdict (never crashes/omits)."""
    space = {(c.integrator, c.execution_mode, c.spectral) for c in parity.all_combos()}
    for integ in parity.INTEGRATORS:
        for mode in parity.EXECUTION_MODES:
            assert (integ, mode, True) in space, f"spectral axis missing {integ}/{mode}"
    # combo_is_valid returns a reason for every spectral combo on any scene class.
    for scene in (_flat_scene(), _sss_scene(), _volume_scene()):
        for c in (c for c in parity.all_combos() if c.spectral):
            ok, reason = parity.combo_is_valid(c, scene)
            assert ok or reason  # a skip always carries a reason
