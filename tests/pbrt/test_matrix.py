"""Render-parity matrix: validity table, enumeration, tolerances, coverage.

All no-GPU: the matrix is data-driven, so its construction and the
compatibility rules are tested without rendering.
"""

from __future__ import annotations

import os as _os
import re as _re

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


def test_spectral_self_consistency_tol_is_separate_from_rgb():
    # change spectral-wavefront GPU-validation: spectral combos consult a separate
    # tolerance table (wider on the sample-sharing classes because spectral
    # wavefront is not bit-identical to the megakernel). The RGB table is NEVER
    # loosened by a spectral override, so the RGB mega≡wave bit-identity gate holds.
    scene = _flat_scene(spectral_self_consistency={"mode": {"relmse": 0.085, "flip": 0.03}})
    rgb = parity.self_consistency_tol(RenderCombo("path", "wavefront"), scene)
    assert rgb == (parity._DEFAULT_SELF_CONSISTENCY["mode"]["relmse"],
                   parity._DEFAULT_SELF_CONSISTENCY["mode"]["flip"])
    # spectral path (mode class) picks up the scene override
    sp = parity.self_consistency_tol(RenderCombo("path", "wavefront", spectral=True), scene)
    assert sp == (0.085, 0.03)
    # spectral class with no scene override falls back to the SPECTRAL default,
    # which is wider than the RGB default on the sample-sharing classes
    sp_bdpt = parity.self_consistency_tol(RenderCombo("bdpt", "wavefront", spectral=True), scene)
    assert sp_bdpt[0] == parity._DEFAULT_SPECTRAL_SELF_CONSISTENCY["integrator"]["relmse"]
    assert sp_bdpt[0] > parity._DEFAULT_SELF_CONSISTENCY["integrator"]["relmse"]


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


def test_spectral_wired_is_admitted_to_the_sweep():
    # SPECTRAL_IMPLEMENTED ships True (Groups 4-6 landed): an in-envelope spectral
    # combo is valid and present in the rendered (enumerated) set.
    from skinny import spectral_capability
    assert spectral_capability.SPECTRAL_IMPLEMENTED is True
    ok, _ = parity.combo_is_valid(
        RenderCombo("path", "megakernel", spectral=True), _flat_scene())
    assert ok
    assert any(c.spectral for c in parity.enumerate_combos(_flat_scene()))


def test_spectral_skipped_when_gate_off(monkeypatch):
    # Forcing the capability gate off returns the in-envelope combo to a recorded
    # "not yet wired" skip, absent from the rendered set (never silent RGB).
    from skinny import spectral_capability
    monkeypatch.setattr(spectral_capability, "SPECTRAL_IMPLEMENTED", False)
    ok, reason = parity.combo_is_valid(
        RenderCombo("path", "megakernel", spectral=True), _flat_scene())
    assert not ok and "not yet wired" in reason
    assert not any(c.spectral for c in parity.enumerate_combos(_flat_scene()))


def test_spectral_out_of_envelope_skipped_even_when_wired():
    # Out-of-envelope combos stay skipped with the gate on. (path/bdpt run in
    # BOTH execution modes and sppm under wavefront — change spectral-wavefront —
    # so the remaining out-of-envelope cases are sppm+megakernel (no megakernel
    # photon pass) and the neural / ReSTIR reuse axes.)
    for bad in [RenderCombo("sppm", "megakernel", spectral=True),
                RenderCombo("path", "megakernel", ("neural",), spectral=True),
                RenderCombo("path", "wavefront", (), "restir-di", spectral=True)]:
        assert not parity.combo_is_valid(bad, _flat_scene())[0]


def test_spectral_bdpt_admitted():
    # BDPT under the megakernel is in the spectral envelope (change
    # spectral-bdpt-megakernel) and, with the transport wired, rendered.
    ok, reason = parity.combo_is_valid(
        RenderCombo("bdpt", "megakernel", spectral=True), _flat_scene())
    assert ok, reason


def test_spectral_sppm_megakernel_skipped():
    # SPPM has no megakernel path, so spectral SPPM under the megakernel is skipped
    # with a reason (it trips the earlier SPPM-wavefront-only rule). Spectral SPPM
    # under WAVEFRONT is admitted — see test_spectral_wavefront_admitted.
    ok, reason = parity.combo_is_valid(
        RenderCombo("sppm", "megakernel", spectral=True), _flat_scene())
    assert not ok and reason


def test_spectral_wavefront_admitted():
    # change spectral-wavefront: spectral path/bdpt/sppm under the wavefront
    # execution mode are now valid rendered combos (flat, no neural, no reuse).
    flat = _flat_scene()
    for integ in ("path", "bdpt", "sppm"):
        ok, reason = parity.combo_is_valid(
            RenderCombo(integ, "wavefront", spectral=True), flat)
        assert ok, f"{integ} spectral wavefront should be admitted: {reason}"


def test_spectral_sppm_wavefront_validity_entry_exists():
    # 6.3: sppm × spectral has a rendered validity entry (under wavefront) so the
    # coverage meta-test does not fail for a missing spectral sppm combo.
    ok, reason = parity.combo_is_valid(
        RenderCombo("sppm", "wavefront", spectral=True), _flat_scene())
    assert ok, reason
    assert any(c.spectral and c.integrator == "sppm"
               for c in parity.enumerate_combos(_flat_scene()))


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


def test_spectral_envelope_is_path_bdpt_both_modes_and_sppm_wavefront():
    # change spectral-wavefront: the envelope is path/bdpt in EITHER execution
    # mode plus sppm under wavefront (flat, no neural, no reuse). With the gate on
    # (ships True) these are exactly the spectral entries in the rendered set.
    expected = {RenderCombo("path", "megakernel", spectral=True),
                RenderCombo("path", "wavefront", spectral=True),
                RenderCombo("bdpt", "megakernel", spectral=True),
                RenderCombo("bdpt", "wavefront", spectral=True),
                RenderCombo("sppm", "wavefront", spectral=True)}
    eligible = {c for c in parity.all_combos()
                if c.spectral and parity.spectral_envelope(c, _flat_scene())[0]}
    assert eligible == expected
    assert all("spectral" in c.label for c in eligible)
    rendered_spectral = {c for c in parity.enumerate_combos(_flat_scene()) if c.spectral}
    assert rendered_spectral == expected


def test_self_consistency_anchor_is_spectral_aware():
    # 6.2: the spectral axis anchors to the megakernel spectral path image, never
    # the RGB golden; RGB combos keep the RGB anchor.
    assert parity.self_consistency_anchor(RenderCombo("path", "wavefront")) == parity.ANCHOR
    sp = RenderCombo("path", "wavefront", spectral=True)
    assert parity.self_consistency_anchor(sp) == parity.SPECTRAL_ANCHOR
    assert parity.SPECTRAL_ANCHOR.spectral
    assert parity.SPECTRAL_ANCHOR.execution_mode == "megakernel"
    assert parity.SPECTRAL_ANCHOR.integrator == "path"


def test_spectral_axis_class_vs_spectral_anchor():
    # 6.2: axis class is measured against the spectral anchor, so a spectral
    # wavefront path is a "mode" delta (not conflated with the RGB→spectral shift).
    assert parity.combo_axis_class(RenderCombo("path", "wavefront", spectral=True)) == "mode"
    assert parity.combo_axis_class(RenderCombo("bdpt", "wavefront", spectral=True)) == "integrator"
    assert parity.combo_axis_class(RenderCombo("sppm", "wavefront", spectral=True)) == "sppm"


def test_spectral_selfconsistency_dispersion_bdpt_is_reported_only():
    # 6.2/6.4 (D4/D7): spectral bdpt on an out-of-gamut dispersion (light-tracer
    # splat) scene is the one retained mega≡wave skip (reported, not asserted).
    disp = _flat_scene(spectral={"kind": "dispersion", "glass": "bk7"})
    assert not parity.spectral_selfconsistency_assertable(
        RenderCombo("bdpt", "wavefront", spectral=True), disp)
    # spectral path on the same scene stays a hard assertion.
    assert parity.spectral_selfconsistency_assertable(
        RenderCombo("path", "wavefront", spectral=True), disp)
    # bdpt on a NON-dispersion spectral scene is assertable.
    assert parity.spectral_selfconsistency_assertable(
        RenderCombo("bdpt", "wavefront", spectral=True), _flat_scene())
    # RGB combos are always assertable; the spectral anchor is not self-compared.
    assert parity.spectral_selfconsistency_assertable(RenderCombo("path", "wavefront"), disp)
    assert not parity.spectral_selfconsistency_assertable(parity.SPECTRAL_ANCHOR, _flat_scene())


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


# ─── corpus scene-data integrity (change parity-scene-asset-integrity) ─────
# The GPU gates render whatever is on disk; a silently-deleted side-file (the
# disney_cloud baked constant sky) or an undeclared material class (the
# subsurface_infinite spectral SystemExit) turns into an opaque gate failure.
# These hostless checks catch both classes at unit-test time.

_CORPUS_DIR = _os.path.join(_os.path.dirname(__file__), "corpus")
_REPO_ROOT = _os.path.abspath(_os.path.join(_CORPUS_DIR, "..", "..", ".."))


def test_usd_asset_texture_refs_exist():
    """Every texture:file authored in an on-disk manifest .usda must resolve.

    A scene whose usd asset is absent on this checkout (untracked assets live
    only in the main checkout) is skipped, not failed.
    """
    dangling: list[str] = []
    checked = 0
    for spec in parity.load_manifest(_CORPUS_DIR):
        if not spec.usd:
            continue
        usd = spec.usd if _os.path.isabs(spec.usd) else _os.path.join(_REPO_ROOT, spec.usd)
        if not _os.path.exists(usd):
            continue  # asset not on this checkout (e.g. a worktree)
        checked += 1
        with open(usd, encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        for ref in _re.findall(r"texture:file\s*=\s*@([^@]+)@", txt):
            p = ref if _os.path.isabs(ref) else _os.path.join(_os.path.dirname(usd), ref)
            if not _os.path.exists(p):
                dangling.append(f"{spec.name}: {ref!r} (in {usd})")
        # A .usda authoring a volume field must sit in the manifest as a
        # volume scene (the validity table keys BDPT/SPPM exclusions off it).
        if "OpenVDBAsset" in txt and spec.material_class != "volume":
            dangling.append(f"{spec.name}: authors OpenVDBAsset but material_class="
                            f"{spec.material_class!r} (in {usd})")
    if checked == 0:
        pytest.skip("no manifest usd assets present on this checkout")
    assert not dangling, "dangling texture:file references:\n  " + "\n  ".join(dangling)


def test_nonflat_pbrt_scene_declares_material_class():
    """A corpus .pbrt authoring subsurface/named-medium content must not sit in
    the manifest as the default flat class (the spectral envelope would admit
    combos the renderer refuses at scene build)."""
    missing: list[str] = []
    for spec in parity.load_manifest(_CORPUS_DIR):
        src = _os.path.join(_CORPUS_DIR, spec.file)
        if not _os.path.exists(src):
            continue
        with open(src, encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        nonflat = 'Material "subsurface"' in txt or "MakeNamedMedium" in txt
        if nonflat and spec.material_class == "flat":
            missing.append(spec.name)
    assert not missing, (
        "scenes author subsurface/medium content but declare no material_class: "
        + ", ".join(missing)
    )
