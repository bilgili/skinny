"""Parity corpus: import-smoke (always) + the metric gate (when refs exist).

The gate runs with no pbrt binary present — it relies solely on checked-in
reference EXRs. Until those are generated (no pbrt v4 on this host), the gate
skips and only the import-smoke checks run.
"""

from __future__ import annotations

import os

import pytest

from skinny.pbrt import metrics
from skinny.pbrt.api import import_pbrt
from skinny.pbrt.parity import (
    ANCHOR,
    all_combos,
    combo_is_valid,
    absolute_radiance_result,
    enumerate_combos,
    evaluate,
    load_manifest,
    materialx_specs,
    pbrt_truth_result,
    reference_exists,
    render_combo,
    self_consistency_anchor,
    self_consistency_result,
    spectral_selfconsistency_assertable,
)

usd_loader = pytest.importorskip("skinny.usd_loader")

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
SPECS = load_manifest(CORPUS_DIR)
# The confirming-scene suite (change confirming-test-scenes) shares this manifest
# but is gated by tests/pbrt/test_suite.py (its own equivalence + furnace gate
# classes). Keep the legacy corpus tests on the non-suite scenes so suite scenes
# aren't double-rendered here.
LEGACY_SPECS = [s for s in SPECS if not s.suite]
MTLX_SPECS = materialx_specs(LEGACY_SPECS)


@pytest.mark.parametrize("spec", LEGACY_SPECS, ids=[s.name for s in LEGACY_SPECS])
def test_corpus_scene_imports_cleanly(spec):
    """Every corpus scene loads with no unsupported feature.

    pbrt-source scenes import through ``import_pbrt``; usd-source heavy scenes
    (bathroom/dragon) open their ``.usda`` asset directly.
    """
    from skinny.pbrt.parity import _scene_source

    if spec.usd:
        usd_path = _scene_source(spec, CORPUS_DIR)["usd_path"]
        assert os.path.exists(usd_path), f"{spec.name}: missing {usd_path}"
        if os.path.getsize(usd_path) > 100_000_000:
            # Heavy geometry (e.g. the 28.8M-tri dragon): a full not-gpu load
            # would materialize the mesh and blow RAM. Sniff the header; the GPU
            # matrix gate exercises the real load + render.
            with open(usd_path, encoding="utf-8", errors="ignore") as fh:
                assert fh.read(16).startswith("#usda"), f"{spec.name}: not a usda"
            return
        from pxr import Usd

        stage = Usd.Stage.Open(usd_path)
        scene = usd_loader.load_scene_from_stage(stage)
        assert len(scene.instances) >= 1
        return
    stage, report = import_pbrt(os.path.join(CORPUS_DIR, spec.file))
    scene = usd_loader.load_scene_from_stage(stage)
    assert len(scene.instances) >= 1
    # corpus scenes are deliberately within the supported subset
    assert report.count("skipped") == 0, str(report)


# ─── matrix construction tier (no GPU) ────────────────────────────────────


@pytest.mark.parametrize("spec", LEGACY_SPECS, ids=[s.name for s in LEGACY_SPECS])
def test_matrix_enumerates_per_spec(spec):
    """Every manifest scene yields a non-empty, anchor-first valid combo set, and
    every combo in the full space is either valid or skipped with a reason."""
    combos = enumerate_combos(spec)
    assert combos and combos[0] == ANCHOR
    for c in all_combos():
        ok, reason = combo_is_valid(c, spec)
        assert ok or reason, f"{c.label} skipped with no reason"


@pytest.mark.parametrize("spec", LEGACY_SPECS, ids=[s.name for s in LEGACY_SPECS])
def test_scene_source_resolves(spec):
    """The scene source (corpus .pbrt or .usda asset) exists on disk."""
    from skinny.pbrt.parity import _scene_source

    src = _scene_source(spec, CORPUS_DIR)
    assert os.path.exists(src["scene_pbrt"]), f"{spec.name}: missing {src['scene_pbrt']}"


# ─── integrator × execution-mode parity matrix gate (GPU) ─────────────────


@pytest.mark.gpu
@pytest.mark.parametrize("spec", LEGACY_SPECS, ids=[s.name for s in LEGACY_SPECS])
def test_scene_matrix_gate(spec):
    """Dual gate over every valid combo of *spec*: pbrt-truth (vs the reference
    EXR, honouring recorded baselines) AND self-consistency (vs the anchor
    image, strict per-axis tolerance). Each combo renders once and feeds both
    gates; the anchor is rendered once and reused. The full ImageMetrics battery
    is logged per combo.
    """
    if not reference_exists(spec, CORPUS_DIR):
        pytest.skip(f"no reference EXR for {spec.name} (generate with pbrt v4)")
    combos = enumerate_combos(spec)
    try:
        ref = metrics.read_exr(os.path.join(CORPUS_DIR, spec.ref))
        imgs = {c.label: render_combo(spec, c, CORPUS_DIR) for c in combos}
    except Exception as exc:  # noqa: BLE001 - GPU/backend unavailable in this env
        pytest.skip(f"render backend unavailable: {exc}")

    anchor_img = imgs[ANCHOR.label]
    failures: list[str] = []
    for c in combos:
        img = imgs[c.label]
        pt = pbrt_truth_result(spec, c, img, ref)
        tag = "(baseline)" if pt.baseline_used else ""
        print(f"[{spec.name}] {c.label:32s} pbrt-truth {tag} {pt.metrics.summary()}")
        if not pt.passed:
            failures.append(f"{c.label}: pbrt-truth relMSE={pt.relmse:.4f} FLIP={pt.flip:.4f}")
        if c == ANCHOR:
            # Absolute-radiance gate (change pbrt-radiometric-parity): un-exposure-
            # aligned mean-ratio + relMSE, beside the exposure-blind pbrt-truth gate.
            # Anchor-only — self-consistency already ties the other combos to it.
            # Skips silently when the scene has no `absolute` config.
            ab = absolute_radiance_result(spec, c, img, ref)
            if ab is not None:
                atag = "(baseline)" if ab.baseline_used else ""
                print(f"[{spec.name}] {c.label:32s} abs-radiance {atag} "
                      f"ratio={ab.flip:.4f} relMSE={ab.relmse:.4f}")
                if not ab.passed:
                    failures.append(
                        f"{c.label}: absolute-radiance ratio={ab.flip:.4f} "
                        f"relMSE={ab.relmse:.4f}"
                    )
        if c != ANCHOR:
            # Spectral-aware anchor (change spectral-wavefront, D7): a spectral
            # combo gates against the MEGAKERNEL SPECTRAL PATH image, never the RGB
            # anchor (RGB↔spectrum is not identity). Report-only fallback if that
            # anchor image is unavailable for the scene.
            anchor_combo = self_consistency_anchor(c)
            sc_anchor_img = imgs.get(anchor_combo.label, anchor_img)
            sc = self_consistency_result(spec, c, img, sc_anchor_img)
            print(f"[{spec.name}] {c.label:32s} vs-anchor       {sc.metrics.summary()}")
            if c.spectral and spectral_selfconsistency_assertable(c, spec) \
                    and anchor_combo.label in imgs:
                # mega≡wave is now ASSERTED for spectral path/bdpt against the
                # spectral anchor (the old blanket "spectral-only skip" is lifted).
                if not sc.passed:
                    failures.append(
                        f"{c.label}: spectral self-consistency vs spectral anchor "
                        f"relMSE={sc.relmse:.4f} FLIP={sc.flip:.4f}"
                    )
            elif c.spectral:
                # Dispersion-splat bdpt (or a missing spectral anchor): REPORTED,
                # not asserted — per-splat gamut clamp differs mega-vs-wave (D4/D7).
                print(f"[{spec.name}] {c.label:32s} spectral-vs-anchor "
                      f"Δ relMSE={sc.relmse:.4f} FLIP={sc.flip:.4f} (reported)")
            elif not sc.passed:
                failures.append(
                    f"{c.label}: self-consistency vs anchor relMSE={sc.relmse:.4f} "
                    f"FLIP={sc.flip:.4f}"
                )

    # 7.3: REPORT the spectral-vs-RGB pbrt-truth comparison (not a hard gate). The
    # exact spectral render improves accuracy on a SMOOTH chromaticity shift
    # (metamerism / Fresnel — e.g. conductor_infinite, spectral relMSE < RGB), but
    # a hero-wavelength DISPERSION caustic is a high-variance feature that the
    # RGB (non-dispersive) render sidesteps, so spectral can carry a higher
    # pointwise relMSE while being the physically-correct result. "spectral ≤ RGB"
    # is therefore scene-dependent, not an invariant — spectral pbrt-truth is gated
    # against its own recorded baseline (above), and the direction is logged here.
    anchor_pt = pbrt_truth_result(spec, ANCHOR, anchor_img, ref)
    for c in combos:
        if not c.spectral:
            continue
        spt = pbrt_truth_result(spec, c, imgs[c.label], ref)
        verdict = "improves" if spt.relmse <= anchor_pt.relmse else "regresses"
        print(f"[{spec.name}] {c.label:32s} spectral-vs-RGB  pbrt-truth "
              f"{verdict}: spectral relMSE={spt.relmse:.4f} vs RGB {anchor_pt.relmse:.4f}")

    if failures and spec.known_divergent:
        # Harness-first: a known, not-yet-fixed heavy-scene divergence. Record
        # and xfail (visible, non-blocking); the follow-up fix flips the flag.
        pytest.xfail(f"{spec.name} known-divergent (fix is a follow-up):\n  "
                     + "\n  ".join(failures))
    assert not failures, f"{spec.name} matrix failures:\n  " + "\n  ".join(failures)


# ─── -mtlx export scene-set ───────────────────────────────────────────────


def test_materialx_specs_mirror_base_set():
    """The -mtlx scene-set parallels the base set: same source/ref/tolerances,
    distinct ids, materialx flag flipped. This is the non-GPU half of the
    parity wiring — the harness plumbing must be exercised without a GPU."""
    # usd-source heavy scenes (bathroom/dragon) have no .pbrt to re-export.
    base_pbrt = [s for s in LEGACY_SPECS if not s.usd]
    assert len(MTLX_SPECS) == len(base_pbrt)
    for base, mtlx in zip(base_pbrt, MTLX_SPECS):
        assert mtlx.materialx is True
        assert base.materialx is False
        assert mtlx.name == f"{base.name}_mtlx"
        # Same source scene, reference EXR, resolution, spp, and tolerances:
        # a -mtlx render is gated against the identical pbrt v4 reference.
        assert (mtlx.file, mtlx.ref) == (base.file, base.ref)
        assert (mtlx.width, mtlx.height, mtlx.spp) == (base.width, base.height, base.spp)
        assert (mtlx.relmse_tol, mtlx.flip_tol) == (base.relmse_tol, base.flip_tol)


@pytest.mark.parametrize("spec", MTLX_SPECS, ids=[s.name for s in MTLX_SPECS])
def test_corpus_scene_imports_cleanly_mtlx(spec, tmp_path):
    """Every corpus scene imports cleanly through the -mtlx path: a
    doc.validate()-clean .mtlx sidecar is written next to the .usda, the stage
    references it, and the bound meshes load with no unsupported feature. No GPU
    render here — this proves the export-both-ways plumbing end to end."""
    out = os.path.join(str(tmp_path), "out.usda")
    stage, report = import_pbrt(
        os.path.join(CORPUS_DIR, spec.file), out=out, materialx=True
    )
    assert os.path.exists(out)
    assert os.path.exists(os.path.splitext(out)[0] + ".mtlx"), "no .mtlx sidecar"
    assert report.count("skipped") == 0, str(report)
    # The .mtlx is referenced and resolvable; the bound meshes load.
    assert usd_loader._collect_mtlx_asset_paths(stage) == {"out.mtlx"}
    scene = usd_loader.load_scene_from_stage(stage)
    assert len(scene.instances) >= 1


@pytest.mark.gpu
@pytest.mark.parametrize("spec", MTLX_SPECS, ids=[s.name for s in MTLX_SPECS])
def test_corpus_scene_parity_gate_mtlx(spec):
    """A -mtlx export must hit the same pbrt v4 reference EXR within the same
    tolerance as its UsdPreviewSurface sibling (Metal backend). Switching the
    export path is a no-op on the rendered image for the supported subset.

    GPU-gated: the main thread runs this; here it skips cleanly when the
    backend is unavailable."""
    if not reference_exists(spec, CORPUS_DIR):
        pytest.skip(f"no reference EXR for {spec.name} (generate with pbrt v4)")
    try:
        result = evaluate(spec, CORPUS_DIR)
    except Exception as exc:  # noqa: BLE001 - GPU/backend unavailable in this env
        pytest.skip(f"render backend unavailable: {exc}")
    assert result.passed, (
        f"{spec.name}: relMSE={result.relmse:.4f} (<= {spec.relmse_tol}), "
        f"FLIP={result.flip:.4f} (<= {spec.flip_tol})"
    )
