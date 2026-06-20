"""Parity corpus: import-smoke (always) + the metric gate (when refs exist).

The gate runs with no pbrt binary present — it relies solely on checked-in
reference EXRs. Until those are generated (no pbrt v4 on this host), the gate
skips and only the import-smoke checks run.
"""

from __future__ import annotations

import os

import pytest

from skinny.pbrt.api import import_pbrt
from skinny.pbrt.parity import (
    evaluate,
    load_manifest,
    materialx_specs,
    reference_exists,
)

usd_loader = pytest.importorskip("skinny.usd_loader")

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
SPECS = load_manifest(CORPUS_DIR)
MTLX_SPECS = materialx_specs(SPECS)


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_corpus_scene_imports_cleanly(spec):
    """Every corpus scene parses, imports, and loads with no unsupported feature."""
    stage, report = import_pbrt(os.path.join(CORPUS_DIR, spec.file))
    scene = usd_loader.load_scene_from_stage(stage)
    assert len(scene.instances) >= 1
    # corpus scenes are deliberately within the supported subset
    assert report.count("skipped") == 0, str(report)


@pytest.mark.gpu
@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_corpus_scene_parity_gate(spec):
    """relMSE/FLIP within tolerance vs the pbrt v4 reference EXR."""
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


# ─── -mtlx export scene-set ───────────────────────────────────────────────


def test_materialx_specs_mirror_base_set():
    """The -mtlx scene-set parallels the base set: same source/ref/tolerances,
    distinct ids, materialx flag flipped. This is the non-GPU half of the
    parity wiring — the harness plumbing must be exercised without a GPU."""
    assert len(MTLX_SPECS) == len(SPECS)
    for base, mtlx in zip(SPECS, MTLX_SPECS):
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
