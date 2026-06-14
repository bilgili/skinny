"""End-to-end emit gate for the variance harness (no GPU).

The GPU render half reuses the proven headless primitives (`read_accumulation_hdr`,
the seed/`frame_index` reset) that `tests/test_neural_headless.py` already gates;
this pins the OTHER half — reference gate → seed-aggregated variance → table +
SVG emit — by driving `emit()` on synthetic images. It is exactly the
`--quick` output path (task 5.2) minus the device, so a broken table/plot can't
ship even when the GPU smoke can't run on a given host.
"""
from __future__ import annotations

import json
import sys
import xml.dom.minidom as minidom
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from guiding_variance_sweep import CellResult, default_config, emit  # noqa: E402


def _synthetic_results():
    # Two cells with a plausible variance-vs-spp curve (var ∝ 1/spp).
    a = CellResult(
        label="bsdf|V1/E0/fp32", proposals="bsdf", chart="V1", encoding="E0",
        precision="fp32", reuse="none", equal_time_var=1.2e-3,
        equal_time_var_spread=1.1e-4, firefly_p999=4.0e-2, equal_time_s=0.8,
        efficiency=1.0 / (1.2e-3 * 0.8), eq_var_spp=120.0, eq_var_time_s=0.6,
        eq_var_reached=True, budget_curve=[(8, 1.0e-2, 0.05), (16, 5.0e-3, 0.10)],
    )
    b = CellResult(
        label="bsdf,neural|V1/E1/fp32", proposals="bsdf,neural", chart="V1",
        encoding="E1", precision="fp32", reuse="none", equal_time_var=9.0e-4,
        equal_time_var_spread=9.0e-5, firefly_p999=6.0e-2, equal_time_s=1.4,
        efficiency=1.0 / (9.0e-4 * 1.4), eq_var_spp=90.0, eq_var_time_s=1.1,
        eq_var_reached=False, budget_curve=[(8, 8.0e-3, 0.09), (16, 4.0e-3, 0.18)],
    )
    return [a, b]


def test_emit_writes_table_manifest_and_svgs(tmp_path):
    cfg = default_config(quick=True)
    ref = np.full((48, 48, 3), 0.42)
    results = _synthetic_results()
    skipped = [("bsdf,neural|V2/E0/fp32", "chart V2 not built (renderer-chart-selection)")]

    emit(tmp_path, "cornell", cfg, ref, 0.004, "deadbeefcafe0000",
         results, skipped, "metal")

    md = (tmp_path / "RESULTS_cornell.md").read_text()
    assert "% source:" in md                     # provenance for transcription
    assert "Equal-time" in md
    assert "1/(var·t)" in md                      # the headline metric present
    assert "Skipped cells" in md                  # coverage gap is visible
    assert "renderer-chart-selection" in md

    manifest = json.loads((tmp_path / "manifest_cornell.json").read_text())
    assert manifest["reference"]["hash"] == "deadbeefcafe0000"
    assert manifest["reference"]["witness_rel_err"] == 0.004
    assert len(manifest["results"]) == 2
    assert len(manifest["skipped"]) == 1

    # All four SVG figures exist and are well-formed XML.
    for name in ("cornell_equal_time.svg", "cornell_efficiency.svg",
                 "cornell_equal_variance.svg", "cornell_variance_vs_spp.svg"):
        svg = (tmp_path / name).read_text()
        assert svg.lstrip().startswith("<svg")
        minidom.parseString(svg)


def test_quick_config_is_minimal_and_complete():
    cfg = default_config(quick=True)
    assert len(cfg["scenes"]) == 1
    assert len(cfg["cells"]) == 2
    assert max(cfg["budgets"]) <= 64        # low spp
    assert cfg["seeds"] >= 2                # still multi-seed (a spread is reported)
