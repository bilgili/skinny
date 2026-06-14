"""Unit gates for the guiding-variance metrics core (no GPU).

These pin the MAX-rigor measurement layer whose outputs become paper claims:
the convergence gate, the seed-aggregated variance with its spread, the
`1/(var·t)` efficiency identity, the equal-variance inversion, and — task 3.5 —
the estimator-validation case (a uniform / deterministic image → ~0 variance).
SVG/table emitters are checked for well-formedness so the harness can't ship a
broken figure.
"""
from __future__ import annotations

import xml.dom.minidom as minidom

import numpy as np
import pytest

from guiding_variance_metrics import (
    BudgetPoint,
    ReferenceNotConverged,
    assert_reference_converged,
    bulk_mse,
    efficiency,
    equal_variance_budget,
    firefly_percentile,
    image_mse,
    markdown_table,
    rel_mean_diff,
    svg_bar_chart,
    svg_line_chart,
    variance_over_seeds,
)


# --- reference convergence gate -------------------------------------------

def test_reference_gate_passes_on_agreeing_refs():
    rng = np.random.default_rng(0)
    base = rng.random((16, 16, 3))
    a = base + 0.001 * rng.standard_normal(base.shape)
    b = base + 0.001 * rng.standard_normal(base.shape)
    w = assert_reference_converged(a, b, rel_tol=0.01)
    assert w < 0.01


def test_reference_gate_rejects_disagreeing_refs():
    a = np.full((8, 8, 3), 1.0)
    b = np.full((8, 8, 3), 1.5)  # 50% energy gap
    with pytest.raises(ReferenceNotConverged):
        assert_reference_converged(a, b, rel_tol=0.01)


def test_reference_gate_running_err_witness():
    ref = np.zeros((4, 4, 3))
    assert assert_reference_converged(ref, running_err=0.005, rel_tol=0.01) == 0.005
    with pytest.raises(ReferenceNotConverged):
        assert_reference_converged(ref, running_err=0.2, rel_tol=0.01)


def test_reference_gate_needs_a_witness():
    with pytest.raises(ValueError):
        assert_reference_converged(np.zeros((2, 2)))


# --- 3.5 estimator validation: uniform / deterministic → ~0 ---------------

def test_uniform_image_has_zero_error_and_variance():
    """The trivial known case (task 3.5): an image identical to the reference
    has zero MSE, zero firefly tail, and (over identical seeds) zero estimator
    variance."""
    ref = np.full((12, 12, 3), 0.37)
    img = ref.copy()
    assert image_mse(img, ref) == 0.0
    assert firefly_percentile(img, ref) == 0.0
    cv = variance_over_seeds([ref.copy(), ref.copy(), ref.copy()], ref)
    assert cv.var_mean == 0.0
    assert cv.var_spread == 0.0
    # np.var of bit-identical seeds is ~0 modulo a two-pass-mean fp artifact.
    assert cv.seed_pixel_var_mean < 1e-20
    assert cv.firefly_p999_mean == 0.0


def test_seed_variance_recovers_known_variance():
    """A controlled case: seeds drawn N(ref, σ²) recover σ² as the per-pixel
    estimator variance and σ² as the MSE-vs-ref (both → σ² for large N)."""
    rng = np.random.default_rng(42)
    ref = np.full((64, 64, 3), 0.5)
    sigma = 0.1
    seeds = [ref + sigma * rng.standard_normal(ref.shape) for _ in range(200)]
    cv = variance_over_seeds(seeds, ref)
    assert cv.n_seeds == 200
    # per-pixel variance across seeds ≈ σ²
    assert cv.seed_pixel_var_mean == pytest.approx(sigma**2, rel=0.1)
    # MSE vs ref ≈ σ² too (ref is the true mean)
    assert cv.var_mean == pytest.approx(sigma**2, rel=0.1)
    assert cv.var_spread > 0.0  # a real spread is reported, not a single number


def test_single_seed_has_zero_spread_but_is_flagged():
    ref = np.zeros((4, 4, 3))
    cv = variance_over_seeds([np.ones((4, 4, 3))], ref)
    assert cv.n_seeds == 1
    assert cv.var_spread == 0.0  # std undefined for n=1 → 0, n_seeds exposes it


# --- firefly tail ----------------------------------------------------------

def test_firefly_percentile_isolates_outliers():
    ref = np.zeros((10, 10, 3))
    img = np.zeros((10, 10, 3))
    img[0, 0, 0] = 100.0  # one firefly
    assert image_mse(img, ref) == pytest.approx(100.0**2 / (10 * 10 * 3))
    # 99.9th percentile catches the single hot element
    assert firefly_percentile(img, ref, 99.9) > 1.0
    # the median (50th) does not
    assert firefly_percentile(img, ref, 50.0) == 0.0


def test_bulk_mse_clamps_fireflies():
    ref = np.zeros((10, 10, 3))
    img = np.zeros((10, 10, 3))
    img[0, 0, 0] = 100.0
    assert bulk_mse(img, ref, clip=1.0) == pytest.approx(1.0 / 300)  # clamped to 1
    assert bulk_mse(img, ref, clip=1.0) < image_mse(img, ref)


# --- efficiency 1/(var·t) (task 3.4) --------------------------------------

def test_efficiency_identity():
    assert efficiency(2.0, 0.5) == pytest.approx(1.0)
    assert efficiency(1e-4, 10.0) == pytest.approx(1.0 / (1e-4 * 10.0))
    # lower variance OR lower time ⇒ higher efficiency
    assert efficiency(1e-4, 1.0) > efficiency(1e-3, 1.0)
    assert efficiency(1e-4, 1.0) > efficiency(1e-4, 2.0)


def test_efficiency_guards_zero():
    assert np.isfinite(efficiency(0.0, 0.0))


# --- equal-variance inversion (task 3.3) ----------------------------------

def test_equal_variance_inverts_one_over_spp():
    """var ∝ 1/spp ⇒ to halve var you double spp; wallclock scales with spp."""
    grid = [
        BudgetPoint(spp=16, time_s=1.0, var=1.0 / 16),
        BudgetPoint(spp=64, time_s=4.0, var=1.0 / 64),
        BudgetPoint(spp=256, time_s=16.0, var=1.0 / 256),
    ]
    ev = equal_variance_budget(grid, target_var=1.0 / 128)
    assert ev.reached
    assert ev.spp == pytest.approx(128, rel=0.02)
    assert ev.time_s == pytest.approx(8.0, rel=0.05)


def test_equal_variance_flags_target_outside_grid():
    grid = [
        BudgetPoint(spp=16, time_s=1.0, var=1.0 / 16),
        BudgetPoint(spp=64, time_s=4.0, var=1.0 / 64),
    ]
    ev = equal_variance_budget(grid, target_var=1.0 / 4096)  # far below the grid
    assert not ev.reached
    assert "extrapolated" in ev.note


def test_equal_variance_needs_two_points():
    with pytest.raises(ValueError):
        equal_variance_budget([BudgetPoint(16, 1.0, 0.1)], 0.05)


# --- output emitters -------------------------------------------------------

def test_markdown_table_carries_provenance():
    rows = [{"cell": "V0/E0", "var": 1.2e-3, "eff": 9.9e2}]
    md = markdown_table(
        rows, [("cell", "config"), ("var", "var"), ("eff", "1/(var·t)")],
        title="Equal-time", source="guiding_variance_sweep.py --slice chart",
    )
    assert "% source:" in md
    assert "guiding_variance_sweep.py" in md
    assert "V0/E0" in md
    assert "| config | var | 1/(var·t) |" in md


def _assert_valid_svg(svg: str):
    assert svg.lstrip().startswith("<svg")
    minidom.parseString(svg)  # raises on malformed XML


def test_svg_bar_chart_well_formed():
    svg = svg_bar_chart(["V0/E0", "V1/E1", "V2/E3"], [1e-3, 8e-4, 6e-4],
                        title="Equal-time variance", ylabel="MSE")
    _assert_valid_svg(svg)
    assert "Equal-time variance" in svg


def test_svg_line_chart_well_formed():
    svg = svg_line_chart(
        [("V0/E0", [16, 64, 256], [1e-2, 2.5e-3, 6e-4]),
         ("V1/E1", [16, 64, 256], [8e-3, 2e-3, 5e-4])],
        title="Variance vs spp", xlabel="spp", ylabel="MSE",
    )
    _assert_valid_svg(svg)
    assert "polyline" in svg


def test_svg_handles_nan_and_infinite_safely():
    svg = svg_bar_chart(["a", "b"], [float("nan"), 1e-3], title="t", ylabel="y")
    _assert_valid_svg(svg)


def test_rel_mean_diff_noise_robust():
    rng = np.random.default_rng(1)
    base = np.full((32, 32, 3), 0.4)
    noisy = base + 0.2 * rng.standard_normal(base.shape)  # zero-mean noise
    assert rel_mean_diff(noisy, base) < 0.02  # cancels in the spatial mean
