"""Unit tests for the ReSTIR reservoir core (restir/reservoir.slang).

Drives the synthetic 1D RIS probe in test_restir_harness.slang: integrand
f(x)=x² on [0,1], uniform source. The RIS estimate f(y)·W must be an unbiased
estimate of ∫₀¹ f = 1/3 for any M / target, and the reservoir replacement
probability must be wᵢ/Σw. Pure core math — no scene bindings.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

INT_F = 1.0 / 3.0   # ∫₀¹ x² dx


class TestRisUnbiased:
    TRIALS = 300_000

    @pytest.mark.parametrize("m", [1, 4, 16])
    def test_proxy_target_unbiased(self, restir_harness, m):
        # Imperfect target p̂=x (≠ f) — RIS still unbiased for any M.
        mean = float(restir_harness.test_ris_mean(0x1234 + m, m, self.TRIALS, 0))
        assert abs(mean - INT_F) < 0.01, f"M={m} proxy: mean {mean:.5f} vs {INT_F:.5f}"

    @pytest.mark.parametrize("m", [1, 8])
    def test_perfect_target_unbiased(self, restir_harness, m):
        # Perfect target p̂=x²=f — unbiased and lower variance.
        mean = float(restir_harness.test_ris_mean(0x9abc + m, m, self.TRIALS, 1))
        assert abs(mean - INT_F) < 0.008, f"M={m} perfect: mean {mean:.5f} vs {INT_F:.5f}"

    def test_more_candidates_reduce_variance(self, restir_harness):
        # Many independent small-trial means; spread shrinks with M. Compare the
        # max deviation from the truth across a handful of seeds for M=1 vs M=16.
        def max_dev(m):
            devs = [
                abs(float(restir_harness.test_ris_mean(s * 7919 + 3, m, 4000, 0)) - INT_F)
                for s in range(12)
            ]
            return max(devs)
        assert max_dev(16) < max_dev(1), "M=16 should be tighter than M=1"


class TestReservoirSelection:
    def test_replacement_probability(self, restir_harness):
        # Two candidates, weights wA, wB → P(survivor == B) ≈ wB/(wA+wB).
        wA, wB = 1.0, 3.0
        freq = float(restir_harness.test_select_frequency(0x55, 200_000, wA, wB))
        assert abs(freq - wB / (wA + wB)) < 0.01, f"freq {freq:.4f} vs {wB/(wA+wB):.4f}"

    def test_zero_weight_never_selected(self, restir_harness):
        # wB=0 → B never replaces A.
        freq = float(restir_harness.test_select_frequency(0x77, 50_000, 1.0, 0.0))
        assert freq == 0.0, f"zero-weight candidate selected (freq {freq})"
