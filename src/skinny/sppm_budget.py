"""SPPM photon-emission budget and group selection pmf — device-free.

Split out of ``renderer.py`` (change renderer-pure-core-extraction) so the
photon-budget math is importable, and therefore testable, on a host with no GPU
package installed. Pure arithmetic: no device handle, no ``self``.
"""

from __future__ import annotations

import math

def _sppm_photon_group_pmf(
    powers: tuple[float, float, float, float],
    present: tuple[bool, bool, bool, bool],
) -> tuple[float, float, float, float]:
    """Photon-emission group selection pmf (emissive, sphere, distant, env),
    proportional to each group's emitted power (change
    sppm-power-proportional-photon-groups). Per-photon flux then equalises
    across groups (Φ_g / p_g ≈ Φ_total) — the pbrt light-power distribution —
    which kills the sparse huge env splats that uniform 1/G selection produced.

    Absent groups get 0. Non-finite or negative powers are treated as 0. When
    the total usable power is 0 (or every power was non-finite) the pmf falls
    back to uniform over the *present* groups — the pre-change behavior.
    """
    clean = [
        p if (present[i] and math.isfinite(p) and p > 0.0) else 0.0
        for i, p in enumerate(powers)
    ]
    total = sum(clean)
    if total > 0.0 and math.isfinite(total):
        return tuple(p / total for p in clean)
    n_present = sum(1 for b in present if b)
    if n_present == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple((1.0 / n_present) if b else 0.0 for b in present)


# Ceiling on the env-aware photon-budget multiplier (change
# sppm-env-photon-budget): pmfEnv → 1 would send the budget to infinity, and ×8
# already cuts the env noise component by √8 ≈ 2.8 (measured exact on
# glass_caustics_test.usda) at negligible photon-stage cost.
_SPPM_ENV_PHOTON_BUDGET_CAP = 8.0


def _sppm_photon_budget(pixels: int, pmf_env: float,
                        cap: float = _SPPM_ENV_PHOTON_BUDGET_CAP) -> int:
    """Env-aware per-pass photon count (change sppm-env-photon-budget).

    ``pixels / (1 - pmfEnv)`` keeps the EXPECTED non-env photon count at
    exactly ``pixels`` (the flat pre-env budget) and rides the env group's
    photons on top of it instead of diluting the local lights — env photons
    deposit only after ≥1 bounce from a disc covering the whole scene bounding
    sphere, so at one-per-pixel they are sparse fat splats (speckle). Capped at
    ``cap``× so an env-dominated pmf can't run away; ``pmfEnv == 0`` returns
    ``pixels`` exactly (env-free scenes stay bit-identical). ``pmf_env`` is
    clamped to [0, 1] and treated as 0 when non-finite — the pmf override hook
    is unvalidated.
    """
    if not math.isfinite(pmf_env):
        pmf_env = 0.0
    pmf_env = min(max(pmf_env, 0.0), 1.0)
    return int(round(int(pixels) / max(1.0 - pmf_env, 1.0 / cap)))
