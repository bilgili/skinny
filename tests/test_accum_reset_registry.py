"""Hostless invariant tests for the accumulation-reset registry.

Change param-registry-accumulation-reset: `Renderer._current_state_hash`
derives its tuple from the params.py registry (ParamSpec.resets_accumulation
+ ACCUM_STATE_PROVIDERS) instead of a hand-maintained ~40-field literal.
These tests prove the derived contributor set equals the frozen legacy field
set (behavior = "resets exactly when it did before"), with no GPU and no
Renderer instance (pattern: tests/test_cli_common.py).
"""

from __future__ import annotations

from skinny.params import (
    ACCUM_STATE_PROVIDERS,
    STATIC_PARAMS,
    ParamSpec,
)

# ── Frozen expected contributor set ─────────────────────────────────────────
# Transcribed field-by-field from the pre-change hand-curated tuple in
# `Renderer._current_state_hash` (renderer.py:10906–10968 at transcription
# time). This is the one place a *legitimate* contributor-set change must
# consciously edit — drift fails loudly as a set difference, inverting the
# old silent-staleness failure mode.

# Param-backed fields of the legacy tuple (attributes that are STATIC_PARAMS
# paths). mtlx.* params are absent: they contribute through the
# `mtlx_overrides` provider's wholesale dict value.
EXPECTED_PARAM_CONTRIBUTORS = {
    "light_elevation", "light_azimuth", "light_intensity",
    "light_color_r", "light_color_g", "light_color_b",
    "env_index",
    "direct_light_index",
    "model_index",
    "tattoo_index",
    "tattoo_density",
    "scatter_index",
    "integrator_index",
    "proposal_preset_index",
    "reuse_index",
    "coat_sampler_index", "spec_sampler_index", "diff_sampler_index",
    "restir_regime_index",
    "restir_biased",
    "restir_m_light", "restir_m_bsdf",
    "restir_spatial_k", "restir_spatial_radius",
    "restir_m_cap",
    "env_intensity",
    "furnace_index",
    "mm_per_unit",
    "film.iso", "film.exposure_time",
    "detail_maps_index",
    "normal_map_strength",
    "displacement_scale_mm",
    "preset_index",
}

# Non-param state of the legacy tuple, one provider name each.
EXPECTED_PROVIDER_NAMES = {
    "camera",                # camera.state_signature()
    "mtlx_overrides",        # sorted (k, _hashable_value(v)) tuple
    "material_version",      # _material_version
    "volume_grid_key",       # _volume_grid_key
    "film_max_component",
    "camera_mirror",         # _camera_mirror
    "usd_time_code",         # clock.current_time_code
    "sppm_radius_override",
    "sppm_photons_override",
    "sppm_glossy_roughness_override",
}


def _provider_prefixes() -> list[str]:
    return [p.covers_prefix for p in ACCUM_STATE_PROVIDERS if p.covers_prefix]


def _derived_param_contributors() -> set[str]:
    """Static resets_accumulation params minus provider-covered prefixes."""
    prefixes = _provider_prefixes()
    return {
        p.path for p in STATIC_PARAMS
        if p.resets_accumulation
        and not any(p.path.startswith(pre) for pre in prefixes)
    }


def test_contributor_set_matches_legacy_hash_fields():
    """The behavior-preservation gate (design D5.1): registry-derived identity
    set == frozen legacy field set, proving reset behavior is unchanged."""
    derived = {
        f"param:{p}" for p in _derived_param_contributors()
    } | {
        f"provider:{p.name}" for p in ACCUM_STATE_PROVIDERS
    }
    expected = {
        f"param:{p}" for p in EXPECTED_PARAM_CONTRIBUTORS
    } | {
        f"provider:{n}" for n in EXPECTED_PROVIDER_NAMES
    }
    assert derived == expected, (
        "accumulation-hash contributor drift:\n"
        f"  missing:    {sorted(expected - derived)}\n"
        f"  unexpected: {sorted(derived - expected)}"
    )


def test_default_is_resets_accumulation():
    """Fail-safe default (design D1): an undeclared new param resets."""
    assert ParamSpec("x", "y", "continuous").resets_accumulation is True
    assert ParamSpec("x", "y", "discrete").resets_accumulation is True


def test_opt_out_set_is_closed():
    """Exactly the post-process controls carry False (design D5.3)."""
    non_resetting = {p.path for p in STATIC_PARAMS if not p.resets_accumulation}
    assert non_resetting == {"tonemap_index", "exposure"}


def test_every_mtlx_param_is_provider_covered():
    """Coverage rule (design D5.4): all mtlx.* static params fall under a
    provider covers_prefix, so they hash through the overrides dict."""
    prefixes = _provider_prefixes()
    for p in STATIC_PARAMS:
        if p.path.startswith("mtlx."):
            assert any(p.path.startswith(pre) for pre in prefixes), (
                f"{p.path} not covered by any provider covers_prefix"
            )


def test_int_coercion_override_set_is_closed():
    """Legacy-cast spot-check (design D4/D5.5): exactly the four continuous
    ReSTIR count params carry the int override; a fractional change within
    the same integer must not perturb the derived contribution."""
    overridden = {p.path for p in STATIC_PARAMS if p.hash_coercion is not None}
    assert overridden == {
        "restir_m_light", "restir_m_bsdf", "restir_spatial_k", "restir_m_cap",
    }
    for p in STATIC_PARAMS:
        if p.hash_coercion is not None:
            assert p.hash_coercion is int
            assert p.kind == "continuous"
            # 4.2 → 4.7 must derive the same contribution (no reset).
            assert p.hash_coercion(4.2) == p.hash_coercion(4.7)


def test_provider_registry_is_hostless_enumerable():
    """Spec scenario: names + coverage prefixes enumerable with no Renderer."""
    names = [p.name for p in ACCUM_STATE_PROVIDERS]
    assert len(names) == len(set(names)), "duplicate provider names"
    assert all(callable(p.extractor) for p in ACCUM_STATE_PROVIDERS)
    covering = [p for p in ACCUM_STATE_PROVIDERS if p.covers_prefix]
    assert [(p.name, p.covers_prefix) for p in covering] == [
        ("mtlx_overrides", "mtlx."),
    ]
