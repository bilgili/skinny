"""Selection-seam invariants for the SPPM integrator (change photon-mapping-sppm).

Source-level / no-GPU guards: the SPPM integrator must be registered in the
renderer's integrator list, carry a shader constant, and have its index folded
into the accumulation state hash so switching to/from SPPM resets accumulation.
These don't construct a Renderer (which needs a GPU context); they assert the
wiring is present so a refactor can't silently drop it.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "skinny"


def _read(rel: str) -> str:
    return (_SRC / rel).read_text()


def test_renderer_registers_sppm_mode():
    src = _read("renderer.py")
    # The GUI / runtime cycler reads integrator_modes; SPPM must be the 3rd entry
    # (index 2) so it lines up with INTEGRATOR_INDEX["sppm"] and INTEGRATOR_SPPM.
    m = re.search(r"integrator_modes:\s*list\[str\]\s*=\s*\[([^\]]*)\]", src)
    assert m, "integrator_modes list not found in renderer.py"
    modes = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
    assert modes[:3] == ["Path", "BDPT", "SPPM"], modes


def test_shader_defines_sppm_constant():
    common = _read("shaders/common.slang")
    assert re.search(r"INTEGRATOR_SPPM\s*=\s*2u", common), \
        "INTEGRATOR_SPPM = 2u missing from common.slang"


def test_state_hash_includes_integrator_index():
    src = _read("renderer.py")
    # Find the _current_state_hash body and assert integrator_index is hashed, so
    # cycling to/from SPPM resets progressive accumulation.
    start = src.index("def _current_state_hash")
    body = src[start:start + 2000]
    assert "self.integrator_index" in body, \
        "_current_state_hash must hash integrator_index (accumulation reset on switch)"


# ── Env photon-emission group (change sppm-env-indirect-transport) ───────────
# Source-level guards on wavefront_sppm.slang: the environment must be a 4th
# photon group gated exactly like env NEE's standing preconditions, with the
# pole-pdf validity guard the design review flagged (F1) and the pbrt SampleLe
# flux normalization. Regex-level so no GPU is needed.

def _sppm_src() -> str:
    return _read("shaders/integrators/wavefront_sppm.slang")


def test_sppm_env_group_gate():
    src = _sppm_src()
    assert re.search(
        r"hasEnv\s*=\s*\(fc\.furnaceMode\s*==\s*0u\s*&&\s*fc\.envIntensity\s*>\s*0\.0\)",
        src), "env photon group must gate on furnaceMode==0 && envIntensity>0"


def test_sppm_group_count_includes_env():
    src = _sppm_src()
    assert re.search(r"G\s*=\s*hasE\s*\+\s*hasS\s*\+\s*hasD\s*\+\s*hasEnv", src), \
        "group count G must include hasEnv"
    assert re.search(r"chosen\s*==\s*3u", src), \
        "env group must map to chosen == 3u (after emissive/sphere/distant)"


def test_sppm_env_emission_guards_degenerate_pdf():
    src = _sppm_src()
    # F1: sampleEnvDir returns pdf 0 at the equirect poles; dividing yields an
    # inf beta that poisons RR and the whole photon walk. The emission branch
    # must reject before dividing, mirroring the other groups' guards.
    i = src.index("EnvSample es = sampleEnvDir")
    window = src[i:i + 400]
    assert re.search(r"if\s*\(es\.pdf\s*<=\s*0\.0\)", window), \
        "env emission must guard es.pdf <= 0.0 immediately after sampleEnvDir"


def test_sppm_env_flux_is_pbrt_sample_le():
    src = _sppm_src()
    i = src.index("EnvSample es = sampleEnvDir")
    window = src[i:i + 1200]
    assert re.search(r"selPdf\s*=\s*pSel\s*\*\s*es\.pdf", window), \
        "env selection pdf must be pSel * es.pdf (solid-angle)"
    assert re.search(r"\(PI\s*\*\s*R\s*\*\s*R\)\s*/\s*max\(selPdf", window), \
        "env flux must carry the disk-area PI*R*R over selPdf (pbrt SampleLe)"


# ── Power-proportional group selection (change
# sppm-power-proportional-photon-groups) ─────────────────────────────────────
# Uniform 1/G selection gave every group an equal photon share regardless of
# emitted power, so the env group's huge flux (β = L·πR²/(gsel·pdf)) landed as
# sparse enormous splats → fireflies. Selection is now a CDF walk over the
# host-normalized power-proportional pmf in FrameConstants; each branch divides
# by its actual probability.

def test_sppm_selection_is_power_proportional_not_uniform():
    src = _sppm_src()
    assert "gsel" not in src, \
        "uniform 1/G selection (gsel) must be gone from wavefront_sppm.slang"
    for f in ("sppmGroupPmfE", "sppmGroupPmfS", "sppmGroupPmfD", "sppmGroupPmfEnv"):
        assert f"fc.{f}" in src, f"selection must read fc.{f}"


def test_fc_declares_group_pmf_before_tile_origin():
    common = _read("shaders/common.slang")
    fields = ["sppmGroupPmfE", "sppmGroupPmfS", "sppmGroupPmfD", "sppmGroupPmfEnv"]
    idx = [common.index(f) for f in fields]
    assert idx == sorted(idx), "pmf fields must be declared in E,S,D,Env order"
    assert idx[-1] < common.index("uint   tileOriginY"), \
        "pmf fields must precede the Metal-gated tileOriginY tail"
    assert common.index("float  filmMaxComponent") < idx[0], \
        "pmf fields must follow filmMaxComponent (packer-order contract)"


def test_pack_uniforms_appends_group_pmf():
    src = _read("renderer.py")
    start = src.index("def _pack_uniforms(self")  # not _pack_uniforms_msl
    body = src[start:]
    i_film = body.index('struct.pack("f", float(self.film_max_component))')
    i_pmf = body.index("sppm_pmf")
    i_tile = body.index("# tileOriginY")
    # earlier "4f" packs exist (proposalAlpha); the pmf pack must sit between
    # the filmMaxComponent float and the trailing tileOriginY u32.
    assert i_film < body.index('struct.pack("4f"', i_film) < i_tile, \
        "_pack_uniforms must pack the 4 pmf floats between filmMaxComponent and tileOriginY"
    assert i_pmf < i_tile
    # The Metal MSL relocation table must carry the same four fields in order.
    for f in ("sppmGroupPmfE", "sppmGroupPmfS", "sppmGroupPmfD", "sppmGroupPmfEnv"):
        assert f'("{f}", 4)' in src, f"_FC_SCALAR_FIELDS must list {f}"


def test_group_pmf_power_proportional():
    from skinny.renderer import _sppm_photon_group_pmf
    pmf = _sppm_photon_group_pmf((1.0, 3.0, 0.0, 4.0), (True, True, False, True))
    assert pmf == (0.125, 0.375, 0.0, 0.5)


def test_group_pmf_single_group_is_one():
    from skinny.renderer import _sppm_photon_group_pmf
    assert _sppm_photon_group_pmf((0.0, 7.5, 0.0, 0.0), (False, True, False, False)) \
        == (0.0, 1.0, 0.0, 0.0)


def test_group_pmf_absent_group_gets_zero_even_with_power():
    from skinny.renderer import _sppm_photon_group_pmf
    # A stale power for an absent group (e.g. env with furnaceMode on) must not
    # receive photons: presence gates the pmf, not just the power inputs.
    pmf = _sppm_photon_group_pmf((2.0, 2.0, 0.0, 100.0), (True, True, False, False))
    assert pmf == (0.5, 0.5, 0.0, 0.0)


def test_group_pmf_zero_total_falls_back_to_uniform_over_present():
    from skinny.renderer import _sppm_photon_group_pmf
    assert _sppm_photon_group_pmf((0.0, 0.0, 0.0, 0.0), (True, False, True, True)) \
        == (1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0)


def test_group_pmf_nonfinite_power_treated_as_zero():
    from skinny.renderer import _sppm_photon_group_pmf
    pmf = _sppm_photon_group_pmf((float("nan"), 1.0, float("inf"), -5.0),
                                 (True, True, True, True))
    assert pmf == (0.0, 1.0, 0.0, 0.0)
    # all-non-finite → uniform fallback
    pmf = _sppm_photon_group_pmf((float("nan"), float("inf"), 0.0, -1.0),
                                 (True, True, False, False))
    assert pmf == (0.5, 0.5, 0.0, 0.0)


def test_group_pmf_no_present_groups_is_all_zero():
    from skinny.renderer import _sppm_photon_group_pmf
    assert _sppm_photon_group_pmf((0.0, 0.0, 0.0, 0.0), (False, False, False, False)) \
        == (0.0, 0.0, 0.0, 0.0)


def test_env_lum_integral_constant_map_is_4pi():
    import numpy as np
    from skinny.environment import build_env_distribution
    rgba = np.ones((8, 16, 4), dtype=np.float32)  # L = lum(1,1,1) = 1 everywhere
    _marg, _cond, integral = build_env_distribution(rgba)
    # ∫L dω over the sphere for a constant unit-luminance map = 4π; the
    # sin θ midpoint rule on the ENV_HEIGHT×ENV_WIDTH grid is exact to ~1e-4.
    assert abs(integral - 4.0 * np.pi) < 1e-2 * 4.0 * np.pi


def test_env_lum_integral_default_precedes_construction_upload():
    # Regression (found live): the `_env_lum_integral = 0.0` default must run
    # BEFORE the construction-time `_ensure_env_uploaded()` (which computes the
    # integral); a later default clobbers it and the cache short-circuit keeps
    # it 0 forever -> env group silently starved of photons (pmf 0).
    src = _read("renderer.py")
    i_default = src.index("self._env_lum_integral: float = 0.0")
    window = src[i_default:i_default + 600]
    assert "self._ensure_env_uploaded()" in window, \
        "_env_lum_integral default must sit immediately before the construction " \
        "_ensure_env_uploaded() call"
    assert src.count("self._env_lum_integral: float = 0.0") == 1, \
        "exactly one _env_lum_integral default (a second one could clobber the computed value)"


def test_pack_uniforms_honors_pmf_override_hook():
    src = _read("renderer.py")
    start = src.index("def _pack_uniforms(self")  # not _pack_uniforms_msl
    body = src[start:]
    assert "_sppm_group_pmf_override" in body, \
        "_pack_uniforms must honor _sppm_group_pmf_override (forced-group flux probe)"


def test_photon_budget_env_free_is_flat_pixels():
    # pmfEnv == 0 must return pixels EXACTLY (env-free renders bit-identical).
    from skinny.renderer import _sppm_photon_budget
    assert _sppm_photon_budget(384 * 384, 0.0) == 384 * 384
    assert _sppm_photon_budget(1, 0.0) == 1


def test_photon_budget_scales_by_env_pmf_share():
    # N = pixels/(1-pmfEnv): expected NON-env photon count stays exactly pixels.
    from skinny.renderer import _sppm_photon_budget
    n = _sppm_photon_budget(100_000, 0.84)
    assert n == round(100_000 / 0.16)  # ×6.25
    # non-env expectation: N·(1−pmfEnv) == pixels (to rounding)
    assert abs(n * (1.0 - 0.84) - 100_000) < 1.0


def test_photon_budget_caps_at_8x():
    from skinny.renderer import _sppm_photon_budget
    assert _sppm_photon_budget(100_000, 1.0) == 800_000
    assert _sppm_photon_budget(100_000, 0.999) == 800_000


def test_photon_budget_nonfinite_and_out_of_range_pmf_is_flat():
    # The pmf override hook is unvalidated: NaN/inf/negative/>1 must not
    # explode the pack path — clamp to the flat budget (or the cap).
    from skinny.renderer import _sppm_photon_budget
    assert _sppm_photon_budget(4096, float("nan")) == 4096
    assert _sppm_photon_budget(4096, float("-inf")) == 4096
    assert _sppm_photon_budget(4096, -0.5) == 4096
    assert _sppm_photon_budget(4096, 1.5) == 4096 * 8


def test_pack_uniforms_derives_photons_from_env_budget():
    src = _read("renderer.py")
    start = src.index("def _pack_uniforms(self")
    body = src[start:]
    assert "_sppm_photon_budget(" in body, \
        "_pack_uniforms must derive the per-pass photon count from the env-aware budget"
    assert body.index("_sppm_photon_group_pmf") < body.index("_sppm_photon_budget("), \
        "group pmf must be computed before the photon budget that consumes pmfEnv"
    assert "_sppm_photons_override" in body.split("_sppm_photon_budget(")[0].rsplit("sppm_photons", 2)[-1] \
        or "_sppm_photons_override" in body, \
        "_sppm_photons_override must keep absolute precedence"
