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
    assert re.search(r"chosen\s*=\s*3u", src), \
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
    assert re.search(r"selPdf\s*=\s*gsel\s*\*\s*es\.pdf", window), \
        "env selection pdf must be gsel * es.pdf (solid-angle)"
    assert re.search(r"\(PI\s*\*\s*R\s*\*\s*R\)\s*/\s*max\(selPdf", window), \
        "env flux must carry the disk-area PI*R*R over selPdf (pbrt SampleLe)"
