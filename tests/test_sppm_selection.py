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
