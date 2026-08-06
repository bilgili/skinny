"""Gate for the wavefront kernel-name owner (change choice-table-wavefront-owners).

`wavefront_driver.py` declares every wavefront compute kernel's entry-point name
once (`WF_…` constants). The driver dispatches through them and both backend pass
modules import them. Three guarantees, all hostless (the backends are AST-parsed,
not imported, so no GPU/SDK is needed):

* **Golden** — each constant still equals its historical string.
* **Source gate** — no kernel-name string literal survives outside the owner's
  own definitions (a re-mirror in a backend fails here).
* **Import-time failure** — a stale kernel name is an ImportError, not a runtime
  dispatch failure on one backend (the negative control).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from skinny import wavefront_driver as wd

SRC = Path(__file__).resolve().parents[1] / "src" / "skinny"

# The historical entry-point strings — the golden the owner must keep.
GOLDEN = {
    "WF_PATH_GENERATE": "wfPathGenerate",
    "WF_PATH_INTERSECT": "wfPathIntersect",
    "WF_BUILD_ARGS": "wfBuildArgs",
    "WF_SCATTER": "wfScatter",
    "WF_PATH_SHADE_FLAT": "wfPathShadeFlat",
    "WF_PATH_SHADE": "wfPathShade",
    "WF_PATH_RESOLVE": "wfPathResolve",
    "WF_BDPT_BUILD_ARGS": "wfBdptBuildArgs",
    "WF_BDPT_SCATTER": "wfBdptScatter",
    "WF_BDPT_WALK": "wfBdptWalk",
    "WF_BDPT_GEN_EYE": "wfBdptGenEye",
    "WF_BDPT_WALK_CLASSIFY": "wfBdptWalkClassify",
    "WF_BDPT_BOUNCE_EYE": "wfBdptBounceEye",
    "WF_BDPT_LIGHT_TAIL": "wfBdptLightTail",
    "WF_BDPT_GEN_LIGHT": "wfBdptGenLight",
    "WF_BDPT_BOUNCE_LIGHT": "wfBdptBounceLight",
    "WF_BDPT_SPLAT": "wfBdptSplat",
    "WF_BDPT_CLASSIFY": "wfBdptClassify",
    "WF_BDPT_CONNECT_NEE": "wfBdptConnectNee",
    "WF_BDPT_CONNECT_FULL": "wfBdptConnectFull",
    "WF_BDPT_RESOLVE": "wfBdptResolve",
    "WF_SPPM_EYE": "wfSppmEye",
    "WF_SPPM_GRID_COUNT": "wfSppmGridCount",
    "WF_SPPM_GRID_SCAN_BLOCK": "wfSppmGridScanBlock",
    "WF_SPPM_GRID_SCAN_BLOCK_SUMS": "wfSppmGridScanBlockSums",
    "WF_SPPM_GRID_SCAN_ADD": "wfSppmGridScanAdd",
    "WF_SPPM_GRID_SCATTER": "wfSppmGridScatter",
    "WF_SPPM_PHOTON_TRACE": "wfSppmPhotonTrace",
    "WF_SPPM_UPDATE": "wfSppmUpdate",
    "WF_MLT_BOOTSTRAP": "wfMltBootstrap",
    "WF_MLT_INIT": "wfMltInit",
    "WF_MLT_MUTATE": "wfMltMutate",
    "WF_MLT_RESOLVE": "wfMltResolve",
    "WF_NEURAL_PROPOSAL": "wfNeuralProposal",
    "WF_INDIRECT_PAINT": "wfIndirectPaint",
}


def test_golden_constant_strings():
    for name, value in GOLDEN.items():
        assert getattr(wd, name) == value, name


def test_kernel_entry_names_set_matches_the_constants():
    assert wd.KERNEL_ENTRY_NAMES == frozenset(GOLDEN.values())
    assert len(wd.KERNEL_ENTRY_NAMES) == 35


def _string_literals(text: str) -> list[str]:
    return [
        n.value for n in ast.walk(ast.parse(text))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _owner_definition_literals(text: str) -> list[str]:
    """The kernel-name strings that ARE the owner's `WF_X = "wfX"` definitions."""
    out = []
    for node in ast.walk(ast.parse(text)):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith("WF_")
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            out.append(node.value.value)
    return out


def test_owner_holds_exactly_the_definitions():
    text = (SRC / "wavefront_driver.py").read_text(encoding="utf-8")
    defs = _owner_definition_literals(text)
    # Every kernel name is defined exactly once, and every kernel-name literal in
    # the driver is one of those definitions (no stray dispatch literal).
    assert sorted(defs) == sorted(GOLDEN.values())
    stray = [s for s in _string_literals(text)
             if s in wd.KERNEL_ENTRY_NAMES and defs.count(s) == 0]
    assert stray == [], f"driver has kernel-name literals outside its definitions: {stray}"
    assert sum(1 for s in _string_literals(text) if s in wd.KERNEL_ENTRY_NAMES) == 35


def test_no_kernel_name_literal_in_the_backends():
    for module in ("vk_wavefront.py", "metal_wavefront.py"):
        text = (SRC / module).read_text(encoding="utf-8")
        offenders = sorted({s for s in _string_literals(text) if s in wd.KERNEL_ENTRY_NAMES})
        assert offenders == [], (
            f"{module} restates kernel-name literal(s) {offenders} — import the "
            f"WF_… constant from wavefront_driver instead")


def test_stale_kernel_name_is_an_import_error():
    """Negative control: referencing a renamed/removed kernel constant fails at
    IMPORT, which is exactly how a rename that misses a backend surfaces."""
    prog = "from skinny.wavefront_driver import WF_PATH_GENERATE_RENAMED"
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                       cwd=str(SRC.parents[1]), env={"PYTHONPATH": str(SRC.parent)})
    assert r.returncode != 0
    assert "ImportError" in r.stderr or "cannot import name" in r.stderr, r.stderr


# ── shared / pinned pass constants (task 3) ──────────────────────────────────
#
# Importing the backend pass classes needs the vulkan / slangpy packages (not a
# GPU device), so these skip when a backend is unavailable.

@pytest.fixture(scope="module")
def backends():
    vk = pytest.importorskip("skinny.vk_wavefront")
    mt = pytest.importorskip("skinny.metal_wavefront")
    return vk, mt


def test_shared_constants_have_one_owner(backends):
    """The must-be-equal pass constants are derived from wavefront_driver, so both
    backends observe one value."""
    vk, mt = backends
    assert vk.WavefrontPathPass.MAX_BOUNCES == mt.MetalWavefrontPathPass.MAX_BOUNCES == wd.WF_MAX_BOUNCES == 6
    assert vk.WavefrontPathPass.STREAM_CAP == mt.MetalWavefrontPathPass.STREAM_CAP == wd.WF_STREAM_CAP_PATH == (1 << 20)
    assert vk.WavefrontPathPass.NUM_SLOTS == mt.MetalWavefrontPathPass.NUM_SLOTS == wd.WF_NUM_SLOTS == 2
    for cls_v, cls_m in [(vk.WavefrontBdptPass, mt.MetalWavefrontBdptPass)]:
        assert cls_v.BDPT_MAX_VERTS == cls_m.BDPT_MAX_VERTS == wd.BDPT_MAX_VERTS == 7
        assert cls_v.EYE_BOUNCES == cls_m.EYE_BOUNCES == wd.WF_EYE_BOUNCES == 5
        assert cls_v.LIGHT_BOUNCES == cls_m.LIGHT_BOUNCES == wd.WF_LIGHT_BOUNCES == 6
        assert cls_v.STREAM_CAP == cls_m.STREAM_CAP == wd.WF_STREAM_CAP_BDPT == (1 << 18)
        assert cls_v.WALK_MODES == cls_m.WALK_MODES == wd.WALK_MODES
    assert vk.RestirDiPass.DEFAULT_CONFIG == mt.MetalRestirDiPass.DEFAULT_CONFIG == wd.RESTIR_DEFAULT_CONFIG
    # Each pass copies the shared config so a per-instance override never mutates it.
    assert vk.RestirDiPass.DEFAULT_CONFIG is not mt.MetalRestirDiPass.DEFAULT_CONFIG
    assert vk.RestirDiPass.DEFAULT_CONFIG is not wd.RESTIR_DEFAULT_CONFIG


def test_per_backend_strides_are_pinned_equal_with_a_reason(backends):
    """VERTEX_STRIDE / AUX_STRIDE / RESERVOIR_STRIDE hold the same value on both
    backends but are NOT moved to the shared home: on Vulkan each is the real
    buffer stride, while on Metal it is only a reflection fallback (the MSL stride
    is authoritative). They are kept per-backend and pinned equal here, so a
    divergence is caught but the Metal fallback is not misrepresented as shared."""
    vk, mt = backends
    assert vk.WavefrontBdptPass.VERTEX_STRIDE == mt.MetalWavefrontBdptPass.VERTEX_STRIDE == 128
    assert vk.WavefrontBdptPass.AUX_STRIDE == mt.MetalWavefrontBdptPass.AUX_STRIDE == 128
    assert vk.RestirDiPass.RESERVOIR_STRIDE == mt.MetalRestirDiPass.RESERVOIR_STRIDE == 32


def test_walk_modes_match_the_cli_axis():
    """The wavefront BDPT walk modes are the same axis the CLI advertises as
    cli_common.WALK_CHOICES; pinned equal so the two agree while the low-level
    driver stays free of a CLI-module import."""
    from skinny import cli_common
    assert wd.WALK_MODES == cli_common.WALK_CHOICES == ("fused", "eye", "eye_light")
