"""The megakernel must compile to SPIR-V that passes spirv-val.

Regression for the BDPT negative-index bug (fix-bdpt-negative-index): when a
BDPT MIS helper (`misWeight`/`splatMisWeight` in `integrators/bdpt.slang`) is
inlined with a compile-time `t == 1` / `s == 1`, an index expression like
`litC[t - 2]` or `litC[i - 1]` folds to a constant `-1`, emitting an invalid
`OpAccessChain ... %int_n1` into `main_pass.spv`. spirv-val rejects it
(`VUID-VkShaderModuleCreateInfo-pCode-08737`, "Index ... may not have a negative
value"), and the guarded `[i - 1]` indices are clamped with `max(..., 0)` so the
folded index is `0`, never `-1` (behaviour-preserving — the value is discarded by
the surrounding `i > 0` / `t >= 2` guard).

Pure tool test: needs `slangc` + `spirv-val` (the Vulkan SDK), but no GPU device.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SHADERS = _ROOT / "src" / "skinny" / "shaders"
_GENSLANG = _ROOT / "src" / "skinny" / "mtlx" / "genslang"
_MAIN = _SHADERS / "main_pass.slang"


def _tool(name: str) -> str | None:
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        cand = Path(sdk) / "bin" / name
        if cand.exists():
            return str(cand)
    return shutil.which(name)


def test_megakernel_spirv_has_no_negative_index(tmp_path):
    slangc = _tool("slangc")
    spirv_val = _tool("spirv-val")
    if not slangc or not spirv_val:
        pytest.skip("needs slangc + spirv-val on PATH or under $VULKAN_SDK/bin")
    if not _MAIN.exists():
        pytest.skip(f"shader source not found: {_MAIN}")

    spv = tmp_path / "main_pass.spv"
    # Mirror the renderer's compile flags (vk_compute.ComputePipeline).
    compile_cmd = [
        slangc, str(_MAIN), "-target", "spirv",
        "-entry", "mainImage", "-stage", "compute",
        "-I", str(_SHADERS), "-I", str(_GENSLANG),
        "-D", "SKINNY_COMPUTE_PIPELINE=1", "-fvk-use-scalar-layout",
        "-o", str(spv),
    ]
    cc = subprocess.run(compile_cmd, capture_output=True, text=True)
    assert cc.returncode == 0, f"slangc failed:\n{cc.stderr}"
    assert spv.exists() and spv.stat().st_size > 0

    val_cmd = [
        spirv_val, str(spv),
        "--relax-block-layout", "--scalar-block-layout",
        "--target-env", "vulkan1.3",
    ]
    vc = subprocess.run(val_cmd, capture_output=True, text=True)
    out = vc.stdout + vc.stderr
    assert "negative value" not in out, (
        f"megakernel SPIR-V has a negative array index (BDPT regression):\n{out}"
    )
    assert vc.returncode == 0, f"spirv-val failed:\n{out}"
