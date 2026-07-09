"""Hostless slangc compile gate for spectrum.slang (task 4.1).

Compiles the spectrum harness to SPIR-V in BOTH the spectral variant
(`-DSKINNY_SPECTRAL`) and the RGB variant (no define), proving the module
typechecks under both `Spectrum` typealiases without a GPU. The GPU≡numpy
behaviour check (that the compiled kernel matches `skinny.pbrt.spectral`) is the
separate gpu-marked test in task 4.2.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SHADERS = _ROOT / "src" / "skinny" / "shaders"
_HARNESSES = Path(__file__).resolve().parent / "harnesses"
_HARNESS = _HARNESSES / "test_spectrum_harness.slang"

_slangc = shutil.which("slangc")
pytestmark = pytest.mark.skipif(_slangc is None, reason="slangc not on PATH")


def _compile(tmp_path: Path, *defines: str) -> Path:
    out = tmp_path / ("spectrum_" + ("spectral" if defines else "rgb") + ".spv")
    cmd = [
        _slangc, str(_HARNESS),
        *defines,
        "-target", "spirv", "-entry", "computeMain", "-stage", "compute",
        "-o", str(out),
        "-I", str(_SHADERS), "-I", str(_HARNESSES),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"slangc failed:\n{proc.stdout}\n{proc.stderr}"
    assert out.exists() and out.stat().st_size > 0
    return out


def test_spectral_variant_compiles(tmp_path):
    _compile(tmp_path, "-DSKINNY_SPECTRAL")


def test_rgb_variant_compiles(tmp_path):
    _compile(tmp_path)
