"""Dev tool: extract pbrt-v4 spectral data into vendored .npz files.

Run once against a local pbrt-v4 checkout to (re)generate the vendored tables
consumed at runtime by :mod:`skinny.pbrt.data.spectral_tables`. The data is
copied verbatim from pbrt-v4 so skinny's spectral mode matches pbrt exactly:

* ``rgb2spec_srgb.npz`` — the sRGB→spectrum sigmoid-coefficient table
  (``sRGBToSpectrumTable_Scale`` / ``_Data``) from the generated
  ``build/rgbspectrum_srgb.cpp`` (RES=64), used by Jakob & Hanika (2019)
  RGB→spectrum upsampling.
* ``spectral_curves.npz`` — the CIE D65 illuminant SPD and the named-metal
  complex IOR curves (Ag/Al/Au/Cu ``eta``/``k``), each resampled onto skinny's
  360–830 nm / 5 nm grid, from ``src/pbrt/util/spectrum.cpp``.

Provenance: pbrt-v4 (Pharr, Jakob, Humphreys), BSD-licensed. These are the
exact published tables, not approximations.

Usage::

    python -m skinny.pbrt.data._extract_pbrt_spectra --pbrt ~/projects/pbrt-v4
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# skinny's internal sample grid (matches spectra._LAMBDA).
_LAMBDA = np.arange(360.0, 830.0 + 1.0, 5.0)

_METALS = ("Ag", "Al", "Au", "Cu")


def _brace_block(text: str, symbol: str) -> str:
    """Return the raw text inside the first ``{...}`` after *symbol*'s ``= {``."""
    m = re.search(re.escape(symbol) + r"\s*(?:\[[^\]]*\])*\s*=\s*\{", text)
    if m is None:
        raise KeyError(f"symbol {symbol!r} not found")
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return text[start : i - 1]


def _floats(block: str) -> np.ndarray:
    toks = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", block)
    return np.asarray([float(t) for t in toks], dtype=np.float64)


def _resample_interleaved(block: str) -> np.ndarray:
    """pbrt ``FromInterleaved`` array [l0,v0,l1,v1,...] → values on _LAMBDA."""
    flat = _floats(block)
    lam, val = flat[0::2], flat[1::2]
    return np.interp(_LAMBDA, lam, val, left=val[0], right=val[-1])


def extract_upsample_table(pbrt_root: Path, out_dir: Path) -> Path:
    src = (pbrt_root / "build" / "rgbspectrum_srgb.cpp").read_text()
    res = int(re.search(r"sRGBToSpectrumTable_Res\s*=\s*(\d+)", src).group(1))
    scale = _floats(_brace_block(src, "sRGBToSpectrumTable_Scale"))
    assert scale.size == res, (scale.size, res)
    data = _floats(_brace_block(src, "sRGBToSpectrumTable_Data"))
    assert data.size == 3 * res * res * res * 3, data.size
    data = data.reshape(3, res, res, res, 3).astype(np.float32)
    out = out_dir / "rgb2spec_srgb.npz"
    np.savez_compressed(out, res=np.int32(res), scale=scale.astype(np.float32), data=data)
    return out


def extract_curves(pbrt_root: Path, out_dir: Path) -> Path:
    src = (pbrt_root / "src" / "pbrt" / "util" / "spectrum.cpp").read_text()
    payload: dict[str, np.ndarray] = {"lambda": _LAMBDA.astype(np.float32)}
    payload["d65"] = _resample_interleaved(_brace_block(src, "CIE_Illum_D6500")).astype(
        np.float32
    )
    for metal in _METALS:
        payload[f"{metal.lower()}_eta"] = _resample_interleaved(
            _brace_block(src, f"{metal}_eta")
        ).astype(np.float32)
        payload[f"{metal.lower()}_k"] = _resample_interleaved(
            _brace_block(src, f"{metal}_k")
        ).astype(np.float32)
    out = out_dir / "spectral_curves.npz"
    np.savez_compressed(out, **payload)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pbrt", type=Path, default=Path.home() / "projects" / "pbrt-v4")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    t = extract_upsample_table(args.pbrt, args.out)
    c = extract_curves(args.pbrt, args.out)
    print(f"wrote {t} ({t.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {c} ({c.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
