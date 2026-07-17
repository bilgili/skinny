"""Dev tool: extract pbrt-v4 spectral data into vendored .npz files.

Run once against a local pbrt-v4 checkout to (re)generate the vendored tables
consumed at runtime by :mod:`skinny.pbrt.data.spectral_tables`. The data is
copied verbatim from pbrt-v4 so skinny's spectral mode matches pbrt exactly:

* ``rgb2spec_srgb.npz`` — the sRGB→spectrum sigmoid-coefficient table
  (``sRGBToSpectrumTable_Scale`` / ``_Data``) from the generated
  ``build/rgbspectrum_srgb.cpp`` (RES=64), used by Jakob & Hanika (2019)
  RGB→spectrum upsampling.
* ``spectral_curves.npz`` — the named illuminant SPDs, the named-metal complex
  IOR curves (``eta``/``k``) and the named-glass ``eta`` curves, each resampled
  onto skinny's 360–830 nm / 5 nm grid, from ``src/pbrt/util/spectrum.cpp``.

Provenance: pbrt-v4 (Pharr, Jakob, Humphreys), BSD-licensed. These are the
exact published tables, not approximations.

**pbrt's ``normalize`` flag is deliberately not applied here.** pbrt passes
``true`` for every illuminant and ``false`` for metals/glasses
(``spectrum.cpp`` ``FromInterleaved`` call sites), scaling illuminants to unit
luminance at load. skinny stores the raw SPD instead: every consumer rescales
anyway (``renderer._spectral_light_spd_scaled`` matches an SPD to its light's
RGB luminance; ``spectra.param_to_rgb`` normalises to unit luminance), so the
absolute scale cancels. Do **not** "fix" this by normalising here — that would
apply the normalisation twice.

Named glasses additionally get a 2-term Cauchy fit ``n(λ) = A + B/λ_µm²`` over
the visible grid; ``--print-tables`` emits the literals for
``spectral_tables._GLASS_CAUCHY`` / ``_GLASS_IOR_D``. The raw curves stay in the
``.npz`` so the fit-residual test is hostless (no pbrt checkout needed).

Usage::

    python -m skinny.pbrt.data._extract_pbrt_spectra --pbrt ~/projects/pbrt-v4
    python -m skinny.pbrt.data._extract_pbrt_spectra --print-tables
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# skinny's internal sample grid (matches spectra._LAMBDA).
_LAMBDA = np.arange(360.0, 830.0 + 1.0, 5.0)

# Named metals: key = symbol.lower(), pbrt arrays are `<symbol>_eta` / `<symbol>_k`.
# APPEND-ONLY: renderer._CONDUCTOR_METAL_ID assigns ids by this order (id ==
# index + 1) and the shader resolves curves by (metalId-1)*stride into the
# upload, so reordering silently swaps metals in every existing scene.
_METALS = ("Ag", "Al", "Au", "Cu", "CuZn", "MgO", "TiO2")

# Named glasses: pbrt's public name -> its C++ array symbol. The two do NOT
# agree for the flints — "glass-F5" reads GlassSF5_eta (SF = dense flint) — so
# this map is explicit rather than derived from the name.
_GLASSES = {
    "bk7": "GlassBK7_eta",
    "baf10": "GlassBAF10_eta",
    "fk51a": "GlassFK51A_eta",
    "lasf9": "GlassLASF9_eta",
    "f5": "GlassSF5_eta",
    "f10": "GlassSF10_eta",
    "f11": "GlassSF11_eta",
}

# Named illuminants: pbrt's public name -> its C++ array symbol. "stdillum-D65"
# is intentionally absent — it aliases the existing `d65` array (same pbrt
# symbol CIE_Illum_D6500), so it is not stored twice.
_ILLUMINANTS = {
    "stdillum-A": "CIE_Illum_A",
    "stdillum-D50": "CIE_Illum_D5000",
    "illum-acesD60": "ACES_Illum_D60",
    **{f"stdillum-F{i}": f"CIE_Illum_F{i}" for i in range(1, 13)},
}

# Sodium d-line: the reference wavelength for a glass's quoted refractive index.
_D_LINE_NM = 589.3


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


def _raw_interleaved(block: str) -> tuple[np.ndarray, np.ndarray]:
    """pbrt ``FromInterleaved`` array [l0,v0,l1,v1,...] → (lambda, value) as authored."""
    flat = _floats(block)
    return flat[0::2], flat[1::2]


def fit_cauchy(lam_nm, n) -> tuple[float, float, float]:
    """Least-squares 2-term Cauchy fit ``n(λ) = A + B/λ_µm²``.

    Returns ``(A, B, max_abs_residual)`` over the supplied samples. A third term
    (``+ C/λ⁴``) was measured and does not pay: on the worst glass (F11) it moves
    the residual 7.5e-3 → 6.1e-3 and on LASF9 it gets *worse*, because the
    residual is piecewise-linear interpolation error in pbrt's own sparse table,
    not a missing Cauchy order. It would also cost a FlatMaterialParams layout
    change to carry the extra coefficient.
    """
    lam_um = np.asarray(lam_nm, dtype=np.float64) * 1e-3
    basis = np.vstack([np.ones_like(lam_um), 1.0 / lam_um**2]).T
    (a, b), *_ = np.linalg.lstsq(basis, np.asarray(n, dtype=np.float64), rcond=None)
    resid = float(np.max(np.abs(basis @ [a, b] - n)))
    return float(a), float(b), resid


def glass_tables(pbrt_root: Path) -> dict[str, tuple[float, float, float, float]]:
    """``{key: (A, B, ior_d, max_residual)}`` for every pbrt named glass.

    The fit and the d-line index are both taken from pbrt's tabulated eta, so
    pbrt is the single source of truth for every glass — including BK7, whose
    previously hand-entered catalogue coefficients this supersedes.
    """
    src = (pbrt_root / "src" / "pbrt" / "util" / "spectrum.cpp").read_text()
    out: dict[str, tuple[float, float, float, float]] = {}
    for key, symbol in _GLASSES.items():
        lam, val = _raw_interleaved(_brace_block(src, symbol))
        on_grid = np.interp(_LAMBDA, lam, val, left=val[0], right=val[-1])
        a, b, resid = fit_cauchy(_LAMBDA, on_grid)
        ior_d = float(np.interp(_D_LINE_NM, lam, val))
        out[key] = (a, b, ior_d, resid)
    return out


def print_tables(pbrt_root: Path) -> None:
    """Emit the `spectral_tables` literals for the named glasses."""
    tables = glass_tables(pbrt_root)
    print("# (A, B) for n(λ)=A+B/λ_µm², least-squares fit to pbrt's tabulated eta")
    print("# over 360-830 nm. Regenerate with `_extract_pbrt_spectra --print-tables`.")
    print("_GLASS_CAUCHY = {")
    for key, (a, b, _ior, resid) in tables.items():
        print(f'    "{key}": ({a:.5f}, {b:.6f}),  # max resid {resid:.1e}')
    print("}")
    print()
    print("_GLASS_IOR_D = {")
    for key, (_a, _b, ior, _resid) in tables.items():
        print(f'    "{key}": {ior:.5f},')
    print("}")


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
    for name, symbol in _ILLUMINANTS.items():
        payload[f"illum_{name}"] = _resample_interleaved(
            _brace_block(src, symbol)
        ).astype(np.float32)
    # Raw glass eta curves: the runtime uses the fitted Cauchy literals, but
    # vendoring the curves keeps the fit-residual test hostless.
    for key, symbol in _GLASSES.items():
        payload[f"glass_{key}_eta"] = _resample_interleaved(
            _brace_block(src, symbol)
        ).astype(np.float32)
    out = out_dir / "spectral_curves.npz"
    np.savez_compressed(out, **payload)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pbrt", type=Path, default=Path.home() / "projects" / "pbrt-v4")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--print-tables", action="store_true",
                    help="print the spectral_tables glass literals and exit")
    args = ap.parse_args()
    if args.print_tables:
        print_tables(args.pbrt)
        return
    t = extract_upsample_table(args.pbrt, args.out)
    c = extract_curves(args.pbrt, args.out)
    print(f"wrote {t} ({t.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {c} ({c.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
