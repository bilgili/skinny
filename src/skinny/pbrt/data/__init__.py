"""Vendored spectral/color data for the pbrt importer.

Kept deliberately small: the CIE XYZ colour-matching functions are evaluated
from the Wyman-Sloan-Shirley (JCGT 2013) analytic fit (see ``spectra.py``)
rather than a bulk table, so the only tabulated data here is a handful of named
conductor complex IORs sampled at the sRGB primary wavelengths
(R ~ 630 nm, G ~ 532 nm, B ~ 465 nm). Values approximate refractiveindex.info;
they are documented approximations, not exact pbrt spectra.
"""

from __future__ import annotations

# name -> (eta_rgb, k_rgb), real and imaginary refractive index per channel.
NAMED_METAL_IOR: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "au": ((0.143, 0.375, 1.442), (3.983, 2.386, 1.603)),
    "gold": ((0.143, 0.375, 1.442), (3.983, 2.386, 1.603)),
    "ag": ((0.155, 0.116, 0.138), (3.600, 3.590, 2.510)),
    "silver": ((0.155, 0.116, 0.138), (3.600, 3.590, 2.510)),
    "al": ((1.345, 0.965, 0.617), (7.470, 6.400, 5.300)),
    "aluminium": ((1.345, 0.965, 0.617), (7.470, 6.400, 5.300)),
    "aluminum": ((1.345, 0.965, 0.617), (7.470, 6.400, 5.300)),
    "cu": ((0.200, 0.924, 1.102), (3.910, 2.450, 2.140)),
    "copper": ((0.200, 0.924, 1.102), (3.910, 2.450, 2.140)),
}


def _normalize_metal_key(name: str) -> str:
    n = name.strip().lower()
    for prefix in ("metal-", "metal_"):
        if n.startswith(prefix):
            n = n[len(prefix) :]
    for suffix in ("-eta", "-k", "_eta", "_k"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n
