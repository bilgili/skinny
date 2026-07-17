"""Vendored spectral/color data for the pbrt importer.

Kept deliberately small: the CIE XYZ colour-matching functions are evaluated
from the Wyman-Sloan-Shirley (JCGT 2013) analytic fit (see ``spectra.py``)
rather than a bulk table, so the only tabulated data here is a handful of named
conductor complex IORs sampled at the sRGB primary wavelengths
(R ~ 630 nm, G ~ 532 nm, B ~ 465 nm).

The au/ag/al/cu values approximate refractiveindex.info; they are documented
approximations, not exact pbrt spectra, and are **kept as-is** — re-deriving them
from the vendored pbrt curves would shift existing RGB renders for no benefit
here. The cuzn/mgo/tio2 entries added later are instead sampled straight from the
vendored pbrt curves, so they need no hand-entered numbers::

    from skinny.pbrt.data import spectral_tables as st
    eta, k = st.named_metal_spectrum(key)   # then np.interp at 630/532/465 nm

(They are literals rather than a runtime derive only because importing
``spectral_tables`` from here would be circular.)

Note mgo/tio2 have k = 0 across the visible — pbrt files them under ``metal-*``
but they are physically dielectrics. ``fresnel_conductor_rgb`` reduces to the
dielectric form at k = 0, so they need no special case.
"""

from __future__ import annotations

#: Canonical named-conductor key -> shader id. **The** source of truth for the
#: named-metal set: the importer's recognised-name gate (`spectra._CONDUCTOR_CANON`),
#: the renderer's `spectralMetals` upload order (`renderer._SPECTRAL_METAL_ORDER`),
#: and the shader's `SPECTRAL_METAL_COUNT` gate all derive from or are pinned to it,
#: so a metal cannot be importable without a shader binding.
#:
#: Lives here (a GPU-free leaf module) rather than in `renderer.py` so the
#: invariant can be tested hostlessly — `renderer` imports `vulkan` at load, and a
#: guard that skips on hosts without the SDK is how a `metalId <= 4u` shader gate
#: shipped once already.
#:
#: **APPEND-ONLY.** An id is a byte offset into the uploaded buffer
#: (`(id-1)*stride`, see `namedMetalEtaK` in bindings.slang), so renumbering an
#: existing metal silently swaps materials in every checked-in scene. New metals go
#: on the end.
CONDUCTOR_METAL_ID: dict[str, int] = {
    "au": 1, "ag": 2, "al": 3, "cu": 4, "cuzn": 5, "mgo": 6, "tio2": 7,
}

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
    # Sampled from the vendored pbrt curves (see module docstring), not hand-entered.
    "cuzn": ((0.445, 0.568, 0.947), (3.522, 2.588, 1.920)),  # brass
    "mgo": ((1.735, 1.742, 1.750), (0.0, 0.0, 0.0)),
    "tio2": ((2.875, 2.969, 3.118), (0.0, 0.0, 0.0)),
    "brass": ((0.445, 0.568, 0.947), (3.522, 2.588, 1.920)),
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
