"""pbrt participating media + subsurface -> best-effort skinny overrides (D9).

Homogeneous media carry their coefficients onto the bound material as
``customData.skinnyOverrides`` (which the loader merges into
``Material.parameter_overrides``); skinny's volume path can consume them where
supported. Heterogeneous (grid/VDB) media are detected and flagged unsupported
rather than emitted as a wrong homogeneous stand-in.
"""

from __future__ import annotations

from . import spectra

_HETEROGENEOUS = {"uniformgrid", "grid", "nanovdb", "vdb", "cloud", "rgbgrid"}


def is_heterogeneous(medium) -> bool:
    return medium is not None and medium.type.lower() in _HETEROGENEOUS


def homogeneous_overrides(medium) -> dict:
    """Return a skinnyOverrides dict for a homogeneous medium."""
    p = medium.params
    scale = p.float("scale", 1.0)
    sigma_a = spectra.param_to_rgb(p.get("sigma_a")) or [1.0, 1.0, 1.0]
    sigma_s = spectra.param_to_rgb(p.get("sigma_s")) or [1.0, 1.0, 1.0]
    return {
        "pbrt_medium": medium.name,
        "volume_sigma_a": [c * scale for c in sigma_a],
        "volume_sigma_s": [c * scale for c in sigma_s],
        "volume_g": float(p.float("g", 0.0)),
    }


def subsurface_overrides(params) -> dict:
    """Coefficients for a pbrt ``subsurface`` material (homogeneous interior)."""
    sigma_a = spectra.param_to_rgb(params.get("sigma_a")) or [0.0011, 0.0024, 0.014]
    sigma_s = spectra.param_to_rgb(params.get("sigma_s")) or [2.55, 3.21, 3.77]
    scale = params.float("scale", 1.0)
    return {
        "volume_sigma_a": [c * scale for c in sigma_a],
        "volume_sigma_s": [c * scale for c in sigma_s],
        "volume_g": float(params.float("g", 0.0)),
        "ior": float(params.float("eta", 1.33)),
    }
