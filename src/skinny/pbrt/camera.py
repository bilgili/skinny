"""pbrt camera -> UsdGeom.Camera mapping (design D7).

pbrt's ``fov`` addresses the *shorter* image axis. We fix the vertical aperture
(skinny's 24 mm default) and solve for a focal length that reproduces pbrt's
framing for the film aspect ratio.
"""

from __future__ import annotations

import math

DEFAULT_VERTICAL_APERTURE_MM = 24.0


def perspective_to_camera(params, aspect: float, notes: list[str]) -> dict:
    """Return UsdGeom.Camera intrinsics for a pbrt ``perspective`` camera.

    *aspect* is width/height. Output keys: focal_length_mm, vertical_aperture_mm,
    horizontal_aperture_mm, and (optionally) fstop / focus_distance.
    """
    fov = params.float("fov", 90.0)
    v_ap = DEFAULT_VERTICAL_APERTURE_MM
    h_ap = v_ap * aspect

    if aspect >= 1.0:
        # landscape: shorter axis is vertical -> fov is the vertical fov
        v_fov = math.radians(fov)
        focal = (v_ap / 2.0) / math.tan(v_fov / 2.0)
    else:
        # portrait: shorter axis is horizontal -> fov is the horizontal fov
        h_fov = math.radians(fov)
        focal = (h_ap / 2.0) / math.tan(h_fov / 2.0)

    out = {
        "focal_length_mm": focal,
        "vertical_aperture_mm": v_ap,
        "horizontal_aperture_mm": h_ap,
    }

    lr = params.float("lensradius", None)
    ad = params.float("aperturediameter", None)
    fd = params.float("focaldistance", None)
    if lr is not None and lr > 0.0:
        # f-number ~ focal / aperture_diameter; aperture_diameter = 2*lensradius(scene units)
        out["fstop"] = focal / max(2.0 * lr * 1000.0, 1e-6)
        notes.append("depth-of-field from lensradius (approximate f-stop)")
    elif ad is not None and ad > 0.0:
        out["fstop"] = focal / max(ad, 1e-6)
    if fd is not None:
        out["focus_distance"] = fd
    return out


def vertical_fov_from_intrinsics(focal_mm: float, vertical_aperture_mm: float) -> float:
    """Inverse of the mapping: derive the vertical FOV (degrees) the loader sees."""
    return math.degrees(2.0 * math.atan((vertical_aperture_mm / 2.0) / focal_mm))
