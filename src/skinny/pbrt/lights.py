"""pbrt light -> UsdLux mapping (design D6).

``distant`` -> DistantLight, ``point`` -> small SphereLight, ``infinite`` ->
DomeLight, ``spot`` -> best-effort SphereLight (flagged). Area lights
(``AreaLightSource`` on a shape) are handled in the shape path as an emissive
material, not here. Radiance is reduced to a (color, intensity) split that the
loader recombines as ``color * intensity * 2**exposure``.
"""

from __future__ import annotations

import os

import numpy as np
from pxr import Sdf, UsdLux

from . import spectra
from . import transform as T
from .emit import sanitize, to_gf_matrix


def _color_intensity(rgb, scale: float = 1.0):
    rgb = np.asarray(rgb, dtype=np.float64) * float(scale)
    lum = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    if lum <= 0:
        return [0.0, 0.0, 0.0], 0.0
    return (rgb / lum).tolist(), lum


def _orient_z_to(direction) -> np.ndarray:
    """4x4 rotation whose local +Z maps to *direction* (unit)."""
    z = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(z)
    if n == 0:
        return T.identity()
    z = z / n
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = x, y, z
    return m


def add_light(stage, parent_path: str, light, report) -> bool:
    """Author one UsdLux light. Returns True if a light was created."""
    p = light.params
    ltype = light.type
    scale = p.float("scale", 1.0)
    name = sanitize(f"light_{ltype}_{id(light) & 0xFFFF:x}")
    path = f"{parent_path}/{name}"

    if ltype == "distant":
        frm = np.array(p.floats("from", [0.0, 0.0, 0.0]))
        to = np.array(p.floats("to", [0.0, 0.0, 1.0]))
        dir_local = to - frm
        dir_world = light.ctm[:3, :3] @ dir_local
        toward_source = -(T.B[:3, :3] @ dir_world)
        prim = UsdLux.DistantLight.Define(stage, path)
        _set_color_intensity(prim, spectra.param_to_rgb(p.get("L"), illuminant=True)
                             or [1.0, 1.0, 1.0], scale)
        prim.AddTransformOp().Set(to_gf_matrix(_orient_z_to(toward_source)))
        report.exact(f"light:distant {path}")
        return True

    if ltype in ("point", "spot"):
        frm = np.array(p.floats("from", [0.0, 0.0, 0.0]))
        pos = T.B[:3, :3] @ (light.ctm[:3, :3] @ frm + light.ctm[:3, 3])
        prim = UsdLux.SphereLight.Define(stage, path)
        prim.CreateRadiusAttr(0.05)
        _set_color_intensity(prim, spectra.param_to_rgb(p.get("I"), illuminant=True)
                             or [1.0, 1.0, 1.0], scale)
        prim.AddTransformOp().Set(to_gf_matrix(T.translate(*pos)))
        if ltype == "spot":
            report.approx(f"light:spot {path}", "no skinny spotlight; emitted as point/sphere")
        else:
            report.approx(f"light:point {path}", "point emitted as small sphere light")
        return True

    if ltype == "infinite":
        prim = UsdLux.DomeLight.Define(stage, path)
        fname = p.string("filename", None)
        if fname:
            ext = os.path.splitext(fname)[1].lower()
            prim.CreateTextureFileAttr().Set(Sdf.AssetPath(fname))
            if ext != ".hdr":
                report.approx(f"light:infinite {path}",
                              f"env map {ext} referenced; loader expects .hdr (convert)")
            else:
                report.exact(f"light:infinite {path}")
        else:
            rgb = spectra.param_to_rgb(p.get("L"), illuminant=True) or [1.0, 1.0, 1.0]
            color, intensity = _color_intensity(rgb, scale)
            prim.CreateColorAttr(tuple(color))
            prim.CreateIntensityAttr(intensity)
            report.approx(f"light:infinite {path}", "constant infinite (no map)")
        return True

    report.skipped(f"light:{ltype}", "unsupported light type")
    return False


def _set_color_intensity(prim, rgb, scale) -> None:
    color, intensity = _color_intensity(rgb, scale)
    prim.CreateColorAttr(tuple(color))
    prim.CreateIntensityAttr(intensity)
