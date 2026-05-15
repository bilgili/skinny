"""Arcball + spherical helpers for the light-direction picker.

Pure numpy. Backend-agnostic; the Qt and Panel direction-picker widgets
both call into this module so they pick up the same rotation behaviour
the legacy Tk popup (``control_panel._DirectionPickerPopup``) had.
"""

from __future__ import annotations

import math

import numpy as np


def arcball_vec(px: float, py: float, cx: float, cy: float, r: float) -> np.ndarray:
    """Project canvas pixel ``(px, py)`` onto the unit hemisphere centred
    at ``(cx, cy)`` with radius ``r``. Screen-y is flipped to give a
    right-handed world frame (y up).
    """
    x = (px - cx) / r
    y = -(py - cy) / r
    d2 = x * x + y * y
    if d2 <= 1.0:
        return np.array([x, y, math.sqrt(1.0 - d2)])
    inv = 1.0 / math.sqrt(d2)
    return np.array([x * inv, y * inv, 0.0])


def rotate_by_delta(D: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of ``D`` by the rotation that takes the unit
    vector ``a`` to the unit vector ``b``.
    """
    axis = np.cross(a, b)
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-8:
        return D
    axis = axis / axis_len
    cos_t = float(np.clip(np.dot(a, b), -1.0, 1.0))
    sin_t = axis_len  # |a||b|sinθ = |a×b| for unit a, b
    return (
        D * cos_t
        + np.cross(axis, D) * sin_t
        + axis * np.dot(axis, D) * (1.0 - cos_t)
    )


def direction_to_eulers(D: np.ndarray) -> tuple[float, float]:
    """Inverse of ``eulers_to_direction``. Returns ``(elev_deg, az_deg)``."""
    el = math.degrees(math.asin(max(-1.0, min(1.0, float(D[1])))))
    az = math.degrees(math.atan2(float(D[0]), float(D[2])))
    return el, az


def eulers_to_direction(el_deg: float, az_deg: float) -> np.ndarray:
    """Spherical → Cartesian: same convention as ``renderer._update_light``.
    Elevation 0 + azimuth 0 → ``+Z``; positive elevation → ``+Y``.
    """
    el = math.radians(el_deg)
    az = math.radians(az_deg)
    return np.array([
        math.cos(el) * math.sin(az),
        math.sin(el),
        math.cos(el) * math.cos(az),
    ])
