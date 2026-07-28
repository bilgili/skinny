"""Small device-free helpers shared by the renderer and its consumers.

Split out of ``renderer.py`` (change renderer-pure-core-extraction). Each of
these is a pure function over plain values; none of them touches a device
handle, so they belong on the hostless side of the renderer.
"""

from __future__ import annotations

import numpy as np

def _spectral_analytic_proposal_token(
    token: str,
    *,
    allow_environment: bool,
) -> str:
    """Resolve a proposal preset to the analytic spectral subset.

    BSDF and environment importance sampling are wavelength-independent and
    supported by the spectral path. Stateful neural inference is not; remove it
    and fall back to BSDF if no analytic proposal remains.
    """
    supported = {"bsdf", "env"} if allow_environment else {"bsdf"}
    analytic = [
        part.strip()
        for part in str(token).split(",")
        if part.strip() in supported
    ]
    return ",".join(analytic) or "bsdf"

def _instance_local_basis(transform: np.ndarray) -> np.ndarray:
    """World-space directions of an instance's local X/Y/Z axes — the
    normalized rows of the row-vector-convention transform's upper 3x3.
    Used by the local-space transform gizmo. Falls back to the matching world
    axis for any degenerate (zero-length) row."""
    m = np.asarray(transform, dtype=np.float64)
    basis = np.eye(3, dtype=np.float64)
    for i in range(3):
        row = m[i, :3]
        n = float(np.linalg.norm(row))
        if n > 1e-9:
            basis[i] = row / n
    return basis

def _light_value_to_vec3(value: object) -> np.ndarray:
    """Convert a color/vec3 value (tuple, list, Gf.Vec3f) to float32 array."""
    if hasattr(value, "asTuple"):
        value = value.asTuple()
    if isinstance(value, (list, tuple)):
        return np.array([float(value[0]), float(value[1]), float(value[2])], np.float32)
    return np.array([float(value)] * 3, np.float32)
