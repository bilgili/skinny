"""Transform algebra for the pbrt importer.

4x4 matrices in numpy, column-vector convention (``M @ [x, y, z, 1]``). The CTM
accumulates new transforms on the right (``CTM = CTM @ M``), matching pbrt's
``curTransform = curTransform * newTransform``.

Also holds the fixed left-handed (pbrt) -> right-handed (skinny/USD)
change-of-basis ``B = diag(1, 1, -1, 1)``: a transform ``M`` in pbrt space maps
to ``B @ M @ B`` in skinny space, a point ``p`` to ``B @ p``. Because ``B`` is a
reflection it reverses triangle winding, so geometry baked through an
orientation-reversing CTM must have its winding flipped (see
:func:`is_orientation_reversing`).
"""

from __future__ import annotations

import numpy as np

# Left-handed (pbrt) -> right-handed (skinny) change of basis. Self-inverse.
B = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float64)


def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = (x, y, z)
    return m


def scale(x: float, y: float, z: float) -> np.ndarray:
    return np.diag([x, y, z, 1.0]).astype(np.float64)


def rotate(angle_deg: float, x: float, y: float, z: float) -> np.ndarray:
    """Rotation by *angle_deg* degrees about axis (x, y, z) (Rodrigues)."""
    axis = np.array([x, y, z], dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(4, dtype=np.float64)
    axis = axis / norm
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    ux, uy, uz = axis
    r = np.array(
        [
            [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
        ],
        dtype=np.float64,
    )
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r
    return m


def from_pbrt_array(values) -> np.ndarray:
    """Build a matrix from pbrt's 16-value ``Transform``/``ConcatTransform``.

    pbrt lists the matrix column-major: the first four values are the first
    column. So ``M[i, j] = values[i + 4 * j]``.
    """
    v = [float(x) for x in values]
    if len(v) != 16:
        raise ValueError(f"transform needs 16 values, got {len(v)}")
    m = np.empty((4, 4), dtype=np.float64)
    for j in range(4):
        for i in range(4):
            m[i, j] = v[i + 4 * j]
    return m


def look_at(pos, look, up) -> np.ndarray:
    """pbrt left-handed camera-to-world LookAt.

    Columns are (right, newUp, dir, pos); the camera looks down +z.
    """
    pos = np.asarray(pos, dtype=np.float64)
    look = np.asarray(look, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    direction = _normalize(look - pos)
    right = _normalize(np.cross(_normalize(up), direction))  # left-handed
    new_up = np.cross(direction, right)
    m = np.eye(4, dtype=np.float64)
    m[:3, 0] = right
    m[:3, 1] = new_up
    m[:3, 2] = direction
    m[:3, 3] = pos
    return m


def invert(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(m)


def to_skinny(m: np.ndarray) -> np.ndarray:
    """Map a pbrt-space transform into skinny right-handed space: ``B @ M @ B``."""
    return B @ m @ B


def transform_point(m: np.ndarray, p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    h = np.append(p, 1.0)
    out = m @ h
    if out[3] != 0 and out[3] != 1:
        out = out / out[3]
    return out[:3]


def transform_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform an (N, 3) array of points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    out = h @ m.T
    w = out[:, 3:4]
    w = np.where(w == 0, 1.0, w)
    return (out[:, :3] / w).astype(np.float64)


def transform_normals(m: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Transform (N, 3) normals by the inverse-transpose of the 3x3 block."""
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    inv_t = np.linalg.inv(m[:3, :3]).T
    out = normals @ inv_t.T
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return out / norms


def is_orientation_reversing(m: np.ndarray) -> bool:
    """True if the 3x3 linear part flips handedness (negative determinant)."""
    return np.linalg.det(m[:3, :3]) < 0.0


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
