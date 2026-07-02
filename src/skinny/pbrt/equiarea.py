"""Equal-area octahedral square <-> sphere chart, and the equal-area -> equirect
reprojection used to bake pbrt v4 `infinite` light image maps into the
equirectangular `.hdr` skinny's dome-light path loads.

pbrt v4 `ImageInfiniteLight` always parameterizes its image with the equal-area
octahedral mapping (`EqualAreaSquareToSphere`); skinny samples its dome texture
equirectangularly (`environment.slang: directionToEquirectUV`,
`u = atan2(dx,dz)/2π + 0.5`, `v = acos(dy)/π`, +y up). This module ports the pbrt
chart and inverts skinny's convention so a baked equirect pixel holds the
radiance pbrt would return for that pixel's shader direction.

Pure numpy (no USD/torch/GPU) so it is unit-testable under `.venv`.
"""

from __future__ import annotations

import numpy as np

# World-axis permutation P: pbrt env light-space direction <-> skinny world
# direction (+y up, phi about +y from +z). Identity is the starting hypothesis;
# the render A/B in the orientation gate (tasks.md §4) fixes the final mapping.
# Keep the forward map and its inverse here so callers stay convention-agnostic.


def _apply_axis(d: np.ndarray) -> np.ndarray:
    """skinny world direction -> pbrt light-space direction (permutation P).

    P = B = diag(1, 1, -1): (x, y, z) -> (x, y, -z). The env reprojection MUST use
    the same pbrt<->skinny change-of-basis the geometry import uses
    (`pbrt.transform.B = diag(1, 1, -1, 1)`), otherwise the baked env is rotated
    relative to the world the camera and meshes live in. The geometry import does
    NOT rotate the up-axis (Y stays Y); it flips handedness on Z. An earlier
    `Rx(+90)` here put the env's sky at skinny +y, which *looks* upright when the
    env is viewed in isolation under a +y-up assumption, but is rotated 90° about
    X away from the actual imported geometry frame — so a ground plane reflected a
    neutral/ground band of the map instead of the sky pbrt shows. Matching B makes
    `skinny_env(d) == pbrt_env(B·d)` for every direction (verified against
    `small_rural_road_equiarea.exr` at the sssdragon camera).
    """
    x = d[..., 0]
    y = d[..., 1]
    z = d[..., 2]
    return np.stack([x, y, -z], axis=-1)


def _apply_axis_inv(d: np.ndarray) -> np.ndarray:
    """pbrt light-space direction -> skinny world direction (P^-1 = P; B is an
    involution: (x, y, z) -> (x, y, -z))."""
    x = d[..., 0]
    y = d[..., 1]
    z = d[..., 2]
    return np.stack([x, y, -z], axis=-1)


def equal_area_square_to_sphere(p: np.ndarray) -> np.ndarray:
    """pbrt `EqualAreaSquareToSphere`: [0,1]^2 -> unit sphere. Vectorized.

    `p` is (..., 2); returns (..., 3).
    """
    p = np.asarray(p, dtype=np.float64)
    u = 2.0 * p[..., 0] - 1.0
    v = 2.0 * p[..., 1] - 1.0
    up = np.abs(u)
    vp = np.abs(v)
    sd = 1.0 - (up + vp)
    d = np.abs(sd)
    r = 1.0 - d
    # phi: guard r == 0 (the -z corners) -> pbrt uses 1.0
    safe_r = np.where(r == 0.0, 1.0, r)
    phi = np.where(r == 0.0, 1.0, (vp - up) / safe_r + 1.0) * (np.pi / 4.0)
    z = np.copysign(1.0 - r * r, sd)
    cos_phi = np.copysign(np.cos(phi), u)
    sin_phi = np.copysign(np.sin(phi), v)
    scale = r * np.sqrt(np.maximum(2.0 - r * r, 0.0))
    x = cos_phi * scale
    y = sin_phi * scale
    return np.stack([x, y, z], axis=-1)


def sphere_to_equal_area_square(d: np.ndarray) -> np.ndarray:
    """pbrt `EqualAreaSphereToSquare`: unit sphere -> [0,1]^2. Inverse of above.

    Uses a real `atan` (not pbrt's polynomial approx) for exact inversion.
    `d` is (..., 3); returns (..., 2).
    """
    d = np.asarray(d, dtype=np.float64)
    x = np.abs(d[..., 0])
    y = np.abs(d[..., 1])
    z = np.abs(d[..., 2])
    r = np.sqrt(np.maximum(0.0, 1.0 - z))
    a = np.maximum(x, y)
    b = np.minimum(x, y)
    b = np.where(a == 0.0, 0.0, b / np.where(a == 0.0, 1.0, a))
    phi = np.arctan(b) * (2.0 / np.pi)
    swap = x < y
    phi = np.where(swap, 1.0 - phi, phi)
    v = phi * r
    u = r - v
    neg = d[..., 2] < 0.0
    u_n = 1.0 - v
    v_n = 1.0 - u
    u = np.where(neg, u_n, u)
    v = np.where(neg, v_n, v)
    u = np.copysign(u, d[..., 0])
    v = np.copysign(v, d[..., 1])
    return np.stack([(u + 1.0) * 0.5, (v + 1.0) * 0.5], axis=-1)


def equirect_uv_to_direction(uv: np.ndarray) -> np.ndarray:
    """Invert skinny `directionToEquirectUV`. `uv` (..., 2) in [0,1] -> dir (..., 3).

    Convention (environment.slang): u = atan2(dx,dz)/2π + 0.5, v = acos(dy)/π.
    """
    uv = np.asarray(uv, dtype=np.float64)
    phi = (uv[..., 0] - 0.5) * (2.0 * np.pi)
    theta = uv[..., 1] * np.pi
    sin_t = np.sin(theta)
    dy = np.cos(theta)
    dx = sin_t * np.sin(phi)
    dz = sin_t * np.cos(phi)
    return np.stack([dx, dy, dz], axis=-1)


def _bilinear_clamped(img: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Bilinear sample `img` (H,W,C) at float (col,row), clamping at the border."""
    h, w = img.shape[:2]
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    fc = col - c0
    fr = row - r0
    c0c = np.clip(c0, 0, w - 1)
    c1c = np.clip(c0 + 1, 0, w - 1)
    r0c = np.clip(r0, 0, h - 1)
    r1c = np.clip(r0 + 1, 0, h - 1)
    fc = fc[..., None]
    fr = fr[..., None]
    top = img[r0c, c0c] * (1.0 - fc) + img[r0c, c1c] * fc
    bot = img[r1c, c0c] * (1.0 - fc) + img[r1c, c1c] * fc
    return top * (1.0 - fr) + bot * fr


def equiarea_to_equirect(src: np.ndarray, height: int | None = None,
                         world_to_light: np.ndarray | None = None) -> np.ndarray:
    """Reproject an equal-area octahedral square `src` (E,E,C) to equirectangular.

    Returns (H, 2H, C) with H = `height` (default = source edge). Each output
    pixel's skinny shader direction is mapped through P then the equal-area chart
    and bilinearly sampled from `src`.

    `world_to_light` (3×3, pbrt space) bakes an authored light CTM rotation into
    the resample: pbrt evaluates an infinite light's image in LIGHT space and
    rotates light→world with the CTM captured at `LightSource` time, so sampling
    goes world → light via the inverse rotation. `None`/identity = no rotation
    (the pre-existing behavior). Dropping this rotated bunny-cloud's `Rotate 10`
    sky by ~11° of horizon (nanovdb-volume-rendering).
    """
    src = np.asarray(src, dtype=np.float64)
    edge = src.shape[0]
    h = int(height) if height is not None else edge
    w = 2 * h
    js = (np.arange(w) + 0.5) / w
    is_ = (np.arange(h) + 0.5) / h
    uu, vv = np.meshgrid(js, is_)  # (H, W)
    uv = np.stack([uu, vv], axis=-1)
    d = equirect_uv_to_direction(uv)            # skinny world dirs
    dp = _apply_axis(d)                          # -> pbrt world space
    if world_to_light is not None:
        dp = dp @ np.asarray(world_to_light, np.float64).T   # -> pbrt light space
    sq = sphere_to_equal_area_square(dp)         # (H, W, 2) in [0,1]
    col = sq[..., 0] * edge - 0.5
    row = sq[..., 1] * edge - 0.5
    return _bilinear_clamped(src, col, row)
