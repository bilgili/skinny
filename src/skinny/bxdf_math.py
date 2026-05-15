"""Pure BXDF evaluation + lobe rasterisation.

Extracted from the legacy Tk ``bxdf_visualizer`` so the Qt port can reuse
the math without dragging in Tk. Lambert + GGX-Smith standard_surface
analytic BSDF for the CPU fallback, plus a Pillow-based renderer for the
hemisphere lobe.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError as _exc:  # pragma: no cover — pillow is a hard dep
    raise RuntimeError(
        "BXDF math requires Pillow; install via `pip install Pillow`."
    ) from _exc


# ── Analytic BSDF (Lambert + GGX-Smith) ────────────────────────────


def _ggx_smith_g1(n_dot_v: float, alpha: float) -> float:
    k = alpha * alpha / 2.0
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 1e-6)


def _ggx_d(n_dot_h: float, alpha: float) -> float:
    a2 = alpha * alpha
    denom = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0
    return a2 / max(math.pi * denom * denom, 1e-8)


def _fresnel_schlick(cos_theta: float, F0: np.ndarray) -> np.ndarray:
    return F0 + (1.0 - F0) * (1.0 - cos_theta) ** 5


def eval_std_surface(
    wi: np.ndarray, wo: np.ndarray, params: dict[str, object],
) -> np.ndarray:
    """Lambert + GGX-Smith reflectance for one (wi, wo) pair in tangent space.

    Tangent-space inputs where +Z is the shading normal. Returns RGB
    reflectance f(wi, wo) · cos(theta_i). Matches the diffuse + specular
    terms of ``mtlx_std_surface.slang::evalStdSurfaceBSDF`` for the common
    opaque case (no coat, no sheen, no transmission).
    """
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    wi = wi.astype(np.float64)
    wo = wo.astype(np.float64)
    n_dot_wi = float(np.dot(n, wi))
    n_dot_wo = float(np.dot(n, wo))
    if n_dot_wi <= 0.0 or n_dot_wo <= 0.0:
        return np.zeros(3, dtype=np.float64)

    base_color = np.array(params.get("base_color", (0.8, 0.8, 0.8)), dtype=np.float64)
    metalness = float(params.get("metalness", params.get("metallic", 0.0)))
    specular = float(params.get("specular", 1.0))
    roughness = float(params.get("specular_roughness", params.get("roughness", 0.5)))
    ior = float(params.get("specular_IOR", params.get("ior", 1.5)))
    base = float(params.get("base", 1.0))

    alpha = max(roughness * roughness, 1e-3)

    f_d = (base * base_color / math.pi) * (1.0 - metalness)

    f0_diel_scalar = ((ior - 1.0) / (ior + 1.0)) ** 2
    f0_diel = specular * np.array([f0_diel_scalar] * 3, dtype=np.float64)
    F0 = f0_diel * (1.0 - metalness) + base_color * metalness

    h = wi + wo
    h_len = float(np.linalg.norm(h))
    if h_len < 1e-6:
        return f_d * n_dot_wi
    h = h / h_len
    n_dot_h = max(float(np.dot(n, h)), 0.0)
    v_dot_h = max(float(np.dot(wo, h)), 0.0)

    D = _ggx_d(n_dot_h, alpha)
    G = _ggx_smith_g1(n_dot_wi, alpha) * _ggx_smith_g1(n_dot_wo, alpha)
    F = _fresnel_schlick(v_dot_h, F0)
    f_s = (D * G * F) / max(4.0 * n_dot_wi * n_dot_wo, 1e-6)

    return (f_d + f_s) * n_dot_wi


def eval_grid(
    locked_dir: np.ndarray, lock_mode: int,
    n_theta: int, n_phi: int, params: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the lobe over a (theta, phi) grid on the upper hemisphere.

    Returns ``(dirs, f)`` where ``dirs`` is ``(n_theta, n_phi, 3)`` of
    tangent-space directions of the swept axis and ``f`` is matching RGB
    reflectance. Lower hemisphere is implicit zero.
    """
    thetas = (np.arange(n_theta) + 0.5) / n_theta * (math.pi * 0.5)
    phis = np.arange(n_phi) / n_phi * (2.0 * math.pi)
    sin_t = np.sin(thetas)
    cos_t = np.cos(thetas)
    cos_p = np.cos(phis)
    sin_p = np.sin(phis)
    dirs = np.empty((n_theta, n_phi, 3), dtype=np.float64)
    dirs[..., 0] = sin_t[:, None] * cos_p[None, :]
    dirs[..., 1] = sin_t[:, None] * sin_p[None, :]
    dirs[..., 2] = cos_t[:, None]

    f = np.zeros((n_theta, n_phi, 3), dtype=np.float64)
    for i in range(n_theta):
        for j in range(n_phi):
            d = dirs[i, j]
            if lock_mode == 0:
                f[i, j] = eval_std_surface(locked_dir, d, params)
            else:
                f[i, j] = eval_std_surface(d, locked_dir, params)
    return dirs, f


# ── Lobe rasterisation (PIL) ───────────────────────────────────────


def _euler_to_rot(yaw: float, pitch: float) -> np.ndarray:
    """Camera orbit rotation. Yaw around world +Z (tangent normal); pitch
    tilts the view up/down toward the equator.
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    return Rx @ Rz


def render_lobe_image(
    dirs: np.ndarray, f: np.ndarray,
    yaw: float, pitch: float,
    size: int = 480, log_scale: bool = True, zoom: float = 1.0,
) -> Image.Image:
    """Rasterise the lobe to a ``size × size`` PIL image.

    Vertex radius = baseline + magnitude. Per-quad fill colour encodes
    magnitude (heatmap). Quad strip is z-sorted (painter's algorithm).
    """
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f = np.maximum(f, 0.0)
    nt, npx = dirs.shape[:2]
    lum = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    lum_max = max(float(lum.max()), 1e-6)
    if log_scale:
        K = 20.0
        lum_norm = np.log1p(K * lum) / float(np.log1p(K * lum_max))
        color_lum = np.log1p(K * f) / float(np.log1p(K * max(f.max(), 1e-6)))
    else:
        lum_norm = lum / lum_max
        color_lum = f / max(float(f.max()), 1e-6)
    lum_norm = np.clip(lum_norm, 0.0, 1.0)
    color_lum = np.clip(color_lum, 0.0, 1.0)

    BASE_R = 0.20
    BULGE_R = 0.80
    radius = BASE_R + BULGE_R * lum_norm
    verts = dirs * radius[..., None]

    pole_r = float(radius[0].max())
    pole = np.tile([0.0, 0.0, pole_r], (npx, 1))
    verts_ext = np.concatenate([pole[None, :, :], verts], axis=0)

    R = _euler_to_rot(yaw, pitch)
    cam = verts_ext @ R.T

    half = size * 0.5
    scale = half * 0.78 * max(zoom, 0.05)
    px = half + scale * cam[..., 0]
    py = half - scale * cam[..., 2]

    img = Image.new("RGB", (size, size), (18, 18, 26))
    draw = ImageDraw.Draw(img)
    cx = cy = half
    guide_r = scale * 1.0
    draw.ellipse([cx - guide_r, cy - guide_r, cx + guide_r, cy + guide_r],
                 outline=(60, 60, 80))
    draw.line([cx - guide_r, cy, cx + guide_r, cy], fill=(55, 55, 75))
    draw.line([cx, cy - guide_r, cx, cy + guide_r], fill=(55, 55, 75))

    rows = nt + 1
    cols = npx
    quads: list[tuple[float, tuple, tuple]] = []
    for i in range(rows - 1):
        for j in range(cols):
            j2 = (j + 1) % cols
            v00 = cam[i, j]
            v01 = cam[i, j2]
            v10 = cam[i + 1, j]
            v11 = cam[i + 1, j2]
            depth = 0.25 * (v00[1] + v01[1] + v10[1] + v11[1])
            poly = (
                (px[i, j], py[i, j]),
                (px[i, j2], py[i, j2]),
                (px[i + 1, j2], py[i + 1, j2]),
                (px[i + 1, j], py[i + 1, j]),
            )
            i_lobe = max(i - 1, 0)
            m = float(lum_norm[i_lobe, j])
            color = color_lum[i_lobe, j]
            rgb = (
                int(round(40 + 215 * color[0] * max(m, 0.25))),
                int(round(40 + 215 * color[1] * max(m, 0.25))),
                int(round(40 + 215 * color[2] * max(m, 0.25))),
            )
            quads.append((depth, poly, rgb))

    quads.sort(key=lambda q: q[0])
    for _, poly, rgb in quads:
        draw.polygon(poly, fill=rgb, outline=(110, 110, 130))

    return img
