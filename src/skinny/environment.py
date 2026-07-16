"""HDR environment maps — procedural presets plus .hdr file loader.

Presets are lit to suit skin rendering (studio softbox, noon sky, sunset).
The `hdrs/` directory next to the repo is scanned for Radiance `.hdr` files,
which are appended to the list. Use `python -m skinny.fetch_hdrs` to populate
it with well-known CC0 HDRIs from Poly Haven.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

ENV_WIDTH = 1024
ENV_HEIGHT = 512


@dataclass
class Environment:
    """HDR environment map with lazy loading.

    Data is computed/loaded on first access to ``.data`` and cached.
    Pass either ``_data`` (eager) or ``_loader`` (lazy).
    """

    name: str
    _data: np.ndarray | None = field(default=None, repr=False)
    _loader: Callable[[], np.ndarray] | None = field(default=None, repr=False)
    path: Path | None = None

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            if self._loader is not None:
                print(f"[skinny] loading environment '{self.name}'...")
                self._data = self._loader()
                self._loader = None
            else:
                self._data = _blank_rgba()
        return self._data


def _blank_rgba() -> np.ndarray:
    arr = np.zeros((ENV_HEIGHT, ENV_WIDTH, 4), np.float32)
    arr[..., 3] = 1.0
    return arr


def _add_sun(img: np.ndarray, phi: float, theta: float, radius_px: int, color: np.ndarray) -> None:
    h, w = img.shape[:2]
    cx = int((phi / (2.0 * np.pi) + 0.5) * w) % w
    cy = int(theta / np.pi * h)
    sigma2 = (radius_px * 0.5) ** 2
    for dy in range(-radius_px, radius_px + 1):
        yy = cy + dy
        if yy < 0 or yy >= h:
            continue
        for dx in range(-radius_px, radius_px + 1):
            xx = (cx + dx) % w
            r2 = dx * dx + dy * dy
            if r2 <= radius_px * radius_px:
                img[yy, xx, :3] += color * np.exp(-r2 / (2.0 * sigma2)).astype(np.float32)


def _neutral_gray() -> np.ndarray:
    img = _blank_rgba()
    img[..., :3] = 0.5
    return img


def _noon_sky() -> np.ndarray:
    img = _blank_rgba()
    v = np.arange(ENV_HEIGHT, dtype=np.float32) / (ENV_HEIGHT - 1)
    zenith = np.array([0.30, 0.55, 1.20], np.float32)
    horizon = np.array([0.90, 0.95, 1.05], np.float32)
    ground = np.array([0.20, 0.18, 0.15], np.float32)
    for y in range(ENV_HEIGHT):
        t = v[y]
        if t < 0.5:
            s = t / 0.5
            col = zenith * (1.0 - s) + horizon * s
        else:
            s = (t - 0.5) / 0.5
            col = horizon * (1.0 - s * 0.5) * 0.5 + ground * (s * 0.5)
        img[y, :, :3] = col
    _add_sun(img, phi=np.pi / 4.0, theta=np.pi / 3.0, radius_px=22,
             color=np.array([50.0, 45.0, 35.0], np.float32))
    return img


def _sunset() -> np.ndarray:
    img = _blank_rgba()
    v = np.arange(ENV_HEIGHT, dtype=np.float32) / (ENV_HEIGHT - 1)
    # Gradient stops (v, rgb)
    stops = [
        (0.00, np.array([0.05, 0.05, 0.18], np.float32)),  # high sky
        (0.35, np.array([0.40, 0.20, 0.35], np.float32)),  # upper warm
        (0.48, np.array([1.20, 0.55, 0.20], np.float32)),  # horizon glow
        (0.52, np.array([0.80, 0.40, 0.20], np.float32)),
        (0.60, np.array([0.10, 0.08, 0.06], np.float32)),  # ground
        (1.00, np.array([0.02, 0.02, 0.02], np.float32)),
    ]
    for y in range(ENV_HEIGHT):
        t = v[y]
        for i in range(len(stops) - 1):
            v0, c0 = stops[i]
            v1, c1 = stops[i + 1]
            if v0 <= t <= v1:
                s = (t - v0) / max(v1 - v0, 1e-6)
                img[y, :, :3] = c0 * (1.0 - s) + c1 * s
                break
    _add_sun(img, phi=0.0, theta=np.pi * 0.48, radius_px=26,
             color=np.array([40.0, 22.0, 10.0], np.float32))
    return img


def _studio_softbox() -> np.ndarray:
    """Soft key overhead + fill around — classic skin-portrait lighting."""
    img = _blank_rgba()
    v = np.arange(ENV_HEIGHT, dtype=np.float32) / (ENV_HEIGHT - 1)
    for y in range(ENV_HEIGHT):
        t = v[y]
        if t < 0.35:
            col = np.array([4.5, 4.4, 4.2], np.float32)  # softbox overhead
        elif t < 0.60:
            col = np.array([0.9, 0.9, 0.9], np.float32)  # white walls
        else:
            col = np.array([0.3, 0.3, 0.3], np.float32)  # dark floor
        img[y, :, :3] = col
    # Secondary fill from front-right
    _add_sun(img, phi=np.pi / 3.0, theta=np.pi * 0.45, radius_px=60,
             color=np.array([1.5, 1.5, 1.5], np.float32))
    return img


BUILT_IN: list[tuple[str, callable]] = [
    ("Neutral Gray",    _neutral_gray),
    ("Noon Sky",        _noon_sky),
    ("Sunset",          _sunset),
    ("Studio Softbox",  _studio_softbox),
]


def _load_radiance_hdr(path: Path) -> np.ndarray:
    """Decode a Radiance `.hdr` (RGBE) file into (H, W, 3) float32 linear RGB.

    Handles the adaptive run-length scheme (widths 8..32767) plus legacy
    straight-RGBE scanlines. We bring this in-tree to avoid needing a third-
    party HDR backend (imageio-freeimage / opencv) just for this one format.
    """
    with open(path, "rb") as f:
        magic = f.readline()
        if not (magic.startswith(b"#?RADIANCE") or magic.startswith(b"#?RGBE")):
            raise ValueError("not a Radiance HDR file")

        while True:
            line = f.readline()
            if not line:
                raise ValueError("truncated header")
            if line == b"\n":
                break

        res = f.readline().decode("ascii", errors="replace").split()
        if len(res) != 4 or "-Y" not in res or "+X" not in res:
            raise ValueError(f"unsupported orientation {' '.join(res)!r}")
        axes = {res[0]: int(res[1]), res[2]: int(res[3])}
        height, width = axes["-Y"], axes["+X"]

        rgbe = np.empty((height, width, 4), dtype=np.uint8)
        for y in range(height):
            head = f.read(4)
            if len(head) < 4:
                raise ValueError(f"truncated at row {y}")

            if head[0] == 0x02 and head[1] == 0x02 and (head[2] & 0x80) == 0:
                if ((head[2] << 8) | head[3]) != width:
                    raise ValueError(f"scanline width mismatch at row {y}")
                for c in range(4):
                    x = 0
                    while x < width:
                        n = f.read(1)[0]
                        if n > 128:
                            rgbe[y, x:x + (n - 128), c] = f.read(1)[0]
                            x += n - 128
                        else:
                            rgbe[y, x:x + n, c] = np.frombuffer(f.read(n), np.uint8)
                            x += n
            else:
                rgbe[y, 0, :] = np.frombuffer(head, np.uint8)
                rest = np.frombuffer(f.read((width - 1) * 4), np.uint8)
                rgbe[y, 1:, :] = rest.reshape(width - 1, 4)

    mantissa = rgbe[..., :3].astype(np.float32)
    e = rgbe[..., 3].astype(np.int32)
    scale = np.where(e > 0, np.ldexp(1.0, e - 128 - 8), 0.0).astype(np.float32)
    return mantissa * scale[..., None]


def build_env_distribution(env_rgba: np.ndarray) -> tuple[bytes, bytes, float]:
    """Build a 2D piecewise-constant sampling distribution over an equirect
    environment, for importance sampling in the path tracer.

    Returns ``(marginal_cdf_bytes, conditional_cdf_bytes, lum_integral)``:
      - marginal CDF over rows (v): ``ENV_HEIGHT + 1`` float32 entries, 0..1.
      - conditional CDF over columns (u) per row: ``ENV_HEIGHT * (ENV_WIDTH+1)``
        float32 entries, row-major, each row 0..1.
      - ``lum_integral``: the sphere luminance integral ∫L dω of the (unscaled)
        map — the sin θ-weighted grid total times the per-cell solid angle
        (π/H)·(2π/W). Feeds the SPPM photon-group power distribution
        (Φ_env = πR² · envIntensity · ∫L dω).

    The per-cell weight is sin(theta)-corrected luminance (equirect rows near
    the poles subtend less solid angle). The shader recovers the image-space
    pdf from finite differences of these CDFs — no separate func/integral
    buffer is needed — and converts to a solid-angle pdf for MIS.
    """
    rgba = _resize_equirect(env_rgba)
    lum = (0.2126 * rgba[..., 0] + 0.7152 * rgba[..., 1]
           + 0.0722 * rgba[..., 2]).astype(np.float64)

    # sin(theta) weight per row; theta = (v + 0.5)/H * pi.
    v = np.arange(ENV_HEIGHT, dtype=np.float64)
    sin_theta = np.sin((v + 0.5) / ENV_HEIGHT * np.pi)
    func = lum * sin_theta[:, None]          # (H, W)

    # Conditional CDF per row (over u). Guard fully-black rows with a uniform
    # distribution so sampling stays well-defined.
    row_sum = func.sum(axis=1)               # (H,)
    safe = row_sum > 0.0
    cond_cdf = np.zeros((ENV_HEIGHT, ENV_WIDTH + 1), dtype=np.float64)
    cum = np.cumsum(func, axis=1)            # (H, W)
    cond_cdf[:, 1:] = cum
    cond_cdf[safe] /= row_sum[safe, None]
    # Uniform fallback for black rows: 0, 1/W, 2/W, ..., 1.
    if (~safe).any():
        uniform = np.linspace(0.0, 1.0, ENV_WIDTH + 1, dtype=np.float64)
        cond_cdf[~safe] = uniform
    cond_cdf[:, ENV_WIDTH] = 1.0             # exact endpoint

    # Marginal CDF over rows, weighted by each row's total func.
    total = row_sum.sum()
    marg_cdf = np.zeros(ENV_HEIGHT + 1, dtype=np.float64)
    if total > 0.0:
        marg_cdf[1:] = np.cumsum(row_sum) / total
    else:
        marg_cdf = np.linspace(0.0, 1.0, ENV_HEIGHT + 1, dtype=np.float64)
    marg_cdf[ENV_HEIGHT] = 1.0

    lum_integral = float(total * (np.pi / ENV_HEIGHT) * (2.0 * np.pi / ENV_WIDTH))
    return (marg_cdf.astype(np.float32).tobytes(),
            cond_cdf.astype(np.float32).tobytes(),
            lum_integral)


def _resize_equirect(img: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resize to ENV_HEIGHT × ENV_WIDTH RGBA float32."""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 3:
        rgba = np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)
    else:
        rgba = img[..., :4]
    h, w = rgba.shape[:2]
    ys = (np.arange(ENV_HEIGHT) * h / ENV_HEIGHT).astype(np.int32)
    xs = (np.arange(ENV_WIDTH) * w / ENV_WIDTH).astype(np.int32)
    return np.ascontiguousarray(rgba[ys][:, xs].astype(np.float32))


# File extensions we recognise as equirectangular HDR environments.
# .hdr is decoded in-tree (Radiance RGBE); the rest are routed through
# imageio[freeimage] which handles .exr / .pfm / etc.
_HDR_EXTS = {".hdr", ".exr", ".pfm"}


def _load_via_imageio(path: Path) -> np.ndarray:
    """Decode any HDR format imageio[freeimage] can read into (H, W, 3)
    float32 linear RGB.
    """
    import imageio.v3 as iio
    arr = iio.imread(str(path))
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr


def _make_hdr_loader(path: Path) -> Callable[[], np.ndarray]:
    def loader() -> np.ndarray:
        ext = path.suffix.lower()
        if ext == ".hdr":
            raw = _load_radiance_hdr(path)
        else:
            raw = _load_via_imageio(path)
        return _resize_equirect(raw)
    return loader


def load_environments(hdr_dir: Path | None = None) -> list[Environment]:
    """Return built-in presets, then any HDR files found in ``hdr_dir``.

    Supports ``.hdr``, ``.exr``, ``.pfm``. All environments are lazy —
    data is generated/loaded on first access.
    """
    envs: list[Environment] = [
        Environment(name=name, _loader=fn) for name, fn in BUILT_IN
    ]

    if hdr_dir is None or not hdr_dir.exists():
        return envs

    candidates: list[Path] = []
    for ext in sorted(_HDR_EXTS):
        candidates.extend(hdr_dir.glob(f"*{ext}"))
    for path in sorted(candidates, key=lambda p: p.name.lower()):
        envs.append(Environment(
            name=path.stem, _loader=_make_hdr_loader(path), path=path,
        ))
        print(f"[skinny] found HDR: {path.name} (lazy)")

    return envs


def make_environment_from_path(path: Path) -> Environment:
    """Build a single ``Environment`` from an arbitrary file path.

    Used by the runtime "Load HDR…" picker in the UI to add HDR files
    that live outside the default ``hdrs/`` directory.
    """
    return Environment(name=path.stem, _loader=_make_hdr_loader(path), path=path)
