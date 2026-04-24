"""Tattoo textures — procedural presets plus PNG/JPG loader.

A tattoo is a 2D RGBA image:
  RGB = linear ink colour (0..1 per channel)
  A   = ink density at this pixel (0 = bare skin, 1 = maximum pigment load)

At render time, the image is sampled at the surface UV; the RGB+alpha drives a
dermis-layer absorption/scattering contribution in `skin_bssrdf.slang`. The
image is mapped via the same spherical UV used elsewhere:
    u = 0.5 + atan2(z, x) / (2π)
    v = 0.5 - asin(y)    / π

So u=0.75 is the front of the face, v≈0.38 is the forehead, v≈0.55 is the
cheek/mouth region. The procedural presets target the right cheek/brow so
they land visibly on the head when the camera is orbiting the face.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TATTOO_WIDTH = 512
TATTOO_HEIGHT = 512


@dataclass
class Tattoo:
    name: str
    data: np.ndarray   # (H, W, 4) linear RGBA float32


# ── Small helpers ──────────────────────────────────────────────────

def _blank_rgba() -> Image.Image:
    return Image.new("RGBA", (TATTOO_WIDTH, TATTOO_HEIGHT), (0, 0, 0, 0))


def _to_linear_rgba32f(pil: Image.Image) -> np.ndarray:
    """8-bit sRGB image → linear RGBA float32 (H, W, 4). Alpha is unchanged."""
    arr = np.asarray(pil.convert("RGBA"), dtype=np.float32) / 255.0
    # Decode sRGB → linear for the colour channels; alpha stays linear.
    c = arr[..., :3]
    linear = np.where(
        c <= 0.04045,
        c / 12.92,
        ((c + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)
    out = np.empty_like(arr)
    out[..., :3] = linear
    out[..., 3] = arr[..., 3]
    return np.ascontiguousarray(out)


# Face-centred pixel coordinate for the procedural presets.
# u=0.75 (front) is at x = TATTOO_WIDTH * 0.75 = 384; we pick slightly to the
# side so the design sits on the cheek/brow rather than dead-centre of the
# nose, which is tight on sphere-mapped area.
_FACE_CX = int(TATTOO_WIDTH * 0.72)


# ── Procedural presets ─────────────────────────────────────────────

def _preset_none() -> Image.Image:
    return _blank_rgba()


def _preset_cross_black() -> Image.Image:
    """Small geometric cross — carbon black, uniform absorption across RGB."""
    img = _blank_rgba()
    draw = ImageDraw.Draw(img)
    cx, cy = _FACE_CX, int(TATTOO_HEIGHT * 0.42)
    half_h, half_w = 36, 6
    col = (5, 5, 5, 255)
    draw.rectangle((cx - half_w, cy - half_h, cx + half_w, cy + half_h), fill=col)
    draw.rectangle((cx - half_h, cy - half_w, cx + half_h, cy + half_w), fill=col)
    return img


def _preset_heart_red() -> Image.Image:
    """Red heart — iron-oxide pigment; absorbs green/blue, passes red."""
    img = _blank_rgba()
    draw = ImageDraw.Draw(img)
    cx, cy = _FACE_CX, int(TATTOO_HEIGHT * 0.55)
    r = 22
    col = (215, 25, 30, 255)
    # Two lobes + triangular bottom.
    draw.ellipse((cx - r - 15, cy - r, cx - 15 + r, cy + r), fill=col)
    draw.ellipse((cx + 15 - r, cy - r, cx + 15 + r, cy + r), fill=col)
    draw.polygon(
        [(cx - r - 15, cy + 8), (cx + r + 15, cy + 8), (cx, cy + r + 28)],
        fill=col,
    )
    return img


def _preset_blue_tribal() -> Image.Image:
    """Tribal-style sweep — copper-phthalocyanine blue; absorbs red, passes blue."""
    img = _blank_rgba()
    draw = ImageDraw.Draw(img)
    cx, cy = _FACE_CX, int(TATTOO_HEIGHT * 0.45)
    col = (20, 60, 215, 255)
    # Three curved "flame" blades splayed outward from a centre spine.
    for angle_deg in (-30, 0, 30):
        a = np.radians(angle_deg)
        length = 58
        tip_x = cx + int(np.sin(a) * length)
        tip_y = cy - int(np.cos(a) * length)
        base = 10
        draw.polygon(
            [
                (cx - base, cy + 18),
                (cx + base, cy + 18),
                (tip_x, tip_y),
            ],
            fill=col,
        )
    # Small accent dots.
    for dx in (-40, 40):
        draw.ellipse(
            (cx + dx - 5, cy + 24 - 5, cx + dx + 5, cy + 24 + 5), fill=col
        )
    return img


def _preset_color_flame() -> Image.Image:
    """Multi-colour radial flame — showcases spatial RGB-dependent absorption."""
    img = _blank_rgba()
    pixels = img.load()
    cx, cy = _FACE_CX, int(TATTOO_HEIGHT * 0.48)
    # Elliptical flame region; hue sweeps horizontally, shape fades with radius.
    rx, ry = 50, 65
    for y in range(cy - ry - 4, cy + ry + 4):
        if y < 0 or y >= TATTOO_HEIGHT:
            continue
        for x in range(cx - rx - 4, cx + rx + 4):
            if x < 0 or x >= TATTOO_WIDTH:
                continue
            nx = (x - cx) / rx
            ny = (y - cy) / ry
            d = nx * nx + ny * ny
            if d >= 1.0:
                continue
            # Hue sweep: red → orange → yellow → green across the width.
            t = 0.5 + 0.5 * nx           # 0 at left edge → 1 at right edge
            r = int(np.clip(255 * (1.0 - 0.4 * t), 0, 255))
            g = int(np.clip(255 * t * 0.95, 0, 255))
            b = int(np.clip(40 * (1.0 - t), 0, 255))
            # Alpha fades toward the outer rim.
            a = int(np.clip(255 * (1.0 - d) ** 1.3, 0, 255))
            pixels[x, y] = (r, g, b, a)
    return img


BUILT_IN: list[tuple[str, callable]] = [
    ("None",            _preset_none),
    ("Black Cross",     _preset_cross_black),
    ("Red Heart",       _preset_heart_red),
    ("Blue Tribal",     _preset_blue_tribal),
    ("Color Flame",     _preset_color_flame),
]


# ── File loading ───────────────────────────────────────────────────

def _load_image_file(path: Path) -> np.ndarray:
    """Load PNG/JPG and resize to TATTOO_WIDTH × TATTOO_HEIGHT RGBA32F."""
    with Image.open(path) as im:
        im = im.convert("RGBA")
        if im.size != (TATTOO_WIDTH, TATTOO_HEIGHT):
            im = im.resize((TATTOO_WIDTH, TATTOO_HEIGHT), Image.LANCZOS)
        return _to_linear_rgba32f(im)


# ── Public API ─────────────────────────────────────────────────────

def load_tattoos(tattoo_dir: Path | None = None) -> list[Tattoo]:
    """Return built-in procedural tattoos, then any PNG/JPG in `tattoo_dir`."""
    out: list[Tattoo] = [Tattoo(name, _to_linear_rgba32f(fn())) for name, fn in BUILT_IN]
    if tattoo_dir is None or not tattoo_dir.exists():
        return out

    for path in sorted(tattoo_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        try:
            data = _load_image_file(path)
            out.append(Tattoo(path.stem, data))
            print(f"[skinny] loaded tattoo: {path.name}")
        except Exception as exc:
            print(f"[skinny] failed to load tattoo {path.name}: {exc}")
    return out


def blank_tattoo_data() -> np.ndarray:
    """Zero-alpha tattoo used as the initial upload so the binding is valid."""
    return np.zeros((TATTOO_HEIGHT, TATTOO_WIDTH, 4), dtype=np.float32)
