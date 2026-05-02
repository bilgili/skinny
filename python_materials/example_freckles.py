"""Example Python material using SlangPile — freckle overlay.

Generates a procedural freckle density mask that skinny's skin material
can sample as a detail texture.

Usage (developer-side regen):
    python -m scripts.codegen
"""

from skinny import slangpile as sp


@sp.shader
def freckle_density(
    position: sp.float32x3,
    scale: sp.float32,
    seed: sp.float32,
    density: sp.float32,
) -> sp.float32:
    """Procedural freckle mask — returns [0,1] density at world position."""
    p = position * scale
    h = sp.fract(sp.sin(sp.dot(p, sp.float32x3(127.1, 311.7, 74.7)) + seed) * 43758.5453)
    return sp.step(1.0 - density, h)


@sp.shader
def freckle_albedo_tint(
    base_albedo: sp.float32x3,
    freckle_color: sp.float32x3,
    mask: sp.float32,
) -> sp.float32x3:
    """Blend base skin albedo toward freckle color based on mask."""
    return sp.lerp(base_albedo, freckle_color, mask)
