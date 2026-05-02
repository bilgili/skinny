"""Reference energy conservation kernels as slangpile shaders.

Used for cross-validation of phase functions and BRDF properties.
"""

from __future__ import annotations

from skinny import slangpile as sp


@sp.shader
def ref_hg_phase(cos_theta: sp.float32, g: sp.float32) -> sp.float32:
    """Henyey-Greenstein phase function: (1-g^2) / (4*pi*(1+g^2-2g*cos)^1.5)."""
    g2: sp.float32 = g * g
    denom: sp.float32 = 1.0 + g2 - 2.0 * g * cos_theta
    return (1.0 - g2) / (4.0 * 3.14159265358979 * denom * sqrt(denom))


@sp.shader
def ref_ggx_ndf(n_dot_h: sp.float32, roughness: sp.float32) -> sp.float32:
    """GGX normal distribution function."""
    a: sp.float32 = roughness * roughness
    a2: sp.float32 = a * a
    d: sp.float32 = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    return a2 / (3.14159265358979 * d * d)


@sp.shader
def ref_smith_g1(n_dot_v: sp.float32, roughness: sp.float32) -> sp.float32:
    """Smith G1 masking function for GGX."""
    a: sp.float32 = roughness * roughness
    k: sp.float32 = a / 2.0
    return n_dot_v / (n_dot_v * (1.0 - k) + k)
