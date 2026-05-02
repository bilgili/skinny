"""Reference implementations of sampling functions as slangpile kernels.

These are compiled to Slang and used for cross-validation against
the handwritten .slang implementations in src/skinny/shaders/.
"""

from __future__ import annotations

from skinny import slangpile as sp


@sp.shader
def ref_cosine_pdf(n_dot_l: sp.float32) -> sp.float32:
    """Cosine hemisphere PDF = max(NdotL, 0) / pi."""
    return max(n_dot_l, 0.0) / 3.14159265358979


@sp.shader
def ref_uniform_sphere_pdf() -> sp.float32:
    """Uniform sphere PDF = 1/(4*pi)."""
    return 1.0 / (4.0 * 3.14159265358979)


@sp.shader
def ref_power_heuristic(pdf_a: sp.float32, pdf_b: sp.float32) -> sp.float32:
    """Balance heuristic with power=2."""
    a2: sp.float32 = pdf_a * pdf_a
    b2: sp.float32 = pdf_b * pdf_b
    return a2 / max(a2 + b2, 1e-12)


@sp.shader
def ref_schlick_fresnel(cos_theta: sp.float32, ior: sp.float32) -> sp.float32:
    """Schlick Fresnel approximation for dielectrics."""
    r0: sp.float32 = (1.0 - ior) / (1.0 + ior)
    r0 = r0 * r0
    c: sp.float32 = 1.0 - cos_theta
    c2: sp.float32 = c * c
    return r0 + (1.0 - r0) * c2 * c2 * c
