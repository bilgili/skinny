"""pbrt `subsurface` material → volumetric medium coefficients (σ_a, σ_s, g, eta).

Reproduces pbrt-v4's `SubsurfaceMaterial::Create` precedence so an imported
subsurface object matches the pbrt reference. Sources (pbrt-v4):
  - `media.cpp` `SubsurfaceParameterTable` (`GetMediumScatteringProperties`) —
    the named measured-media presets. Column order is **σ_s′ (reduced) then σ_a**,
    units mm⁻¹, RGB/sRGB. Named presets force g = 0 (the table stores reduced
    coefficients), so σ_s′ is used directly as σ_s.
  - `materials.cpp` `SubsurfaceMaterial::Create` (lines ~498-562) — the 4-way
    precedence + `scale` / `eta` / `g` defaults.
  - `bssrdf.cpp` `SubsurfaceFromDiffuse` — the `reflectance` + `mfp` inversion.
    pbrt uses a photon-beam-diffusion table; here we use the classical Jensen 2001
    diffuse-reflectance inversion (approximate — a PBD port is a follow-up). The
    primary parity target (sssdragon) uses a named preset, not this path.

All coefficients returned in mm⁻¹ (× `scale`); the renderer converts mm⁻¹ → scene
units via the scene's mm-per-unit at pack time, exactly as the skin path does.
"""

from __future__ import annotations

import math

# pbrt-v4 SubsurfaceParameterTable (media.cpp). Keys lowercased for lookup.
# value = (sigma_prime_s [reduced], sigma_a), RGB, mm^-1. pbrt uses sigma_prime_s
# directly as sigma_s with g = 0 for named presets.
_NAMED_MEDIA: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "skin1":      ((0.74, 0.88, 1.01),  (0.032, 0.17, 0.48)),
    "skin2":      ((1.09, 1.59, 1.79),  (0.013, 0.07, 0.145)),
    "marble":     ((2.19, 2.62, 3.00),  (0.0021, 0.0041, 0.0071)),
    "cream":      ((7.38, 5.47, 3.15),  (0.0002, 0.0028, 0.0163)),
    "ketchup":    ((0.18, 0.07, 0.03),  (0.061, 0.97, 1.45)),
    "wholemilk":  ((2.55, 3.21, 3.77),  (0.0011, 0.0024, 0.014)),
    "apple":      ((2.29, 2.39, 1.97),  (0.003, 0.0034, 0.046)),
    "chicken1":   ((0.15, 0.21, 0.38),  (0.015, 0.077, 0.19)),
    "skimmilk":   ((0.70, 1.22, 1.90),  (0.0014, 0.0025, 0.0142)),
    "spectralon": ((11.6, 20.4, 14.9),  (0.0, 0.0, 0.0)),
    "potato":     ((0.68, 0.70, 0.55),  (0.0024, 0.009, 0.12)),
}

# pbrt SubsurfaceMaterial defaults (materials.cpp): used when nothing is specified
# (= Wholemilk), plus eta / scale / g.
_DEFAULT_SIGMA_S = (2.55, 3.21, 3.77)
_DEFAULT_SIGMA_A = (0.0011, 0.0024, 0.014)
ETA_DEFAULT = 1.33
SCALE_DEFAULT = 1.0
G_DEFAULT = 0.0
MFP_DEFAULT = 1.0


def _f3(v, default):
    if v is None:
        return tuple(float(c) for c in default)
    try:
        return float(v[0]), float(v[1]), float(v[2])
    except (TypeError, IndexError, ValueError):
        try:
            f = float(v)
            return (f, f, f)
        except (TypeError, ValueError):
            return tuple(float(c) for c in default)


def _fresnel_diffuse_reflectance(eta: float) -> float:
    """Egan/Hilgeman approximation of the diffuse Fresnel reflectance F_dr(eta),
    as used by the Jensen 2001 dipole inversion."""
    return -1.440 / (eta * eta) + 0.710 / eta + 0.668 + 0.0636 * eta


def _diffuse_reflectance_from_albedo(alpha_prime: float, A: float) -> float:
    """Jensen 2001 dipole diffuse reflectance R_d for a reduced single-scatter
    albedo α′ (monotonic increasing in α′ ∈ [0,1])."""
    s = math.sqrt(3.0 * (1.0 - alpha_prime))
    return 0.5 * alpha_prime * (1.0 + math.exp(-(4.0 / 3.0) * A * s)) * math.exp(-s)


def _invert_albedo(rd: float, eta: float) -> float:
    """Invert R_d → reduced single-scatter albedo α′ via bisection (Jensen 2001).
    Classical approximation to pbrt's photon-beam-diffusion inversion."""
    rd = min(max(rd, 0.0), 1.0)
    if rd <= 0.0:
        return 0.0
    A = (1.0 + _fresnel_diffuse_reflectance(eta)) / (1.0 - _fresnel_diffuse_reflectance(eta))
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _diffuse_reflectance_from_albedo(mid, A) < rd:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def subsurface_coefficients(
    *,
    name: str | None = None,
    sigma_a=None,
    sigma_s=None,
    reflectance=None,
    mfp=None,
    g: float = G_DEFAULT,
    eta: float = ETA_DEFAULT,
    scale: float = SCALE_DEFAULT,
) -> dict:
    """pbrt `subsurface` → `{sigma_a:[3], sigma_s:[3], g, eta}` (mm⁻¹, × scale).

    Precedence (pbrt `SubsurfaceMaterial::Create`, fixed order):
      1. ``name`` non-empty → named preset (forces g = 0).
      2. else ``sigma_a`` AND ``sigma_s`` both given → used directly (× scale).
      3. else ``reflectance`` (+ ``mfp``, default 1) → classical R_d inversion
         (× scale on mfp).
      4. else → Wholemilk-like defaults.
    """
    eta = float(eta)
    scale = float(scale)
    g = float(g)

    # (1) named preset — wins over everything; g forced to 0.
    if name:
        key = str(name).strip().lower()
        if key in _NAMED_MEDIA:
            ssp, sa = _NAMED_MEDIA[key]
            return {
                "sigma_a": [c * scale for c in sa],
                "sigma_s": [c * scale for c in ssp],
                "g": 0.0,
                "eta": eta,
            }
        # Unknown name: fall through to defaults (pbrt would error; we degrade).

    # (2) explicit sigma_a + sigma_s.
    if sigma_a is not None and sigma_s is not None:
        sa = _f3(sigma_a, _DEFAULT_SIGMA_A)
        ss = _f3(sigma_s, _DEFAULT_SIGMA_S)
        return {
            "sigma_a": [c * scale for c in sa],
            "sigma_s": [c * scale for c in ss],
            "g": g,
            "eta": eta,
        }

    # (3) reflectance (+ mfp) → classical Jensen inversion. scale multiplies mfp.
    if reflectance is not None:
        rd = _f3(reflectance, (0.5, 0.5, 0.5))
        m = _f3(mfp, (MFP_DEFAULT,) * 3)
        out_a, out_s = [], []
        for c in range(3):
            mc = max(m[c] * scale, 1e-6)
            alpha_p = _invert_albedo(rd[c], eta)          # reduced single-scatter albedo
            sigma_t_prime = 1.0 / mc                       # reduced extinction = 1/mfp
            sigma_s_prime = alpha_p * sigma_t_prime
            sa_c = (1.0 - alpha_p) * sigma_t_prime
            # de-reduce scattering: σ_s = σ_s′ / (1 - g) (g = 0 ⇒ unchanged).
            ss_c = sigma_s_prime / max(1.0 - g, 1e-6)
            out_a.append(sa_c)
            out_s.append(ss_c)
        return {"sigma_a": out_a, "sigma_s": out_s, "g": g, "eta": eta}

    # (4) defaults (Wholemilk).
    return {
        "sigma_a": [c * scale for c in _DEFAULT_SIGMA_A],
        "sigma_s": [c * scale for c in _DEFAULT_SIGMA_S],
        "g": g,
        "eta": eta,
    }
