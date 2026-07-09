"""Simple dispersion example for skinny's spectral rendering (Group 6.4).

Hostless demonstration of the physics the megakernel spectral path will produce:
a wavelength-dependent dielectric IOR (the vendored BK7 Cauchy fit) refracts
short wavelengths more than long ones, so white light through a prism fans into
a spectrum. Uses only the merged spectral code — the same `named_glass_ior` the
GPU dispersion path evaluates at the hero wavelength, and skinny's CIE fit for
the per-wavelength display colour.

Run (matplotlib lives in the repo-root Python 3.13 env)::

    PYTHONPATH=src ./bin/python3.13 examples/dispersion_demo.py

Writes ``examples/dispersion_demo.png``.
"""

from __future__ import annotations

import numpy as np

from skinny.pbrt import spectra
from skinny.pbrt.data import spectral_tables as st

GLASS = "glass-BK7"
PRISM_APEX_DEG = 60.0  # equilateral prism


def glass_ior(lam_nm):
    """Wavelength-dependent refractive index of the prism glass (Cauchy fit)."""
    return st.named_glass_ior(GLASS, lam_nm)


def snell_refraction_angle(theta_i_rad: float, lam_nm) -> np.ndarray:
    """Air->glass refraction angle at wavelength(s) *lam_nm* (Snell's law)."""
    n = glass_ior(lam_nm)
    return np.arcsin(np.sin(theta_i_rad) / n)


def prism_min_deviation(lam_nm, apex_deg: float = PRISM_APEX_DEG) -> np.ndarray:
    """Minimum-deviation angle (deg) of a prism at wavelength(s) *lam_nm*.

    δ_min(λ) = 2·asin(n(λ)·sin(A/2)) − A. Rises with n, so blue (higher n)
    deviates more than red — the spread that separates white light.
    """
    a = np.radians(apex_deg)
    n = np.asarray(glass_ior(lam_nm), dtype=np.float64)
    return np.degrees(2.0 * np.arcsin(n * np.sin(a / 2.0)) - a)


def wavelength_to_srgb(lam_nm) -> np.ndarray:
    """Representative display sRGB of a single wavelength (CIE fit → sRGB → gamma)."""
    xyz = np.array(spectra.cie_xyz_bar(float(lam_nm)))
    rgb = np.clip(spectra.xyz_to_linear_srgb(xyz), 0.0, None)
    peak = rgb.max()
    if peak > 0:
        rgb = rgb / peak
    return np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.2)


def _plot(path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    lam = np.linspace(380.0, 700.0, 240)
    n = glass_ior(lam)
    dev = prism_min_deviation(lam)
    colors = np.array([wavelength_to_srgb(x) for x in lam])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle(f"Spectral dispersion through a {GLASS} prism "
                 f"(apex {PRISM_APEX_DEG:.0f}°) — skinny Group 6.4", fontsize=12)

    # Left: n(λ) and deviation δ(λ), each point coloured by its wavelength.
    axL.scatter(lam, n, c=colors, s=10)
    axL.set_xlabel("wavelength λ (nm)")
    axL.set_ylabel("refractive index n(λ)", color="0.2")
    axL.set_title("normal dispersion: n and deviation fall with λ")
    ax2 = axL.twinx()
    ax2.scatter(lam, dev, c=colors, s=10, marker="s", alpha=0.6)
    ax2.set_ylabel("prism min-deviation δ(λ) (deg)", color="0.4")
    axL.annotate(f"blue 486nm: n={glass_ior(486.0):.4f}", (486, glass_ior(486.0)),
                 textcoords="offset points", xytext=(6, 8), fontsize=8)
    axL.annotate(f"red 656nm: n={glass_ior(656.0):.4f}", (656, glass_ior(656.0)),
                 textcoords="offset points", xytext=(6, 8), fontsize=8)

    # Right: a white ray entering a prism, fanning into a spectrum by δ(λ).
    axR.set_title("white light fans into a spectrum")
    axR.set_aspect("equal")
    axR.axis("off")
    # prism triangle
    tri = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.8], [0.0, 0.0]])
    axR.plot(tri[:, 0], tri[:, 1], color="0.5", lw=1.5)
    axR.fill(tri[:, 0], tri[:, 1], color="0.9", alpha=0.4)
    # incoming white ray
    entry = np.array([1.0, 0.55])
    axR.plot([-1.3, entry[0]], [0.95, entry[1]], color="0.15", lw=2.0)
    axR.text(-1.35, 1.0, "white", fontsize=9, ha="right", va="center")
    # outgoing fanned rays, one per sampled wavelength, deviated by (δ − δ_mean)
    base = np.radians(-35.0)
    spread = np.radians((dev - dev.mean()) * 6.0)  # exaggerate for visibility
    seglen = 2.6
    segs, segc = [], []
    for ang, col in zip(base - spread, colors):
        end = entry + seglen * np.array([np.cos(ang), np.sin(ang)])
        segs.append([entry, end])
        segc.append(col)
    axR.add_collection(LineCollection(segs, colors=segc, linewidths=1.6))
    # Blue (higher n) deviates most → steepest ray (lowest); red deviates least.
    axR.text(entry[0] + 2.5, entry[1] - 1.05, "red\n(least bent)", fontsize=8, color="0.3")
    axR.text(entry[0] + 2.2, entry[1] - 2.0, "blue\n(most bent)", fontsize=8, color="0.3")
    axR.set_xlim(-1.6, 4.2)
    axR.set_ylim(-1.8, 2.1)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(__file__), "dispersion_demo.png")
    _plot(out)
