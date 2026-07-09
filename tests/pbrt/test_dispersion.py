"""Hostless dispersion example/test for spectral rendering (Group 6.4 mirror).

Exercises the wavelength-dependent-IOR physics the GPU dispersion path will
produce, using the vendored BK7 Cauchy fit and the CPU spectral mirror. No GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from skinny.pbrt import spectral

# Load the example module by path (examples/ is not a package).
_DEMO = Path(__file__).resolve().parents[2] / "examples" / "dispersion_demo.py"
# Unique module name to avoid colliding with any other "dispersion_demo".
_MODNAME = "_skinny_examples_dispersion_demo"
_spec = importlib.util.spec_from_file_location(_MODNAME, _DEMO)
demo = importlib.util.module_from_spec(_spec)
sys.modules[_MODNAME] = demo
_spec.loader.exec_module(demo)


def test_normal_dispersion_index_falls_with_wavelength():
    # n(blue) > n(green) > n(red): the defining property of normal dispersion.
    n_blue = demo.glass_ior(450.0)
    n_green = demo.glass_ior(550.0)
    n_red = demo.glass_ior(650.0)
    assert n_blue > n_green > n_red


def test_blue_refracts_more_than_red_single_interface():
    # Air->glass at 45°: higher n (blue) ⇒ smaller refraction angle (bent more).
    theta_i = np.radians(45.0)
    t_blue = demo.snell_refraction_angle(theta_i, 450.0)
    t_red = demo.snell_refraction_angle(theta_i, 650.0)
    assert t_blue < t_red


def test_prism_deviation_larger_for_blue():
    # A prism deviates blue more than red — the visible spectral fan.
    dev = demo.prism_min_deviation(np.array([450.0, 650.0]))
    assert dev[0] > dev[1]
    # deviation is a smooth, monotonically decreasing function of wavelength
    lam = np.linspace(400.0, 700.0, 50)
    d = demo.prism_min_deviation(lam)
    assert np.all(np.diff(d) < 0.0)


def test_wavelength_colours_are_sensible():
    # 450 nm reads blue-dominant, 650 nm red-dominant.
    blue = demo.wavelength_to_srgb(450.0)
    red = demo.wavelength_to_srgb(650.0)
    assert blue[2] > blue[0] and blue[2] > blue[1]
    assert red[0] > red[1] and red[0] > red[2]


def test_secondary_termination_collapses_to_hero():
    # pbrt's dispersion strategy: on a dispersive refraction only the hero
    # wavelength survives, its pdf scaled by 1/N (keeps the estimator unbiased).
    sw = spectral.sample_wavelengths(0.4)
    assert not sw.secondary_terminated()
    term = spectral.terminate_secondary(sw)
    assert term.secondary_terminated()
    assert term.pdf[0] == pytest.approx(sw.pdf[0] / 4.0)
    assert np.all(term.pdf[1:] == 0.0)


def test_demo_renders_png(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "dispersion.png"
    demo._plot(str(out))
    assert out.exists() and out.stat().st_size > 0
