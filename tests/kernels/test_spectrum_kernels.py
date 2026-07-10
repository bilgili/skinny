"""GPU≡numpy kernel checks for the hero-wavelength spectral core (task 4.2).

Proves the ``spectrum.slang`` shader functions match the numpy CPU mirror in
``skinny/pbrt/spectral.py`` (+ ``spectral_tables.py``) to float32 precision.
Each ``test_*`` wrapper in ``tests/harnesses/test_spectrum_harness.slang`` is
dispatched via slangpy and compared to the mirror over a spread of inputs.

The three data buffers the upsample wrappers consume are passed as explicit
slang function PARAMETERS (see the harness note): slangpy binds a numpy float32
array straight to a ``StructuredBuffer<float>`` parameter. The buffers fed here
MUST be exactly what the renderer uploads so the GPU path matches this mirror:
``gScale = scale``, ``gData = data.ravel('C')`` (the flat [3,res,res,res,3] grid),
``gD65 = spectral.d65_normalized()`` (unit-luminance D65), with ``gRes = 64``,
``gD65Count = 95``.

Marked ``gpu``: skipped in the default hostless sweep, run under the guarded
Metal runner (one process at a time — the thermal rule).
"""

from __future__ import annotations

import numpy as np
import pytest

from skinny.pbrt import spectral
from skinny.pbrt.data import spectral_tables as st

pytestmark = pytest.mark.gpu

# float32-vs-float64 tolerances. lambda/pdf/upsample carry the shader's fp32
# atanh/cosh/trilinear rounding plus the fp32-vs-fp64 table upload; these bounds
# are comfortably tighter than any real divergence would produce.
REL = 1e-4
ABS = 1e-5

# pbrt visible-wavelength range. The sampling pdf is a hard step at these edges:
# where a sampled wavelength lands within BOUNDARY_EPS of an edge, fp32 (GPU) and
# fp64 (numpy) legitimately fall on opposite sides of the cutoff (pdf>0 vs pdf=0),
# so those pdf components are masked from the exact comparison — the wavelengths
# themselves still match to fp32 there. (Resolve is unaffected: the CMF is ~0 at
# both edges, so the mismatched component contributes ~1e-9 either way.)
LAMBDA_MIN, LAMBDA_MAX = 360.0, 830.0
BOUNDARY_EPS = 0.05  # nm


@pytest.fixture(scope="session")
def spectrum_harness(load_shader):
    return load_shader("test_spectrum_harness.slang")


@pytest.fixture(scope="session")
def upsample_buffers(device):
    """(res, scaleBuf, dataBuf, d65Buf, d65_count) — the three tables uploaded
    exactly as the renderer feeds them, wrapped as slangpy GPU buffers.

    The wrappers' ``StructuredBuffer<float>`` parameters take a *resource*, not a
    vectorized value: a raw numpy array is broadcast per-thread by slangpy and
    trips a shape mismatch, so bind ``NDBuffer.from_numpy(...).storage`` (a whole
    ``Buffer``) instead.
    """
    import slangpy as spy

    res, scale, data = st.load_srgb_upsample_table()
    scale_f32 = np.ascontiguousarray(scale, dtype=np.float32)
    data_f32 = np.ascontiguousarray(data.ravel("C"), dtype=np.float32)
    d65_f32 = np.ascontiguousarray(spectral.d65_normalized(), dtype=np.float32)
    scale_buf = spy.NDBuffer.from_numpy(device, scale_f32).storage
    data_buf = spy.NDBuffer.from_numpy(device, data_f32).storage
    d65_buf = spy.NDBuffer.from_numpy(device, d65_f32).storage
    return int(res), scale_buf, data_buf, d65_buf, int(d65_f32.size)


# A spread of sample coordinates in [0, 1).
_US = [0.0, 0.05, 0.17, 0.33, 0.5, 0.618, 0.75, 0.9, 0.999]

# Reflectances: primaries, gray, saturated, and mixed chroma (all in [0, 1]).
_REFLECTANCES = [
    [0.5, 0.5, 0.5],
    [0.8, 0.2, 0.1],
    [0.1, 0.7, 0.3],
    [0.2, 0.3, 0.9],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.04, 0.04, 0.04],
]

# Illuminants: unit white, gray, chromatic, and HDR emitters (values > 1).
_ILLUMINANTS = [
    [1.0, 1.0, 1.0],
    [0.5, 0.5, 0.5],
    [1.0, 0.6, 0.2],
    [0.2, 0.4, 1.0],
    [16.0, 16.0, 16.0],
    [40.0, 10.0, 5.0],
    [3.0, 6.0, 2.0],
]


def _rel_err(gpu: np.ndarray, ref: np.ndarray) -> float:
    gpu = np.asarray(gpu, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    denom = np.maximum(np.abs(ref), ABS)
    return float(np.max(np.abs(gpu - ref) / denom))


def _assert_close(gpu, ref, ctx: str, errs: list[float]) -> None:
    gpu = np.asarray([float(x) for x in gpu], dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    errs.append(_rel_err(gpu, ref))
    ok = np.allclose(gpu, ref, rtol=REL, atol=ABS)
    assert ok, f"{ctx}: gpu={gpu.tolist()} ref={ref.tolist()}"


class TestSampleWavelengths:
    def test_lambda_and_pdf(self, spectrum_harness):
        errs: list[float] = []
        for u in _US:
            sw = spectral.sample_wavelengths(u)
            lam = spectrum_harness.test_sampleWavelengths(float(u))
            pdf = np.asarray(
                [float(x) for x in spectrum_harness.test_wavelengthPdfs(float(u))]
            )
            _assert_close(lam, sw.lambda_, f"lambda(u={u})", errs)
            near_edge = (np.abs(sw.lambda_ - LAMBDA_MIN) < BOUNDARY_EPS) | (
                np.abs(sw.lambda_ - LAMBDA_MAX) < BOUNDARY_EPS
            )
            keep = ~near_edge
            _assert_close(pdf[keep], sw.pdf[keep], f"pdf(u={u})", errs)
        print(f"\nsampleWavelengths max rel err: {max(errs):.3e}")

    def test_terminate_secondary(self, spectrum_harness):
        errs: list[float] = []
        for u in _US:
            sw = spectral.terminate_secondary(spectral.sample_wavelengths(u))
            pdf = spectrum_harness.test_terminateSecondary(float(u))
            _assert_close(pdf, sw.pdf, f"terminate(u={u})", errs)
        print(f"\nterminateSecondary max rel err: {max(errs):.3e}")


class TestUpsampleReflectance:
    def test_matches_mirror(self, spectrum_harness, upsample_buffers):
        res, scale, data, _d65, _n = upsample_buffers
        errs: list[float] = []
        for rgb in _REFLECTANCES:
            for u in _US:
                sw = spectral.sample_wavelengths(u)
                ref = spectral.upsample_reflectance(rgb, sw.lambda_)
                gpu = spectrum_harness.test_upsampleReflectance(
                    rgb, float(u), res, scale, data
                )
                _assert_close(gpu, ref, f"refl(rgb={rgb}, u={u})", errs)
        print(f"\nupsampleReflectance max rel err: {max(errs):.3e}")


class TestUpsampleIlluminant:
    def test_matches_mirror(self, spectrum_harness, upsample_buffers):
        res, scale, data, d65, n = upsample_buffers
        errs: list[float] = []
        for rgb in _ILLUMINANTS:
            for u in _US:
                sw = spectral.sample_wavelengths(u)
                ref = spectral.upsample_illuminant(rgb, sw.lambda_)
                gpu = spectrum_harness.test_upsampleIlluminant(
                    rgb, float(u), res, scale, data, d65, n
                )
                _assert_close(gpu, ref, f"illum(rgb={rgb}, u={u})", errs)
        print(f"\nupsampleIlluminant max rel err: {max(errs):.3e}")


class TestPlanckBlackbody:
    """Exact-Planck blackbody emission (Group 6.1): GPU ``planckSpectrum`` ==
    numpy ``spectral.blackbody_emission`` at the 4 hero wavelengths."""

    def test_matches_mirror(self, spectrum_harness):
        errs: list[float] = []
        for temp in (3000.0, 5500.0, 6500.0):
            for u in _US:
                sw = spectral.sample_wavelengths(u)
                ref = spectral.blackbody_emission(sw, temp)
                gpu = spectrum_harness.test_planck(float(u), float(temp))
                _assert_close(gpu, ref, f"planck(u={u}, T={temp})", errs)
        print(f"\nplanckSpectrum max rel err: {max(errs):.3e}")


class TestSpectrumResolve:
    def test_full_film_resolve(self, spectrum_harness):
        """Resolve a known spectrum (an upsampled reflectance) to linear sRGB."""
        errs: list[float] = []
        for rgb in _REFLECTANCES:
            for u in _US:
                sw = spectral.sample_wavelengths(u)
                values = spectral.upsample_reflectance(rgb, sw.lambda_)
                ref = spectral.resolve_to_linear_srgb(sw.lambda_, values, sw.pdf)
                gpu = spectrum_harness.test_resolve(
                    float(u), [float(v) for v in values]
                )
                _assert_close(gpu, ref, f"resolve(rgb={rgb}, u={u})", errs)
        print(f"\nspectrumResolveToLinearSRGB max rel err: {max(errs):.3e}")
