// Compile/behaviour harness wrapping spectrum.slang for slangpy calls
// (tests/kernels GPU≡numpy checks) and for the hostless slangc compile gate.

import spectrum;
import common;  // Spectrum / SPECTRUM_N (spectrum.slang imports it non-exported)

// One combined ByteAddressBuffer holding scale | data | D65 end to end, matching
// the renderer's `spectralTables` after change spectral-table-fold. The Python
// GPU≡numpy test packs the three arrays contiguously and passes each region's
// byte offset.
ByteAddressBuffer gTables;
uniform uint gScaleOff;
uniform uint gDataOff;
uniform uint gD65Off;
RWStructuredBuffer<float> gOut;
uniform int gRes;
uniform int gD65Count;

float4 test_sampleWavelengths(float u)
{
    SampledWavelengths sw = sampleWavelengths(u);
    return sw.lambda;
}

float4 test_wavelengthPdfs(float u)
{
    SampledWavelengths sw = sampleWavelengths(u);
    return sw.pdf;
}

float4 test_terminateSecondary(float u)
{
    SampledWavelengths sw = terminateSecondary(sampleWavelengths(u));
    return sw.pdf;
}

float3 test_cieXYZBar(float lam)
{
    return cieXYZBar(lam);
}

float test_sigmoidPoly(float3 coeffs, float lam)
{
    return sigmoidPoly(coeffs, lam);
}

float3 test_resolve(float u, float4 values)
{
    return spectrumResolveToLinearSRGB(sampleWavelengths(u), values);
}

// Exact-Planck blackbody emission at the hero wavelengths (change
// spectral-rendering, Group 6.1). Returns the 4 hero-λ SPD as float4 so the
// Python GPU≡numpy test can compare against spectral.blackbody_emission. The
// per-wavelength copy widens the `Spectrum` return (float4 spectral / float3
// RGB) into a fixed float4 so the wrapper links under BOTH compile variants —
// the RGB compile-gate leaves .w = 0.
float4 test_planck(float u, float temperature)
{
    Spectrum s = planckSpectrum(sampleWavelengths(u), temperature);
    float4 r = float4(0.0);
    [ForceUnroll]
    for (int i = 0; i < SPECTRUM_N; ++i)
        r[i] = s[i];
    return r;
}

// Carrier width of this compile (4 spectral / 3 RGB) so the Python test knows
// how many test_planck lanes are meaningful.
float test_spectrumWidth()
{
    return float(SPECTRUM_N);
}

// The upsample wrappers take the combined `ByteAddressBuffer` + each region's
// byte offset as explicit PARAMETERS (matching the refactored spectrum.slang
// signatures, change spectral-table-fold): the Python GPU≡numpy test
// (tests/kernels/test_spectrum_kernels.py) packs scale|data|d65 into one buffer
// and feeds `tables`, `scaleOff`, `dataOff`, `d65Off` per call. The module-scope
// globals below stay so `computeMain` (the slangc compile gate) still links.
float3 test_rgbSigmoidCoeffs(float3 rgb, int res,
                             ByteAddressBuffer tables, uint scaleOff, uint dataOff)
{
    return rgbSigmoidCoeffs(rgb, res, tables, scaleOff, dataOff);
}

float4 test_upsampleReflectance(float3 rgb, float u, int res,
                                ByteAddressBuffer tables, uint scaleOff, uint dataOff)
{
    return upsampleReflectance(rgb, sampleWavelengths(u), res, tables, scaleOff, dataOff);
}

float4 test_upsampleIlluminant(float3 rgb, float u, int res,
                               ByteAddressBuffer tables, uint scaleOff, uint dataOff,
                               uint d65Off, int d65Count)
{
    return upsampleIlluminant(rgb, sampleWavelengths(u), res, tables, scaleOff,
                              dataOff, d65Off, d65Count);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID)
{
    // Touch every entry so nothing is dead-stripped before the compile gate.
    float4 a = test_sampleWavelengths(0.3) + test_wavelengthPdfs(0.3)
             + test_terminateSecondary(0.3)
             + test_upsampleReflectance(float3(0.5), 0.3, gRes, gTables, gScaleOff, gDataOff)
             + test_upsampleIlluminant(float3(0.5), 0.3, gRes, gTables, gScaleOff, gDataOff, gD65Off, gD65Count)
             + test_planck(0.3, 5000.0);
    float3 b = test_cieXYZBar(550.0) + test_resolve(0.3, a)
             + test_rgbSigmoidCoeffs(float3(0.5, 0.2, 0.7), gRes, gTables, gScaleOff, gDataOff);
    float c = test_sigmoidPoly(b, 550.0);
    gOut[tid.x] = a.x + b.x + c;  // sink: defeats dead-code elimination
}
