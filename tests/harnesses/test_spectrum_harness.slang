// Compile/behaviour harness wrapping spectrum.slang for slangpy calls
// (tests/kernels GPU≡numpy checks) and for the hostless slangc compile gate.

import spectrum;

StructuredBuffer<float> gScale;
StructuredBuffer<float> gData;
StructuredBuffer<float> gD65;
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

float3 test_rgbSigmoidCoeffs(float3 rgb)
{
    return rgbSigmoidCoeffs(rgb, gRes, gScale, gData);
}

float4 test_upsampleReflectance(float3 rgb, float u)
{
    return upsampleReflectance(rgb, sampleWavelengths(u), gRes, gScale, gData);
}

float4 test_upsampleIlluminant(float3 rgb, float u)
{
    return upsampleIlluminant(rgb, sampleWavelengths(u), gRes, gScale, gData, gD65, gD65Count);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID)
{
    // Touch every entry so nothing is dead-stripped before the compile gate.
    float4 a = test_sampleWavelengths(0.3) + test_wavelengthPdfs(0.3)
             + test_terminateSecondary(0.3) + test_upsampleReflectance(float3(0.5), 0.3)
             + test_upsampleIlluminant(float3(0.5), 0.3);
    float3 b = test_cieXYZBar(550.0) + test_resolve(0.3, a)
             + test_rgbSigmoidCoeffs(float3(0.5, 0.2, 0.7));
    float c = test_sigmoidPoly(b, 550.0);
    gOut[tid.x] = a.x + b.x + c;  // sink: defeats dead-code elimination
}
