// Test harness wrapping common.slang free functions for slangpy calls.

import common;

float test_powerHeuristic(float a, float b)
{
    return powerHeuristic(a, b);
}

float test_uniformSpherePdf()
{
    return uniformSpherePdf();
}

float test_cosineHemispherePdf(float3 dir, float3 n)
{
    return cosineHemispherePdf(dir, n);
}

float3 test_sampleCosineHemisphere(float2 u, float3 n)
{
    return sampleCosineHemisphere(u, n);
}

float3 test_sampleUniformSphere(float2 u)
{
    return sampleUniformSphere(u);
}

float test_ggxSmithG1(float NdotV, float roughness)
{
    return ggxSmithG1(NdotV, roughness);
}

struct BasisResult
{
    float3 t;
    float3 b;
};

BasisResult test_buildBasis(float3 n)
{
    BasisResult r;
    buildBasis(n, r.t, r.b);
    return r;
}

float2 test_sampleUniformDisk(float2 u)
{
    return sampleUniformDisk(u);
}

uint test_pcgHash(uint input)
{
    return pcgHash(input);
}
