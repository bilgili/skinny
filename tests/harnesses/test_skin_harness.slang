// Test harness for skin optics functions from skin_bssrdf.slang.

import common;
import skin_bssrdf;

float3 test_melaninAbsorption(float fraction)
{
    return melaninAbsorption(fraction);
}

float3 test_hemoglobinAbsorption(float fraction, float oxygenation)
{
    return hemoglobinAbsorption(fraction, oxygenation);
}

struct SkinLayerResult
{
    float3 sigmaA;
    float3 sigmaS;
    float  thickness;
    float  g;
    float  ior;
};

SkinLayerResult test_buildEpidermis(
    float melaninFraction, float epidermisThickness,
    float3 scatteringCoeff, float anisotropy, float ior)
{
    SkinParams p;
    p.melaninFraction    = melaninFraction;
    p.epidermisThickness = epidermisThickness;
    p.scatteringCoeff    = scatteringCoeff;
    p.anisotropy         = anisotropy;
    p.ior                = ior;
    SkinLayer layer = buildEpidermis(p);
    SkinLayerResult r;
    r.sigmaA    = layer.sigmaA;
    r.sigmaS    = layer.sigmaS;
    r.thickness = layer.thickness;
    r.g         = layer.g;
    r.ior       = layer.ior;
    return r;
}

SkinLayerResult test_buildDermis(
    float hemoglobinFraction, float bloodOxygenation, float dermisThickness,
    float3 scatteringCoeff, float anisotropy, float ior)
{
    SkinParams p;
    p.hemoglobinFraction = hemoglobinFraction;
    p.bloodOxygenation   = bloodOxygenation;
    p.dermisThickness    = dermisThickness;
    p.scatteringCoeff    = scatteringCoeff;
    p.anisotropy         = anisotropy;
    p.ior                = ior;
    SkinLayer layer = buildDermis(p);
    SkinLayerResult r;
    r.sigmaA    = layer.sigmaA;
    r.sigmaS    = layer.sigmaS;
    r.thickness = layer.thickness;
    r.g         = layer.g;
    r.ior       = layer.ior;
    return r;
}

float3 test_inkSigmaAbsorption(float3 rgb, float density)
{
    return inkSigmaAbsorption(rgb, density);
}

float3 test_inkSigmaScattering(float3 rgb, float density)
{
    return inkSigmaScattering(rgb, density);
}

float test_fresnelSchlick(float cosTheta, float ior)
{
    return fresnelSchlick(cosTheta, ior);
}

float test_ggxDistribution(float NdotH, float roughness)
{
    return ggxDistribution(NdotH, roughness);
}

float3 test_evaluateSkinSpecular(float3 N, float3 V, float3 L, float roughness, float ior)
{
    return evaluateSkinSpecular(N, V, L, roughness, ior);
}

float3 test_burleyDiffusionProfile(float r, float3 d)
{
    return burleyDiffusionProfile(r, d);
}
