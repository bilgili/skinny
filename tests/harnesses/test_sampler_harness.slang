// Test harness for ISampler implementations.

import common;
import interfaces;
import samplers.ggx;
import samplers.lambert;
import samplers.uniform_sphere;
import samplers.henyey_greenstein;
import samplers.mis_combine;

// ── GGX ────────────────────────────────────────────────────────

float3 test_ggx_sample(float3 N, float3 V, float roughness, float2 u)
{
    GGXSampler s;
    s.N = N;
    s.V = V;
    s.roughness = roughness;
    return s.sampleDirection(u);
}

float test_ggx_pdf(float3 N, float3 V, float roughness, float3 L)
{
    GGXSampler s;
    s.N = N;
    s.V = V;
    s.roughness = roughness;
    return s.pdf(L);
}

// ── Lambert ────────────────────────────────────────────────────

float3 test_lambert_sample(float3 N, float2 u)
{
    LambertSampler s;
    s.N = N;
    return s.sampleDirection(u);
}

float test_lambert_pdf(float3 N, float3 L)
{
    LambertSampler s;
    s.N = N;
    return s.pdf(L);
}

// ── Uniform sphere ─────────────────────────────────────────────

float3 test_uniform_sphere_sample(float2 u)
{
    UniformSphereSampler s;
    return s.sampleDirection(u);
}

float test_uniform_sphere_pdf(float3 L)
{
    UniformSphereSampler s;
    return s.pdf(L);
}

// ── Henyey-Greenstein ──────────────────────────────────────────

float3 test_hg_sample(float3 forward, float g, float2 u)
{
    HenyeyGreensteinSampler s;
    s.forward = forward;
    s.g = g;
    return s.sampleDirection(u);
}

float test_hg_pdf(float3 forward, float g, float3 L)
{
    HenyeyGreensteinSampler s;
    s.forward = forward;
    s.g = g;
    return s.pdf(L);
}

// ── MIS combine ────────────────────────────────────────────────

struct MISWeightResult
{
    float primary;
    float companion;
};

MISWeightResult test_mis_lambert_uniform(float3 N, float3 L)
{
    LambertSampler a;
    a.N = N;
    UniformSphereSampler b;
    MISWeightResult r;
    r.primary   = misPrimaryWeight(a, b, L);
    r.companion = misCompanionWeight(a, b, L);
    return r;
}

MISWeightResult test_mis_ggx_uniform(float3 N, float3 V, float roughness, float3 L)
{
    GGXSampler a;
    a.N = N;
    a.V = V;
    a.roughness = roughness;
    UniformSphereSampler b;
    MISWeightResult r;
    r.primary   = misPrimaryWeight(a, b, L);
    r.companion = misCompanionWeight(a, b, L);
    return r;
}
