// Test harness for ILight implementations.

import common;
import interfaces;
import lights.sphere_light;
import lights.directional_light;
import lights.emissive_triangle_light;

// ── Sphere light ───────────────────────────────────────────────

struct SphereLightSampleResult
{
    float3 point;
    float3 normal;
    float3 radiance;
    float  pdfArea;
    bool   valid;
};

SphereLightSampleResult test_sphere_light_sample(
    float3 center, float radius, float3 rad, float3 shadingPos, float2 u)
{
    SphereLightImpl l;
    l.center = center;
    l.radius = radius;
    l.rad    = rad;
    LightSample s = l.samplePoint(shadingPos, u);
    SphereLightSampleResult r;
    r.point    = s.point;
    r.normal   = s.normal;
    r.radiance = s.radiance;
    r.pdfArea  = s.pdfArea;
    r.valid    = s.valid;
    return r;
}

float test_sphere_light_pdf(
    float3 center, float radius, float3 rad, float3 shadingPos, float3 dir)
{
    SphereLightImpl l;
    l.center = center;
    l.radius = radius;
    l.rad    = rad;
    return l.pdfSolidAngle(shadingPos, dir);
}

// ── Directional light ──────────────────────────────────────────

SphereLightSampleResult test_directional_light_sample(
    float3 dir, float3 rad, float3 shadingPos, float2 u)
{
    DirectionalLightImpl l;
    l.dir = normalize(dir);
    l.rad = rad;
    LightSample s = l.samplePoint(shadingPos, u);
    SphereLightSampleResult r;
    r.point    = s.point;
    r.normal   = s.normal;
    r.radiance = s.radiance;
    r.pdfArea  = s.pdfArea;
    r.valid    = s.valid;
    return r;
}

float test_directional_light_pdf(float3 dir, float3 rad, float3 shadingPos, float3 queryDir)
{
    DirectionalLightImpl l;
    l.dir = normalize(dir);
    l.rad = rad;
    return l.pdfSolidAngle(shadingPos, queryDir);
}

// ── Emissive triangle light ────────────────────────────────────

SphereLightSampleResult test_emissive_tri_sample(
    float3 v0, float3 v1, float3 v2, float3 emission, float area,
    float selectionPdf, float3 shadingPos, float2 u)
{
    EmissiveTriangleLightImpl l;
    l.v0            = v0;
    l.v1            = v1;
    l.v2            = v2;
    l.emissionColor = emission;
    l.triArea       = area;
    l.selectionPdf  = selectionPdf;
    LightSample s = l.samplePoint(shadingPos, u);
    SphereLightSampleResult r;
    r.point    = s.point;
    r.normal   = s.normal;
    r.radiance = s.radiance;
    r.pdfArea  = s.pdfArea;
    r.valid    = s.valid;
    return r;
}

float test_emissive_tri_pdf(
    float3 v0, float3 v1, float3 v2, float area,
    float selectionPdf, float3 shadingPos, float3 dir)
{
    EmissiveTriangleLightImpl l;
    l.v0            = v0;
    l.v1            = v1;
    l.v2            = v2;
    l.emissionColor = float3(1.0);
    l.triArea       = area;
    l.selectionPdf  = selectionPdf;
    return l.pdfSolidAngle(shadingPos, dir);
}
