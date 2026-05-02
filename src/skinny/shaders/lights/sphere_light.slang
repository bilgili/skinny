// Sphere area light — ILight implementation.
// Uniform surface sampling with solid-angle PDF for MIS.

import common;
import interfaces;

struct SphereLightImpl : ILight
{
    float3 center;
    float  radius;
    float3 rad;

    LightSample samplePoint(float3 shadingPos, float2 u)
    {
        LightSample s;
        float r = max(radius, 1e-4);
        float z   = 2.0 * u.x - 1.0;
        float phi = 2.0 * PI * u.y;
        float st  = sqrt(max(0.0, 1.0 - z * z));
        s.normal   = float3(st * cos(phi), st * sin(phi), z);
        s.point    = center + r * s.normal;
        s.radiance = rad;
        s.pdfArea  = 1.0 / (4.0 * PI * r * r);
        s.valid    = true;
        return s;
    }

    float pdfSolidAngle(float3 shadingPos, float3 direction)
    {
        float r = max(radius, 1e-4);
        float3 oc = shadingPos - center;
        float b  = dot(oc, direction);
        float c  = dot(oc, oc) - r * r;
        float disc = b * b - c;
        if (disc < 0.0) return 0.0;
        float sd = sqrt(disc);
        float t = -b - sd;
        if (t <= 1e-3) t = -b + sd;
        if (t <= 1e-3) return 0.0;
        float3 P  = shadingPos + direction * t;
        float3 lN = (P - center) / r;
        float cosY = max(dot(lN, -direction), 0.0);
        if (cosY <= 0.0) return 0.0;
        float area = 4.0 * PI * r * r;
        return (t * t) / (cosY * area);
    }
}

SphereLightImpl loadSphereLightImpl(SphereLight raw)
{
    SphereLightImpl l;
    l.center = raw.position;
    l.radius = raw.radius;
    l.rad    = raw.radiance;
    return l;
}
