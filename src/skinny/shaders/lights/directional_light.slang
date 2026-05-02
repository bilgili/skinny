// Directional (analytic) light — ILight adapter.
// Delta distribution: pdfSolidAngle returns 0 and the caller must not
// apply geometry-term conversion.  Use the `delta` convention: when the
// caller detects pdfArea == 0, treat `point` as a direction and skip
// 1/d² falloff.

import common;
import interfaces;

struct DirectionalLightImpl : ILight
{
    float3 dir;     // unit vector *toward* the light
    float3 rad;     // emitted radiance

    LightSample samplePoint(float3 shadingPos, float2 u)
    {
        LightSample s;
        s.point    = shadingPos + dir * 1e6;
        s.normal   = -dir;
        s.radiance = rad;
        s.pdfArea  = 0.0;       // delta flag — caller skips geometry term
        s.valid    = true;
        return s;
    }

    float pdfSolidAngle(float3 shadingPos, float3 direction)
    {
        return 0.0;             // delta distribution — zero probability from BSDF sampling
    }
}

DirectionalLightImpl loadDirectionalLightImpl(float3 direction, float3 radiance)
{
    DirectionalLightImpl l;
    l.dir = normalize(direction);
    l.rad = radiance;
    return l;
}
