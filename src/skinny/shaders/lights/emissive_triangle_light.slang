// Emissive triangle area light — ILight implementation.
// Uniform barycentric sampling with selection-weighted area PDF.

import common;
import interfaces;

struct EmissiveTriangleLightImpl : ILight
{
    float3 v0;
    float3 v1;
    float3 v2;
    float3 emissionColor;
    float  triArea;
    float  selectionPdf;    // 1 / numEmissiveTriangles

    LightSample samplePoint(float3 shadingPos, float2 u)
    {
        LightSample s;
        float su = sqrt(u.x);
        float b0 = 1.0 - su;
        float b1 = u.y * su;
        s.point = v0 * b0 + v1 * b1 + v2 * (1.0 - b0 - b1);

        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;
        s.normal = normalize(cross(e1, e2));

        s.radiance = emissionColor;
        s.pdfArea  = selectionPdf / max(triArea, 1e-12);
        s.valid    = true;
        return s;
    }

    float pdfSolidAngle(float3 shadingPos, float3 direction)
    {
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;
        float3 pvec = cross(direction, e2);
        float det = dot(e1, pvec);
        if (abs(det) < 1e-8) return 0.0;
        float invDet = 1.0 / det;

        float3 tvec = shadingPos - v0;
        float u_ = dot(tvec, pvec) * invDet;
        if (u_ < 0.0 || u_ > 1.0) return 0.0;

        float3 qvec = cross(tvec, e1);
        float v_ = dot(direction, qvec) * invDet;
        if (v_ < 0.0 || u_ + v_ > 1.0) return 0.0;

        float t = dot(e2, qvec) * invDet;
        if (t <= 1e-3) return 0.0;

        float3 triN = normalize(cross(e1, e2));
        float cosLight = max(dot(triN, -direction), 0.0);
        if (cosLight <= 0.0) return 0.0;

        return selectionPdf * (t * t) / (cosLight * max(triArea, 1e-12));
    }
}

EmissiveTriangleLightImpl loadEmissiveTriangleLightImpl(
    EmissiveTriangle tri, uint numTriangles)
{
    EmissiveTriangleLightImpl l;
    l.v0            = tri.v0;
    l.v1            = tri.v1;
    l.v2            = tri.v2;
    l.emissionColor = tri.emission;
    l.triArea       = tri.area;
    l.selectionPdf  = 1.0 / float(max(numTriangles, 1u));
    return l;
}
