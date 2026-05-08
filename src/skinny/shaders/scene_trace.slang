// Scene traversal utilities shared across material evaluators.
//
// Every function takes FrameConstants as its first parameter so modules
// that import scene_trace don't need to declare their own copy of the
// uniform — the caller (main_pass.slang) passes its module-scope fc.

import common;
import bindings;
import sdf_head;
import mesh_head;
import flat_shading;

HitInfo traceFurnaceSphere(Ray ray)
{
    HitInfo h;
    h.hit            = false;
    h.t              = 0.0;
    h.backT          = -1.0;
    h.position       = float3(0.0);
    h.positionObject = float3(0.0);
    h.normal         = float3(0.0, 1.0, 0.0);
    h.uv             = float2(0.0);
    h.tangent        = float3(1.0, 0.0, 0.0);
    h.hasTangent     = false;
    h.materialId     = 0u;

    float3 oc   = ray.origin;
    float  b    = dot(oc, ray.direction);
    float  c    = dot(oc, oc) - 1.0;
    float  disc = b * b - c;
    if (disc < 0.0) return h;

    float sd = sqrt(disc);
    float t  = -b - sd;
    if (t < 0.0) t = -b + sd;
    if (t < 0.0) return h;

    h.hit      = true;
    h.t        = t;
    h.position = ray.origin + ray.direction * t;
    h.normal   = normalize(h.position);
    return h;
}

HitInfo traceScene(FrameConstants fc, Ray r)
{
    if (fc.furnaceMode != 0u)
        return traceFurnaceSphere(r);
    return (fc.useMesh != 0u)
        ? marchHeadMesh(r, fc.numInstances)
        : marchHead(r);
}

bool isMaterialTransparent(uint materialId)
{
    uint matType = materialTypes[materialId] & 0xFFu;
    if (matType != MATERIAL_TYPE_FLAT) return false;
    FlatMaterialParams p = flatMaterials[materialId];
    if (p.opacity < 1.0) return true;
    if (p.opacityTextureIdx != 0xFFFFFFFFu) return true;
    return false;
}

bool isShadowTransparent(HitInfo h)
{
    return isMaterialTransparent(h.materialId) || isCutoutTransparent(h);
}

bool visibleSegment(FrameConstants fc, float3 a, float3 b)
{
    float3 d = b - a;
    float dist = length(d);
    if (dist <= 1e-4)
        return false;

    float3 dir = d / dist;
    Ray shadow;
    shadow.origin = a + dir * 0.002;
    shadow.direction = dir;
    float remaining = dist - 0.004;

    for (int i = 0; i < 8; i++)
    {
        HitInfo h = traceScene(fc, shadow);
        if (!h.hit || h.t > remaining)
            return true;
        if (!isShadowTransparent(h))
            return false;
        shadow.origin = h.position + dir * 0.004;
        remaining -= (h.t + 0.004);
    }
    return false;
}

bool visibleDirectional(FrameConstants fc, float3 origin, float3 N, float3 L)
{
    Ray shadow;
    shadow.origin    = origin + N * 0.002;
    shadow.direction = L;

    for (int i = 0; i < 8; i++)
    {
        HitInfo h = traceScene(fc, shadow);
        if (!h.hit)
            return true;
        if (!isShadowTransparent(h))
            return false;
        shadow.origin = h.position + L * 0.004;
    }
    return false;
}
