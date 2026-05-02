// Shared light buffer bindings and sphere-light intersection helpers.
//
// Both the flat-material path (main_pass.slang) and the skin-material
// path (skin_material.slang) sample these buffers for area-light NEE,
// so the bindings live here rather than in either material module.

import common;

[[vk::binding(17)]] StructuredBuffer<SphereLight>      sphereLights;
[[vk::binding(18)]] StructuredBuffer<EmissiveTriangle>  emissiveTriangles;

struct SphereSample
{
    float t;
    float cos_y;
    bool  hit;
};

/// Ray / sphere intersection via quadratic discriminant.
///
///   b    = dot(o − c, d)
///   c_   = ‖o − c‖² − r²
///   disc = b² − c_
///   t    = −b − √disc   (near root; use far root if t ≤ ε)
///
/// where  o = ray origin, d = ray direction, c = sphere centre, r = radius.
/// Returns the nearest positive t and cos_y = dot(lN, −d) for the solid
/// angle PDF.
SphereSample intersectSphere(SphereLight light, Ray r)
{
    SphereSample out_;
    out_.hit   = false;
    out_.t     = 0.0;
    out_.cos_y = 0.0;
    float3 oc = r.origin - light.position;
    float b = dot(oc, r.direction);
    float c = dot(oc, oc) - light.radius * light.radius;
    float disc = b * b - c;
    if (disc < 0.0) return out_;
    float sd = sqrt(disc);
    float t = -b - sd;
    if (t <= 1e-3) t = -b + sd;
    if (t <= 1e-3) return out_;
    out_.hit = true;
    out_.t   = t;
    float3 P  = r.origin + r.direction * t;
    float3 lN = (P - light.position) / max(light.radius, 1e-4);
    out_.cos_y = max(dot(lN, -r.direction), 0.0);
    return out_;
}

struct NearestSphereHit
{
    uint  idx;
    float t;
    float cos_y;
};

NearestSphereHit intersectSphereLights(Ray r, float t_max, uint numSphereLights)
{
    NearestSphereHit best;
    best.idx   = 0xFFFFFFFFu;
    best.t     = t_max;
    best.cos_y = 0.0;
    for (uint i = 0u; i < numSphereLights; i++)
    {
        SphereSample s = intersectSphere(sphereLights[i], r);
        if (!s.hit || s.t >= best.t) continue;
        best.idx   = i;
        best.t     = s.t;
        best.cos_y = s.cos_y;
    }
    return best;
}

/// Solid angle PDF for a sphere light at shading point x in direction L.
///
///   p(L) = t² / (cos_y · A)
///
/// where  t      = hit distance along L to the sphere surface
///        cos_y  = dot(sphere normal at hit, −L)   (angle w.r.t. surface)
///        A      = 4π·r²   (sphere surface area)
/// This converts the area-measure PDF (1/A) to solid-angle measure via
/// the Jacobian |∂A/∂ω| = t²/cos_y.
float sphereLightSolidPdfAt(uint idx, float3 x, float3 L, uint numSphereLights)
{
    if (idx >= numSphereLights) return 0.0;
    SphereLight light = sphereLights[idx];
    Ray r;
    r.origin    = x;
    r.direction = L;
    SphereSample s = intersectSphere(light, r);
    if (!s.hit || s.cos_y <= 0.0) return 0.0;
    float area = 4.0 * PI * light.radius * light.radius;
    return (s.t * s.t) / (s.cos_y * area);
}
