// Pinhole camera model — ideal perspective projection.
//
// Reads camera transforms from uniform `fc` (bindings.slang). The
// struct carries no state — it exists only as an `ICamera` conformance
// tag so estimators can be monomorphized on the camera model at
// compile time.

import common;
import bindings;
import interfaces;

struct PinholeCamera : ICamera
{
    Ray generateRay(uint2 pixel, inout RNG rng, out float weight)
    {
        weight = 1.0;

        float2 jitter = rng.next2() - 0.5;
        float2 ndc    = zoomedNDC(fc, pixel, jitter);

        float4 target    = mul(fc.camera.projInverse, float4(ndc, 1.0, 1.0));
        float3 direction = mul(fc.camera.viewInverse,
                               float4(normalize(target.xyz / target.w), 0.0)).xyz;

        Ray ray;
        ray.origin    = fc.camera.position;
        ray.direction = normalize(direction);
        return ray;
    }

    bool sampleWi(float3 p, inout RNG rng,
                  out int2 outPixel, out float cosCam, out float distToCam)
    {
        outPixel = int2(0);
        cosCam = 0.0;
        distToCam = 0.0;
        float3 toP = p - fc.camera.position;
        distToCam = length(toP);
        if (distToCam <= 1e-4) return false;
        float4 viewSp = mul(fc.camera.view, float4(p, 1.0));
        float4 clip = mul(fc.camera.proj, viewSp);
        if (clip.w <= 1e-4) return false;
        float3 ndc = clip.xyz / clip.w;
        if (any(abs(ndc.xy) > float2(1.0))) return false;
        float xs = (ndc.x * 0.5 + 0.5) * float(fc.width);
        float ys = (-ndc.y * 0.5 + 0.5) * float(fc.height);
        if (xs < 0.0 || ys < 0.0 || xs >= float(fc.width) || ys >= float(fc.height))
            return false;
        outPixel = int2(int(xs), int(ys));

        float3 camForward = normalize(
            mul(fc.camera.viewInverse, float4(0, 0, -1, 0)).xyz);
        cosCam = max(dot(camForward, toP / distToCam), 0.0);
        return cosCam > 1e-4;
    }
}
