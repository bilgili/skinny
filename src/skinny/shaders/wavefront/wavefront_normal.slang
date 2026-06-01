// Hit-normal visualization wavefront entry (P1 §5). Generates the camera ray,
// traverses the shared BVH, and writes the world-space shading normal (mapped
// to [0,1]) for hits / black for misses into the accumulation image. Verifies
// that traceScene returns correct hit *data* (not just hit/miss) in the
// wavefront — the per-hit normal the shade stage builds the BSDF frame from.

import common;            // RNG, createRNG, Ray
import bindings;          // fc, accumBuffer
import scene_trace;       // traceScene (HitInfo carries position/normal/uv)
import cameras.pinhole;   // PinholeCamera

[shader("compute")]
[numthreads(8, 8, 1)]
void wavefrontNormal(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= fc.width || pixel.y >= fc.height)
        return;

    RNG rng = createRNG(pixel, fc.frameIndex);
    PinholeCamera cam;
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);

    HitInfo hit = traceScene(fc, ray);
    float3 c = hit.hit ? (normalize(hit.normal) * 0.5 + 0.5) : float3(0.0);
    accumBuffer[pixel] = float4(c, 1.0);
}
