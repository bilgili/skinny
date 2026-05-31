// Primary-visibility wavefront entry — the intersect step of the staged
// pipeline, verified in isolation (P1 §5). Generates the camera ray, traverses
// the shared BVH via traceScene, and writes a binary hit mask into the
// accumulation image (1 = geometry hit, 0 = miss). No shading — this proves the
// wavefront can read the geometry/BVH/instance bindings and reproduce the
// megakernel's hit/miss boundary, checked against the demo scene's known sphere
// screen positions (test_wavefront_render.py).

import common;            // RNG, createRNG, Ray
import bindings;          // fc, accumBuffer
import scene_trace;       // traceScene (pulls mesh_head: vertices/indices/BVH/instances)
import cameras.pinhole;   // PinholeCamera

[shader("compute")]
[numthreads(8, 8, 1)]
void wavefrontVisibility(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= fc.width || pixel.y >= fc.height)
        return;

    RNG rng = createRNG(pixel, fc.frameIndex);
    PinholeCamera cam;
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);

    HitInfo hit = traceScene(fc, ray);
    float v = hit.hit ? 1.0 : 0.0;
    accumBuffer[pixel] = float4(v, v, v, 1.0);
}
