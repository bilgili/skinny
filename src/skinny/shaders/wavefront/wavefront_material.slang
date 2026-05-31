// Per-material albedo wavefront entry (P1 §5 shade — material evaluation).
//
// generate → intersect → evaluate the hit material's base colour via the same
// MaterialX graph codegen the megakernel uses (evalSceneGraphBaseColor over the
// per-scene generated_materials), and output it; environment on miss. Proves
// the wavefront drives per-material evaluation: the marble / wood / brass
// spheres come out in distinct colours rather than a single fill. Graphs that
// drive only non-base inputs (brass) leave the default, so it reads as a third
// distinct region.

import common;            // RNG, createRNG, Ray
import bindings;          // fc, accumBuffer, materialGraphId, materialTypes
import scene_trace;       // traceScene
import environment;       // sceneEnvRadiance
import cameras.pinhole;   // PinholeCamera
import generated_materials;  // evalSceneGraphBaseColor (per-scene graphs)

[shader("compute")]
[numthreads(8, 8, 1)]
void wavefrontMaterial(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= fc.width || pixel.y >= fc.height)
        return;

    RNG rng = createRNG(pixel, fc.frameIndex);
    PinholeCamera cam;
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);

    HitInfo hit = traceScene(fc, ray);
    if (!hit.hit)
    {
        accumBuffer[pixel] = float4(
            sceneEnvRadiance(ray.direction, fc.envIntensity, fc.furnaceMode), 1.0);
        return;
    }

    uint matId = hit.materialId;
    uint graphId = materialGraphId(matId);
    float3 albedo = float3(0.5);
    float3 graphColor;
    if (evalSceneGraphBaseColor(graphId, matId, hit.positionObject,
                                normalize(hit.normal), hit.tangent, hit.uv, graphColor))
        albedo = graphColor;

    accumBuffer[pixel] = float4(albedo, 1.0);
}
