// Direct-lit diffuse wavefront entry (P1 §5 shade, first lighting step).
//
// generate → intersect → Lambert shade with a fixed albedo under the scene's
// analytic directional light (binding 20) + a small environment ambient;
// environment on miss. No material lookup yet — this proves the wavefront can
// LIGHT a surface using the hit normal (a recognizable lit sphere with an N·L
// gradient), before the per-material evalGraphSurface + MIS + multi-bounce.

import common;            // RNG, createRNG, Ray, DistantLight
import bindings;          // fc, accumBuffer, distantLights (binding 20)
import scene_trace;       // traceScene
import environment;       // sceneEnvRadiance
import cameras.pinhole;   // PinholeCamera

static const float INV_PI = 0.3183098862;

[shader("compute")]
[numthreads(8, 8, 1)]
void wavefrontDiffuse(uint3 tid: SV_DispatchThreadID)
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

    float3 N = normalize(hit.normal);
    float3 L = float3(0.0, 1.0, 0.0);
    float3 lightRadiance = float3(0.0);
    if (fc.numDistantLights > 0u)
    {
        DistantLight l0 = distantLights[0];
        L = normalize(l0.direction);
        lightRadiance = l0.radiance;
    }
    float ndl = max(dot(N, L), 0.0);

    const float3 albedo = float3(0.7);
    float3 direct = albedo * INV_PI * ndl * lightRadiance;
    // Small hemispherical ambient from the environment so unlit faces are not
    // pure black (keeps the image readable when the key light points away).
    float3 ambient = albedo * 0.1 * sceneEnvRadiance(N, fc.envIntensity, 0u);

    accumBuffer[pixel] = float4(direct + ambient, 1.0);
}
