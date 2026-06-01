// Env-only wavefront entry — first renderer-integration milestone (P1 §5/§6).
//
// Generates the camera ray and shades it with the environment only (no BVH, no
// material shading), reproducing main_pass.slang's miss path + progressive
// accumulation EXACTLY, so a wavefront-vs-megakernel A/B over a geometry-free
// scene is comparable. This proves the integration plumbing — camera + env +
// accumulation-image write + execution-mode dispatch gating — before the staged
// generate→intersect→logic→shade pipeline (which supersedes this kernel) lands.
//
// Reuses the shared scene bindings: fc (binding 0), accumBuffer (binding 2),
// envMap (binding 4 via environment.slang). The wavefront pipeline's descriptor
// set (built in WavefrontPasses) binds these from the renderer's existing
// resources.

import common;        // RNG, createRNG, Ray
import bindings;      // fc, accumBuffer
import environment;   // envMap, sceneEnvRadiance
import cameras.pinhole;  // PinholeCamera

[shader("compute")]
[numthreads(8, 8, 1)]
void wavefrontEnv(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= fc.width || pixel.y >= fc.height)
        return;

    // Match main_pass.slang's per-pixel seeding so the jittered camera ray is
    // identical to the megakernel's for the same frame.
    RNG rng = createRNG(pixel, fc.frameIndex);
    PinholeCamera cam;
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);

    // Miss path: environment radiance (honours furnace override + IBL scale).
    float3 sample = sceneEnvRadiance(ray.direction, fc.envIntensity, fc.furnaceMode);
    sample *= lensWeight;

    bool3 bad = isnan(sample) | isinf(sample);
    sample = select(bad, float3(0.0), sample);
    sample = max(sample, float3(0.0));

    // Progressive accumulation — running mean in linear HDR (identical to
    // main_pass.slang:157-169).
    uint n = fc.accumFrame;
    float3 accum;
    if (n == 0u)
        accum = sample;
    else
    {
        float3 prev = accumBuffer[pixel].rgb;
        accum = (prev * float(n) + sample) / float(n + 1u);
    }
    accumBuffer[pixel] = float4(accum, 1.0);
}
