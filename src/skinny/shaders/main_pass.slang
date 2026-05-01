// Main render pass — dispatches camera rays through the unified material
// evaluator and composites the final display.

import common;
import bindings;
import scene_trace;
import environment;
import material_eval;

// ── Tonemapping ─────────────────────────────────────────────────

float3 acesFilmic(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float3 linearToSRGB(float3 linear)
{
    return pow(max(linear, float3(0.0)), float3(1.0 / 2.2));
}

// ── Entry point ─────────────────────────────────────────────────

[shader("compute")]
[numthreads(8, 8, 1)]
void mainImage(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= fc.width || pixel.y >= fc.height)
        return;

    RNG rng = createRNG(pixel, fc.frameIndex);
    Ray ray = generateCameraRay(fc, pixel, rng);
    HitInfo hit = traceScene(fc, ray);

    float3 sample;

    if (hit.hit)
    {
        sample = evalMaterial(hit, ray, rng);
    }
    else
    {
        sample = sceneEnvRadiance(ray.direction, fc.envIntensity, fc.furnaceMode);
    }

    // Progressive accumulation — running mean in linear HDR.
    uint n = fc.accumFrame;
    float3 accum;
    if (n == 0)
    {
        accum = sample;
    }
    else
    {
        float3 prev = accumBuffer[pixel].rgb;
        accum = (prev * float(n) + sample) / float(n + 1);
    }
    accumBuffer[pixel] = float4(accum, 1.0);

    float3 display = linearToSRGB(acesFilmic(accum));

    if (fc.furnaceMode != 0u && hit.hit)
    {
        float maxChan = max(accum.r, max(accum.g, accum.b));
        if (maxChan > 1.02)
        {
            display = float3(1.0, 0.35, 0.65);
        }
    }

    float hud = hudMask[pixel];
    if (hud > 0.75)
    {
        display = float3(1.0);
    }
    else if (hud > 0.0)
    {
        display = lerp(display, float3(0.02, 0.02, 0.03), hud);
    }

    outputBuffer[pixel] = float4(display, 1.0);
}
