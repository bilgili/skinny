// Main render pass — dispatches camera rays through the unified material
// evaluator and composites the final display.
//
// References:
//   [1] Veach, Guibas, "Bidirectional Estimators for Light Transport",
//       EGSR 1995.
//   [2] Lafortune, Willems, "Bi-Directional Path Tracing",
//       Compugraphics 1993.
//   [3] Veach, "Robust Monte Carlo Methods for Light Transport Simulation",
//       PhD thesis, Stanford 1997.  (MIS, §9; BDPT, §10)
//   [4] Narkowicz, "ACES Filmic Tone Mapping Curve" (approximation fit),
//       blog post 2016; original ACES matrices by the Academy S-2014-003.

import common;
import bindings;
import scene_trace;
import environment;
import material_eval;

// ── Tonemapping ─────────────────────────────────────────────────

/// ACES filmic tone-mapping curve (Narkowicz rational approximation) [4].
///
///   f(x) = (x · (a·x + b)) / (x · (c·x + d) + e)
///
/// where  a = 2.51,  b = 0.03,  c = 2.43,  d = 0.59,  e = 0.14
/// Maps linear HDR radiance → display [0, 1]. Applied per channel;
/// result is clamped to [0, 1] before sRGB gamma encoding.
float3 acesFilmic(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

/// Simplified sRGB gamma encoding (power-law approximation, γ = 2.2).
///
///   f(c) = max(c, 0)^(1/2.2)
///
/// The IEC 61966-2-1 standard uses a piecewise form with a linear toe;
/// the power-law approximation is accurate to < 1% for c > 0.003 and
/// avoids a branch in the shader.
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
        // IIntegrator dispatch deferred to Phase 4 — until path / MIS /
        // light tracers diverge in body, monomorphizing all three at the
        // call site triples SPV size with no behavioral gain. The
        // integrator structs (integrators/{path,mis,light}.slang) define
        // the seam; today the slider still steers behavior via the
        // fc.integratorMode reads inside skin estimators.
        sample = evalMaterial(hit, ray, rng);
    }
    else
    {
        sample = sceneEnvRadiance(ray.direction, fc.envIntensity, fc.furnaceMode);
    }

    // Reject NaN / negative / inf samples — a single corrupted value
    // permanently poisons the running mean for that pixel.
    if (any(isnan(sample)) || any(isinf(sample)) || any(sample < float3(0.0)))
        sample = float3(0.0);

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
