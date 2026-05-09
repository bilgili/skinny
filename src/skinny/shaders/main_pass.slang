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
import integrators.path;
import integrators.bdpt;

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

    bool useBdpt = (fc.integratorType == INTEGRATOR_BDPT)
                && hit.hit
                && ((materialTypes[hit.materialId] & 0xFFu) == MATERIAL_TYPE_FLAT);

    if (useBdpt)
    {
        BDPTIntegrator bdpt;
        sample = bdpt.estimateRadiance(ray, hit, rng);
    }
    else if (hit.hit)
    {
        PathTracer integrator;
        sample = integrator.estimateRadiance(ray, hit, rng);
    }
    else
    {
        sample = sceneEnvRadiance(ray.direction, fc.envIntensity, fc.furnaceMode);
    }

    // Sanitize NaN / Inf / negative components without nuking healthy
    // channels: a tiny float-noise negative in green should not zero out
    // a valid red contribution.
    bool3 bad = isnan(sample) | isinf(sample);
    sample = select(bad, float3(0.0), sample);
    sample = max(sample, float3(0.0));

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

    // BDPT light-tracer (s = 1) composite. lightSplatBuffer accumulates
    // fixed-point Q22.10 radiance over all frames (atomic-added by
    // bdpt.slang's splatLightVertex); divide by frames-so-far to get the
    // running mean of the light-tracer estimator and add to the eye-side
    // accumulation. For non-BDPT modes the buffer stays zeroed (cleared on
    // accumulation reset) so the contribution is zero.
    uint base = (pixel.y * fc.width + pixel.x) * 3u;
    uint sR = lightSplatBuffer[base + 0u];
    uint sG = lightSplatBuffer[base + 1u];
    uint sB = lightSplatBuffer[base + 2u];
    float invScale = 1.0 / 1024.0;
    float3 splatSum = float3(sR, sG, sB) * invScale;
    float frames = float(fc.accumFrame + 1u);
    float3 splatMean = splatSum / frames;
    accum = accum + splatMean;

    float3 display = linearToSRGB(acesFilmic(accum));

    if (hit.hit && effectiveFurnaceMode(hit.materialId) != 0u)
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

    // Gizmo overlay: cheap CPU→GPU 2D line list. For each segment we
    // measure the perpendicular distance from the current pixel to the
    // segment and blend the segment colour in with a 1-pixel anti-alias
    // falloff. Cap the loop at numGizmoSegments to skip the work entirely
    // when no gizmo is active.
    float2 px = float2(float(pixel.x) + 0.5, float(pixel.y) + 0.5);
    for (uint gi = 0u; gi < fc.numGizmoSegments; ++gi)
    {
        GizmoSegment seg = gizmoSegments[gi];
        float2 ab = seg.b - seg.a;
        float ab2 = max(dot(ab, ab), 1e-6);
        float t = clamp(dot(px - seg.a, ab) / ab2, 0.0, 1.0);
        float2 closest = seg.a + ab * t;
        float d = length(px - closest);
        float aa = clamp(seg.width + 1.0 - d, 0.0, 1.0);
        display = lerp(display, seg.color, aa);
    }

    outputBuffer[pixel] = float4(display, 1.0);
}
