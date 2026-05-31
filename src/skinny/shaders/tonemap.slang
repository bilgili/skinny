// Display tonemapping + sRGB encoding — shared by the megakernel (main_pass)
// and the wavefront resolve display tail (wavefront/wf_display) so there is one
// definition of each operator.
//
// Reference: Narkowicz, "ACES Filmic Tone Mapping Curve" (approximation fit),
// blog post 2016; original ACES matrices by the Academy S-2014-003.

// ── Tonemapping ─────────────────────────────────────────────────

/// ACES filmic tone-mapping curve (Narkowicz rational approximation).
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

/// Reinhard tone mapping: x / (1 + x). Cheap, desaturates highlights less
/// aggressively than ACES, useful as a neutral reference.
float3 reinhardTonemap(float3 x)
{
    return clamp(x / (1.0 + x), 0.0, 1.0);
}

/// Hable / Uncharted 2 filmic curve (John Hable, GDC 2010). Constants
/// match the original blog-post fit; final divide by the white-point
/// response normalises so a pre-exposed white maps to ~1.
float3 hableCurve(float3 x)
{
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

float3 hableFilmic(float3 x)
{
    float W = 11.2;
    float3 numer = hableCurve(x * 2.0);
    float3 denom = hableCurve(float3(W));
    return clamp(numer / denom, 0.0, 1.0);
}

/// Apply the user-selected tonemap operator to linear HDR radiance.
/// `mode` matches the Python `tonemap_modes` list ordering:
///   0 = ACES, 1 = Reinhard, 2 = Hable, 3 = Linear (clamp only).
float3 applyTonemap(float3 x, uint mode)
{
    if (mode == 1u) return reinhardTonemap(x);
    if (mode == 2u) return hableFilmic(x);
    if (mode == 3u) return clamp(x, 0.0, 1.0);
    return acesFilmic(x);
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
