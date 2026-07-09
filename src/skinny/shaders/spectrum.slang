// Hero-wavelength spectral core (change spectral-rendering, Group 4).
//
// Compiled into the megakernel ONLY in the spectral variant (`-DSKINNY_SPECTRAL`);
// the RGB build never imports this module, so its SPIR-V is byte-unchanged.
//
// The functions here mirror, bit-for-bit, the CPU reference in
// `skinny/pbrt/spectral.py` + `skinny/pbrt/data/spectral_tables.py` (validated
// GPU≡numpy in tests/kernels): pbrt visible-wavelength importance sampling, the
// Wilkie-et-al. hero-wavelength rotation, secondary termination for dispersion,
// Jakob-Hanika RGB→spectrum upsampling (pbrt's exact table lookup), and the film
// resolve back to linear sRGB through skinny's Wyman-Sloan-Shirley CMF fit
// (shared with `spectra.py` so importer, mirror, and shader agree).

// `Spectrum` (float4 spectral / float3 RGB) + SPECTRUM_N live in common.slang so
// the carrier structs type-check in both builds; import it here rather than
// redefine (this module is compiled only in the spectral variant).
import common;

static const float SPECTRUM_LAMBDA_MIN = 360.0;
static const float SPECTRUM_LAMBDA_MAX = 830.0;

// ∫ ȳ(λ) dλ over the Wyman fit on the 360–830/5 nm grid — MUST match
// spectra._Y_INTEGRAL so the GPU resolve equals the numpy mirror.
static const float CIE_Y_INTEGRAL = 106.9229674725;

// N hero-rotated wavelengths + their sampling pdfs (pbrt SampledWavelengths).
struct SampledWavelengths
{
    float4 lambda;   // wavelengths (nm); only .xyz meaningful in RGB build
    float4 pdf;      // per-sample sampling pdf; 0 ⇒ terminated secondary
};

// ── pbrt visible-wavelength importance sampling ─────────────────

float sampleVisibleWavelength(float u)
{
    return 538.0 - 138.888889 * atanh(0.85691062 - 1.82750197 * u);
}

float visibleWavelengthPdf(float lam)
{
    if (lam < SPECTRUM_LAMBDA_MIN || lam > SPECTRUM_LAMBDA_MAX)
        return 0.0;
    float c = cosh(0.0072 * (lam - 538.0));
    return 0.0039398042 / (c * c);
}

// Draw the 4 hero-rotated wavelengths from one sample `u` (pbrt SampleVisible).
SampledWavelengths sampleWavelengths(float u)
{
    SampledWavelengths sw;
    [ForceUnroll]
    for (int i = 0; i < 4; ++i)
    {
        float up = u + float(i) / 4.0;
        if (up > 1.0)
            up -= 1.0;
        float lam = sampleVisibleWavelength(up);
        sw.lambda[i] = lam;
        sw.pdf[i] = visibleWavelengthPdf(lam);
    }
    return sw;
}

bool secondaryTerminated(SampledWavelengths sw)
{
    return sw.pdf.y == 0.0 && sw.pdf.z == 0.0 && sw.pdf.w == 0.0;
}

// Collapse to the hero wavelength on a dispersive event (pbrt TerminateSecondary).
SampledWavelengths terminateSecondary(SampledWavelengths sw)
{
    if (secondaryTerminated(sw))
        return sw;
    sw.pdf.x /= 4.0;
    sw.pdf.y = 0.0;
    sw.pdf.z = 0.0;
    sw.pdf.w = 0.0;
    return sw;
}

// ── CIE colour-matching (Wyman-Sloan-Shirley fit, mirrors spectra.py) ──

float cieGauss(float x, float mu, float s1, float s2)
{
    float s = (x < mu) ? s1 : s2;
    float t = (x - mu) / s;
    return exp(-0.5 * t * t);
}

float3 cieXYZBar(float lam)
{
    float x = 1.056 * cieGauss(lam, 599.8, 37.9, 31.0)
            + 0.362 * cieGauss(lam, 442.0, 16.0, 26.7)
            - 0.065 * cieGauss(lam, 501.1, 20.4, 26.2);
    float y = 0.821 * cieGauss(lam, 568.8, 46.9, 40.5)
            + 0.286 * cieGauss(lam, 530.9, 16.3, 31.1);
    float z = 1.217 * cieGauss(lam, 437.0, 11.8, 36.0)
            + 0.681 * cieGauss(lam, 459.0, 26.0, 13.8);
    return float3(x, y, z);
}

float3 xyzToLinearSRGB(float3 xyz)
{
    return float3(
        3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z,
        -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z,
        0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z);
}

// ── Film resolve: N spectral samples → linear sRGB ──────────────

float3 sampledSpectrumToXYZ(SampledWavelengths sw, float4 values)
{
    float3 acc = float3(0.0);
    [ForceUnroll]
    for (int i = 0; i < 4; ++i)
    {
        float p = sw.pdf[i];
        float safe = (p > 0.0) ? (1.0 / p) : 0.0;
        acc += values[i] * cieXYZBar(sw.lambda[i]) * safe;
    }
    return (acc / 4.0) / CIE_Y_INTEGRAL;
}

float3 spectrumResolveToLinearSRGB(SampledWavelengths sw, float4 values)
{
    return max(xyzToLinearSRGB(sampledSpectrumToXYZ(sw, values)), float3(0.0));
}

// ── RGB→spectrum upsampling (Jakob-Hanika; pbrt exact lookup) ───

// S(λ) = s(c0λ² + c1λ + c2), s(x) = 0.5 + x/(2√(1+x²)).
float sigmoidPoly(float3 coeffs, float lam)
{
    float poly = coeffs.z + lam * (coeffs.y + lam * coeffs.x);
    return 0.5 + poly / (2.0 * sqrt(1.0 + poly * poly));
}

float sigmoidLerp(float t, float a, float b) { return a + t * (b - a); }

// pbrt RGBToSpectrumTable::operator() trilinear lookup. `scale` is the RES node
// array; `data` is the flat [3][res][res][res][3] coefficient grid (row-major).
// Bound as StructuredBuffers by the renderer (descriptor plumbing, Group 3.2).
float3 rgbSigmoidCoeffs(float3 rgb, int res,
                        StructuredBuffer<float> scale,
                        StructuredBuffer<float> data)
{
    rgb = clamp(rgb, float3(0.0), float3(1.0));
    if (rgb.x == rgb.y && rgb.y == rgb.z)
    {
        // Uniform RGB → constant spectrum. pbrt rides the r∈{0,1} divide-by-zero
        // into IEEE ±inf → sigmoid {0,1}; a finite ±1e9 reproduces that limit.
        // The prior `c2 = 0` collapsed both endpoints to 0.5 (white reflected
        // 50 % → white furnace never closed; black reflected 50 %). Mirrors
        // spectral_tables.rgb_to_sigmoid_coeffs.
        float r = rgb.x;
        float c2;
        if (r >= 1.0)      c2 = 1.0e9;
        else if (r <= 0.0) c2 = -1.0e9;
        else               c2 = (r - 0.5) / sqrt(r * (1.0 - r));
        return float3(0.0, 0.0, c2);
    }
    int maxc = (rgb.x > rgb.y) ? ((rgb.x > rgb.z) ? 0 : 2)
                               : ((rgb.y > rgb.z) ? 1 : 2);
    float z = rgb[maxc];
    float x = rgb[(maxc + 1) % 3] * float(res - 1) / z;
    float y = rgb[(maxc + 2) % 3] * float(res - 1) / z;
    int xi = min(int(x), res - 2);
    int yi = min(int(y), res - 2);

    // FindInterval: largest zi in [0, res-2] with scale[zi] < z.
    int zi = 0;
    for (int i = 1; i < res - 1; ++i)
        if (scale[i] < z)
            zi = i;
    float dx = x - float(xi);
    float dy = y - float(yi);
    float dz = (z - scale[zi]) / (scale[zi + 1] - scale[zi]);

    float3 c;
    [ForceUnroll]
    for (int comp = 0; comp < 3; ++comp)
    {
        // co(dxo,dyo,dzo) = data[maxc][zi+dzo][yi+dyo][xi+dxo][comp]
        // flat index = (((maxc*res + z)*res + y)*res + x)*3 + comp
        float c000 = data[(((maxc * res + (zi + 0)) * res + (yi + 0)) * res + (xi + 0)) * 3 + comp];
        float c100 = data[(((maxc * res + (zi + 0)) * res + (yi + 0)) * res + (xi + 1)) * 3 + comp];
        float c010 = data[(((maxc * res + (zi + 0)) * res + (yi + 1)) * res + (xi + 0)) * 3 + comp];
        float c110 = data[(((maxc * res + (zi + 0)) * res + (yi + 1)) * res + (xi + 1)) * 3 + comp];
        float c001 = data[(((maxc * res + (zi + 1)) * res + (yi + 0)) * res + (xi + 0)) * 3 + comp];
        float c101 = data[(((maxc * res + (zi + 1)) * res + (yi + 0)) * res + (xi + 1)) * 3 + comp];
        float c011 = data[(((maxc * res + (zi + 1)) * res + (yi + 1)) * res + (xi + 0)) * 3 + comp];
        float c111 = data[(((maxc * res + (zi + 1)) * res + (yi + 1)) * res + (xi + 1)) * 3 + comp];
        c[comp] = sigmoidLerp(dz,
            sigmoidLerp(dy, sigmoidLerp(dx, c000, c100), sigmoidLerp(dx, c010, c110)),
            sigmoidLerp(dy, sigmoidLerp(dx, c001, c101), sigmoidLerp(dx, c011, c111)));
    }
    return c;
}

// Evaluate an sRGB reflectance as a spectrum at the sampled wavelengths.
float4 upsampleReflectance(float3 rgb, SampledWavelengths sw, int res,
                           StructuredBuffer<float> scale,
                           StructuredBuffer<float> data)
{
    float3 co = rgbSigmoidCoeffs(rgb, res, scale, data);
    float4 s;
    [ForceUnroll]
    for (int i = 0; i < 4; ++i)
        s[i] = sigmoidPoly(co, sw.lambda[i]);
    return s;
}

// Linear interp of a curve sampled on the 360–830/5 nm grid (mirrors np.interp
// with endpoint clamp). `count` = grid length (95 for the vendored curves).
float sampleCurve5nm(float lam, StructuredBuffer<float> curve, int count)
{
    float t = (lam - SPECTRUM_LAMBDA_MIN) / 5.0;
    int i = clamp(int(t), 0, count - 2);
    float f = clamp(t - float(i), 0.0, 1.0);
    return lerp(curve[i], curve[i + 1], f);
}

// Evaluate an sRGB emitter as a spectrum (pbrt RGBIlluminantSpectrum):
// scale·sigmoid(rgb/scale)·D65 with scale = 2·max(rgb). The scale keeps HDR
// emitters (rgb > 1) inside the sigmoid's [0,1] gamut and restores their
// magnitude — clamping would collapse a bright light to white and lose its
// intensity. `d65` is uploaded pre-normalized to unit luminance (renderer feeds
// `spectral.d65_normalized()`), so a unit RGB illuminant resolves to unit
// radiance and this mirrors `spectral.upsample_illuminant` bit-for-bit.
float4 upsampleIlluminant(float3 rgb, SampledWavelengths sw, int res,
                          StructuredBuffer<float> scale,
                          StructuredBuffer<float> data,
                          StructuredBuffer<float> d65, int d65Count)
{
    float m  = max(rgb.x, max(rgb.y, rgb.z));
    float sc = 2.0 * m;
    float3 rgbN = (sc > 0.0) ? rgb / sc : float3(0.0);
    float3 co = rgbSigmoidCoeffs(rgbN, res, scale, data);
    float4 s;
    [ForceUnroll]
    for (int i = 0; i < 4; ++i)
        s[i] = sc * sigmoidPoly(co, sw.lambda[i]) * sampleCurve5nm(sw.lambda[i], d65, d65Count);
    return s;
}
