// MaterialX pbrlib BSDF closure functions for Slang.
// Extracted from MaterialXGenSlang output; shared by all standard_surface
// materials. Env/light functions stubbed — our path tracer's bounce rays
// handle IBL, and our NEE handles direct lighting.
//
// References:
//   [1] Walter, Marschner, Li, Torrance,
//       "Microfacet Models for Refraction through Rough Surfaces", EGSR 2007.
//   [2] Heitz, "Understanding the Masking-Shadowing Function in
//       Microfacet-Based BRDFs", JCGT 2014.
//   [3] Schlick, "An Inexpensive BRDF Model for Physically-Based Rendering",
//       CGF 1994.
//   [4] Oren, Nayar, "Generalization of Lambert's Reflectance Model",
//       SIGGRAPH 1994.
//   [5] Christensen, Burley,
//       "Approximate Reflectance Profiles for Efficient Subsurface Scattering",
//       Disney Research / SIGGRAPH 2015.
//   [6] Turquin, "Practical multiple scattering compensation for microfacet
//       models", blog 2019.
//   [7] Zeltner, Chermain, Ureña,
//       "Practical and Controllable Subsurface Scattering for Production Path
//       Tracing", SIGGRAPH 2022.
//   [8] Duff et al., "Building an Orthonormal Basis, Revisited", JCGT 2017.

struct BSDF { float3 response; float3 throughput; };

// Slangc 2026.1's positional `BSDF(r, t)` constructor produces a value
// that mutates correctly when assigned to a local, BUT when that local
// is passed via `inout` to a BSDF helper (mx_oren_nayar_diffuse_bsdf
// etc.) the function's writes do not propagate back: caller's
// `bsdf.response` stays zero. Field-by-field init via this helper works
// around the issue. Replace every `BSDF X = BSDF(r, t);` with
// `BSDF X = makeBSDF(r, t);` to keep the closure tree producing
// non-zero output for graph-bound materials.
BSDF makeBSDF(float3 response, float3 throughput)
{
    BSDF b;
    b.response = response;
    b.throughput = throughput;
    return b;
}
#define EDF float3
struct VDF { float3 response; float3 throughput; };
struct surfaceshader { float3 color; float3 transparency; };
struct volumeshader { float3 color; float3 transparency; };
struct displacementshader { float3 offset; float scale; };
// lightshader and material alias omitted — our path tracer has its own types.

#define M_FLOAT_EPS 1e-8

#define mx_inversesqrt rsqrt
#define mx_sin sin
#define mx_cos cos
#define mx_tan tan
#define mx_asin asin
#define mx_acos acos
#define mx_radians radians
#define mx_float_bits_to_int asint

/// The GLSL we are piggybacking on has all its matrices transposed compared to Slang and the MaterialX spec.
/// (The matrices are defined like float3x3(1, 2, 3, 4, 5, 6, 7, 8, 9) where the spec says it should be row-major order, but GLSL creates it as col-major)
/// So when GLSL code says "mul(M, v)" it means "v * transpose(M)", and since in Slang the matrices are stored
/// in row-major order, we need to reverse the order of multiplication to get the same result.
float2 mx_matrix_mul(float2 v, float2x2 m) { return mul(m, v); }
float3 mx_matrix_mul(float3 v, float3x3 m) { return mul(m, v); }
float4 mx_matrix_mul(float4 v, float4x4 m) { return mul(m, v); }
float2 mx_matrix_mul(float2x2 m, float2 v) { return mul(v, m); }
float3 mx_matrix_mul(float3x3 m, float3 v) { return mul(v, m); }
float4 mx_matrix_mul(float4x4 m, float4 v) { return mul(v, m); }
float2x2 mx_matrix_mul(float2x2 m1, float2x2 m2) { return mul(m2, m1); }
float3x3 mx_matrix_mul(float3x3 m1, float3x3 m2) { return mul(m2, m1); }
float4x4 mx_matrix_mul(float4x4 m1, float4x4 m2) { return mul(m2, m1); }

float mx_square(float x)
{
    return x*x;
}

float2 mx_square(float2 x)
{
    return x*x;
}

float3 mx_square(float3 x)
{
    return x*x;
}

float3 mx_srgb_encode(float3 color)
{
    bool3 isAbove = (color > float3(0.0031308));
    float3 linSeg = color * 12.92;
    float3 powSeg = 1.055 * pow(max(color, float3(0.0)), float3(1.0 / 2.4)) - 0.055;
    return select(isAbove, powSeg, linSeg);
}

/// Library assumes GLSL style mod (result has the sign of y), so cannot use fmod
float mx_mod(float a, float b) { return (a - b * floor(a / b)); }
float2 mx_mod(float2 a, float2 b) { return (a - b * floor(a / b)); }
float3 mx_mod(float3 a, float3 b) { return (a - b * floor(a / b)); }
float4 mx_mod(float4 a, float4 b) { return (a - b * floor(a / b)); }
float2 mx_mod(float2 a, float b) { return (a - b * floor(a / b)); }
float3 mx_mod(float3 a, float b) { return (a - b * floor(a / b)); }
float4 mx_mod(float4 a, float b) { return (a - b * floor(a / b)); }

/// The float3x3 and float4x4 inverse are taken from the Slang's SGL library
/// https://github.com/shader-slang/sgl/
/// Specifically the matrix_math.h
/// https://github.com/shader-slang/sgl/blob/main/src/sgl/math/matrix_math.h
float3x3 mx_inverse(float3x3 m)
{
    float one_over_det = 1.f / determinant(m);

    float3x3 result;
    result[0][0] = +(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * one_over_det;
    result[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * one_over_det;
    result[0][2] = +(m[0][1] * m[1][2] - m[0][2] * m[1][1]) * one_over_det;
    result[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * one_over_det;
    result[1][1] = +(m[0][0] * m[2][2] - m[0][2] * m[2][0]) * one_over_det;
    result[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * one_over_det;
    result[2][0] = +(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * one_over_det;
    result[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * one_over_det;
    result[2][2] = +(m[0][0] * m[1][1] - m[0][1] * m[1][0]) * one_over_det;
    return result;
}

float4x4 mx_inverse(float4x4 m)
{
    float c00 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
    float c02 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
    float c03 = m[2][1] * m[3][2] - m[2][2] * m[3][1];

    float c04 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
    float c06 = m[1][1] * m[3][3] - m[1][3] * m[3][1];
    float c07 = m[1][1] * m[3][2] - m[1][2] * m[3][1];

    float c08 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
    float c10 = m[1][1] * m[2][3] - m[1][3] * m[2][1];
    float c11 = m[1][1] * m[2][2] - m[1][2] * m[2][1];

    float c12 = m[0][2] * m[3][3] - m[0][3] * m[3][2];
    float c14 = m[0][1] * m[3][3] - m[0][3] * m[3][1];
    float c15 = m[0][1] * m[3][2] - m[0][2] * m[3][1];

    float c16 = m[0][2] * m[2][3] - m[0][3] * m[2][2];
    float c18 = m[0][1] * m[2][3] - m[0][3] * m[2][1];
    float c19 = m[0][1] * m[2][2] - m[0][2] * m[2][1];

    float c20 = m[0][2] * m[1][3] - m[0][3] * m[1][2];
    float c22 = m[0][1] * m[1][3] - m[0][3] * m[1][1];
    float c23 = m[0][1] * m[1][2] - m[0][2] * m[1][1];

    float4 fac0 = float4(c00, c00, c02, c03);
    float4 fac1 = float4(c04, c04, c06, c07);
    float4 fac2 = float4(c08, c08, c10, c11);
    float4 fac3 = float4(c12, c12, c14, c15);
    float4 fac4 = float4(c16, c16, c18, c19);
    float4 fac5 = float4(c20, c20, c22, c23);

    float4 vec0 = float4(m[0][1], m[0][0], m[0][0], m[0][0]);
    float4 vec1 = float4(m[1][1], m[1][0], m[1][0], m[1][0]);
    float4 float2 = float4(m[2][1], m[2][0], m[2][0], m[2][0]);
    float4 float3 = float4(m[3][1], m[3][0], m[3][0], m[3][0]);

    float4 inv0 = float4(vec1 * fac0 - float2 * fac1 + float3 * fac2);
    float4 inv1 = float4(vec0 * fac0 - float2 * fac3 + float3 * fac4);
    float4 inv2 = float4(vec0 * fac1 - vec1 * fac3 + float3 * fac5);
    float4 inv3 = float4(vec0 * fac2 - vec1 * fac4 + float2 * fac5);

    float4 sign_a = float4(+1, -1, +1, -1);
    float4 sign_b = float4(-1, +1, -1, +1);
    // make matrix from columns
    float4x4 inverse = transpose(float4x4(inv0 * sign_a, inv1 * sign_b, inv2 * sign_a, inv3 * sign_b));

    float4 row0 = float4(inverse[0][0], inverse[0][1], inverse[0][2], inverse[0][3]);
    float4 col0 = float4(inverse[0][0], inverse[1][0], inverse[2][0], inverse[3][0]);

    float4 dot0 = float4(col0 * row0);
    float dot1 = (dot0.x + dot0.y) + (dot0.z + dot0.w);

    float one_over_det = 1.f / dot1;

    return inverse * one_over_det;
}

float mx_atan(float y_over_x)
{
    return atan(y_over_x);
}

float mx_atan(float y, float x)
{
    return atan2(y, x);
}

float2 mx_atan(float2 y, float2 x)
{
    return atan2(y, x);
}

float3 mx_atan(float3 y, float3 x)
{
    return atan2(y, x);
}

float4 mx_atan(float4 y, float4 x)
{
    return atan2(y, x);
}

#define DIRECTIONAL_ALBEDO_METHOD 0

#define AIRY_FRESNEL_ITERATIONS 2

#define M_PI 3.1415926535897932
#define M_PI_INV (1.0 / M_PI)

float mx_pow5(float x)
{
    return mx_square(mx_square(x)) * x;
}

float mx_pow6(float x)
{
    float x2 = mx_square(x);
    return mx_square(x2) * x2;
}

/// Schlick Fresnel approximation for dielectrics [3].
///
///   F(cosθ) = F₀ + (1 − F₀) · (1 − cosθ)⁵
///
/// where  F₀ = normal-incidence reflectance ∈ [0, 1]
///        cosθ = dot(view, half-vector)
// Standard Schlick Fresnel
float mx_fresnel_schlick(float cosTheta, float F0)
{
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x5 = mx_pow5(x);
    return F0 + (1.0 - F0) * x5;
}
float3 mx_fresnel_schlick(float cosTheta, float3 F0)
{
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x5 = mx_pow5(x);
    return F0 + (1.0 - F0) * x5;
}

// Generalized Schlick Fresnel
float mx_fresnel_schlick(float cosTheta, float F0, float F90)
{
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x5 = mx_pow5(x);
    return lerp(F0, F90, x5);
}
float3 mx_fresnel_schlick(float cosTheta, float3 F0, float3 F90)
{
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x5 = mx_pow5(x);
    return lerp(F0, F90, x5);
}

// Generalized Schlick Fresnel with a variable exponent
float mx_fresnel_schlick(float cosTheta, float F0, float F90, float exponent)
{
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    return lerp(F0, F90, pow(x, exponent));
}
float3 mx_fresnel_schlick(float cosTheta, float3 F0, float3 F90, float exponent)
{
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    return lerp(F0, F90, pow(x, exponent));
}

// Enforce that the given normal is forward-facing from the specified view direction.
float3 mx_forward_facing_normal(float3 N, float3 V)
{
    return (dot(N, V) < 0.0) ? -N : N;
}

// Gram-Schmidt orthogonalize tangent X against normal N.
// Falls back to a safe perpendicular when X is (nearly) parallel to N.
float3 mx_orthogonalize_tangent(float3 X, float3 N)
{
    float3 proj = X - dot(X, N) * N;
    float lenSq = dot(proj, proj);
    if (lenSq > 1e-6)
        return proj * rsqrt(lenSq);
    float3 up = abs(N.x) > 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    return normalize(up - dot(up, N) * N);
}

// https://www.graphics.rwth-aachen.de/publication/2/jgt.pdf
float mx_golden_ratio_sequence(int i)
{
    const float GOLDEN_RATIO = 1.6180339887498948;
    return frac((float(i) + 1.0) * GOLDEN_RATIO);
}

// https://people.irisa.fr/Ricardo.Marques/articles/2013/SF_CGF.pdf
float2 mx_spherical_fibonacci(int i, int numSamples)
{
    return float2((float(i) + 0.5) / float(numSamples), mx_golden_ratio_sequence(i));
}

// Generate a uniform-weighted sample on the unit hemisphere.
float3 mx_uniform_sample_hemisphere(float2 Xi)
{
    float phi = 2.0 * M_PI * Xi.x;
    float cosTheta = 1.0 - Xi.y;
    float sinTheta = sqrt(1.0 - mx_square(cosTheta));
    return float3(mx_cos(phi) * sinTheta,
                mx_sin(phi) * sinTheta,
                cosTheta);
}

// Generate a cosine-weighted sample on the unit hemisphere.
float3 mx_cosine_sample_hemisphere(float2 Xi)
{
    float phi = 2.0 * M_PI * Xi.x;
    float cosTheta = sqrt(Xi.y);
    float sinTheta = sqrt(1.0 - Xi.y);
    return float3(mx_cos(phi) * sinTheta,
                mx_sin(phi) * sinTheta,
                cosTheta);
}

// PDF of a cosine-weighted hemisphere sample.
float mx_cosine_hemisphere_PDF(float cosTheta)
{
    return max(cosTheta, 0.0) * M_PI_INV;
}

// PDF of a uniform hemisphere sample.
float mx_uniform_hemisphere_PDF()
{
    return 0.5 * M_PI_INV;
}

// Construct an orthonormal basis from a unit vector.
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
float3x3 mx_orthonormal_basis(float3 N)
{
    float sign = (N.z < 0.0) ? -1.0 : 1.0;
    float a = -1.0 / (sign + N.z);
    float b = N.x * N.y * a;
    float3 X = float3(1.0 + sign * N.x * N.x * a, sign * b, -sign * N.x);
    float3 Y = float3(b, sign + N.y * N.y * a, -N.y);
    return float3x3(X, Y, N);
}

static const int FRESNEL_MODEL_DIELECTRIC = 0;
static const int FRESNEL_MODEL_CONDUCTOR = 1;
static const int FRESNEL_MODEL_SCHLICK = 2;

// Parameters for Fresnel calculations
struct FresnelData
{
    // Fresnel model
    int model;
    bool airy;

    // Physical Fresnel
    float3 ior;
    float3 extinction;

    // Generalized Schlick Fresnel
    float3 F0;
    float3 F82;
    float3 F90;
    float exponent;

    // Thin film
    float tf_thickness;
    float tf_ior;

    // Refraction
    bool refraction;
};

/// GGX (Trowbridge-Reitz) anisotropic normal distribution function [1].
///
///   D(h) = 1 / (π · αₓ · αᵧ · (‖h.xy / α‖² + h.z²)²)
///
/// where  h   = microfacet normal in local (tangent) space; h.z = n·h
///        α   = (αₓ, αᵧ) anisotropic roughness (Disney §B.2)
// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// Appendix B.2 Equation 13
float mx_ggx_NDF(float3 H, float2 alpha)
{
    float2 He = H.xy / alpha;
    float denom = dot(He, He) + mx_square(H.z);
    return 1.0 / (M_PI * alpha.x * alpha.y * mx_square(denom));
}

/// GGX Visible Normal Distribution Function (VNDF) importance sampling.
///
/// Samples a microfacet normal h distributed proportionally to the
/// product D(h) · G₁(v, h) · (v·h) (i.e., weighted toward visible normals).
/// This reduces variance vs. sampling D(h) alone, especially at grazing
/// angles and high roughness.
///
/// Algorithm: project V onto a hemisphere, sample a spherical cap,
/// transform back. See Heitz & d'Eon 2023 for the derivation.
// https://ggx-research.github.io/publication/2023/06/09/publication-ggx.html
float3 mx_ggx_importance_sample_VNDF(float2 Xi, float3 V, float2 alpha)
{
    // Transform the view direction to the hemisphere configuration.
    V = normalize(float3(V.xy * alpha, V.z));

    // Sample a spherical cap in (-V.z, 1].
    float phi = 2.0 * M_PI * Xi.x;
    float z = (1.0 - Xi.y) * (1.0 + V.z) - V.z;
    float sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    float x = sinTheta * mx_cos(phi);
    float y = sinTheta * mx_sin(phi);
    float3 c = float3(x, y, z);

    // Compute the microfacet normal.
    float3 H = c + V;

    // Transform the microfacet normal back to the ellipsoid configuration.
    H = normalize(float3(H.xy * alpha, max(H.z, 0.0)));

    return H;
}

// PDF of a reflection direction sampled from the GGX VNDF.
float mx_ggx_VNDF_reflection_PDF(float3 H, float2 alpha, float G1V, float NdotV)
{
    return mx_ggx_NDF(H, alpha) * G1V / (4.0 * NdotV);
}

/// GGX Smith G₁ masking term for a single direction [1].
///
///   G₁(v, α) = 2 / (1 + √(1 + α²·tan²θᵥ))
///            = 2·(n·v) / ((n·v) + √(α² + (1−α²)·(n·v)²))
///
/// where  cosθ = n·v,  α = roughness (linear, not squared)
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
// Equation 34
float mx_ggx_smith_G1(float cosTheta, float alpha)
{
    float cosTheta2 = mx_square(cosTheta);
    float tanTheta2 = (1.0 - cosTheta2) / cosTheta2;
    return 2.0 / (1.0 + sqrt(1.0 + mx_square(alpha) * tanTheta2));
}

/// Height-correlated GGX Smith G₂ masking-shadowing function [2].
///
///   Λ(v) = (√(1 + α²·tan²θᵥ) − 1) / 2
///   G₂   = 2·(n·l)·(n·v) / (Λ(l)·(n·v) + Λ(v)·(n·l) + (n·l)·(n·v))
///        = 2·(n·l)·(n·v) / (√(α²+(1−α²)·(n·l)²)·(n·v)
///                          + √(α²+(1−α²)·(n·v)²)·(n·l))
///
/// Height-correlation (vs. separable G₁·G₁) reduces energy loss by ~8%
/// at α = 1 and gives more accurate total reflectance (JCGT §4, eq. 99).
// Height-correlated Smith masking-shadowing
// http://jcgt.org/published/0003/02/03/paper.pdf
// Equations 72 and 99
float mx_ggx_smith_G2(float NdotL, float NdotV, float alpha)
{
    float alpha2 = mx_square(alpha);
    float lambdaL = sqrt(alpha2 + (1.0 - alpha2) * mx_square(NdotL));
    float lambdaV = sqrt(alpha2 + (1.0 - alpha2) * mx_square(NdotV));
    return 2.0 * NdotL * NdotV / (lambdaL * NdotV + lambdaV * NdotL);
}

// Rational quadratic fit to Monte Carlo data for GGX directional albedo.
float3 mx_ggx_dir_albedo_analytic(float NdotV, float alpha, float3 F0, float3 F90)
{
    float x = NdotV;
    float y = alpha;
    float x2 = mx_square(x);
    float y2 = mx_square(y);
    float4 r = float4(0.1003, 0.9345, 1.0, 1.0) +
             float4(-0.6303, -2.323, -1.765, 0.2281) * x +
             float4(9.748, 2.229, 8.263, 15.94) * y +
             float4(-2.038, -3.748, 11.53, -55.83) * x * y +
             float4(29.34, 1.424, 28.96, 13.08) * x2 +
             float4(-8.245, -0.7684, -7.507, 41.26) * y2 +
             float4(-26.44, 1.436, -36.11, 54.9) * x2 * y +
             float4(19.99, 0.2913, 15.86, 300.2) * x * y2 +
             float4(-5.448, 0.6286, 33.37, -285.1) * x2 * y2;
    float2 AB = clamp(r.xy / r.zw, 0.0, 1.0);
    return F0 * AB.x + F90 * AB.y;
}

float3 mx_ggx_dir_albedo_table_lookup(float NdotV, float alpha, float3 F0, float3 F90)
{
#if DIRECTIONAL_ALBEDO_METHOD == 1
    if (textureSize(u_albedoTable, 0).x > 1)
    {
        float2 AB = texture(u_albedoTable, float2(NdotV, alpha)).rg;
        return F0 * AB.x + F90 * AB.y;
    }
#endif
    return float3(0.0);
}

// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
float3 mx_ggx_dir_albedo_monte_carlo(float NdotV, float alpha, float3 F0, float3 F90)
{
    NdotV = clamp(NdotV, M_FLOAT_EPS, 1.0);
    float3 V = float3(sqrt(1.0 - mx_square(NdotV)), 0, NdotV);

    float2 AB = float2(0.0);
    const int SAMPLE_COUNT = 64;
    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        float2 Xi = mx_spherical_fibonacci(i, SAMPLE_COUNT);

        // Compute the half vector and incoming light direction.
        float3 H = mx_ggx_importance_sample_VNDF(Xi, V, float2(alpha));
        float3 L = -reflect(V, H);
        
        // Compute dot products for this sample.
        float NdotL = clamp(L.z, M_FLOAT_EPS, 1.0);
        float VdotH = clamp(dot(V, H), M_FLOAT_EPS, 1.0);

        // Compute the Fresnel term.
        float Fc = mx_fresnel_schlick(VdotH, 0.0, 1.0);

        // Compute the per-sample geometric term.
        // https://hal.inria.fr/hal-00996995v2/document, Algorithm 2
        float G2 = mx_ggx_smith_G2(NdotL, NdotV, alpha);
        
        // Add the contribution of this sample.
        AB += float2(G2 * (1.0 - Fc), G2 * Fc);
    }

    // Apply the global component of the geometric term and normalize.
    AB /= mx_ggx_smith_G1(NdotV, alpha) * float(SAMPLE_COUNT);

    // Return the final directional albedo.
    return F0 * AB.x + F90 * AB.y;
}

float3 mx_ggx_dir_albedo(float NdotV, float alpha, float3 F0, float3 F90)
{
#if DIRECTIONAL_ALBEDO_METHOD == 0
    return mx_ggx_dir_albedo_analytic(NdotV, alpha, F0, F90);
#elif DIRECTIONAL_ALBEDO_METHOD == 1
    return mx_ggx_dir_albedo_table_lookup(NdotV, alpha, F0, F90);
#else
    return mx_ggx_dir_albedo_monte_carlo(NdotV, alpha, F0, F90);
#endif
}

float mx_ggx_dir_albedo(float NdotV, float alpha, float F0, float F90)
{
    return mx_ggx_dir_albedo(NdotV, alpha, float3(F0), float3(F90)).x;
}

/// GGX multi-scatter energy compensation term [6].
///
///   f_ms(v) = (1 − Ess(v)) / (π · (1 − E_avg))  · F_avg · F_avg
///
/// In practice, the tabulated directional albedo Ess(v) and its average
/// E_avg are replaced by the rational-quadratic fit mx_ggx_dir_albedo.
/// This additive term recovers energy lost to multiple micro-facet
/// bounces that the single-scatter GGX BRDF misses (Turquin eq. 14–16).
// https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf
// Equations 14 and 16
float3 mx_ggx_energy_compensation(float NdotV, float alpha, float3 Fss)
{
    float Ess = mx_ggx_dir_albedo(NdotV, alpha, 1.0, 1.0);
    return 1.0 + Fss * (1.0 - Ess) / Ess;
}

float mx_ggx_energy_compensation(float NdotV, float alpha, float Fss)
{
    return mx_ggx_energy_compensation(NdotV, alpha, float3(Fss)).x;
}

// Compute the average of an anisotropic alpha pair.
float mx_average_alpha(float2 alpha)
{
    return sqrt(alpha.x * alpha.y);
}

// Convert a real-valued index of refraction to normal-incidence reflectivity.
float mx_ior_to_f0(float ior)
{
    return mx_square((ior - 1.0) / (ior + 1.0));
}

// Convert normal-incidence reflectivity to real-valued index of refraction.
float mx_f0_to_ior(float F0)
{
    float sqrtF0 = sqrt(clamp(F0, 0.01, 0.99));
    return (1.0 + sqrtF0) / (1.0 - sqrtF0);
}
float3 mx_f0_to_ior(float3 F0)
{
    float3 sqrtF0 = sqrt(clamp(F0, 0.01, 0.99));
    return (float3(1.0) + sqrtF0) / (float3(1.0) - sqrtF0);
}

// https://renderwonk.com/publications/wp-generalization-adobe/gen-adobe.pdf
float3 mx_fresnel_hoffman_schlick(float cosTheta, FresnelData fd)
{
    const float COS_THETA_MAX = 1.0 / 7.0;
    const float COS_THETA_FACTOR = 1.0 / (COS_THETA_MAX * pow(1.0 - COS_THETA_MAX, 6.0));

    float x = clamp(cosTheta, 0.0, 1.0);
    float3 a = lerp(fd.F0, fd.F90, pow(1.0 - COS_THETA_MAX, fd.exponent)) * (float3(1.0) - fd.F82) * COS_THETA_FACTOR;
    return lerp(fd.F0, fd.F90, pow(1.0 - x, fd.exponent)) - a * x * mx_pow6(1.0 - x);
}

/// Exact dielectric Fresnel equations [1] — unpolarised average.
///
///   g  = √(n² + cos²θ − 1)
///   Rₛ = ((g−cosθ)/(g+cosθ))²
///   Rₚ = Rₛ · ((g+cosθ)·cosθ − 1)² / ((g−cosθ)·cosθ + 1)²
///   F  = (Rₛ + Rₚ) / 2
///
/// Returns 1 when g² < 0 (total internal reflection).
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
float mx_fresnel_dielectric(float cosTheta, float ior)
{
    float c = cosTheta;
    float g2 = ior*ior + c*c - 1.0;
    if (g2 < 0.0)
    {
        // Total internal reflection
        return 1.0;
    }

    float g = sqrt(g2);
    return 0.5 * mx_square((g - c) / (g + c)) *
                (1.0 + mx_square(((g + c) * c - 1.0) / ((g - c) * c + 1.0)));
}

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
float2 mx_fresnel_dielectric_polarized(float cosTheta, float ior)
{
    float cosTheta2 = mx_square(clamp(cosTheta, 0.0, 1.0));
    float sinTheta2 = 1.0 - cosTheta2;

    float t0 = max(ior * ior - sinTheta2, 0.0);
    float t1 = t0 + cosTheta2;
    float t2 = 2.0 * sqrt(t0) * cosTheta;
    float Rs = (t1 - t2) / (t1 + t2);

    float t3 = cosTheta2 * t0 + sinTheta2 * sinTheta2;
    float t4 = t2 * sinTheta2;
    float Rp = Rs * (t3 - t4) / (t3 + t4);

    return float2(Rp, Rs);
}

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
void mx_fresnel_conductor_polarized(float cosTheta, float3 n, float3 k, out float3 Rp, out float3 Rs)
{
    float cosTheta2 = mx_square(clamp(cosTheta, 0.0, 1.0));
    float sinTheta2 = 1.0 - cosTheta2;
    float3 n2 = n * n;
    float3 k2 = k * k;

    float3 t0 = n2 - k2 - float3(sinTheta2);
    float3 a2plusb2 = sqrt(t0 * t0 + 4.0 * n2 * k2);
    float3 t1 = a2plusb2 + float3(cosTheta2);
    float3 a = sqrt(max(0.5 * (a2plusb2 + t0), 0.0));
    float3 t2 = 2.0 * a * cosTheta;
    Rs = (t1 - t2) / (t1 + t2);

    float3 t3 = cosTheta2 * a2plusb2 + float3(sinTheta2 * sinTheta2);
    float3 t4 = t2 * sinTheta2;
    Rp = Rs * (t3 - t4) / (t3 + t4);
}

/// Conductor Fresnel using complex IOR (n + ik) — unpolarised average.
///
///   a² + b² = √((n²−k²−sin²θ)² + 4·n²·k²)
///   Rₛ = (a²+b² − 2a·cosθ + cos²θ) / (a²+b² + 2a·cosθ + cos²θ)
///   Rₚ = Rₛ · (cos²θ·(a²+b²) − 2a·cosθ·sin²θ + sin⁴θ)
///            / (cos²θ·(a²+b²) + 2a·cosθ·sin²θ + sin⁴θ)
///   F  = (Rₛ + Rₚ) / 2
///
/// where  n = real part of IOR (per RGB),  k = extinction coefficient
float3 mx_fresnel_conductor(float cosTheta, float3 n, float3 k)
{
    float3 Rp, Rs;
    mx_fresnel_conductor_polarized(cosTheta, n, k, Rp, Rs);
    return 0.5 * (Rp  + Rs);
}

// https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
void mx_fresnel_conductor_phase_polarized(float cosTheta, float eta1, float3 eta2, float3 kappa2, out float3 phiP, out float3 phiS)
{
    float3 k2 = kappa2 / eta2;
    float3 sinThetaSqr = float3(1.0) - cosTheta * cosTheta;
    float3 A = eta2*eta2*(float3(1.0)-k2*k2) - eta1*eta1*sinThetaSqr;
    float3 B = sqrt(A*A + mx_square(2.0*eta2*eta2*k2));
    float3 U = sqrt((A+B)/2.0);
    float3 V = max(float3(0.0), sqrt((B-A)/2.0));

    phiS = mx_atan(2.0*eta1*V*cosTheta, U*U + V*V - mx_square(eta1*cosTheta));
    phiP = mx_atan(2.0*eta1*eta2*eta2*cosTheta * (2.0*k2*U - (float3(1.0)-k2*k2) * V),
                   mx_square(eta2*eta2*(float3(1.0)+k2*k2)*cosTheta) - eta1*eta1*(U*U+V*V));
}

// https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
float3 mx_eval_sensitivity(float opd, float3 shift)
{
    // Use Gaussian fits, given by 3 parameters: val, pos and var
    float phase = 2.0*M_PI * opd;
    float3 val = float3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    float3 pos = float3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    float3 var = float3(4.3278e+09, 9.3046e+09, 6.6121e+09);
    float3 xyz = val * sqrt(2.0*M_PI * var) * mx_cos(pos * phase + shift) * exp(- var * phase*phase);
    xyz.x   += 9.7470e-14 * sqrt(2.0*M_PI * 4.5282e+09) * mx_cos(2.2399e+06 * phase + shift[0]) * exp(- 4.5282e+09 * phase*phase);
    return xyz / 1.0685e-7;
}

// A Practical Extension to Microfacet Theory for the Modeling of Varying Iridescence
// https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
float3 mx_fresnel_airy(float cosTheta, FresnelData fd)
{
    // XYZ to CIE 1931 RGB color space (using neutral E illuminant)
    const float3x3 XYZ_TO_RGB = float3x3(2.3706743, -0.5138850, 0.0052982, -0.9000405, 1.4253036, -0.0146949, -0.4706338, 0.0885814, 1.0093968);

    // Assume vacuum on the outside
    float eta1 = 1.0;
    float eta2 = max(fd.tf_ior, eta1);
    float3 eta3 = (fd.model == FRESNEL_MODEL_SCHLICK) ? mx_f0_to_ior(fd.F0) : fd.ior;
    float3 kappa3 = (fd.model == FRESNEL_MODEL_SCHLICK) ? float3(0.0) : fd.extinction;
    float cosThetaT = sqrt(1.0 - (1.0 - mx_square(cosTheta)) * mx_square(eta1 / eta2));

    // First interface
    float2 R12 = mx_fresnel_dielectric_polarized(cosTheta, eta2 / eta1);
    if (cosThetaT <= 0.0)
    {
        // Total internal reflection
        R12 = float2(1.0);
    }
    float2 T121 = float2(1.0) - R12;

    // Second interface
    float3 R23p, R23s;
    if (fd.model == FRESNEL_MODEL_SCHLICK)
    {
        float3 f = mx_fresnel_hoffman_schlick(cosThetaT, fd);
        R23p = 0.5 * f;
        R23s = 0.5 * f;
    }
    else
    {
        mx_fresnel_conductor_polarized(cosThetaT, eta3 / eta2, kappa3 / eta2, R23p, R23s);
    }

    // Phase shift
    float cosB = mx_cos(mx_atan(eta2 / eta1));
    float2 phi21 = float2(cosTheta < cosB ? 0.0 : M_PI, M_PI);
    float3 phi23p, phi23s;
    if (fd.model == FRESNEL_MODEL_SCHLICK)
    {
        phi23p = float3((eta3[0] < eta2) ? M_PI : 0.0,
                      (eta3[1] < eta2) ? M_PI : 0.0,
                      (eta3[2] < eta2) ? M_PI : 0.0);
        phi23s = phi23p;
    }
    else
    {
        mx_fresnel_conductor_phase_polarized(cosThetaT, eta2, eta3, kappa3, phi23p, phi23s);
    }
    float3 r123p = max(sqrt(R12.x*R23p), 0.0);
    float3 r123s = max(sqrt(R12.y*R23s), 0.0);

    // Iridescence term
    float3 I = float3(0.0);
    float3 Cm, Sm;

    // Optical path difference
    float distMeters = fd.tf_thickness * 1.0e-9;
    float opd = 2.0 * eta2 * cosThetaT * distMeters;

    // Iridescence term using spectral antialiasing for Parallel polarization

    // Reflectance term for m=0 (DC term amplitude)
    float3 Rs = (mx_square(T121.x) * R23p) / (float3(1.0) - R12.x*R23p);
    I += R12.x + Rs;

    // Reflectance term for m>0 (pairs of diracs)
    Cm = Rs - T121.x;
    for (int m = 1; m <= AIRY_FRESNEL_ITERATIONS; m++)
    {
        Cm *= r123p;
        Sm  = 2.0 * mx_eval_sensitivity(float(m) * opd, float(m)*(phi23p+float3(phi21.x)));
        I  += Cm*Sm;
    }

    // Iridescence term using spectral antialiasing for Perpendicular polarization

    // Reflectance term for m=0 (DC term amplitude)
    float3 Rp = (mx_square(T121.y) * R23s) / (float3(1.0) - R12.y*R23s);
    I += R12.y + Rp;

    // Reflectance term for m>0 (pairs of diracs)
    Cm = Rp - T121.y;
    for (int m = 1; m <= AIRY_FRESNEL_ITERATIONS; m++)
    {
        Cm *= r123s;
        Sm  = 2.0 * mx_eval_sensitivity(float(m) * opd, float(m)*(phi23s+float3(phi21.y)));
        I  += Cm*Sm;
    }

    // Average parallel and perpendicular polarization
    I *= 0.5;

    // Convert back to RGB reflectance
    I = clamp(mx_matrix_mul(XYZ_TO_RGB, I), 0.0, 1.0);

    return I;
}

FresnelData mx_init_fresnel_dielectric(float ior, float tf_thickness, float tf_ior)
{
    FresnelData fd;
    fd.model = FRESNEL_MODEL_DIELECTRIC;
    fd.airy = tf_thickness > 0.0;
    fd.ior = float3(ior);
    fd.extinction = float3(0.0);
    fd.F0 = float3(0.0);
    fd.F82 = float3(0.0);
    fd.F90 = float3(0.0);
    fd.exponent = 0.0;
    fd.tf_thickness = tf_thickness;
    fd.tf_ior = tf_ior;
    fd.refraction = false;
    return fd;
}

FresnelData mx_init_fresnel_conductor(float3 ior, float3 extinction, float tf_thickness, float tf_ior)
{
    FresnelData fd;
    fd.model = FRESNEL_MODEL_CONDUCTOR;
    fd.airy = tf_thickness > 0.0;
    fd.ior = ior;
    fd.extinction = extinction;
    fd.F0 = float3(0.0);
    fd.F82 = float3(0.0);
    fd.F90 = float3(0.0);
    fd.exponent = 0.0;
    fd.tf_thickness = tf_thickness;
    fd.tf_ior = tf_ior;
    fd.refraction = false;
    return fd;
}

FresnelData mx_init_fresnel_schlick(float3 F0, float3 F82, float3 F90, float exponent, float tf_thickness, float tf_ior)
{
    FresnelData fd;
    fd.model = FRESNEL_MODEL_SCHLICK;
    fd.airy = tf_thickness > 0.0;
    fd.ior = float3(0.0);
    fd.extinction = float3(0.0);
    fd.F0 = F0;
    fd.F82 = F82;
    fd.F90 = F90;
    fd.exponent = exponent;
    fd.tf_thickness = tf_thickness;
    fd.tf_ior = tf_ior;
    fd.refraction = false;
    return fd;
}

float3 mx_compute_fresnel(float cosTheta, FresnelData fd)
{
    if (fd.airy)
    {
         return mx_fresnel_airy(cosTheta, fd);
    }
    else if (fd.model == FRESNEL_MODEL_DIELECTRIC)
    {
        return float3(mx_fresnel_dielectric(cosTheta, fd.ior.x));
    }
    else if (fd.model == FRESNEL_MODEL_CONDUCTOR)
    {
        return mx_fresnel_conductor(cosTheta, fd.ior, fd.extinction);
    }
    else // FRESNEL_MODEL_SCHLICK
    {
        return mx_fresnel_hoffman_schlick(cosTheta, fd);
    }
}

// Directional albedo accounting for different Fresnel functions.
float3 mx_ggx_dir_albedo(float NdotV, float alpha, FresnelData fd)
{
    if (fd.airy)
    {
        // Approximation using a blend between mirror (alpha = 0)
        // and rougher cases. This helps to maintain angular
        // color variation at lower roughness values.
        float3 mirrorDirAlbedo = mx_compute_fresnel(NdotV, fd);
        float3 F0 = mx_fresnel_airy(1.0, fd);
        float3 roughDirAlbedo = mx_ggx_dir_albedo(NdotV, alpha, F0, float3(1.0));
        return lerp(mirrorDirAlbedo, roughDirAlbedo, sqrt(alpha));
    }
    else if (fd.model == FRESNEL_MODEL_DIELECTRIC)
    {
        float F0 = mx_ior_to_f0(fd.ior.x);
        return mx_ggx_dir_albedo(NdotV, alpha, float3(F0), float3(1.0));
    }
    else if (fd.model == FRESNEL_MODEL_CONDUCTOR)
    {
        float3 F0 = mx_fresnel_conductor(1.0, fd.ior, fd.extinction);
        return mx_ggx_dir_albedo(NdotV, alpha, F0, float3(1.0));
    }
    else // FRESNEL_MODEL_SCHLICK
    {
        return mx_ggx_dir_albedo(NdotV, alpha, fd.F0, fd.F90);
    }
}

// Compute the refraction of a ray through a solid sphere.
float3 mx_refraction_solid_sphere(float3 R, float3 N, float ior)
{
    R = refract(R, N, 1.0 / ior);
    float3 N1 = normalize(R * dot(R, N) - N * 0.5);
    return refract(R, N1, ior);
}


float3 mx_environment_radiance(float3 N, float3 V, float3 X, float2 alpha,
                               int distribution, FresnelData fd)
{
    return float3(0.0);
}

float3 mx_environment_irradiance(float3 N)
{
    return float3(0.0);
}

float3 mx_surface_transmission(float3 N, float3 V, float3 X, float2 alpha,
                               int distribution, FresnelData fd, float3 tint)
{
    return float3(0.0);
}

// Return the mip level with the appropriate coverage for a filtered importance sample.
// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html
// Section 20.4 Equation 13
float mx_latlong_compute_lod(float3 dir, float pdf, float maxMipLevel, int envSamples)
{
    const float MIP_LEVEL_OFFSET = 1.5;
    float effectiveMaxMipLevel = maxMipLevel - MIP_LEVEL_OFFSET;
    float distortion = sqrt(1.0 - mx_square(dir.y));
    return max(effectiveMaxMipLevel - 0.5 * log2(float(envSamples) * pdf * distortion), 0.0);
}


void mx_roughness_anisotropy(float roughness, float anisotropy, out float2 result)
{
    float roughness_sqr = clamp(roughness*roughness, M_FLOAT_EPS, 1.0);
    if (anisotropy > 0.0)
    {
        float aspect = sqrt(1.0 - clamp(anisotropy, 0.0, 0.98));
        result.x = min(roughness_sqr / aspect, 1.0);
        result.y = roughness_sqr * aspect;
    }
    else
    {
        result.x = roughness_sqr;
        result.y = roughness_sqr;
    }
}

// These are defined based on the HwShaderGenerator::ClosureContextType enum
// if that changes - these need to be updated accordingly.

static const int CLOSURE_TYPE_DEFAULT = 0;
static const int CLOSURE_TYPE_REFLECTION = 1;
static const int CLOSURE_TYPE_TRANSMISSION = 2;
static const int CLOSURE_TYPE_INDIRECT = 3;
static const int CLOSURE_TYPE_EMISSION = 4;

struct ClosureData {
    int closureType;
    float3 L;
    float3 V;
    float3 N;
    float3 P;
    float occlusion;
};

ClosureData makeClosureData(int closureType, float3 L, float3 V, float3 N, float3 P, float occlusion)
{
    return ClosureData(closureType, L, V, N, P, occlusion);
}

// https://fpsunflower.github.io/ckulla/data/s2017_pbs_imageworks_sheen.pdf
// Equation 2
float mx_imageworks_sheen_NDF(float NdotH, float roughness)
{
    float invRoughness = 1.0 / max(roughness, 0.005);
    float cos2 = NdotH * NdotH;
    float sin2 = 1.0 - cos2;
    return (2.0 + invRoughness) * pow(sin2, invRoughness * 0.5) / (2.0 * M_PI);
}

float mx_imageworks_sheen_brdf(float NdotL, float NdotV, float NdotH, float roughness)
{
    // Microfacet distribution.
    float D = mx_imageworks_sheen_NDF(NdotH, roughness);

    // Fresnel and geometry terms are ignored.
    float F = 1.0;
    float G = 1.0;

    // We use a smoother denominator, as in:
    // https://blog.selfshadow.com/publications/s2013-shading-course/rad/s2013_pbs_rad_notes.pdf
    return D * F * G / (4.0 * (NdotL + NdotV - NdotL*NdotV));
}

// Rational quadratic fit to Monte Carlo data for Imageworks sheen directional albedo.
float mx_imageworks_sheen_dir_albedo_analytic(float NdotV, float roughness)
{
    float2 r = float2(13.67300, 1.0) +
             float2(-68.78018, 61.57746) * NdotV +
             float2(799.08825, 442.78211) * roughness +
             float2(-905.00061, 2597.49308) * NdotV * roughness +
             float2(60.28956, 121.81241) * mx_square(NdotV) +
             float2(1086.96473, 3045.55075) * mx_square(roughness);
    return r.x / r.y;
}

float mx_imageworks_sheen_dir_albedo_table_lookup(float NdotV, float roughness)
{
#if DIRECTIONAL_ALBEDO_METHOD == 1
    if (textureSize(u_albedoTable, 0).x > 1)
    {
        return texture(u_albedoTable, float2(NdotV, roughness)).b;
    }
#endif
    return 0.0;
}

float mx_imageworks_sheen_dir_albedo_monte_carlo(float NdotV, float roughness)
{
    NdotV = clamp(NdotV, M_FLOAT_EPS, 1.0);
    float3 V = float3(sqrt(1.0f - mx_square(NdotV)), 0, NdotV);

    float radiance = 0.0;
    const int SAMPLE_COUNT = 64;
    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        float2 Xi = mx_spherical_fibonacci(i, SAMPLE_COUNT);

        // Compute the incoming light direction and half vector.
        float3 L = mx_uniform_sample_hemisphere(Xi);
        float3 H = normalize(L + V);
        
        // Compute dot products for this sample.
        float NdotL = clamp(L.z, M_FLOAT_EPS, 1.0);
        float NdotH = clamp(H.z, M_FLOAT_EPS, 1.0);

        // Compute sheen reflectance.
        float reflectance = mx_imageworks_sheen_brdf(NdotL, NdotV, NdotH, roughness);

        // Add the radiance contribution of this sample.
        //   radiance = reflectance * NdotL / uniform_pdf;
        radiance += reflectance * NdotL / mx_uniform_hemisphere_PDF();
    }

    // Return the final directional albedo.
    return radiance / float(SAMPLE_COUNT);
}

float mx_imageworks_sheen_dir_albedo(float NdotV, float roughness)
{
#if DIRECTIONAL_ALBEDO_METHOD == 0
    float dirAlbedo = mx_imageworks_sheen_dir_albedo_analytic(NdotV, roughness);
#elif DIRECTIONAL_ALBEDO_METHOD == 1
    float dirAlbedo = mx_imageworks_sheen_dir_albedo_table_lookup(NdotV, roughness);
#else
    float dirAlbedo = mx_imageworks_sheen_dir_albedo_monte_carlo(NdotV, roughness);
#endif
    return clamp(dirAlbedo, 0.0, 1.0);
}

// The following functions are adapted from https://github.com/tizian/ltc-sheen.
// "Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines", Zeltner et al.

// Gaussian fit to directional albedo table.
float mx_zeltner_sheen_dir_albedo(float x, float y)
{
    float s = y*(0.0206607 + 1.58491*y)/(0.0379424 + y*(1.32227 + y));
    float m = y*(-0.193854 + y*(-1.14885 + y*(1.7932 - 0.95943*y*y)))/(0.046391 + y);
    float o = y*(0.000654023 + (-0.0207818 + 0.119681*y)*y)/(1.26264 + y*(-1.92021 + y));
    return exp(-0.5*mx_square((x - m)/s))/(s*sqrt(2.0*M_PI)) + o;
}

// Rational fits to LTC matrix coefficients.
float mx_zeltner_sheen_ltc_aInv(float x, float y)
{
    return (2.58126*x + 0.813703*y)*y/(1.0 + 0.310327*x*x + 2.60994*x*y);
}

float mx_zeltner_sheen_ltc_bInv(float x, float y)
{
    return sqrt(1.0 - x)*(y - 1.0)*y*y*y/(0.0000254053 + 1.71228*x - 1.71506*x*y + 1.34174*y*y);
}

// V and N are assumed to be unit vectors.
float3x3 mx_orthonormal_basis_ltc(float3 V, float3 N, float NdotV)
{
    // Generate a tangent vector in the plane of V and N.
    // This required to correctly orient the LTC lobe.
    float3 X = V - N*NdotV;
    float lenSqr = dot(X, X);
    if (lenSqr > 0.0)
    {
        X *= mx_inversesqrt(lenSqr);
        float3 Y = cross(N, X);
        return float3x3(X, Y, N);
    }

    // If lenSqr == 0, then V == N, so any orthonormal basis will do.
    return mx_orthonormal_basis(N);
}

// Multiplication by directional albedo is handled by the calling function.
float mx_zeltner_sheen_brdf(float3 L, float3 V, float3 N, float NdotV, float roughness)
{
    float3x3 toLTC = transpose(mx_orthonormal_basis_ltc(V, N, NdotV));
    float3 w = mx_matrix_mul(toLTC, L);

    float aInv = mx_zeltner_sheen_ltc_aInv(NdotV, roughness);
    float bInv = mx_zeltner_sheen_ltc_bInv(NdotV, roughness);

    // Transform w to original configuration (clamped cosine).
    //                 |aInv    0 bInv|
    // wo = M^-1 . w = |   0 aInv    0| . w
    //                 |   0    0    1|
    float3 wo = float3(aInv*w.x + bInv*w.z, aInv * w.y, w.z);
    float lenSqr = dot(wo, wo);

    // D(w) = Do(M^-1.w / ||M^-1.w||) . |M^-1| / ||M^-1.w||^3
    //      = Do(M^-1.w) . |M^-1| / ||M^-1.w||^4
    //      = Do(wo) . |M^-1| / dot(wo, wo)^2
    //      = Do(wo) . aInv^2 / dot(wo, wo)^2
    //      = Do(wo) . (aInv / dot(wo, wo))^2
    return mx_cosine_hemisphere_PDF(wo.z) * mx_square(aInv / lenSqr);
}

float3 mx_zeltner_sheen_importance_sample(float2 Xi, float3 V, float3 N, float roughness, out float pdf)
{
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    roughness = clamp(roughness, 0.01, 1.0); // Clamp to range of original impl.

    float3 wo = mx_cosine_sample_hemisphere(Xi);

    float aInv = mx_zeltner_sheen_ltc_aInv(NdotV, roughness);
    float bInv = mx_zeltner_sheen_ltc_bInv(NdotV, roughness);

    // Transform wo from original configuration (clamped cosine).
    //              |1/aInv      0 -bInv/aInv|
    // w = M . wo = |     0 1/aInv          0| . wo
    //              |     0      0          1|    
    float3 w = float3(wo.x/aInv - wo.z*bInv/aInv, wo.y / aInv, wo.z);

    float lenSqr = dot(w, w);
    w *= mx_inversesqrt(lenSqr);

    // D(w) = Do(wo) . ||M.wo||^3 / |M|
    //      = Do(wo / ||M.wo||) . ||M.wo||^4 / |M| 
    //      = Do(w) . ||M.wo||^4 / |M| (possible because M doesn't change z component)
    //      = Do(w) . dot(w, w)^2 * aInv^2
    //      = Do(w) . (aInv * dot(w, w))^2
    pdf = mx_cosine_hemisphere_PDF(w.z) * mx_square(aInv * lenSqr);

    float3x3 fromLTC = mx_orthonormal_basis_ltc(V, N, NdotV);
    w = mx_matrix_mul(fromLTC, w);

    return w;
}

void mx_sheen_bsdf(ClosureData closureData, float weight, float3 color, float roughness, float3 N, int mode, inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS)
    {
        return;
    }

    float3 V = closureData.V;
    float3 L = closureData.L;

    N = mx_forward_facing_normal(N, V);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    if (closureData.closureType == CLOSURE_TYPE_REFLECTION)
    {
        float dirAlbedo;
        if (mode == 0)
        {
            float3 H = normalize(L + V);

            float NdotL = clamp(dot(N, L), M_FLOAT_EPS, 1.0);
            float NdotH = clamp(dot(N, H), M_FLOAT_EPS, 1.0);

            float3 fr = color * mx_imageworks_sheen_brdf(NdotL, NdotV, NdotH, roughness);
            dirAlbedo = mx_imageworks_sheen_dir_albedo(NdotV, roughness);

            // We need to include NdotL from the light integral here
            // as in this case it's not cancelled out by the BRDF denominator.
            bsdf.response = fr * NdotL * closureData.occlusion * weight;
        }
        else
        {
            roughness = clamp(roughness, 0.01, 1.0); // Clamp to range of original impl.

            float3 fr = color * mx_zeltner_sheen_brdf(L, V, N, NdotV, roughness);
            dirAlbedo = mx_zeltner_sheen_dir_albedo(NdotV, roughness);
            bsdf.response = dirAlbedo * fr * closureData.occlusion * weight;
        }
        bsdf.throughput = float3(1.0 - dirAlbedo * weight);
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT)
    {
        float dirAlbedo;
        if (mode == 0)
        {
            dirAlbedo = mx_imageworks_sheen_dir_albedo(NdotV, roughness);
        }
        else
        {
            roughness = clamp(roughness, 0.01, 1.0); // Clamp to range of original impl.
            dirAlbedo = mx_zeltner_sheen_dir_albedo(NdotV, roughness);
        }

        float3 Li = mx_environment_irradiance(N);
        bsdf.response = Li * color * dirAlbedo * weight;
        bsdf.throughput = float3(1.0 - dirAlbedo * weight);
    }
}

void mx_luminance_color3(float3 _in, float3 lumacoeffs, out float3 result)
{
    result = float3(dot(_in, lumacoeffs));
}

float4x4 mx_rotationMatrix(float3 axis, float angle)
{
    axis = normalize(axis);
    float s = mx_sin(angle);
    float c = mx_cos(angle);
    float oc = 1.0 - c;

    return float4x4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

void mx_rotate_vector3(float3 _in, float amount, float3 axis, out float3 result)
{
    float rotationRadians = mx_radians(amount);
    float4x4 m = mx_rotationMatrix(axis, rotationRadians);
    result = mx_matrix_mul(m, float4(_in, 1.0)).xyz;
}

void mx_artistic_ior(float3 reflectivity, float3 edge_color, out float3 ior, out float3 extinction)
{
    // "Artist Friendly Metallic Fresnel", Ole Gulbrandsen, 2014
    // http://jcgt.org/published/0003/04/03/paper.pdf

    float3 r = clamp(reflectivity, 0.0, 0.99);
    float3 r_sqrt = sqrt(r);
    float3 n_min = (1.0 - r) / (1.0 + r);
    float3 n_max = (1.0 + r_sqrt) / (1.0 - r_sqrt);
    ior = lerp(n_max, n_min, edge_color);

    float3 np1 = ior + 1.0;
    float3 nm1 = ior - 1.0;
    float3 k2 = (np1*np1 * r - nm1*nm1) / (1.0 - r);
    k2 = max(k2, 0.0);
    extinction = sqrt(k2);
}


void mx_uniform_edf(ClosureData closureData, float3 color, out EDF result)
{
    if (closureData.closureType == CLOSURE_TYPE_EMISSION)
    {
        result = color;
    }
}


void mx_multiply_edf_color3(ClosureData closureData, EDF in1, float3 in2, out EDF result)
{
    result = in1 * in2;
}


void mx_dielectric_bsdf(ClosureData closureData, float weight, float3 tint, float ior, float2 roughness, bool retroreflective, float thinfilm_thickness, float thinfilm_ior, float3 N, float3 X, int distribution, int scatter_mode, inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS)
    {
        return;
    }
    if (closureData.closureType != CLOSURE_TYPE_TRANSMISSION && scatter_mode == 1)
    {
        return;
    }

    float3 V = closureData.V;
    float3 L = closureData.L;

    // Retroreflective mode is only supported for reflection and indirect
    if (retroreflective && (closureData.closureType != CLOSURE_TYPE_TRANSMISSION))
        V = reflect(-V, N);
    
    N = mx_forward_facing_normal(N, V);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    FresnelData fd = mx_init_fresnel_dielectric(ior, thinfilm_thickness, thinfilm_ior);
    float F0 = mx_ior_to_f0(ior);

    float2 safeAlpha = clamp(roughness, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(safeAlpha);
    float3 safeTint = max(tint, 0.0);

    if (closureData.closureType == CLOSURE_TYPE_REFLECTION)
    {
        X = mx_orthogonalize_tangent(X, N);
        float3 Y = cross(N, X);
        float3 H = normalize(L + V);

        float NdotL = clamp(dot(N, L), M_FLOAT_EPS, 1.0);
        float VdotH = clamp(dot(V, H), M_FLOAT_EPS, 1.0);

        float3 Ht = float3(dot(H, X), dot(H, Y), dot(H, N));

        float3 F = mx_compute_fresnel(VdotH, fd);
        float D = mx_ggx_NDF(Ht, safeAlpha);
        float G = mx_ggx_smith_G2(NdotL, NdotV, avgAlpha);

        float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * comp;
        bsdf.throughput = 1.0 - dirAlbedo * weight;

        bsdf.response = D * F * G * comp * safeTint * closureData.occlusion * weight / (4.0 * NdotV);
    }
    else if (closureData.closureType == CLOSURE_TYPE_TRANSMISSION)
    {
        float3 F = mx_compute_fresnel(NdotV, fd);

        float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * comp;
        bsdf.throughput = 1.0 - dirAlbedo * weight;

        if (scatter_mode != 0)
        {
            bsdf.response = mx_surface_transmission(N, V, X, safeAlpha, distribution, fd, safeTint) * weight;
        }
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT)
    {
        float3 F = mx_compute_fresnel(NdotV, fd);

        float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, F0, 1.0) * comp;
        bsdf.throughput = 1.0 - dirAlbedo * weight;

        float3 Li = mx_environment_radiance(N, V, X, safeAlpha, distribution, fd);
        bsdf.response = Li * safeTint * comp * weight;
    }
}


void mx_conductor_bsdf(ClosureData closureData, float weight, float3 ior_n, float3 ior_k, float2 roughness, bool retroreflective, float thinfilm_thickness, float thinfilm_ior, float3 N, float3 X, int distribution, inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);

    if (weight < M_FLOAT_EPS)
    {
        return;
    }

    float3 V = closureData.V;
    float3 L = closureData.L;

    V = retroreflective ? reflect(-V, N) : V;
    N = mx_forward_facing_normal(N, V);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    FresnelData fd = mx_init_fresnel_conductor(ior_n, ior_k, thinfilm_thickness, thinfilm_ior);

    float2 safeAlpha = clamp(roughness, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(safeAlpha);

    if (closureData.closureType == CLOSURE_TYPE_REFLECTION)
    {
        X = mx_orthogonalize_tangent(X, N);
        float3 Y = cross(N, X);
        float3 H = normalize(L + V);

        float NdotL = clamp(dot(N, L), M_FLOAT_EPS, 1.0);
        float VdotH = clamp(dot(V, H), M_FLOAT_EPS, 1.0);

        float3 Ht = float3(dot(H, X), dot(H, Y), dot(H, N));

        float3 F = mx_compute_fresnel(VdotH, fd);
        float D = mx_ggx_NDF(Ht, safeAlpha);
        float G = mx_ggx_smith_G2(NdotL, NdotV, avgAlpha);

        float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);

        // Note: NdotL is cancelled out
        bsdf.response = D * F * G * comp * closureData.occlusion * weight / (4.0 * NdotV);
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT)
    {
        float3 F = mx_compute_fresnel(NdotV, fd);
        float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        float3 Li = mx_environment_radiance(N, V, X, safeAlpha, distribution, fd);
        bsdf.response = Li * comp * weight;
    }
}


void mx_translucent_bsdf(ClosureData closureData, float weight, float3 color, float3 N, inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);

    if (weight < M_FLOAT_EPS)
    {
        return;
    }

    float3 V = closureData.V;
    float3 L = closureData.L;

    // Invert normal since we're transmitting light from the other side
    N = -N;

    if (closureData.closureType == CLOSURE_TYPE_REFLECTION)
    {
        float NdotL = clamp(dot(N, L), 0.0, 1.0);
        bsdf.response = color * weight * NdotL * M_PI_INV;
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT)
    {
        float3 Li = mx_environment_irradiance(N);
        bsdf.response = Li * color * weight;
    }
}


static const float FUJII_CONSTANT_1 = 0.5 - 2.0 / (3.0 * M_PI);
static const float FUJII_CONSTANT_2 = 2.0 / 3.0 - 28.0 / (15.0 * M_PI);

// Qualitative Oren-Nayar diffuse with simplified math:
/// Qualitative Oren-Nayar rough diffuse BRDF (simplified form) [4].
///
///   s    = l·v − (n·l)·(n·v)   (combined cosφ·sinα·tanβ factor)
///   t    = max(n·l, n·v)        (denominator for the sinα·tanβ ratio)
///   f_d  = A + B · max(0, s/t)  · albedo / π
///
/// where  A = 1 − 0.5·σ² / (σ²+0.33),   B = 0.45·σ² / (σ²+0.09)
///        σ = roughness (surface slope std-dev)
// https://www1.cs.columbia.edu/CAVE/publications/pdfs/Oren_SIGGRAPH94.pdf
float mx_oren_nayar_diffuse(float NdotV, float NdotL, float LdotV, float roughness)
{
    float s = LdotV - NdotL * NdotV;
    float stinv = (s > 0.0) ? s / max(NdotL, NdotV) : 0.0;

    float sigma2 = mx_square(roughness);
    float A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33));
    float B = 0.45 * sigma2 / (sigma2 + 0.09);

    return A + B * stinv;
}

// Rational quadratic fit to Monte Carlo data for Oren-Nayar directional albedo.
float mx_oren_nayar_diffuse_dir_albedo_analytic(float NdotV, float roughness)
{
    float2 r = float2(1.0, 1.0) +
             float2(-0.4297, -0.6076) * roughness +
             float2(-0.7632, -0.4993) * NdotV * roughness +
             float2(1.4385, 2.0315) * mx_square(roughness);
    return r.x / r.y;
}

float mx_oren_nayar_diffuse_dir_albedo_table_lookup(float NdotV, float roughness)
{
#if DIRECTIONAL_ALBEDO_METHOD == 1
    if (textureSize(u_albedoTable, 0).x > 1)
    {
        return texture(u_albedoTable, float2(NdotV, roughness)).b;
    }
#endif
    return 0.0;
}

float mx_oren_nayar_diffuse_dir_albedo_monte_carlo(float NdotV, float roughness)
{
    NdotV = clamp(NdotV, M_FLOAT_EPS, 1.0);
    float3 V = float3(sqrt(1.0 - mx_square(NdotV)), 0, NdotV);

    float radiance = 0.0;
    const int SAMPLE_COUNT = 64;
    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        float2 Xi = mx_spherical_fibonacci(i, SAMPLE_COUNT);

        // Compute the incoming light direction.
        float3 L = mx_uniform_sample_hemisphere(Xi);
        
        // Compute dot products for this sample.
        float NdotL = clamp(L.z, M_FLOAT_EPS, 1.0);
        float LdotV = clamp(dot(L, V), M_FLOAT_EPS, 1.0);

        // Compute diffuse reflectance.
        float reflectance = mx_oren_nayar_diffuse(NdotV, NdotL, LdotV, roughness);

        // Add the radiance contribution of this sample.
        //   uniform_pdf = 1 / (2 * PI)
        //   radiance = (reflectance * NdotL) / (uniform_pdf * PI);
        radiance += reflectance * NdotL;
    }

    // Apply global components and normalize.
    radiance *= 2.0 / float(SAMPLE_COUNT);

    // Return the final directional albedo.
    return radiance;
}

float mx_oren_nayar_diffuse_dir_albedo(float NdotV, float roughness)
{
#if DIRECTIONAL_ALBEDO_METHOD == 2
    float dirAlbedo = mx_oren_nayar_diffuse_dir_albedo_monte_carlo(NdotV, roughness);
#else
    float dirAlbedo = mx_oren_nayar_diffuse_dir_albedo_analytic(NdotV, roughness);
#endif
    return clamp(dirAlbedo, 0.0, 1.0);
}

// Improved Oren-Nayar diffuse from Fujii:
// https://mimosa-pudica.net/improved-oren-nayar.html
float mx_oren_nayar_fujii_diffuse_dir_albedo(float cosTheta, float roughness)
{
    float A = 1.0 / (1.0 + FUJII_CONSTANT_1 * roughness);
    float B = roughness * A;
    float Si = sqrt(max(0.0, 1.0 - mx_square(cosTheta)));
    float G = Si * (mx_acos(clamp(cosTheta, -1.0, 1.0)) - Si * cosTheta) +
              2.0 * ((Si / cosTheta) * (1.0 - Si * Si * Si) - Si) / 3.0;
    return A + (B * G * M_PI_INV);
}

float mx_oren_nayar_fujii_diffuse_avg_albedo(float roughness)
{
    float A = 1.0 / (1.0 + FUJII_CONSTANT_1 * roughness);
    return A * (1.0 + FUJII_CONSTANT_2 * roughness);
}   

// Energy-compensated Oren-Nayar diffuse from OpenPBR Surface:
// https://academysoftwarefoundation.github.io/OpenPBR/
float3 mx_oren_nayar_compensated_diffuse(float NdotV, float NdotL, float LdotV, float roughness, float3 color)
{
    float s = LdotV - NdotL * NdotV;
    float stinv = (s > 0.0) ? s / max(NdotL, NdotV) : s;

    // Compute the single-scatter lobe.
    float A = 1.0 / (1.0 + FUJII_CONSTANT_1 * roughness);
    float3 lobeSingleScatter = color * A * (1.0 + roughness * stinv);

    // Compute the multi-scatter lobe.
    float dirAlbedoV = mx_oren_nayar_fujii_diffuse_dir_albedo(NdotV, roughness);
    float dirAlbedoL = mx_oren_nayar_fujii_diffuse_dir_albedo(NdotL, roughness);
    float avgAlbedo = mx_oren_nayar_fujii_diffuse_avg_albedo(roughness);
    float3 colorMultiScatter = mx_square(color) * avgAlbedo /
                             (float3(1.0) - color * max(0.0, 1.0 - avgAlbedo));
    float3 lobeMultiScatter = colorMultiScatter *
                            max(M_FLOAT_EPS, 1.0 - dirAlbedoV) *
                            max(M_FLOAT_EPS, 1.0 - dirAlbedoL) /
                            max(M_FLOAT_EPS, 1.0 - avgAlbedo);

    // Return the sum.
    return lobeSingleScatter + lobeMultiScatter;
}

float3 mx_oren_nayar_compensated_diffuse_dir_albedo(float cosTheta, float roughness, float3 color)
{
    float dirAlbedo = mx_oren_nayar_fujii_diffuse_dir_albedo(cosTheta, roughness);
    float avgAlbedo = mx_oren_nayar_fujii_diffuse_avg_albedo(roughness);
    float3 colorMultiScatter = mx_square(color) * avgAlbedo /
                             (float3(1.0) - color * max(0.0, 1.0 - avgAlbedo));
    return lerp(colorMultiScatter, color, dirAlbedo);
}
  
// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// Section 5.3
float mx_burley_diffuse(float NdotV, float NdotL, float LdotH, float roughness)
{
    float F90 = 0.5 + (2.0 * roughness * mx_square(LdotH));
    float refL = mx_fresnel_schlick(NdotL, 1.0, F90);
    float refV = mx_fresnel_schlick(NdotV, 1.0, F90);
    return refL * refV;
}

// Compute the directional albedo component of Burley diffuse for the given
// view angle and roughness.  Curve fit provided by Stephen Hill.
float mx_burley_diffuse_dir_albedo(float NdotV, float roughness)
{
    float x = NdotV;
    float fit0 = 0.97619 - 0.488095 * mx_pow5(1.0 - x);
    float fit1 = 1.55754 + (-2.02221 + (2.56283 - 1.06244 * x) * x) * x;
    return lerp(fit0, fit1, roughness);
}

/// Burley normalised diffusion profile [5].
///
///   R(r) = (e^(−shape·r) + e^(−shape·r/3)) · shape / (8π·r)
///
/// where  r     = radial distance (dist)
///        shape = 1/d per RGB channel,  d = diffuse mean free path
/// Same formula as the skin BSSRDF version; shape = 1/d is the
/// Pixar parameterisation (ApproxBSSRDF tech report).
// Evaluate the Burley diffusion profile for the given distance and diffusion shape.
// Based on https://graphics.pixar.com/library/ApproxBSSRDF/
float3 mx_burley_diffusion_profile(float dist, float3 shape)
{
    float3 num1 = exp(-shape * dist);
    float3 num2 = exp(-shape * dist / 3.0);
    float denom = max(dist, M_FLOAT_EPS);
    return (num1 + num2) / denom;
}

// Integrate the Burley diffusion profile over a sphere of the given radius.
// Inspired by Eric Penner's presentation in http://advances.realtimerendering.com/s2011/
float3 mx_integrate_burley_diffusion(float3 N, float3 L, float radius, float3 mfp)
{
    float theta = mx_acos(dot(N, L));

    // Estimate the Burley diffusion shape from mean free path.
    float3 shape = float3(1.0) / max(mfp, 0.1);

    // Integrate the profile over the sphere.
    float3 sumD = float3(0.0);
    float3 sumR = float3(0.0);
    const int SAMPLE_COUNT = 32;
    const float SAMPLE_WIDTH = (2.0 * M_PI) / float(SAMPLE_COUNT);
    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        float x = -M_PI + (float(i) + 0.5) * SAMPLE_WIDTH;
        float dist = radius * abs(2.0 * mx_sin(x * 0.5));
        float3 R = mx_burley_diffusion_profile(dist, shape);
        sumD += R * max(mx_cos(theta + x), 0.0);
        sumR += R;
    }

    return sumD / sumR;
}

float3 mx_subsurface_scattering_approx(float3 N, float3 L, float3 P, float3 albedo, float3 mfp)
{
    // fwidth unavailable in compute shaders; use a fixed curvature estimate.
    float curvature = 1.0;
    float radius = 1.0 / max(curvature, 0.01);
    return albedo * mx_integrate_burley_diffusion(N, L, radius, mfp) / float3(M_PI);
}

void mx_subsurface_bsdf(ClosureData closureData, float weight, float3 color, float3 radius, float anisotropy, float3 N, inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);

    if (weight < M_FLOAT_EPS)
    {
        return;
    }

    float3 V = closureData.V;
    float3 L = closureData.L;
    float3 P = closureData.P;
    float occlusion = closureData.occlusion;

    N = mx_forward_facing_normal(N, V);

    if (closureData.closureType == CLOSURE_TYPE_REFLECTION)
    {
        float3 sss = mx_subsurface_scattering_approx(N, L, P, color, radius);
        float NdotL = clamp(dot(N, L), M_FLOAT_EPS, 1.0);
        float visibleOcclusion = 1.0 - NdotL * (1.0 - occlusion);
        bsdf.response = sss * visibleOcclusion * weight;
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT)
    {
        // For now, we render indirect subsurface as simple indirect diffuse.
        float3 Li = mx_environment_irradiance(N);
        bsdf.response = Li * color * weight;
    }
}


void mx_oren_nayar_diffuse_bsdf(ClosureData closureData, float weight, float3 color, float roughness, float3 N, bool energy_compensation, inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);

    if (weight < M_FLOAT_EPS)
    {
        return;
    }

    float3 V = closureData.V;
    float3 L = closureData.L;

    N = mx_forward_facing_normal(N, V);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    if (closureData.closureType == CLOSURE_TYPE_REFLECTION)
    {
        float NdotL = clamp(dot(N, L), M_FLOAT_EPS, 1.0);
        float LdotV = clamp(dot(L, V), M_FLOAT_EPS, 1.0);

        float3 diffuse = energy_compensation ?
                       mx_oren_nayar_compensated_diffuse(NdotV, NdotL, LdotV, roughness, color) :
                       mx_oren_nayar_diffuse(NdotV, NdotL, LdotV, roughness) * color;
        bsdf.response = diffuse * closureData.occlusion * weight * NdotL * M_PI_INV;
    }
    else if (closureData.closureType == CLOSURE_TYPE_INDIRECT)
    {
        float3 diffuse = energy_compensation ?
                       mx_oren_nayar_compensated_diffuse_dir_albedo(NdotV, roughness, color) :
                       mx_oren_nayar_diffuse_dir_albedo(NdotV, roughness) * color;
        float3 Li = mx_environment_irradiance(N);
        bsdf.response = Li * diffuse * weight;
    }
}

void NG_convert_float_color3(float in1, out float3 out1)
{
    float3 combine_out = float3(in1,in1,in1);
    out1 = combine_out;
}


void mx_add_bsdf(ClosureData closureData, BSDF in1, BSDF in2, out BSDF result)
{
    result.response = in1.response + in2.response;

    // We derive the throughput for closure addition as follows:
    //   throughput_1 = 1 - dir_albedo_1
    //   throughput_2 = 1 - dir_albedo_2
    //   throughput_sum = 1 - (dir_albedo_1 + dir_albedo_2)
    //                  = 1 - ((1 - throughput_1) + (1 - throughput_2))
    //                  = throughput_1 + throughput_2 - 1
    result.throughput = max(in1.throughput + in2.throughput - 1.0, 0.0);
}


void mx_generalized_schlick_edf(ClosureData closureData, float3 color0, float3 color90, float exponent, EDF base, out EDF result)
{
    if (closureData.closureType == CLOSURE_TYPE_EMISSION)
    {
        float3 N = mx_forward_facing_normal(closureData.N, closureData.V);
        float NdotV = clamp(dot(N, closureData.V), M_FLOAT_EPS, 1.0);
        float3 f = mx_fresnel_schlick(NdotV, color0, color90, exponent);
        result = base * f;
    }
}


void mx_multiply_bsdf_float(ClosureData closureData, BSDF in1, float in2, out BSDF result)
{
    float weight = clamp(in2, 0.0, 1.0);
    result.response = in1.response * weight;
    result.throughput = in1.throughput;
}


void mx_mix_edf(ClosureData closureData, EDF fg, EDF bg, float mixValue, out EDF result)
{
    result = lerp(bg, fg, mixValue);
}


void mx_layer_bsdf(ClosureData closureData, BSDF top, BSDF base, out BSDF result)
{
    result.response = top.response + base.response * top.throughput;
    result.throughput = top.throughput * base.throughput;
}


void mx_multiply_bsdf_color3(ClosureData closureData, BSDF in1, float3 in2, out BSDF result)
{
    float3 tint = clamp(in2, 0.0, 1.0);
    result.response = in1.response * tint;
    result.throughput = in1.throughput;
}
