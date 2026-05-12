// Common types and utilities shared across all shader modules.
//
// References:
//   [1] O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically
//       Good Algorithms for Random Number Generation", HMC-CS-2014-0905, 2014.
//   [2] Veach, "Robust Monte Carlo Methods for Light Transport Simulation",
//       PhD thesis, Stanford University 1997.
//   [3] Walter, Marschner, Li, Torrance,
//       "Microfacet Models for Refraction through Rough Surfaces", EGSR 2007.

static const float PI = 3.14159265358979;

struct Ray
{
    float3 origin;
    float3 direction;
};

struct HitInfo
{
    float  t;              // ray parameter at intersection (world-space units)
    float  backT;          // ray parameter at back-face exit (-1 = no back hit)
    float3 position;       // world-space hit position
    float3 positionObject; // object-space (BLAS-local) hit position — for 3D procedural textures
    float3 normal;         // world-space shading normal
    float3 tangent;        // surface tangent aligned with +U; bitangent = cross(N, T)
    float2 uv;
    bool   hit;
    bool   hasTangent;     // true when uv comes from an asset (not the fallback sphere map)
    uint   materialId;     // index into the active material set
};

// Per-material flat-shading record. Used by main_pass.slang when a hit's
// materialTypes[id] == MATERIAL_TYPE_FLAT. Layout matches
// renderer.pack_flat_material exactly: 96 B / record, std430-compatible.
struct FlatMaterialParams
{
    float3 diffuseColor;          // 0..12
    float  roughness;             // 12..16 (packs into vec3 trailing pad)
    float  metallic;              // 16..20
    float  specular;              // 20..24
    float  opacity;               // 24..28
    uint   diffuseTextureIdx;     // 28..32; 0xFFFFFFFF = use constant
    uint   roughnessTextureIdx;   // 32..36; sentinel = use constant
    uint   metallicTextureIdx;    // 36..40; sentinel = use constant
    uint   normalTextureIdx;      // 40..44; sentinel = use mesh normal
    uint   emissiveTextureIdx;    // 44..48; sentinel = use emissive const
    float3 emissiveColor;         // 48..60
    float  ior;                   // 60..64; index of refraction (1.0 = vacuum)
    float  coat;                  // 64..68; clear coat weight (0..1)
    float  coatRoughness;         // 68..72
    float  coatIOR;               // 72..76
    uint   opacityTextureIdx;     // 76..80; sentinel = use constant
    float3 coatColor;             // 80..92
    float  opacityThreshold;      // 92..96; cutout alpha threshold
};

// Gen-reflected skin material UBO — layout produced by MaterialXGenSlang
// running on M_skinny_skin_default. 164 scalar-layout bytes / 27 fields.
// The renderer fills this each frame via materialx_runtime.pack_material_values
// using SkinParameters values mapped through Renderer._mtlx_skin_overrides.
// Compiled with -fvk-use-scalar-layout: float3 has 4-byte alignment (no
// 16-byte promotion), so the struct packs tightly.
struct MtlxSkinParams
{
    float  layer_top_melanin;             // +0
    float  layer_top_thickness;           // +4
    float3 layer_top_scattering_coeff;    // +8
    float  layer_top_anisotropy;          // +20
    float  layer_top_ior;                 // +24
    float  layer_middle_hemoglobin;       // +28
    float  layer_middle_blood_oxygenation;// +32
    float  layer_middle_thickness;        // +36
    float3 layer_middle_scattering_coeff; // +40
    float  layer_middle_anisotropy;       // +52
    float  layer_middle_ior;              // +56
    float4 layer_middle_pigment;          // +60
    float  layer_bottom_thickness;        // +76
    float3 layer_bottom_absorption;       // +80
    float3 layer_bottom_scattering;       // +92
    float  layer_bottom_anisotropy;       // +104
    float  layer_bottom_ior;              // +108
    float4 skin_bsdf_pigment;             // +112
    float  skin_bsdf_weight;              // +128
    float  skin_bsdf_roughness;           // +132
    float  skin_bsdf_ior;                 // +136
    float  skin_bsdf_pore_density;        // +140
    float  skin_bsdf_pore_depth;          // +144
    float  skin_bsdf_hair_density;        // +148
    float  skin_bsdf_hair_tilt;           // +152
    float  skin_surface_opacity;          // +156
    uint   skin_surface_thin_walled;      // +160
};

struct EmissiveTriangle
{
    float3 v0;
    float  pad0;
    float3 v1;
    float  pad1;
    float3 v2;
    float  pad2;
    float3 emission;
    float  area;
};

struct SphereLight
{
    float3 position;
    float  radius;
    float3 radiance;
    float  pad;
};

struct GizmoSegment
{
    float2 a;       // screen-space pixel coords (origin at bottom-left)
    float2 b;
    float3 color;   // linear RGB
    float  width;   // half-width in pixels
};

struct Camera
{
    float4x4 viewInverse;
    float4x4 projInverse;
    float4x4 view;          // forward view matrix (world → camera space)
    float4x4 proj;          // forward projection matrix (camera → clip)
    float3   position;
    float    fov;
};

struct FrameConstants
{
    Camera camera;
    uint   frameIndex;
    uint   accumFrame;      // running count for progressive accumulation (0 = reset)
    float  time;
    uint   width;
    uint   height;
    uint   useDirectLight;  // 0 = IBL only; 1 = IBL + direct analytical light
    uint   useMesh;         // always 1 — triangle mesh pipeline (mesh_head.slang)
    float  tattooDensity;   // 0 = no ink; 1 = full pigment load at alpha-1 texels
    float  envIntensity;    // scalar on every sampleEnvironment() result
    uint   furnaceMode;     // 1 = energy-conservation test: unit sphere + white env + pink on violation
    float  mmPerUnit;       // world-scale factor: 1 world unit = mmPerUnit millimetres. Used to
                            // reconcile σ (mm⁻¹) with ray distances (world units) in the volume march.
    uint   detailFlags;     // bit0 = master enable; bit1 = normal avail; bit2 = roughness avail; bit3 = disp avail; bit4 = normals pre-baked into vertex data
    float  normalMapStrength;    // 0..1 scale on tangent-space XY before rebuilding +Z
    float  displacementScaleMM;  // millimetres per (disp - 0.5); 0 disables displacement offset
    uint   numInstances;         // size of the TLAS instance list; 0 disables mesh ray traversal
    uint   numSphereLights;      // active count in sphereLights buffer (binding 17)
    uint   numEmissiveTriangles; // active count in emissiveTriangles buffer (binding 18)
    uint   integratorType;       // 0 = unidirectional path tracer, 1 = BDPT (flat first-hit only)
    uint   numGizmoSegments;     // active count in gizmoSegments buffer (binding 22)
    uint   numLensElements;      // active count in lensElements buffer (binding 23); 0 = pinhole
    float  filmDistance;         // world-space distance from film plane to rear lens surface
    float  rearZ;                // world-space z of rear surface in lens-local frame
    float  rearAperture;         // world-space half-aperture (radius) of the rear element
    float  frontZ;               // world-space z of front (scene-side) surface; used to anchor
                                 // the lens so its front element sits at fc.camera.position
    float  filmHalfH;            // half the sensor height in world units (verticalAperture/2);
                                 // governs framing in the lens path (camera.fov is ignored there)
    float  irisZ;                // [unused; reserved] kept so UBO byte layout matches Python
    uint   numPupilBounds;       // active count in exitPupilBounds buffer (binding 24)
    float  filmDiagRadiusW;      // half of the film-plane diagonal in world units; bin index =
                                 //   floor(length(pFilm.xy) / filmDiagRadiusW · numPupilBounds)
    uint   focusOverlay;         // 0 = off; 1 = composite the translucent focal-plane visualiser
    float3 focusPlaneOrigin;     // world-space point on the focal plane (camera + forward·focusDist)
    float3 focusPlaneNormal;     // unit world-space normal of the focal plane
    float2 zoomMin;              // viewport sub-rect lower-left in [0, 1]² normalised pixel coords
    float2 zoomMax;              // viewport sub-rect upper-right; (0,0)→(1,1) = no zoom
    uint   lensVignetteDebug;    // 1 = render the per-pixel lens-ray success rate as grayscale
                                 //     instead of the integrator's radiance (black corners ⇒
                                 //     vignetting, full white ⇒ all rays survive)
};

/// Remap an integer pixel coordinate to NDC ∈ [-1, +1]² accounting for
/// the current zoom-rect. When the zoom rect is the full frame this
/// reduces to the canonical `pixel/dim · 2 − 1` mapping; otherwise the
/// sub-rect is stretched to fill the output. Use this everywhere an
/// integrator needs a primary-ray NDC.
float2 zoomedNDC(FrameConstants fc, uint2 pixel, float2 jitter)
{
    float2 uv = (float2(pixel) + 0.5 + jitter) / float2(fc.width, fc.height);
    float2 sub = lerp(fc.zoomMin, fc.zoomMax, uv);
    float2 ndc = sub * 2.0 - 1.0;
    ndc.y = -ndc.y;
    return ndc;
}

/// One spherical interface in a thick-lens stack (PBRT § 6.4).
/// Packed as float4 (radius, thickness, ior, halfAperture) — see
/// renderer.py::_sync_lens_buffer. Matches PBRT-v3's
/// `LensElementInterface` exactly: `thickness` is the axial distance
/// to the *next surface toward the film* (the rearmost element's
/// thickness is the film distance). `ior` is the medium *between this
/// surface and the next one toward the film*. `radius == 0` means a
/// planar interface (the aperture stop).
struct LensSurface
{
    float radius;
    float thickness;
    float ior;
    float halfAperture;
};

static const uint INTEGRATOR_PATH = 0u;
static const uint INTEGRATOR_BDPT = 1u;

// ── Random number generation (PCG) ──────────────────────────────

/// PCG-32 hash — one step of the PCG random-number generator [1].
///
///   state = input · 747796405 + 2891336453            (LCG advance)
///   word  = ((state >> ((state >> 28) + 4)) ^ state) · 277803737   (output XSH-RR)
///   hash  = (word >> 22) ^ word
///
/// Produces a statistically high-quality 32-bit hash with period 2³².
uint pcgHash(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct RNG
{
    uint state;

    [mutating]
    float next()
    {
        state = pcgHash(state);
        return float(state) / float(0xFFFFFFFFu);
    }

    [mutating]
    float2 next2()
    {
        return float2(next(), next());
    }
};

RNG createRNG(uint2 pixel, uint frame)
{
    RNG rng;
    rng.state = pcgHash(pixel.x + pixel.y * 65536u + frame * 16777259u);
    return rng;
}

// ── Sampling helpers ────────────────────────────────────────────

/// Cosine-weighted hemisphere sampling via Malley's method.
///
///   φ = 2π·u₁
///   cosθ = √u₂          → projection of uniform disk sample onto hemisphere
///   sinθ = √(1 − u₂)
///
/// Probability density:  p(ω) = cosθ / π
/// Used as the diffuse-BRDF importance sampler: weight = f_d · cosθ / p = 1/π · π = 1.
float3 sampleCosineHemisphere(float2 u, float3 normal)
{
    float phi      = 2.0 * PI * u.x;
    float cosTheta = sqrt(u.y);
    float sinTheta = sqrt(1.0 - u.y);

    // Build orthonormal basis around normal
    float3 tangent   = abs(normal.x) > 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 bitangent = normalize(cross(normal, tangent));
    tangent          = cross(bitangent, normal);

    return normalize(
        tangent * cos(phi) * sinTheta +
        bitangent * sin(phi) * sinTheta +
        normal * cosTheta
    );
}

/// PDF of cosine-weighted hemisphere sampling.
///
///   p(ω) = (n · ω) / π
///
/// where  n · ω = max(dot(normal, dir), 0)
float cosineHemispherePdf(float3 dir, float3 normal)
{
    float nDotW = max(dot(normalize(dir), normalize(normal)), 0.0);
    return nDotW / PI;
}

/// Uniform sphere sampling via equal-area mapping.
///
///   z   = 1 − 2·u₁              (uniform in [−1, 1])
///   r   = √(1 − z²)
///   φ   = 2π·u₂
///   dir = (r·cosφ, z, r·sinφ)
///
/// PDF: p(ω) = 1 / (4π)
float3 sampleUniformSphere(float2 u)
{
    float z = 1.0 - 2.0 * u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * u.y;
    return float3(r * cos(phi), z, r * sin(phi));
}

/// PDF of uniform sphere sampling.
///
///   p(ω) = 1 / (4π)
float uniformSpherePdf()
{
    return 1.0 / (4.0 * PI);
}

/// GGX Smith G₁ masking term (single-direction) [3].
///
///   G₁(v, α) = 2·(n·v) / ((n·v) + √(α² + (1 − α²)·(n·v)²))
///
/// where  α  = roughness²  (Disney remap)
///        v  = view / light direction
///        n·v = dot(shading normal, v)
/// The separable product G₁(v)·G₁(l) approximates the joint Smith G₂.
float ggxSmithG1(float NdotV, float roughness)
{
    float a = roughness * roughness;
    float k = a / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

/// Power heuristic for multiple importance sampling (MIS) [2].
///
///   w(pₐ, p_b) = pₐ² / (pₐ² + p_b²)       (β = 2)
///
/// Combines two sampling strategies A and B so their sum is an unbiased
/// estimator of the integrand with near-optimal variance reduction.
/// The β = 2 exponent minimises expected squared error (Veach §9.2).
float powerHeuristic(float pdfA, float pdfB)
{
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / max(a2 + b2, 1e-12);
}

void buildBasis(float3 n, out float3 t, out float3 b)
{
    t = abs(n.x) > 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    b = normalize(cross(n, t));
    t = cross(b, n);
}

float3 worldToTangent(float3 dir, float3 T, float3 B, float3 N)
{
    return float3(dot(dir, T), dot(dir, B), dot(dir, N));
}

float3 tangentToWorld(float3 dir, float3 T, float3 B, float3 N)
{
    return T * dir.x + B * dir.y + N * dir.z;
}

float3 sampleCosineHemisphereTS(float2 u)
{
    float phi      = 2.0 * PI * u.x;
    float cosTheta = sqrt(u.y);
    float sinTheta = sqrt(1.0 - u.y);
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

float cosineHemispherePdfTS(float3 wi)
{
    return max(wi.z, 0.0) / PI;
}

/// Uniform disk sampling via concentric mapping.
///
///   r   = √u₁              (area-preserving radial CDF inversion)
///   φ   = 2π·u₂
///   p   = (r·cosφ, r·sinφ)
///
/// PDF: p(x, y) = 1 / π  (uniform over unit disk)
float2 sampleUniformDisk(float2 u)
{
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    return float2(r * cos(phi), r * sin(phi));
}

// Camera ray generation lives in the ICamera implementations under
// `cameras/` (pinhole.slang and thick_lens.slang). main_pass.slang
// tag-switches on `fc.numLensElements` and dispatches through the
// interface; common.slang stays free of the lens-buffer binding.
