// Common types and utilities shared across all shader modules.

static const float PI = 3.14159265358979;

struct Ray
{
    float3 origin;
    float3 direction;
};

struct HitInfo
{
    float  t;           // ray parameter at intersection
    float  backT;       // ray parameter at back-face exit (-1 = no back hit)
    float3 position;
    float3 normal;
    float3 tangent;     // surface tangent aligned with +U; bitangent = cross(N, T)
    float2 uv;
    bool   hit;
    bool   hasTangent;  // true when uv comes from an asset (not the fallback sphere map)
};

struct Camera
{
    float4x4 viewInverse;
    float4x4 projInverse;
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
    uint   useMesh;         // 0 = SDF head (sdf_head.slang); 1 = triangle mesh (mesh_head.slang)
    float  tattooDensity;   // 0 = no ink; 1 = full pigment load at alpha-1 texels
    uint   scatterMode;     // bit 0 = BSSRDF (point SSS), bit 1 = volume ray march
    float  envIntensity;    // scalar on every sampleEnvironment() result
    uint   furnaceMode;     // 1 = energy-conservation test: unit sphere + white env + pink on violation
    float  mmPerUnit;       // world-scale factor: 1 world unit = mmPerUnit millimetres. Used to
                            // reconcile σ (mm⁻¹) with ray distances (world units) in the volume march.
    uint   detailFlags;     // bit0 = master enable; bit1 = normal avail; bit2 = roughness avail; bit3 = disp avail; bit4 = normals pre-baked into vertex data
    float  normalMapStrength;    // 0..1 scale on tangent-space XY before rebuilding +Z
    float  displacementScaleMM;  // millimetres per (disp - 0.5); 0 disables displacement offset
    uint   integratorMode;       // 0 = path, 1 = MIS, 2 = volume light connections, 3 = stored light-subpath BDPT
};

// ── Random number generation (PCG) ──────────────────────────────

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

float cosineHemispherePdf(float3 dir, float3 normal)
{
    float nDotW = max(dot(normalize(dir), normalize(normal)), 0.0);
    return nDotW / PI;
}

float3 sampleUniformSphere(float2 u)
{
    float z = 1.0 - 2.0 * u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * u.y;
    return float3(r * cos(phi), z, r * sin(phi));
}

float uniformSpherePdf()
{
    return 1.0 / (4.0 * PI);
}

float powerHeuristic(float pdfA, float pdfB)
{
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / max(a2 + b2, 1e-12);
}

Ray generateCameraRay(FrameConstants fc, uint2 pixel, inout RNG rng)
{
    float2 jitter = rng.next2() - 0.5;
    float2 ndc    = (float2(pixel) + 0.5 + jitter) / float2(fc.width, fc.height) * 2.0 - 1.0;
    ndc.y         = -ndc.y;

    float4 target    = mul(fc.camera.projInverse, float4(ndc, 1.0, 1.0));
    float3 direction = mul(fc.camera.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz;

    Ray ray;
    ray.origin    = fc.camera.position;
    ray.direction = normalize(direction);
    return ray;
}
