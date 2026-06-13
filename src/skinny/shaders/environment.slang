// HDR environment map — equirectangular lookup for image-based lighting.

import common;

// slang-rhi's Metal backend cannot bind a combined `Sampler2D` (it asserts
// "Unsupported binding type"), so on the Metal target every combined sampler is
// split into a plain `Texture2D` + its own `SamplerState`. Vulkan keeps the
// combined form, so its SPIR-V is byte-identical. `MAP_SAMPLE_LEVEL(T, uv, lod)`
// expands to `T.SampleLevel(T##Sampler, uv, lod)` on Metal and the unchanged
// `T.SampleLevel(uv, lod)` on Vulkan (macros do not cross `import`; defined per
// sampling file).
#if defined(SKINNY_METAL)
[[vk::binding(4)]]  Texture2D<float4> envMap;
[[vk::binding(39)]] SamplerState envMapSampler;
#define MAP_SAMPLE_LEVEL(T, UV, LOD)  T.SampleLevel(T##Sampler, UV, LOD)
#else
[[vk::binding(4)]] Sampler2D<float4> envMap;
#define MAP_SAMPLE_LEVEL(T, UV, LOD)  T.SampleLevel(UV, LOD)
#endif

// Direction → equirectangular UV. Convention: +y up, phi around y from +z.
float2 directionToEquirectUV(float3 dir)
{
    float3 d = normalize(dir);
    float phi   = atan2(d.x, d.z);                    // [-π, π]
    float theta = acos(clamp(d.y, -1.0, 1.0));        // [0, π], 0 at +y (zenith)
    return float2(phi / (2.0 * PI) + 0.5, theta / PI);
}

float3 sampleEnvironment(float3 dir, float intensity)
{
    float2 uv = directionToEquirectUV(dir);
    return MAP_SAMPLE_LEVEL(envMap, uv, 0.0).rgb * intensity;
}

// Unit-white fallback when furnace / energy-conservation mode is active.
// Every direction carries L = 1 per channel, which is what lets the furnace
// probe check outgoing vs. incoming radiance.
float3 sceneEnvRadiance(float3 dir, float intensity, uint furnaceMode)
{
    if (furnaceMode != 0u) return float3(1.0);
    return sampleEnvironment(dir, intensity);
}

// ── Environment importance sampling ─────────────────────────────────
// 2D piecewise-constant distribution over the equirect map, built CPU-side
// in environment.build_env_distribution() and uploaded at bindings 31/32.
// Sampling proportional to sin(theta)-weighted luminance lets next-event
// estimation find the bright sun/sky directly instead of relying on the BSDF
// happening to bounce a ray onto it — the fix for specular env fireflies.
static const int ENV_DIST_W = 1024;   // must equal ENV_WIDTH  (environment.py)
static const int ENV_DIST_H = 512;    // must equal ENV_HEIGHT (environment.py)

// Env importance-sampling CDFs share ONE buffer (change
// combine-graph-param-buffers): the marginal occupies elements
// [0, ENV_DIST_H+1) and the conditional follows at ENV_COND_CDF_BASE. Folding
// the two StructuredBuffers into one frees a buffer slot so the neural +
// online-training wavefront kernel fits Metal's 31-buffer argument table; the
// host concatenates the same bytes (renderer._upload_env_distribution).
[[vk::binding(31)]] StructuredBuffer<float> envDistCdf;  // [marginal | conditional]
static const int ENV_COND_CDF_BASE = ENV_DIST_H + 1;     // conditional element offset

// Piecewise-constant env radiance, consistent with the importance-sampling
// distribution. The CDF is built per cell (nearest-resized luminance), but the
// regular lookup filters bilinearly — so a direction one texel off the sun
// reads bright yet carries a low env pdf, and MIS fails to down-weight it
// (a firefly). Sampling at the texel CENTRE returns the exact cell value the
// CDF was built from, so L(dir) and envPdf(dir) agree and MIS is correct.
// Used only by the light-transport paths; the directly-visible background
// keeps the smooth bilinear lookup.
float3 sampleEnvironmentCell(float3 dir, float intensity)
{
    float2 uv = directionToEquirectUV(dir);
    int iu = clamp(int(uv.x * float(ENV_DIST_W)), 0, ENV_DIST_W - 1);
    int iv = clamp(int(uv.y * float(ENV_DIST_H)), 0, ENV_DIST_H - 1);
    float2 c = float2((float(iu) + 0.5) / float(ENV_DIST_W),
                      (float(iv) + 0.5) / float(ENV_DIST_H));
    return MAP_SAMPLE_LEVEL(envMap, c, 0.0).rgb * intensity;
}

float3 sceneEnvRadianceCell(float3 dir, float intensity, uint furnaceMode)
{
    if (furnaceMode != 0u) return float3(1.0);
    return sampleEnvironmentCell(dir, intensity);
}

struct EnvSample
{
    float3 dir;        // world-space sampled direction
    float3 radiance;   // L(dir) * intensity
    float  pdf;        // solid-angle pdf (0 ⇒ invalid / no distribution)
};

// Largest cell index i in [0, n-1] with cdf[i] <= u (cdf has n+1 entries at
// `base`). Returns i; `frac` is the in-cell position for stratified continuity.
int envSearchCdf(StructuredBuffer<float> cdf, int base, int n, float u, out float frac)
{
    int lo = 0;
    int hi = n - 1;
    [loop] while (lo < hi)
    {
        int mid = (lo + hi + 1) >> 1;
        if (cdf[base + mid] <= u) lo = mid; else hi = mid - 1;
    }
    float c0 = cdf[base + lo];
    float c1 = cdf[base + lo + 1];
    frac = (c1 > c0) ? saturate((u - c0) / (c1 - c0)) : 0.5;
    return lo;
}

float envImagePdf(int iu, int iv)
{
    float mPdf = (envDistCdf[iv + 1] - envDistCdf[iv]) * float(ENV_DIST_H);
    int   cb   = ENV_COND_CDF_BASE + iv * (ENV_DIST_W + 1);
    float cPdf = (envDistCdf[cb + iu + 1] - envDistCdf[cb + iu]) * float(ENV_DIST_W);
    return mPdf * cPdf;
}

// Solid-angle pdf of sampling `dir` under the env distribution. Used to
// MIS-weight the BSDF-sampled env-miss contribution against env NEE.
float envPdf(float3 dir)
{
    float2 uv = directionToEquirectUV(dir);
    int iu = clamp(int(uv.x * float(ENV_DIST_W)), 0, ENV_DIST_W - 1);
    int iv = clamp(int(uv.y * float(ENV_DIST_H)), 0, ENV_DIST_H - 1);
    // Use the direction's own theta (not the cell-centre) so this density
    // matches the one sampleEnvDir reports for the same direction; otherwise
    // the two MIS pdfs disagree and the power heuristic mis-weights.
    float st = sin(uv.y * PI);
    if (st <= 1e-6) return 0.0;
    return envImagePdf(iu, iv) / (2.0 * PI * PI * st);
}

// Importance-sample a direction from the environment distribution.
EnvSample sampleEnvDir(float2 u, float intensity)
{
    EnvSample es;
    es.dir = float3(0.0, 1.0, 0.0);
    es.radiance = float3(0.0);
    es.pdf = 0.0;

    float fv;
    int iv = envSearchCdf(envDistCdf, 0, ENV_DIST_H, u.x, fv);
    int cb = ENV_COND_CDF_BASE + iv * (ENV_DIST_W + 1);
    float fu;
    int iu = envSearchCdf(envDistCdf, cb, ENV_DIST_W, u.y, fu);

    float uImg = (float(iu) + fu) / float(ENV_DIST_W);
    float vImg = (float(iv) + fv) / float(ENV_DIST_H);
    float phi   = (uImg - 0.5) * 2.0 * PI;
    float theta = vImg * PI;
    float st = sin(theta);

    es.dir = float3(st * sin(phi), cos(theta), st * cos(phi));
    if (st <= 1e-6) return es;          // pole: degenerate solid angle
    es.pdf = envImagePdf(iu, iv) / (2.0 * PI * PI * st);
    es.radiance = sampleEnvironmentCell(es.dir, intensity);
    return es;
}
