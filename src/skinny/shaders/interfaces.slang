// Pluggable interface seams for materials, samplers, lights, and integrators.
//
// Dispatch policy:
//   - IMaterial: tag-switch monomorphization in PathTracer's evaluateBounce().
//     Never used as an existential value — divergent material hits in a warp
//     would serialize otherwise.  All directions in tangent space (N=+Z).
//   - IIntegrator: PathTracer with MIS.
//   - ISampler: tangent-space sampler (N=+Z); generic parameter on estimators,
//     compile-time monomorphized. Callers transform world↔tangent.
//   - ILight: generic parameter on estimators or loaded per-light from
//     buffer; compile-time monomorphized. Delta lights (directional) set
//     pdfArea=0 as a sentinel — callers skip geometry-term conversion.

import common;

// ── Sampling ────────────────────────────────────────────────────

interface ISampler
{
    float3 sample(float3 wo, float2 uv);
    float  pdf(float3 wi, float3 wo);
}

// ── Materials ───────────────────────────────────────────────────

struct BSDFSample
{
    float3 wi;            // sampled direction in tangent space (N=+Z)
    float3 weight;        // BSDF * cos / pdf
    float  pdf;           // solid-angle PDF for BSDF-path MIS
    float3 emission;      // self-emission at this surface point
    bool   valid;
    bool   transmitted;   // refraction — caller offsets ray along −N
};

struct BSDFEval
{
    float3 response;      // f(wo, wi) * cos(theta_i)
    float  pdf;           // solid-angle sampling PDF for this wi
};

interface IMaterial
{
    BSDFSample sample(float3 wo, inout RNG rng);
    BSDFEval evaluate(float3 wo, float3 wi);
}

struct BounceResult
{
    float3 directLight;    // NEE contribution
    float3 fullRadiance;   // self-integrating materials (skin, debug)
    BSDFSample bsdfSample; // wi converted to world space by evaluateBounce
};

// ── Integrators ─────────────────────────────────────────────────

interface IIntegrator
{
    float3 estimateRadiance(Ray ray, HitInfo firstHit, inout RNG rng);
}

// ── Lights ──────────────────────────────────────────────────────

struct LightSample
{
    float3 point;       // world-space sampled point on light surface
    float3 normal;      // outward surface normal at sampled point
    float3 radiance;    // emitted radiance
    float  pdfArea;     // area-measure probability density of this sample
    bool   valid;       // false = degenerate / no contribution
};

interface ILight
{
    LightSample samplePoint(float3 shadingPos, float2 u);
    float pdfSolidAngle(float3 shadingPos, float3 direction);
}

// ── Cameras ─────────────────────────────────────────────────────
//
// Pluggable camera model (PBRT CameraBase analogue).
//   - `generateRay`: primary eye ray for pixel `pixel` (PBRT GenerateRay).
//      `weight` is the per-ray throughput multiplier (1.0 for an
//      ideal pinhole; cos⁴θ vignetting for the realistic lens path).
//   - `sampleWi`: project a world point onto the film for BDPT s=1
//      light-tracer splatting (PBRT SampleWi). Returns false when the
//      point is off-frame or blocked by an aperture / TIR.
//
// Concrete cameras read uniform `fc` (bindings.slang) directly; the
// struct itself carries no state. Dispatch is by compile-time
// generic parameter on integrator entry points — same monomorphization
// pattern as IMaterial/ISampler.

interface ICamera
{
    Ray  generateRay(uint2 pixel, inout RNG rng, out float weight);
    bool sampleWi(float3 p, inout RNG rng,
                  out int2 outPixel, out float cosCam, out float distToCam);
}
