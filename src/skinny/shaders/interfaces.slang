// Pluggable interface seams for materials, samplers, lights, estimators,
// and integrators.
//
// Dispatch policy:
//   - IMaterial: tag-switch monomorphization in material_eval.slang. Never
//     used as an existential value — divergent material hits in a warp
//     would serialize otherwise.
//   - IIntegrator: uniform per dispatch (selected by fc.integratorMode in
//     main_pass.slang); safe to keep as runtime switch.
//   - ISampler: generic parameter on estimators; compile-time monomorphized.
//   - ILight: generic parameter on estimators or loaded per-light from
//     buffer; compile-time monomorphized. Delta lights (directional) set
//     pdfArea=0 as a sentinel — callers skip geometry-term conversion.

import common;

// ── Sampling ────────────────────────────────────────────────────

interface ISampler
{
    float3 sampleDirection(float2 u);
    float  pdf(float3 L);
}

// ── Materials ───────────────────────────────────────────────────

interface IMaterial
{
    float3 evalRadiance(Ray ray, inout RNG rng);
}

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

// ── Generic dispatchers ─────────────────────────────────────────

float3 runEstimators<TM : IMaterial>(TM mat, Ray ray, inout RNG rng)
{
    return mat.evalRadiance(ray, rng);
}
