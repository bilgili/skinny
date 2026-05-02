// Pluggable interface seams for materials, samplers, estimators, and
// integrators.
//
// Dispatch policy:
//   - IMaterial: tag-switch monomorphization in material_eval.slang. Never
//     used as an existential value — divergent material hits in a warp
//     would serialize otherwise.
//   - IIntegrator: uniform per dispatch (selected by fc.integratorMode in
//     main_pass.slang); safe to keep as runtime switch.
//   - ISampler: generic parameter on estimators; compile-time monomorphized.

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

// ── Generic dispatchers ─────────────────────────────────────────

float3 runEstimators<TM : IMaterial>(TM mat, Ray ray, inout RNG rng)
{
    return mat.evalRadiance(ray, rng);
}
