// Shared skin-material types, resource bindings, and BSDF evaluation
// helpers. Pulled out of skin_material.slang so estimators (under
// shaders/estimators/) can import this module without creating a cycle
// with the orchestrator. skin_material.slang also imports this; the
// orchestrator itself owns nothing beyond the chained estimator calls.

import common;
import skin_bssrdf;
import skinny_skin_layered_bsdf_genslang;

// ── Skin-specific resource bindings ─────────────────────────────

[[vk::binding(8)]]  Sampler2D<float4>              tattooMap;
[[vk::binding(9)]]  Sampler2D<float4>              normalMap;
[[vk::binding(10)]] Sampler2D<float4>              roughnessMap;
[[vk::binding(11)]] Sampler2D<float4>              displacementMap;
[[vk::binding(15)]] StructuredBuffer<MtlxSkinParams> mtlxSkin;

// ── Closure-type tags for the genslang BSDF ─────────────────────

static const int SKINNY_CLOSURE_REFLECTION   = 1;
static const int SKINNY_CLOSURE_TRANSMISSION = 2;
static const int SKINNY_CLOSURE_INDIRECT     = 3;

// ── Helpers ─────────────────────────────────────────────────────

float4 inkAtHit(HitInfo h, float tattooDensity)
{
    float4 s = tattooMap.SampleLevel(h.uv, 0.0);
    return float4(s.rgb, s.a * tattooDensity);
}

struct SkinEvalContext
{
    SkinParams       sp;
    MtlxSkinParams   mtlx;
    SkinLayerStack   layers;
    float4           ink;
    float3           Nspec;
    float            roughSpec;
    bool             useBSSRDF;
};

float3 evalSkinBSDF(
    float3 L, float3 V, float3 P, float3 T,
    SkinEvalContext ctx)
{
    ClosureData cd = makeClosureData(
        SKINNY_CLOSURE_REFLECTION, L, V, ctx.Nspec, P, 1.0);
    BSDF bsdf;
    bsdf.response   = float3(0.0);
    bsdf.throughput = float3(1.0);
    skinny_skin_layered_bsdf(
        cd,
        ctx.layers.epi.sigmaA, ctx.layers.epi.sigmaS, ctx.layers.epi.g,
        ctx.layers.epi.thickness, ctx.layers.epi.ior,
        ctx.layers.der.sigmaA, ctx.layers.der.sigmaS, ctx.layers.der.g,
        ctx.layers.der.thickness, ctx.layers.der.ior,
        ctx.layers.sub.sigmaA, ctx.layers.sub.sigmaS, ctx.layers.sub.g,
        ctx.layers.sub.thickness, ctx.layers.sub.ior,
        ctx.ink,
        ctx.mtlx.skin_bsdf_weight,
        ctx.roughSpec,
        ctx.mtlx.skin_bsdf_ior,
        ctx.Nspec, T,
        0.0, 0.0, 0.0, 0.0,
        bsdf);
    return bsdf.response;
}
