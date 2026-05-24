// Flat-material BSDF — standard_surface / UsdPreviewSurface.
// Provides sample/evaluate via IMaterial interface in tangent space (N=+Z).
// The bounce loop lives in PathTracer (integrators/path.slang).

import common;
import bindings;
import interfaces;
import mtlx_closures;
import mtlx_std_surface;
import materials.flat.flat_shading;
import generated_materials;
import samplers.ggx;
import samplers.lambert;

// MIS pdf across coat + base spec + base diffuse lobes. Each lobe samples
// via its ISampler implementor; weights by lobe-selection probabilities.
float flatBsdfPdf(float3 wo, float3 wi,
                  float pCoat, float coatRoughness,
                  float pSpec, float baseRoughness)
{
    GGXSampler     coatSampler; coatSampler.roughness = coatRoughness;
    GGXSampler     specSampler; specSampler.roughness = baseRoughness;
    LambertSampler diffSampler;
    float basePdf = pSpec * specSampler.pdf(wi, wo)
                  + (1.0 - pSpec) * diffSampler.pdf(wi, wo);
    return pCoat * coatSampler.pdf(wi, wo) + (1.0 - pCoat) * basePdf;
}

// Lobe-selection probability from colored Fresnel / albedo balance.
float flatLobePSpec(float NdotV, float3 albedo, float metallic, float3 F0)
{
    float3 Fview   = schlickFresnelColor(NdotV, F0);
    float3 diffHue = albedo * (float3(1.0) - Fview) * (1.0 - metallic);
    float specLum  = relativeLuminance(Fview);
    float diffLum  = relativeLuminance(diffHue);
    return clamp(specLum / max(specLum + diffLum, 1e-6), 0.05, 0.95);
}

struct FlatMaterial : IMaterial
{
    FlatHitData data;
    StdSurfaceParams sp;
    float3 F0;

    BSDFSample sample(float3 wo, inout RNG rng)
    {
        BSDFSample s;
        s.valid       = false;
        s.transmitted = false;
        s.pdf         = 0.0;
        s.emission    = data.mat.emission;
        s.wi          = float3(0.0);
        s.weight      = float3(0.0);
        s.response    = float3(0.0);

        FlatHitMat m = data.mat;
        float NdotV  = max(wo.z, 1e-6);

        // ── Opacity / refraction ──
        if (m.opacity < 1.0)
        {
            float transmit = 1.0 - m.opacity;
            if (rng.next() < transmit)
            {
                bool entering  = wo.z > 0.0;
                float3 faceN   = entering ? float3(0, 0, 1) : float3(0, 0, -1);
                float eta      = entering ? (1.0 / m.ior) : m.ior;
                float cosI     = abs(dot(wo, faceN));
                float Fr       = fresnelDielectric(cosI, eta);

                float3 newDir;
                if (rng.next() < Fr)
                {
                    newDir = reflect(-wo, faceN);
                }
                else
                {
                    newDir = refract(-wo, faceN, eta);
                    if (all(newDir == float3(0.0)))
                        newDir = reflect(-wo, faceN);
                }

                s.wi          = newDir;
                s.weight      = m.albedo;
                s.pdf         = 0.0;
                s.valid       = true;
                s.transmitted = newDir.z < 0.0;
                return s;
            }
        }

        // ── Lobe-selection probabilities ──
        float coatFresnel  = (m.coat > 0.0)
                           ? fresnelDielectric(NdotV, m.coatIOR) : 0.0;
        float pCoat        = m.coat * coatFresnel;
        float roughness    = m.roughness;
        float pSpec        = flatLobePSpec(NdotV, m.albedo, m.metallic, F0);
        float3 coatAttenuation = (m.coat > 0.0) ? m.coatColor : float3(1.0);

        GGXSampler     coatSampler; coatSampler.roughness = m.coatRoughness;
        GGXSampler     specSampler; specSampler.roughness = roughness;
        LambertSampler diffSampler;

        float3 wi;
        float3 bounceWeight;

        // ── Coat lobe (selected with probability pCoat) ──
        if (m.coat > 0.0 && rng.next() < pCoat)
        {
            wi = coatSampler.sample(wo, rng.next2());
            if (wi.z <= 0.0)
                return s;

            float3 H     = normalize(wi + wo);
            float  NdotL = wi.z;
            float  VdotH = max(dot(wo, H), 0.0);

            // VNDF importance sampling: BRDF×cosθ / pdf reduces to
            // F · G₁(L) (separable Smith G₂). Bounded above by 1.
            float r    = (1.0 - m.coatIOR) / (1.0 + m.coatIOR);
            float F0c  = r * r;
            float F    = schlickFresnelColor(VdotH, float3(F0c)).r;
            float G1L  = ggxSmithG1(NdotL, m.coatRoughness);
            float3 weight = m.coatColor * (F * G1L);
            if (all(weight == float3(0.0)))
                return s;

            s.wi     = wi;
            s.weight = weight;
            s.pdf    = flatBsdfPdf(wo, wi, pCoat, m.coatRoughness,
                                   pSpec, roughness);
            s.valid  = true;
            return s;
        }

        // ── Base spec / diffuse MIS split ──
        float u = rng.next();
        if (u < pSpec)
        {
            wi = specSampler.sample(wo, rng.next2());
            if (wi.z <= 0.0)
                return s;

            float3 H     = normalize(wi + wo);
            float  NdotL = wi.z;
            float  VdotH = max(dot(wo, H), 0.0);

            // VNDF importance sampling: BRDF×cosθ / pdf reduces to
            // F · G₁(L) for separable Smith G₂. The classical
            // D(H)-of-microfacet sampler this replaced left
            // ``VdotH / (NdotV·NdotH)`` in the weight — a singular
            // factor at grazing/near-horizon microfacets that
            // tools/firefly_debug.py traced as the spec-lobe firefly
            // source (mode 5 peak ≈ 4e4 per bounce, compounding to
            // ~1e16 throughput over six bounces).
            float3 F = schlickFresnelColor(VdotH, F0);
            float  G1L = ggxSmithG1(NdotL, roughness);
            float3 specWeight = F * G1L;
            if (all(specWeight == float3(0.0)))
                return s;
            bounceWeight = specWeight / pSpec;
        }
        else
        {
            wi = diffSampler.sample(wo, rng.next2());
            float3 Fview   = schlickFresnelColor(NdotV, F0);
            float3 diffHue = m.albedo * (float3(1.0) - Fview)
                           * (1.0 - m.metallic);
            bounceWeight = diffHue / max(1.0 - pSpec, 1e-6);
        }

        s.wi     = wi;
        s.weight = bounceWeight * coatAttenuation;
        s.pdf    = flatBsdfPdf(wo, wi, pCoat, m.coatRoughness,
                               pSpec, roughness);
        s.valid  = true;
        return s;
    }

    BSDFSample evaluate(float3 wo, float3 wi)
    {
        BSDFSample e;
        e.valid       = false;
        e.transmitted = false;
        e.wi          = float3(0.0);
        e.weight      = float3(0.0);
        e.emission    = float3(0.0);
        // Opacity gates the reflective/diffuse lobes — sample() routes
        // (1 - opacity) of evaluations to the delta transmission branch,
        // so the non-delta response must scale accordingly. Without this,
        // NEE lights up fully-transparent surfaces (e.g. glass, opacity=0).
        float op = data.mat.opacity;
        if (op <= 0.0)
        {
            e.response = float3(0.0);
            e.pdf      = 0.0;
            return e;
        }
        ClosureData cd = makeClosureData(
            CLOSURE_TYPE_REFLECTION, wi, wo,
            float3(0, 0, 1), float3(0), 1.0);
        BSDF bsdf  = evalStdSurfaceBSDF(cd, sp, float3(0, 0, 1), float3(1, 0, 0));
        e.response = bsdf.response * op;

        float  NdotV = max(wo.z, 1e-6);
        float3 baseColor = sp.base_color * sp.base;
        float3 evalF0 = schlickF0(baseColor, sp.metalness,
                                  sp.specular, sp.specular_IOR);
        float  pSpec  = flatLobePSpec(NdotV, baseColor, sp.metalness, evalF0);
        float  rough  = max(sp.specular_roughness, 0.04);
        float  coatRough = max(sp.coat_roughness, 0.04);
        float  pCoat = sp.coat * fresnelDielectric(NdotV, sp.coat_IOR);
        e.pdf = flatBsdfPdf(wo, wi, pCoat, coatRough, pSpec, rough) * op;
        return e;
    }
}

FlatMaterial loadFlatMaterial(HitInfo h)
{
    FlatMaterial m;
    m.data = fetchFlatHitData(h);
    m.sp   = loadStdSurfaceParams(h.materialId, h.positionObject,
                                   h.normal, h.tangent, h.uv);

    // Mirror loadStdSurfaceParams's graph dispatch onto the flat path's
    // albedo so direct lobes (used by flatBsdfPdf / FlatMaterial.sample)
    // pick up the same color the std_surface BSDF sees. The helper
    // returns false for graphs that drive only other std_surface inputs
    // (brass: specular_roughness + coat_*), in which case the SSBO-
    // uploaded constant is the correct albedo.
    uint graphId = materialGraphId(h.materialId);
    if (graphId >= 2u)
    {
        float3 graphColor;
        if (evalSceneGraphBaseColor(graphId, h.materialId,
                                     h.positionObject, h.normal, h.tangent,
                                     h.uv, graphColor))
        {
            m.data.mat.albedo = graphColor;
        }
    }

    // UsdPreviewSurface authors `diffuseColor` connected to a UsdUVTexture.
    // pack_std_surface_params can't sample textures on the CPU, so it falls
    // back to the constant default (0.8 grey) for textured materials.
    // fetchFlatHitData already sampled the texture for the diffuse lobe;
    // copy that color into the std_surface params so FlatMaterial.evaluate's
    // BSDF response (used by NEE) matches the textured albedo sample() uses.
    m.sp.base_color = m.data.mat.albedo;

    m.F0 = schlickF0(m.data.mat.albedo, m.data.mat.metallic,
                     m.data.mat.specular, m.data.mat.ior);
    return m;
}
