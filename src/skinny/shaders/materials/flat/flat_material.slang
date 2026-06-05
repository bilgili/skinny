// Flat-material BSDF — standard_surface / UsdPreviewSurface.
// Provides sample/evaluate via IMaterial interface in tangent space (N=+Z).
// The bounce loop lives in PathTracer (integrators/path.slang).

import common;
import bindings;
import interfaces;
import materials.flat.flat_shading;
import materials.flat.flat_lobes;
import generated_materials;

struct FlatMaterial : IMaterial
{
    FlatHitData data;
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

        // Per-lobe sampler ids (runtime selection in fc.flatLobeSamplers). The
        // SAME ids feed flatBsdfPdf here and in evaluate(), so the draw density
        // and the weight/MIS density agree for any registered strategy.
        uint coatSId = flatLobeSamplerId(fc.flatLobeSamplers, LOBE_COAT);
        uint specSId = flatLobeSamplerId(fc.flatLobeSamplers, LOBE_SPEC);
        uint diffSId = flatLobeSamplerId(fc.flatLobeSamplers, LOBE_DIFFUSE);

        float3 wi;
        float3 bounceWeight;

        // ── Coat lobe (selected with probability pCoat) ──
        if (m.coat > 0.0 && rng.next() < pCoat)
        {
            wi = flatSampleLobe(LOBE_COAT, coatSId, m.coatRoughness,
                                wo, rng.next2());
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
                                   pSpec, roughness, coatSId, specSId, diffSId);
            s.valid  = true;
            return s;
        }

        // ── Base spec / diffuse MIS split ──
        float u = rng.next();
        if (u < pSpec)
        {
            wi = flatSampleLobe(LOBE_SPEC, specSId, roughness,
                                wo, rng.next2());
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
            wi = flatSampleLobe(LOBE_DIFFUSE, diffSId, 0.0,
                                wo, rng.next2());
            float3 Fview   = schlickFresnelColor(NdotV, F0);
            float3 diffHue = m.albedo * (float3(1.0) - Fview)
                           * (1.0 - m.metallic);
            // Cosine sampling importance-samples cos so it cancels
            // (weight = diffHue); uniform-hemisphere does not, leaving
            // f·cos/pdf = 2·NdotL·diffHue — still bounded (≤ 2·diffHue).
            float diffScale = (diffSId == FLAT_SAMPLER_UNIFORM) ? (2.0 * wi.z) : 1.0;
            bounceWeight = diffScale * diffHue / max(1.0 - pSpec, 1e-6);
        }

        s.wi     = wi;
        s.weight = bounceWeight * coatAttenuation;
        s.pdf    = flatBsdfPdf(wo, wi, pCoat, m.coatRoughness,
                               pSpec, roughness, coatSId, specSId, diffSId);
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

        // Unified lobe BSDF — the SAME 3-lobe model, params, and per-lobe
        // sampler ids FlatMaterial.sample() draws from, so e.pdf == sample().pdf
        // and e.response / e.pdf reduces to the bounded per-lobe weight.
        // flatBsdfResponse (= f·cos) is sampler-invariant; only e.pdf reads the
        // selected ids (the diffuse density differs cosine vs uniform).
        FlatHitMat m       = data.mat;
        float NdotV        = max(wo.z, 1e-6);
        float coatFresnel  = (m.coat > 0.0) ? fresnelDielectric(NdotV, m.coatIOR) : 0.0;
        float pCoat        = m.coat * coatFresnel;
        float pSpec        = flatLobePSpec(NdotV, m.albedo, m.metallic, F0);

        uint coatSId = flatLobeSamplerId(fc.flatLobeSamplers, LOBE_COAT);
        uint specSId = flatLobeSamplerId(fc.flatLobeSamplers, LOBE_SPEC);
        uint diffSId = flatLobeSamplerId(fc.flatLobeSamplers, LOBE_DIFFUSE);

        e.response = flatBsdfResponse(wo, wi, m, F0) * op;
        e.pdf      = flatBsdfPdf(wo, wi, pCoat, m.coatRoughness, pSpec, m.roughness,
                                 coatSId, specSId, diffSId) * op;
        return e;
    }
}

FlatMaterial loadFlatMaterial(HitInfo h)
{
    FlatMaterial m;
    m.data = fetchFlatHitData(h);

    // Graph dispatch onto the flat path's albedo so the lobe model picks up
    // the same color a connected diffuseColor / base_color node drives. The
    // helper returns false for graphs that drive only other inputs (brass:
    // specular_roughness + coat_*), in which case the SSBO-uploaded constant
    // is the correct albedo.
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

    // fetchFlatHitData already sampled any diffuseColor texture into
    // data.mat.albedo, so the lobe model (sample + evaluate) sees the
    // textured color directly — no std_surface param mirror needed.
    m.F0 = schlickF0(m.data.mat.albedo, m.data.mat.metallic,
                     m.data.mat.specular, m.data.mat.ior);
    return m;
}
