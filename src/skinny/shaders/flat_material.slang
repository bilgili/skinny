// Flat-material BSDF — standard_surface / UsdPreviewSurface.
// Provides sample/evaluate via IMaterial interface in tangent space (N=+Z).
// The bounce loop lives in PathTracer (integrators/path.slang).

import common;
import bindings;
import interfaces;
import mtlx_closures;
import mtlx_std_surface;
import flat_shading;

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

        // ── Coat layer ──
        float3 coatAttenuation = float3(1.0);
        if (m.coat > 0.0)
        {
            float coatFresnel = fresnelDielectric(NdotV, m.coatIOR);
            float pCoat = m.coat * coatFresnel;
            if (rng.next() < pCoat)
            {
                ColoredSpecSample cs = sampleGGXSpecularColored(
                    float3(0, 0, 1), wo, m.coatRoughness, float3(1.0), rng.next2()
                );
                s.wi    = cs.direction;
                s.weight = cs.weight * m.coatColor;
                s.valid  = cs.direction.z > 0.0
                         && !all(cs.weight == float3(0.0));
                return s;
            }
            coatAttenuation = m.coatColor;
        }

        // ── Spec / diffuse MIS split ──
        float3 Fview   = schlickFresnelColor(NdotV, F0);
        float3 diffHue = m.albedo * (float3(1.0) - Fview) * (1.0 - m.metallic);
        float specLum  = relativeLuminance(Fview);
        float diffLum  = relativeLuminance(diffHue);
        float pSpec    = clamp(specLum / max(specLum + diffLum, 1e-6),
                               0.05, 0.95);
        float roughness = m.roughness;

        float u = rng.next();
        float3 wi;
        float3 bounceWeight;

        if (u < pSpec)
        {
            ColoredSpecSample ss = sampleGGXSpecularColored(
                float3(0, 0, 1), wo, roughness, F0, rng.next2()
            );
            wi = ss.direction;
            if (wi.z <= 0.0 || all(ss.weight == float3(0.0)))
                return s;
            bounceWeight = ss.weight / pSpec;
        }
        else
        {
            wi           = sampleCosineHemisphereTS(rng.next2());
            bounceWeight = diffHue / max(1.0 - pSpec, 1e-6);
        }

        float  NdotL    = max(wi.z, 0.0);
        float  diff_pdf = NdotL / PI;
        float3 H        = normalize(wi + wo);
        float  NdotH    = max(H.z, 0.0);
        float  VdotH    = max(dot(wo, H), 0.0);
        float  a        = max(roughness * roughness, 1e-4);
        float  a2       = a * a;
        float  den      = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float  D_ggx    = a2 / max(PI * den * den, 1e-6);
        float  spec_pdf = D_ggx * NdotH / max(4.0 * VdotH, 1e-6);
        float  samplePdf = pSpec * spec_pdf + (1.0 - pSpec) * diff_pdf;

        s.wi     = wi;
        s.weight = bounceWeight * coatAttenuation;
        s.pdf    = samplePdf;
        s.valid  = true;
        return s;
    }

    BSDFEval evaluate(float3 wo, float3 wi)
    {
        BSDFEval e;
        ClosureData cd = makeClosureData(
            CLOSURE_TYPE_REFLECTION, wi, wo,
            float3(0, 0, 1), float3(0), 1.0);
        BSDF bsdf  = evalStdSurfaceBSDF(cd, sp, float3(0, 0, 1), float3(1, 0, 0));
        e.response = bsdf.response;
        e.pdf      = stdSurfaceSamplingPdf(float3(0, 0, 1), wo, wi, sp);
        return e;
    }
}

FlatMaterial loadFlatMaterial(HitInfo h)
{
    FlatMaterial m;
    m.data = fetchFlatHitData(h);
    m.sp   = loadStdSurfaceParams(h.materialId, h.positionObject);

    ProceduralParams pp = proceduralParams[h.materialId];
    if (pp.type != PROC_TYPE_NONE)
    {
        float3 procColor = evalProceduralColor(pp, h.positionObject);
        if (procColor.x >= 0.0)
            m.data.mat.albedo = procColor;
    }

    m.F0 = schlickF0(m.data.mat.albedo, m.data.mat.metallic,
                     m.data.mat.specular, m.data.mat.ior);
    return m;
}
