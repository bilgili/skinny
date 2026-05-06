// Flat-material orchestrator — standard_surface / UsdPreviewSurface bounce
// loop. Direct/area/emissive contributions live in estimators/flat_direct.
// Sampling helpers and the per-hit FlatHitData fetch live in
// flat_shading.slang. This file owns only the bounce-loop control flow
// + Russian-roulette + MIS bookkeeping for the next-event hit.

import common;
import bindings;
import interfaces;
import scene_trace;
import scene_lights;
import environment;
import mtlx_closures;
import mtlx_std_surface;
import flat_shading;
import estimators.flat_direct;

static const uint FLAT_MAX_BOUNCES = 6;

float3 evalFlatMaterial(HitInfo cameraHit, Ray cameraRay, inout RNG rng)
{
    float3 radiance   = float3(0.0);
    float3 throughput = float3(1.0);
    HitInfo h         = cameraHit;
    Ray r             = cameraRay;

    for (uint bounce = 0u; bounce < FLAT_MAX_BOUNCES; bounce++)
    {
        if (!h.hit)
        {
            radiance += throughput * sceneEnvRadiance(
                r.direction, fc.envIntensity, effectiveFurnaceMode(h.materialId)
            );
            break;
        }

        FlatHitData data = fetchFlatHitData(h);
        float3 N         = data.N;
        float3 V         = -r.direction;
        float3 T         = h.hasTangent ? h.tangent : float3(1, 0, 0);
        float NdotV      = max(dot(N, V), 1e-6);

        FlatHitMat m = data.mat;

        StdSurfaceParams sp = loadStdSurfaceParams(h.materialId, h.positionObject);
        {
            ProceduralParams pp = proceduralParams[h.materialId];
            if (pp.type != PROC_TYPE_NONE)
            {
                float3 procColor = evalProceduralColor(pp, h.positionObject);
                if (procColor.x >= 0.0)
                    m.albedo = procColor;
            }
        }

        if (m.opacity < 1.0)
        {
            float transmit = 1.0 - m.opacity;
            if (rng.next() < transmit)
            {
                bool entering = dot(r.direction, h.normal) < 0.0;
                float3 faceN = entering ? N : -N;
                float eta = entering ? (1.0 / m.ior) : m.ior;
                float cosI = abs(dot(V, faceN));
                float Fr = fresnelDielectric(cosI,
                    entering ? m.ior : (1.0 / m.ior));

                float3 newDir_;
                float3 offset;
                if (rng.next() < Fr)
                {
                    newDir_ = reflect(-V, faceN);
                    offset = faceN * 0.002;
                }
                else
                {
                    newDir_ = refract(-V, faceN, eta);
                    if (all(newDir_ == float3(0.0)))
                    {
                        newDir_ = reflect(-V, faceN);
                        offset = faceN * 0.002;
                    }
                    else
                    {
                        offset = -faceN * 0.002;
                    }
                }

                throughput *= m.albedo;

                r.origin    = h.position + offset;
                r.direction = newDir_;
                h = traceScene(fc, r);
                continue;
            }
        }

        if (bounce == 0u || fc.numEmissiveTriangles == 0u)
            radiance += throughput * m.emission;

        radiance += throughput * flatDirectEstimator(h, N, V, sp, fc, lightDirection, lightRadiance);
        radiance += throughput * flatSphereLightsEstimator(h, N, V, sp, fc, rng);
        radiance += throughput * flatEmissiveTrianglesEstimator(h, N, V, sp, fc, rng);
        float3 baseColor = m.albedo;
        float3 F0    = schlickF0(baseColor, m.metallic, m.specular, m.ior);
        float3 Fview = schlickFresnelColor(NdotV, F0);
        float3 diffHue = baseColor * (float3(1.0) - Fview) * (1.0 - m.metallic);
        float specLum = relativeLuminance(Fview);
        float diffLum = relativeLuminance(diffHue);
        float pSpec   = clamp(specLum / max(specLum + diffLum, 1e-6),
                              0.05, 0.95);
        float roughness = m.roughness;

        if (m.coat > 0.0)
        {
            float coatFresnel = fresnelDielectric(NdotV, m.coatIOR);
            float pCoat = m.coat * coatFresnel;
            if (rng.next() < pCoat)
            {
                ColoredSpecSample cs = sampleGGXSpecularColored(
                    N, V, m.coatRoughness, float3(1.0), rng.next2()
                );
                if (dot(cs.direction, N) <= 0.0 || all(cs.weight == float3(0.0)))
                    break;
                throughput *= cs.weight * m.coatColor;
                r.origin    = h.position + h.normal * 0.002;
                r.direction = cs.direction;
                h = traceScene(fc, r);
                continue;
            }
            throughput *= m.coatColor;
        }

        float u = rng.next();

        float3 newDir;
        bool   valid = true;
        float3 bounceWeight;

        if (u < pSpec)
        {
            ColoredSpecSample s = sampleGGXSpecularColored(
                N, V, roughness, F0, rng.next2()
            );
            newDir = s.direction;
            if (dot(newDir, N) <= 0.0 || all(s.weight == float3(0.0)))
                valid = false;
            else
                bounceWeight = s.weight / pSpec;
        }
        else
        {
            newDir = sampleCosineHemisphere(rng.next2(), N);
            bounceWeight = diffHue / max(1.0 - pSpec, 1e-6);
        }

        if (!valid)
            break;

        float NdotL = max(dot(N, newDir), 0.0);
        float diff_pdf = NdotL / PI;
        float3 H    = normalize(newDir + V);
        float NdotH = max(dot(N, H), 0.0);
        float VdotH = max(dot(V, H), 0.0);
        float a     = max(roughness * roughness, 1e-4);
        float a2    = a * a;
        float den   = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D_ggx = a2 / max(PI * den * den, 1e-6);
        float spec_pdf = D_ggx * NdotH / max(4.0 * VdotH, 1e-6);
        float samplePdf = pSpec * spec_pdf + (1.0 - pSpec) * diff_pdf;

        throughput *= bounceWeight;

        if (bounce > 0u)
        {
            float p = clamp(
                max(max(throughput.x, throughput.y), throughput.z), 0.05, 0.95
            );
            if (rng.next() > p)
                break;
            throughput /= p;
        }

        r.origin    = h.position + h.normal * 0.002;
        r.direction = newDir;

        NearestSphereHit sph = intersectSphereLights(r, 1e30, fc.numSphereLights);
        HitInfo hNext = traceScene(fc, r);
        float geom_t = hNext.hit ? hNext.t : 1e30;
        if (sph.idx != 0xFFFFFFFFu && sph.t < geom_t)
        {
            SphereLight light = sphereLights[sph.idx];
            float area         = 4.0 * PI * light.radius * light.radius;
            float pdf_light_sa = (sph.t * sph.t) /
                                 max(sph.cos_y * area, 1e-12);
            float w_bsdf = (samplePdf > 0.0)
                ? powerHeuristic(samplePdf, pdf_light_sa)
                : 0.0;
            radiance += throughput * light.radiance * w_bsdf;
            break;
        }
        h = hNext;
    }

    return radiance;
}

// ── IMaterial adapter ───────────────────────────────────────────

struct FlatMaterial : IMaterial
{
    HitInfo hit;

    float3 evalRadiance(Ray ray, inout RNG rng)
    {
        return evalFlatMaterial(hit, ray, rng);
    }
}

FlatMaterial loadFlat(HitInfo h)
{
    FlatMaterial m;
    m.hit = h;
    return m;
}
