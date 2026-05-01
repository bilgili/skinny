// Flat-material evaluator — standard_surface / UsdPreviewSurface bounce loop.
// Extracted from main_pass.slang so the integrator dispatch is material-agnostic.

import common;
import bindings;
import scene_trace;
import scene_lights;
import environment;
import mtlx_closures;
import mtlx_std_surface;
import mtlx_noise;

// ── Helpers ─────────────────────────────────────────────────────

static const uint FLAT_MAX_BOUNCES = 6;

float relativeLuminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float3 schlickF0(float3 albedo, float metallic, float specular, float ior)
{
    float r0 = (1.0 - ior) / (1.0 + ior);
    float dielectricF0 = r0 * r0 * specular;
    return lerp(float3(dielectricF0), albedo, metallic);
}

float3 schlickFresnelColor(float cosTheta, float3 F0)
{
    float c = clamp(1.0 - cosTheta, 0.0, 1.0);
    float c5 = c * c * c * c * c;
    return F0 + (float3(1.0) - F0) * c5;
}

struct ColoredSpecSample
{
    float3 direction;
    float3 weight;
};

ColoredSpecSample sampleGGXSpecularColored(
    float3 N, float3 V, float roughness, float3 F0, float2 u
)
{
    float a = max(roughness * roughness, 1e-4);
    float a2 = a * a;
    float cosTheta = sqrt((1.0 - u.y) / (1.0 + (a2 - 1.0) * u.y));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi      = 2.0 * PI * u.x;

    float3 T, B;
    buildBasis(N, T, B);
    float3 H = normalize(
        T * cos(phi) * sinTheta +
        B * sin(phi) * sinTheta +
        N * cosTheta
    );
    float3 L = reflect(-V, H);

    ColoredSpecSample s;
    s.direction = L;

    float NdotL = dot(N, L);
    if (NdotL <= 0.0)
    {
        s.weight = float3(0.0);
        return s;
    }

    float NdotV = max(dot(N, V), 1e-6);
    float NdotH = max(cosTheta, 1e-6);
    float VdotH = max(dot(V, H), 1e-6);

    float G  = ggxSmithG1(NdotV, roughness) * ggxSmithG1(NdotL, roughness);
    float3 F = schlickFresnelColor(VdotH, F0);
    s.weight = F * (G * VdotH / (NdotV * NdotH));
    return s;
}

float fresnelDielectric(float cosI, float eta)
{
    float sinT2 = eta * eta * (1.0 - cosI * cosI);
    if (sinT2 > 1.0)
        return 1.0;
    float cosT = sqrt(max(1.0 - sinT2, 0.0));
    float rS = (eta * cosI - cosT) / (eta * cosI + cosT);
    float rP = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5 * (rS * rS + rP * rP);
}

struct FlatHitMat
{
    float3 albedo;
    float  roughness;
    float  metallic;
    float  specular;
    float  ior;
    float  opacity;
    float3 emission;
    float  coat;
    float  coatRoughness;
    float  coatIOR;
    float3 coatColor;
};

struct FlatHitData
{
    FlatHitMat mat;
    float3 N;
};

FlatHitData fetchFlatHitData(HitInfo h)
{
    FlatHitData out_;
    if (h.materialId == 0u)
    {
        out_.mat.albedo       = float3(0.7);
        out_.mat.roughness    = 0.5;
        out_.mat.metallic     = 0.0;
        out_.mat.specular     = 0.5;
        out_.mat.ior          = 1.4;
        out_.mat.opacity      = 1.0;
        out_.mat.emission     = float3(0.0);
        out_.mat.coat         = 0.0;
        out_.mat.coatRoughness = 0.0;
        out_.mat.coatIOR      = 1.5;
        out_.mat.coatColor    = float3(1.0);
        out_.N                = h.normal;
        return out_;
    }

    FlatMaterialParams p = flatMaterials[h.materialId];

    out_.mat.albedo = p.diffuseColor;
    if (p.diffuseTextureIdx != 0xFFFFFFFFu)
    {
        out_.mat.albedo = flatMaterialTextures[NonUniformResourceIndex(p.diffuseTextureIdx)]
                              .SampleLevel(h.uv, 0.0).rgb;
    }

    out_.mat.roughness = clamp(p.roughness, 0.04, 1.0);
    if (p.roughnessTextureIdx != 0xFFFFFFFFu)
    {
        float r = flatMaterialTextures[NonUniformResourceIndex(p.roughnessTextureIdx)]
                      .SampleLevel(h.uv, 0.0).r;
        out_.mat.roughness = clamp(r, 0.04, 1.0);
    }

    out_.mat.metallic = clamp(p.metallic, 0.0, 1.0);
    if (p.metallicTextureIdx != 0xFFFFFFFFu)
    {
        float m = flatMaterialTextures[NonUniformResourceIndex(p.metallicTextureIdx)]
                      .SampleLevel(h.uv, 0.0).r;
        out_.mat.metallic = clamp(m, 0.0, 1.0);
    }

    out_.mat.specular = clamp(p.specular, 0.0, 1.0);
    out_.mat.ior = max(p.ior, 1.0);
    out_.mat.opacity = clamp(p.opacity, 0.0, 1.0);
    out_.mat.coat = clamp(p.coat, 0.0, 1.0);
    out_.mat.coatRoughness = clamp(p.coatRoughness, 0.04, 1.0);
    out_.mat.coatIOR = max(p.coatIOR, 1.0);
    out_.mat.coatColor = p.coatColor;

    out_.mat.emission = p.emissiveColor;
    if (p.emissiveTextureIdx != 0xFFFFFFFFu)
    {
        out_.mat.emission *= flatMaterialTextures[NonUniformResourceIndex(p.emissiveTextureIdx)]
                                .SampleLevel(h.uv, 0.0).rgb;
    }

    out_.N = h.normal;
    if (p.normalTextureIdx != 0xFFFFFFFFu && h.hasTangent)
    {
        float3 sampled = flatMaterialTextures[NonUniformResourceIndex(p.normalTextureIdx)]
                            .SampleLevel(h.uv, 0.0).xyz * 2.0 - 1.0;
        float3 N = h.normal;
        float3 T = normalize(h.tangent - N * dot(h.tangent, N));
        float3 B = cross(N, T);
        out_.N = normalize(T * sampled.x + B * sampled.y + N * sampled.z);
    }
    return out_;
}

// ── Procedural material evaluation ──────────────────────────────

float3 evalProceduralColor(ProceduralParams pp, float3 posObject)
{
    if (pp.type == PROC_TYPE_MARBLE_3D)
    {
        float add_xyz_out = dot(posObject, pp.add_xyz_in2);
        float3 scale_pos_out = posObject * pp.scale_pos_in2;
        float scale_xyz_out = add_xyz_out * pp.scale_xyz_in2;
        float noise_out = 0.0;
        mx_fractal3d_float(pp.noise_amplitude, pp.noise_octaves,
                           pp.noise_lacunarity, pp.noise_diminish,
                           scale_pos_out, noise_out);
        float scale_noise_out = noise_out * pp.scale_noise_in2;
        float sum_out = scale_xyz_out + scale_noise_out;
        float sin_out = sin(sum_out);
        float scale_out = sin_out * pp.scale_in2;
        float bias_out = scale_out + pp.bias_in2;
        float power_out = pow(max(bias_out, 0.0), pp.power_in2);
        return lerp(pp.color_mix_bg, pp.color_mix_fg, clamp(power_out, 0.0, 1.0));
    }
    return float3(-1.0);
}

StdSurfaceParams loadStdSurfaceParams(uint matId, float3 posObject)
{
    StdSurfaceParams sp = stdSurfaceParams[matId];
    ProceduralParams pp = proceduralParams[matId];
    if (pp.type != PROC_TYPE_NONE)
    {
        float3 color = evalProceduralColor(pp, posObject);
        if (color.x >= 0.0)
        {
            sp.base_color = color;
            sp.subsurface_color = color;
        }
    }
    return sp;
}

// ── MIS support helpers ─────────────────────────────────────────

float stdSurfaceSamplingPdf(float3 N, float3 V, float3 L, StdSurfaceParams sp)
{
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return 0.0;
    float NdotV = max(dot(N, V), 1e-6);

    float3 baseColor = sp.base_color * sp.base;
    float3 F0    = schlickF0(baseColor, sp.metalness, sp.specular, sp.specular_IOR);
    float3 Fview = schlickFresnelColor(NdotV, F0);
    float3 diffHue = baseColor * (float3(1.0) - Fview) * (1.0 - sp.metalness);
    float specLum  = relativeLuminance(Fview);
    float diffLum  = relativeLuminance(diffHue);
    float pSpec    = clamp(specLum / max(specLum + diffLum, 1e-6),
                           0.05, 0.95);

    float diff_pdf = NdotL / PI;

    float3 H    = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    float roughness = max(sp.specular_roughness, 0.04);
    float a     = max(roughness * roughness, 1e-4);
    float a2    = a * a;
    float den   = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D     = a2 / max(PI * den * den, 1e-6);
    float spec_pdf = D * NdotH / max(4.0 * VdotH, 1e-6);

    return pSpec * spec_pdf + (1.0 - pSpec) * diff_pdf;
}

// ── Flat-material area-light contributions ──────────────────────

float3 flatSphereLightsContribution(
    HitInfo h, float3 N, float3 V, StdSurfaceParams sp, inout RNG rng
)
{
    float3 total = float3(0.0);
    if (fc.numSphereLights == 0u)
        return total;
    float3 T = h.hasTangent ? h.tangent : float3(1, 0, 0);
    for (uint i = 0u; i < fc.numSphereLights; i++)
    {
        SphereLight L = sphereLights[i];
        float r = max(L.radius, 1e-4);

        float2 u  = rng.next2();
        float z   = 2.0 * u.x - 1.0;
        float phi = 2.0 * PI * u.y;
        float st  = sqrt(max(0.0, 1.0 - z * z));
        float3 lN = float3(st * cos(phi), st * sin(phi), z);
        float3 P  = L.position + r * lN;

        float3 toLight = P - h.position;
        float d2 = dot(toLight, toLight);
        if (d2 <= 1e-8) continue;
        float dist = sqrt(d2);
        float3 dir = toLight / dist;

        float NdotL = max(dot(N, dir), 0.0);
        float cosLight = max(dot(lN, -dir), 0.0);
        if (NdotL <= 0.0 || cosLight <= 0.0) continue;
        if (!visibleSegment(fc, h.position, P)) continue;

        ClosureData cd = makeClosureData(CLOSURE_TYPE_REFLECTION, dir, V, N, h.position, 1.0);
        BSDF bsdfResult = evalStdSurfaceBSDF(cd, sp, N, T);

        float area = 4.0 * PI * r * r;
        float lightGeom = cosLight * area / d2;
        float pdf_light_sa = d2 / max(cosLight * area, 1e-12);
        float pdf_bsdf_sa  = stdSurfaceSamplingPdf(N, V, dir, sp);
        float w_nee = powerHeuristic(pdf_light_sa, pdf_bsdf_sa);
        total += bsdfResult.response * L.radiance * lightGeom * w_nee;
    }
    return total;
}

float3 flatEmissiveTrianglesContribution(
    HitInfo h, float3 N, float3 V, StdSurfaceParams sp, inout RNG rng
)
{
    float3 total = float3(0.0);
    if (fc.numEmissiveTriangles == 0u)
        return total;

    uint idx = min(uint(rng.next() * float(fc.numEmissiveTriangles)),
                   fc.numEmissiveTriangles - 1u);
    EmissiveTriangle tri = emissiveTriangles[idx];

    float2 u = rng.next2();
    float su = sqrt(u.x);
    float b0 = 1.0 - su;
    float b1 = u.y * su;
    float3 P = tri.v0 * b0 + tri.v1 * b1 + tri.v2 * (1.0 - b0 - b1);

    float3 e1 = tri.v1 - tri.v0;
    float3 e2 = tri.v2 - tri.v0;
    float3 triN = normalize(cross(e1, e2));

    float3 toLight = P - h.position;
    float d2 = dot(toLight, toLight);
    if (d2 <= 1e-8) return total;
    float dist = sqrt(d2);
    float3 dir = toLight / dist;

    float NdotL = max(dot(N, dir), 0.0);
    float cosLight = max(dot(triN, -dir), 0.0);
    if (NdotL <= 0.0 || cosLight <= 0.0) return total;

    if (!visibleSegment(fc, h.position, P)) return total;

    float3 T = h.hasTangent ? h.tangent : float3(1, 0, 0);
    ClosureData cd = makeClosureData(CLOSURE_TYPE_REFLECTION, dir, V, N, h.position, 1.0);
    BSDF bsdfResult = evalStdSurfaceBSDF(cd, sp, N, T);

    float lightGeom = float(fc.numEmissiveTriangles) * tri.area * cosLight / d2;
    total = bsdfResult.response * tri.emission * lightGeom;
    return total;
}

// ── Flat-material direct light ──────────────────────────────────

float3 flatDirectContribution(HitInfo h, float3 N, float3 V, StdSurfaceParams sp)
{
    if (fc.useDirectLight == 0u)
        return float3(0.0);
    float3 L = lightDirection;
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0)
        return float3(0.0);
    if (!visibleDirectional(fc, h.position, N, L))
        return float3(0.0);

    float3 T = h.hasTangent ? h.tangent : float3(1, 0, 0);
    ClosureData cd = makeClosureData(CLOSURE_TYPE_REFLECTION, L, V, N, h.position, 1.0);
    BSDF result = evalStdSurfaceBSDF(cd, sp, N, T);
    return result.response * lightRadiance;
}

// ── Flat-material bounce loop ───────────────────────────────────

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
                r.direction, fc.envIntensity, fc.furnaceMode
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

        radiance += throughput * flatDirectContribution(h, N, V, sp);
        radiance += throughput * flatSphereLightsContribution(h, N, V, sp, rng);
        radiance += throughput * flatEmissiveTrianglesContribution(h, N, V, sp, rng);
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
