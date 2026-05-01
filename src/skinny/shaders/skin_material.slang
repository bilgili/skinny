// Skin material evaluator — all skin-specific shading lives here.
//
// Encapsulates detail maps, pore perturbation, the genslang layered
// BSDF, direct + area + IBL + volume + transmission + hair estimators,
// and the stored-BDPT path so that main_pass.slang stays material-
// agnostic. Called through the material dispatch in mainImage:
//
//   sample = evalSkinMaterial(hit, ray, rng, fc, lightDir, lightRad);

import common;
import scene_trace;
import scene_lights;
import skin_bssrdf;
import volume_render;
import environment;
import detail;
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

// ── Skin-path area-light helpers ────────────────────────────────

float3 skinSphereLightsContribution(
    HitInfo h, float3 N, float3 V,
    SkinEvalContext ctx, FrameConstants fc, inout RNG rng)
{
    float3 total = float3(0.0);
    if (fc.numSphereLights == 0u)
        return total;
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
        float NdotL    = max(dot(N, dir), 0.0);
        float cosLight = max(dot(lN, -dir), 0.0);
        if (NdotL <= 0.0 || cosLight <= 0.0) continue;
        if (!visibleSegment(fc, h.position, P)) continue;

        float area = 4.0 * PI * r * r;
        float geom = cosLight * area / d2;

        if (ctx.useBSSRDF)
        {
            float3 response = evalSkinBSDF(dir, V, h.position, h.tangent, ctx);
            total += response * L.radiance * geom;
        }
        else
        {
            float3 spec = evaluateSkinSpecular(
                ctx.Nspec, V, dir, ctx.roughSpec, ctx.sp.ior) * NdotL;
            total += spec * L.radiance * geom;
        }
    }
    return total;
}

float3 skinEmissiveTrianglesContribution(
    HitInfo h, float3 N, float3 V,
    SkinEvalContext ctx, FrameConstants fc, inout RNG rng)
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

    float NdotL    = max(dot(N, dir), 0.0);
    float cosLight = max(dot(triN, -dir), 0.0);
    if (NdotL <= 0.0 || cosLight <= 0.0) return total;
    if (!visibleSegment(fc, h.position, P)) return total;

    float geom = float(fc.numEmissiveTriangles) * tri.area * cosLight / d2;

    if (ctx.useBSSRDF)
    {
        float3 response = evalSkinBSDF(dir, V, h.position, h.tangent, ctx);
        total = response * tri.emission * geom;
    }
    else
    {
        float3 spec = evaluateSkinSpecular(
            ctx.Nspec, V, dir, ctx.roughSpec, ctx.sp.ior) * NdotL;
        total = spec * tri.emission * geom;
    }
    return total;
}

// ── Stored light-subpath BDPT ───────────────────────────────────

float3 storedLightSubpathBDPT(
    HitInfo cameraHit, float3 cameraN, float4 cameraInk,
    SkinParams sp, FrameConstants fc, inout RNG rng)
{
    if (fc.integratorMode != 3u)
        return float3(0.0);

    static const float LIGHT_DISK_RADIUS = 2.5;
    static const float LIGHT_START_DIST  = 4.0;

    float3 wiEnv = sampleUniformSphere(rng.next2());
    float3 T, B;
    buildBasis(wiEnv, T, B);
    float2 disk = sampleUniformDisk(rng.next2()) * LIGHT_DISK_RADIUS;

    Ray lightRay;
    lightRay.origin = wiEnv * LIGHT_START_DIST + T * disk.x + B * disk.y;
    lightRay.direction = -wiEnv;

    HitInfo lightHit = traceScene(fc, lightRay);
    if (!lightHit.hit)
        return float3(0.0);

    float cosLightIn = max(dot(lightHit.normal, wiEnv), 0.0);
    if (cosLightIn <= 1e-5)
        return float3(0.0);

    float3 xToY = lightHit.position - cameraHit.position;
    float dist2 = max(dot(xToY, xToY), 1e-6);
    float dist = sqrt(dist2);
    float3 dirXY = xToY / dist;
    float cosCamera = max(dot(cameraN, dirXY), 0.0);
    float cosLightOut = max(dot(lightHit.normal, -dirXY), 0.0);
    if (cosCamera <= 1e-5 || cosLightOut <= 1e-5)
        return float3(0.0);
    if (!visibleSegment(fc, cameraHit.position, lightHit.position))
        return float3(0.0);

    float pDir = uniformSpherePdf();
    float pDisk = 1.0 / (PI * LIGHT_DISK_RADIUS * LIGHT_DISK_RADIUS);
    float invPdfArea = cosLightIn / max(pDir * pDisk, 1e-8);

    float4 lightInk = inkAtHit(lightHit, fc.tattooDensity);
    float3 fCamera = evaluateSubsurfaceScattering(
        sp, cameraHit.position, cameraN, 0.0, float3(1.0), cameraInk
    );
    float3 fLight = evaluateSubsurfaceScattering(
        sp, lightHit.position, lightHit.normal, 0.0, float3(1.0), lightInk
    );

    float3 Le = sceneEnvRadiance(wiEnv, fc.envIntensity, fc.furnaceMode);
    float geometry = cosCamera * cosLightOut / dist2;
    return Le * fLight * fCamera * geometry * invPdfArea;
}

// ── Main skin evaluator ─────────────────────────────────────────

float3 evalSkinMaterial(
    HitInfo hit, Ray ray, inout RNG rng,
    FrameConstants fc,
    float3 lightDirection, float3 lightRadiance,
    uint scatterBits)
{
    float3 V = -ray.direction;
    float3 N = hit.normal;

    MtlxSkinParams skin = mtlxSkin[hit.materialId];
    SkinParams sp = skinParamsFromMtlx(skin);

    // Detail maps (normal / roughness)
    bool detailMaster    = (fc.detailFlags & 1u) != 0u;
    bool detailEligible  = detailMaster && hit.hasTangent && fc.useMesh != 0u && fc.furnaceMode == 0u;
    bool normalsBaked    = (fc.detailFlags & 16u) != 0u;
    bool useNormalMap    = detailEligible && !normalsBaked && ((fc.detailFlags & 2u) != 0u);
    bool useRoughnessMap = detailEligible && ((fc.detailFlags & 4u) != 0u);

    float3 N_detail = N;
    float  mapRough = -1.0;
    if (detailEligible)
    {
        float3 T = normalize(hit.tangent - N * dot(hit.tangent, N));
        float3 B = cross(N, T);
        if (useNormalMap)
        {
            float3 nSample = normalMap.SampleLevel(hit.uv, 0.0).xyz;
            float2 nxy = (nSample.xy * 2.0 - 1.0) * clamp(fc.normalMapStrength, 0.0, 4.0);
            float  nz  = sqrt(max(1.0 - dot(nxy, nxy), 1e-4));
            N_detail = normalize(T * nxy.x + B * nxy.y + N * nz);
        }
        if (useRoughnessMap)
            mapRough = roughnessMap.SampleLevel(hit.uv, 0.0).x;
    }
    N = N_detail;

    float4 ink = inkAtHit(hit, fc.tattooDensity);

    bool useBSSRDF = (scatterBits & 1u) != 0u;
    bool useVolume = (scatterBits & 2u) != 0u;
    bool useDirect = fc.useDirectLight != 0;
    bool volumeOnly = useVolume && !useBSSRDF;

    bool combined      = useBSSRDF && useVolume;
    float bssrdfWeight = combined ? 0.5 : 1.0;
    float volumeWeight = combined ? 0.5 : 1.0;
    int volumeBounces  = combined ? 1 : 8;

    float3 Nspec     = N;
    float  baseRough = mapRough >= 0.0 ? mapRough : sp.roughness;
    float  roughSpec = baseRough;
    if (fc.furnaceMode == 0u)
    {
        PoreSample ps = applyPores(N, hit.position, sp.poreDensity, sp.poreDepth);
        Nspec     = ps.N;
        roughSpec = clamp(baseRough + ps.dRoughness, 0.02, 1.0);
    }

    float F = fresnelSchlick(max(dot(N, V), 0.0), sp.ior);

    SkinLayerStack layers = buildSkinLayerStackFromMtlx(skin, ink);
    SkinEvalContext ctx;
    ctx.sp        = sp;
    ctx.mtlx      = skin;
    ctx.layers    = layers;
    ctx.ink       = ink;
    ctx.Nspec     = Nspec;
    ctx.roughSpec = roughSpec;
    ctx.useBSSRDF = useBSSRDF;

    // In volume-only mode, suppress all surface terms (specular, hair, BSSRDF)
    // so the output isolates the in-medium transport — otherwise specular
    // drowns the volume contribution and parameter changes are invisible.
    float3 analyticLightDir = float3(0.0);
    float3 analyticLightRad = float3(0.0);
    if (useDirect)
    {
        analyticLightDir = normalize(lightDirection);
        if (visibleDirectional(fc, hit.position, N, analyticLightDir))
            analyticLightRad = lightRadiance;
    }

    // 1. Direct analytic light + area lights (surface reflection)
    float3 directContrib = float3(0.0);
    if (!volumeOnly)
    {
        if (useDirect)
        {
            if (useBSSRDF)
            {
                directContrib = evalSkinBSDF(
                    analyticLightDir, V, hit.position, hit.tangent, ctx
                ) * analyticLightRad;
            }
            else
            {
                float NdotL = max(dot(N, analyticLightDir), 0.0);
                directContrib = evaluateSkinSpecular(
                    Nspec, V, analyticLightDir, roughSpec, sp.ior
                ) * analyticLightRad * NdotL;
            }
        }
        directContrib += skinSphereLightsContribution(hit, N, V, ctx, fc, rng);
        directContrib += skinEmissiveTrianglesContribution(hit, N, V, ctx, fc, rng);
    }

    // 2. IBL specular (GGX importance-sampled)
    float3 envSpec = float3(0.0);
    float  uniPdf  = uniformSpherePdf();
    if (!volumeOnly)
    {
        SpecularIBLSample specIBL = sampleGGXSpecularIBL(
            Nspec, V, roughSpec, sp.ior, rng.next2());
        float ggxPdf = ggxSpecularPdf(Nspec, V, specIBL.direction, roughSpec);
        envSpec = sceneEnvRadiance(specIBL.direction, fc.envIntensity, fc.furnaceMode)
                      * specIBL.weight
                      * (fc.integratorMode != 0u ? powerHeuristic(ggxPdf, uniPdf) : 1.0);
        if (fc.integratorMode != 0u)
        {
            float3 specLightDir = sampleUniformSphere(rng.next2());
            float  specNdotL    = max(dot(Nspec, specLightDir), 0.0);
            if (specNdotL > 0.0)
            {
                float  specGgxPdf = ggxSpecularPdf(Nspec, V, specLightDir, roughSpec);
                float  specWMis   = powerHeuristic(uniPdf, specGgxPdf);
                float3 specEnv    = sceneEnvRadiance(specLightDir, fc.envIntensity, fc.furnaceMode);
                envSpec += evaluateSkinSpecular(Nspec, V, specLightDir, roughSpec, sp.ior)
                         * specEnv * specNdotL * (specWMis / uniPdf);
            }
        }
    }

    // 3. IBL diffuse via BSSRDF
    float3 diffDir       = sampleCosineHemisphere(rng.next2(), N);
    float3 envIrradiance = sceneEnvRadiance(diffDir, fc.envIntensity, fc.furnaceMode);
    float3 envSSS        = useBSSRDF
        ? evaluateSubsurfaceScattering(sp, hit.position, N, 0.0, envIrradiance, ink)
              * PI
              * (fc.integratorMode != 0u
                    ? powerHeuristic(cosineHemispherePdf(diffDir, N), uniPdf)
                    : 1.0)
              * bssrdfWeight
        : float3(0.0);
    if (useBSSRDF && fc.integratorMode != 0u)
    {
        float3 diffLightDir = sampleUniformSphere(rng.next2());
        float  diffNdotL    = max(dot(N, diffLightDir), 0.0);
        if (diffNdotL > 0.0)
        {
            float3 diffEnv  = sceneEnvRadiance(diffLightDir, fc.envIntensity, fc.furnaceMode);
            float  diffPdf  = cosineHemispherePdf(diffLightDir, N);
            float  diffWMis = powerHeuristic(uniPdf, diffPdf);
            envSSS += evaluateSubsurfaceScattering(sp, hit.position, N, 0.0, diffEnv, ink)
                    * diffNdotL * (diffWMis / uniPdf) * bssrdfWeight;
        }
    }

    // 4. Volume march
    float3 volume = useVolume
        ? volumeRenderSkin(
              ray, hit, analyticLightDir, analyticLightRad,
              sp, ink, fc.envIntensity, fc.furnaceMode,
              fc.integratorMode >= 2u ? fc.integratorMode : 0u,
              fc.mmPerUnit, volumeBounces, rng) * volumeWeight
        : float3(0.0);

    // 5. Thin-geometry translucency
    float3 transmission = float3(0.0);
    if (hit.backT > 0.0 && fc.furnaceMode == 0u)
    {
        float localThicknessMM = (hit.backT - hit.t) * fc.mmPerUnit;
        float skinSlabMM = sp.epidermisThickness + sp.dermisThickness + sp.subcutThickness;
        if (localThicknessMM < skinSlabMM)
        {
            SkinLayer epidermis = buildEpidermis(sp);
            SkinLayer dermis    = buildDermisInked(sp, ink);
            float3 sigmaSprime_e = epidermis.sigmaS * (1.0 - epidermis.g);
            float3 sigmaSprime_d = dermis.sigmaS    * (1.0 - dermis.g);
            float3 sigmaTr_e = sqrt(3.0 * epidermis.sigmaA * (epidermis.sigmaA + sigmaSprime_e));
            float3 sigmaTr_d = sqrt(3.0 * dermis.sigmaA    * (dermis.sigmaA    + sigmaSprime_d));
            float wE = sp.epidermisThickness;
            float wD = sp.dermisThickness;
            float3 sigmaTr = (sigmaTr_e * wE + sigmaTr_d * wD) / max(wE + wD, 1e-6);
            float3 tr = exp(-sigmaTr * localThicknessMM);
            float3 backN = -hit.normal;
            if (useDirect)
            {
                float backNdotL = max(dot(backN, analyticLightDir), 0.0);
                transmission += tr * lightRadiance * backNdotL;
            }
            float3 backDiffDir = sampleCosineHemisphere(rng.next2(), backN);
            float3 backEnv     = sceneEnvRadiance(backDiffDir, fc.envIntensity, fc.furnaceMode);
            transmission += tr * backEnv;
            transmission /= PI;
        }
    }

    // 6. Vellus hair sheen (surface effect — skip in volume-only)
    float3 hairDirect = float3(0.0);
    float3 hairEnv    = float3(0.0);
    if (!volumeOnly && fc.furnaceMode == 0u && sp.hairDensity > 1e-3)
    {
        float3 hairBRDF = vellusHairBRDF(N, V, float3(1.0), sp.hairDensity, sp.hairTilt);
        if (useDirect)
        {
            float NdotLh = max(dot(N, analyticLightDir), 0.0);
            hairDirect   = hairBRDF * NdotLh * analyticLightRad;
        }
        hairEnv = PI * hairBRDF * envIrradiance;
    }

    // 7. BDPT
    float3 bdptSSS = useBSSRDF
        ? storedLightSubpathBDPT(hit, N, ink, sp, fc, rng) : float3(0.0);

    return directContrib + envSpec + hairDirect + hairEnv
         + (envSSS + bdptSSS + volume + transmission) * (1.0 - F);
}
