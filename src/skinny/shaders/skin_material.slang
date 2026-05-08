// Skin material orchestrator. Loads detail maps + skin params, then
// chains estimators (§1-§6) for unidirectional path tracing.
//
// Contribution terms:
//   §1 direct + area + emissive (estimators/skin_direct.slang)
//   §2 IBL specular            (estimators/skin_ibl_specular.slang)
//   §3 IBL diffuse via BSSRDF   (estimators/skin_ibl_diffuse.slang)
//   §4 volume march             (estimators/skin_volume.slang)
//   §5 thin-geometry transmit   (estimators/skin_transmission.slang)
//   §6 vellus hair sheen        (estimators/skin_hair_sheen.slang)

import common;
import bindings;
import interfaces;
import scene_trace;
import skin_bssrdf;
import detail;
import skin_shading;
import samplers.ggx;
import samplers.lambert;
import estimators.skin_direct;
import estimators.skin_ibl_specular;
import estimators.skin_ibl_diffuse;
import estimators.skin_volume;
import estimators.skin_transmission;
import estimators.skin_hair_sheen;

// ── Detail / params staging ─────────────────────────────────────

struct SkinShading
{
    SkinEvalContext ctx;
    float3 N;             // detail-mapped normal
    float  bssrdfWeight;
    float  volumeWeight;
    int    volumeBounces;
    bool   useDirect;
    bool   useVolume;
    bool   volumeOnly;
};

SkinShading buildSkinShading(HitInfo hit, FrameConstants fc, uint scatterBits)
{
    float3 N = hit.normal;
    MtlxSkinParams skin = mtlxSkin[hit.materialId];
    SkinParams sp = skinParamsFromMtlx(skin);

    bool detailMaster    = (fc.detailFlags & 1u) != 0u;
    bool detailEligible  = detailMaster && hit.hasTangent && fc.useMesh != 0u && effectiveFurnaceMode(hit.materialId) == 0u;
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
    if (effectiveFurnaceMode(hit.materialId) == 0u)
    {
        PoreSample ps = applyPores(N, hit.position, sp.poreDensity, sp.poreDepth);
        Nspec     = ps.N;
        roughSpec = clamp(baseRough + ps.dRoughness, 0.02, 1.0);
    }

    SkinLayerStack layers = buildSkinLayerStackFromMtlx(skin, ink);
    SkinShading s;
    s.ctx.sp        = sp;
    s.ctx.mtlx      = skin;
    s.ctx.layers    = layers;
    s.ctx.ink       = ink;
    s.ctx.Nspec     = Nspec;
    s.ctx.roughSpec = roughSpec;
    s.ctx.useBSSRDF   = useBSSRDF;
    s.ctx.materialId  = hit.materialId;
    s.N             = N;
    s.bssrdfWeight  = bssrdfWeight;
    s.volumeWeight  = volumeWeight;
    s.volumeBounces = volumeBounces;
    s.useDirect     = useDirect;
    s.useVolume     = useVolume;
    s.volumeOnly    = volumeOnly;
    return s;
}

// ── Path tracing (§1-§6, no volume connections, no BDPT) ────────

float3 evalSkinPath(
    HitInfo hit, Ray ray, inout RNG rng,
    FrameConstants fc,
    float3 lightDirection, float3 lightRadiance,
    uint scatterBits)
{
    SkinShading s = buildSkinShading(hit, fc, scatterBits);
    float3 V = -ray.direction;

    float3 analyticLightDir = float3(0.0);
    float3 analyticLightRad = float3(0.0);
    if (s.useDirect)
    {
        analyticLightDir = normalize(lightDirection);
        if (visibleDirectional(fc, hit.position, s.N, analyticLightDir))
            analyticLightRad = lightRadiance;
    }

    // §1 all lights (directional + sphere + emissive triangle via ILight)
    float3 directContrib = skinAllLightsEstimator(
        hit, s.N, V, s.ctx, s.useDirect, s.volumeOnly, fc, rng);

    // §2 IBL specular
    GGXSampler specSampler;
    specSampler.roughness = s.ctx.roughSpec;
    float3 envSpec = skinIBLSpecularEstimator(specSampler, s.ctx, V, s.volumeOnly, fc, rng);

    // §3 IBL diffuse via BSSRDF
    LambertSampler diffSampler;
    SkinIBLDiffuseResult diff = skinIBLDiffuseEstimator(
        diffSampler, s.ctx, hit, s.N, V, s.bssrdfWeight, fc, rng);

    // §4 volume march (no light connections)
    float3 volume = skinVolumeEstimator(
        ray, hit, s.ctx, analyticLightDir, analyticLightRad,
        s.volumeBounces, s.volumeWeight, s.useVolume, fc, rng);

    // §5 thin-geometry translucency
    float3 transmission = skinTransmissionEstimator(
        hit, s.ctx, s.useDirect, analyticLightDir, lightRadiance, fc, rng);

    // §6 vellus hair sheen
    SkinHairContribution hair = skinHairSheenEstimator(
        s.N, V, s.ctx, s.useDirect, s.volumeOnly,
        analyticLightDir, analyticLightRad, diff.envIrradiance, fc);

    float F = fresnelSchlick(max(dot(s.N, V), 0.0), s.ctx.sp.ior);
    return directContrib + envSpec + hair.direct + hair.envSheen
         + (diff.envSSS + volume + transmission) * (1.0 - F);
}

// ── IMaterial compatibility wrapper ─────────────────────────────
// Skin uses 6 chained estimators, not a bounce loop. The wrapper
// returns the full integrated radiance from emission() and signals
// no further bouncing via sample().valid = false.

struct SkinMaterial : IMaterial
{
    HitInfo hit;
    Ray ray;
    uint scatterBits;

    BSDFSample sample(float3 wo, inout RNG rng)
    {
        BSDFSample s;
        s.valid    = false;
        s.emission = float3(0.0);
        return s;
    }

    BSDFEval evaluate(float3 wo, float3 wi)
    {
        BSDFEval e;
        e.response = float3(0.0);
        e.pdf      = 0.0;
        return e;
    }
}

float3 evalSkinRadiance(HitInfo hit, Ray ray, inout RNG rng, uint scatterBits)
{
    return evalSkinPath(hit, ray, rng, fc,
        lightDirection, lightRadiance, scatterBits);
}

SkinMaterial loadSkinMaterial(HitInfo h, Ray r, uint scatterBits)
{
    SkinMaterial m;
    m.hit          = h;
    m.ray          = r;
    m.scatterBits  = scatterBits;
    return m;
}

