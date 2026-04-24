// Main render pass — dispatches camera rays and composites
// surface specular + subsurface + volume contributions.

import common;
import sdf_head;
import mesh_head;
import skin_bssrdf;
import volume_render;
import environment;
import detail;

// ── Bindings ────────────────────────────────────────────────────

uniform FrameConstants fc;
uniform SkinParams     skinParams;
uniform float3         lightDirection;
uniform float3         lightRadiance;

[[vk::binding(1)]] RWTexture2D<float4> outputBuffer;
[[vk::binding(2)]] RWTexture2D<float4> accumBuffer;
[[vk::binding(3)]] [[vk::image_format("r8")]] RWTexture2D<float> hudMask;
[[vk::binding(8)]] Sampler2D<float4>    tattooMap;
// Per-model detail maps. Blank placeholders are uploaded when the active head
// has no texture, so these samplers are always valid to sample — the shader
// gates real use behind fc.detailFlags (bit 0: master enable; bits 1-3: per-map
// availability set by the renderer when a texture file actually attached).
[[vk::binding(9)]]  Sampler2D<float4> normalMap;
[[vk::binding(10)]] Sampler2D<float4> roughnessMap;
[[vk::binding(11)]] Sampler2D<float4> displacementMap;

// ── Furnace test helpers ────────────────────────────────────────

// Ray-vs-unit-sphere at origin. Used only in furnace mode, where we swap the
// head for a lossless test geometry to drive the energy-conservation probe.
HitInfo traceFurnaceSphere(Ray ray)
{
    HitInfo h;
    h.hit        = false;
    h.t          = 0.0;
    h.backT      = -1.0;
    h.position   = float3(0.0);
    h.normal     = float3(0.0, 1.0, 0.0);
    h.uv         = float2(0.0);
    h.tangent    = float3(1.0, 0.0, 0.0);
    h.hasTangent = false;

    float3 oc   = ray.origin;                 // sphere centred at origin
    float  b    = dot(oc, ray.direction);
    float  c    = dot(oc, oc) - 1.0;          // radius = 1
    float  disc = b * b - c;
    if (disc < 0.0) return h;

    float sd = sqrt(disc);
    float t  = -b - sd;
    if (t < 0.0) t = -b + sd;
    if (t < 0.0) return h;

    h.hit      = true;
    h.t        = t;
    h.position = ray.origin + ray.direction * t;
    h.normal   = normalize(h.position);
    return h;
}

// ── Tonemapping ─────────────────────────────────────────────────

float3 acesFilmic(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float3 linearToSRGB(float3 linear)
{
    return pow(max(linear, float3(0.0)), float3(1.0 / 2.2));
}

// ── Entry point ─────────────────────────────────────────────────

HitInfo traceScene(Ray r)
{
    if (fc.furnaceMode != 0u)
        return traceFurnaceSphere(r);
    return (fc.useMesh != 0u) ? marchHeadMesh(r) : marchHead(r);
}

bool visibleSegment(float3 a, float3 b)
{
    float3 d = b - a;
    float dist = length(d);
    if (dist <= 1e-4)
        return false;

    float3 dir = d / dist;
    Ray shadow;
    shadow.origin = a + dir * 0.002;
    shadow.direction = dir;
    HitInfo h = traceScene(shadow);
    return !h.hit || h.t > dist - 0.004;
}

float2 sampleUniformDisk(float2 u)
{
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    return float2(r * cos(phi), r * sin(phi));
}

void buildBasis(float3 n, out float3 t, out float3 b)
{
    t = abs(n.x) > 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    b = normalize(cross(n, t));
    t = cross(b, n);
}

float4 inkAtHit(HitInfo h)
{
    float4 inkSample = tattooMap.SampleLevel(h.uv, 0.0);
    return float4(inkSample.rgb, inkSample.a * fc.tattooDensity);
}

float3 storedLightSubpathBDPT(HitInfo cameraHit, float3 cameraN, float4 cameraInk, inout RNG rng)
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

    HitInfo lightHit = traceScene(lightRay);
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
    if (!visibleSegment(cameraHit.position, lightHit.position))
        return float3(0.0);

    float pDir = uniformSpherePdf();
    float pDisk = 1.0 / (PI * LIGHT_DISK_RADIUS * LIGHT_DISK_RADIUS);
    float invPdfArea = cosLightIn / max(pDir * pDisk, 1e-8);

    float4 lightInk = inkAtHit(lightHit);
    float3 fCamera = evaluateSubsurfaceScattering(
        skinParams, cameraHit.position, cameraN, 0.0, float3(1.0), cameraInk
    );
    float3 fLight = evaluateSubsurfaceScattering(
        skinParams, lightHit.position, lightHit.normal, 0.0, float3(1.0), lightInk
    );

    float3 Le = sceneEnvRadiance(wiEnv, fc.envIntensity, fc.furnaceMode);
    float geometry = cosCamera * cosLightOut / dist2;
    return Le * fLight * fCamera * geometry * invPdfArea;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void mainImage(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= fc.width || pixel.y >= fc.height)
        return;

    RNG rng = createRNG(pixel, fc.frameIndex);

    Ray ray = generateCameraRay(fc, pixel, rng);

    // Scene: either the analytic SDF head (sdf_head.slang) or a triangle
    // mesh traversed through a linear BVH (mesh_head.slang). The switch is
    // a uniform flag driven by the "Head model" UI param. Furnace mode
    // overrides both with a unit sphere so the energy-conservation check
    // runs against a known-simple geometry.
    HitInfo hit;
    if (fc.furnaceMode != 0u)
        hit = traceFurnaceSphere(ray);
    else
        hit = (fc.useMesh != 0u) ? marchHeadMesh(ray) : marchHead(ray);

    float3 sample;

    if (hit.hit)
    {
        float3 V = -ray.direction;
        float3 N = hit.normal;

        // ── Per-model detail maps (normal / roughness) ───────────────────
        //
        // Displacement is baked into vertex positions on the CPU (see
        // renderer.py -> mesh.bake_mesh), so the ray-traced hit already sits
        // on the displaced surface. The remaining two maps perturb shading
        // only: normal for tangent-space micro-detail, roughness for
        // per-texel surface response. Both gated on mesh mode + the master
        // toggle; furnace mode always uses the plain sphere.
        bool detailMaster    = (fc.detailFlags & 1u) != 0u;
        bool detailEligible  = detailMaster && hit.hasTangent && fc.useMesh != 0u && fc.furnaceMode == 0u;
        // Bit 4 = normals already baked into the mesh (see renderer._bake_and_upload
        // + mesh.bake_mesh). Skip the tangent-space normal sample when set so we
        // don't double-apply the same perturbation.
        bool normalsBaked    = (fc.detailFlags & 16u) != 0u;
        bool useNormalMap    = detailEligible && !normalsBaked && ((fc.detailFlags & 2u) != 0u);
        bool useRoughnessMap = detailEligible && ((fc.detailFlags & 4u) != 0u);

        float3 N_detail   = N;
        float  mapRough   = -1.0;  // sentinel: -1 means "no map override"
        if (detailEligible)
        {
            // Orthonormalise tangent against the shading normal once, then
            // build the bitangent from the cross. Same basis for every map
            // read so XY offsets stay self-consistent.
            float3 T = normalize(hit.tangent - N * dot(hit.tangent, N));
            float3 B = cross(N, T);

            if (useNormalMap)
            {
                float3 nSample = normalMap.SampleLevel(hit.uv, 0.0).xyz;
                // Decode [0,1] → [-1,1]; scale XY by user strength, rebuild Z
                // so the result is guaranteed unit length.
                float2 nxy = (nSample.xy * 2.0 - 1.0) * clamp(fc.normalMapStrength, 0.0, 4.0);
                float  nz  = sqrt(max(1.0 - dot(nxy, nxy), 1e-4));
                N_detail = normalize(T * nxy.x + B * nxy.y + N * nz);
            }
            if (useRoughnessMap)
            {
                // Most asset roughness maps store the value in R. Use that
                // directly — Pillow already decoded the file's gamma to
                // linear-ish 8-bit, which is accurate enough for this param.
                mapRough = roughnessMap.SampleLevel(hit.uv, 0.0).x;
            }
        }
        N = N_detail;

        float F = fresnelSchlick(max(dot(N, V), 0.0), skinParams.ior);

        // Sample the tattoo texture at the surface UV. RGB is the linear ink
        // colour; alpha scales with the global density slider. A zero-alpha
        // ink sample makes every downstream tattoo path a no-op.
        float4 inkSample = tattooMap.SampleLevel(hit.uv, 0.0);
        float4 ink = float4(inkSample.rgb, inkSample.a * fc.tattooDensity);

        bool useBSSRDF = (fc.scatterMode & 1u) != 0u;
        bool useVolume = (fc.scatterMode & 2u) != 0u;
        bool useDirect = fc.useDirectLight != 0 && fc.furnaceMode == 0u;

        // BSSRDF's layered Lambertian and the volume path trace are
        // *alternative* estimators of the same subsurface radiance — summing
        // them at full weight double-counts, which at low anisotropy +
        // low absorption pushes furnace-mode energy past 1.0. When both are
        // selected, split responsibilities: BSSRDF provides the smooth
        // multi-bounce diffuse envelope; volume is demoted to single-scatter
        // (1 bounce) for crisp near-surface detail. Equal MIS weights on
        // the two estimators keep the running mean energy-conserving.
        bool combined       = useBSSRDF && useVolume;
        float bssrdfWeight  = combined ? 0.5 : 1.0;
        float volumeWeight  = combined ? 0.5 : 1.0;
        int   volumeBounces = combined ? 1 : 8;

        // Pore-scale normal & roughness perturbation for the specular path.
        // SSS and volume stay on macro N — pores are a micron-scale feature
        // that the diffusion footprint already averages over. Fresnel also
        // stays on macro N so the (1-F) split with the diffuse side matches.
        // Skipped in furnace mode so the energy probe sees a clean surface.
        float3 Nspec     = N;
        // Base roughness: texture when the user opted in, slider otherwise.
        // Pores then add their localised increment on top of either.
        float  baseRough = mapRough >= 0.0 ? mapRough : skinParams.roughness;
        float  roughSpec = baseRough;
        if (fc.furnaceMode == 0u)
        {
            PoreSample ps = applyPores(
                N, hit.position, skinParams.poreDensity, skinParams.poreDepth
            );
            Nspec     = ps.N;
            roughSpec = clamp(baseRough + ps.dRoughness, 0.02, 1.0);
        }

        // 1. Direct analytic light — surface specular + point-BSSRDF. The
        // volume march runs separately so it can also do IBL next-event-
        // estimation regardless of whether the directional light is on.
        float3 directSpec = float3(0.0);
        float3 directSSS  = float3(0.0);
        float3 bdptSSS    = useBSSRDF ? storedLightSubpathBDPT(hit, N, ink, rng) : float3(0.0);
        float3 analyticLightDir = float3(0.0);
        float3 analyticLightRad = float3(0.0);
        if (useDirect)
        {
            analyticLightDir = normalize(lightDirection);
            analyticLightRad = lightRadiance;
            float NdotL = max(dot(N, analyticLightDir), 0.0);

            directSpec = evaluateSkinSpecular(
                Nspec, V, analyticLightDir, roughSpec, skinParams.ior
            ) * lightRadiance * NdotL;

            if (useBSSRDF)
            {
                float3 irradiance = lightRadiance * NdotL;
                directSSS = evaluateSubsurfaceScattering(
                    skinParams, hit.position, N, 0.0, irradiance, ink
                ) * bssrdfWeight;
            }
        }

        // 2. IBL specular. GGX-importance-sampled env tap; the MC weight
        // (G·F·VdotH / (NdotV·NdotH)) already includes Fresnel.
        SpecularIBLSample specIBL = sampleGGXSpecularIBL(
            Nspec, V, roughSpec, skinParams.ior, rng.next2()
        );
        float ggxPdf = ggxSpecularPdf(Nspec, V, specIBL.direction, roughSpec);
        float uniPdf = uniformSpherePdf();
        float3 envSpec = sceneEnvRadiance(specIBL.direction, fc.envIntensity, fc.furnaceMode)
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
                envSpec += evaluateSkinSpecular(Nspec, V, specLightDir, roughSpec, skinParams.ior)
                         * specEnv * specNdotL * (specWMis / uniPdf);
            }
        }

        // 3. IBL diffuse via BSSRDF. PDF = cos(θ)/π cancels the BRDF's 1/π
        // and the cos(θ) from the rendering equation, so the estimator is
        // albedo × L(ω). evaluateSubsurfaceScattering returns albedo×L/π so
        // we multiply by PI to undo the extra division.
        float3 diffDir       = sampleCosineHemisphere(rng.next2(), N);
        float3 envIrradiance = sceneEnvRadiance(diffDir, fc.envIntensity, fc.furnaceMode);
        float3 envSSS        = useBSSRDF
            ? evaluateSubsurfaceScattering(
                  skinParams, hit.position, N, 0.0, envIrradiance, ink)
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
                envSSS += evaluateSubsurfaceScattering(
                              skinParams, hit.position, N, 0.0, diffEnv, ink)
                        * diffNdotL * (diffWMis / uniPdf) * bssrdfWeight;
            }
        }

        // 4. Volume march — internally does next-event estimation to the
        // analytic light (if active) AND to the environment, so the volume
        // contributes under pure-IBL setups and the furnace probe too.
        float3 volume = useVolume
            ? volumeRenderSkin(
                  ray, hit, analyticLightDir, analyticLightRad,
                  skinParams, ink, fc.envIntensity, fc.furnaceMode,
                  fc.integratorMode >= 2u ? fc.integratorMode : 0u,
                  fc.mmPerUnit, volumeBounces, rng) * volumeWeight
            : float3(0.0);

        // 5. Thin-geometry translucency. When the back-face exit is known
        // and the local mesh thickness is less than the full skin slab,
        // light transmits through via Beer-Lambert attenuation. This gives
        // backlit ears, nostrils, and eyelids their characteristic warm glow.
        float3 transmission = float3(0.0);
        if (hit.backT > 0.0 && fc.furnaceMode == 0u)
        {
            float localThicknessMM = (hit.backT - hit.t) * fc.mmPerUnit;
            float skinSlabMM = skinParams.epidermisThickness
                             + skinParams.dermisThickness
                             + skinParams.subcutThickness;

            if (localThicknessMM < skinSlabMM)
            {
                // Use the effective transport coefficient from diffusion theory:
                //   σ_tr = sqrt(3 · σ_a · (σ_a + σ_s'))
                // This models diffuse transmission (multiply-scattered light
                // exiting the back), not direct ballistic transmission.
                SkinLayer epidermis = buildEpidermis(skinParams);
                SkinLayer dermis    = buildDermisInked(skinParams, ink);

                float3 sigmaSprime_e = epidermis.sigmaS * (1.0 - epidermis.g);
                float3 sigmaSprime_d = dermis.sigmaS    * (1.0 - dermis.g);

                float3 sigmaTr_e = sqrt(3.0 * epidermis.sigmaA * (epidermis.sigmaA + sigmaSprime_e));
                float3 sigmaTr_d = sqrt(3.0 * dermis.sigmaA    * (dermis.sigmaA    + sigmaSprime_d));

                // Thickness-weighted blend of per-layer transport coefficients.
                float wE = skinParams.epidermisThickness;
                float wD = skinParams.dermisThickness;
                float3 sigmaTr = (sigmaTr_e * wE + sigmaTr_d * wD) / max(wE + wD, 1e-6);

                float3 tr = exp(-sigmaTr * localThicknessMM);

                float3 backN = -hit.normal;
                if (useDirect)
                {
                    float backNdotL = max(dot(backN, analyticLightDir), 0.0);
                    transmission += tr * lightRadiance * backNdotL;
                }
                // Back-hemisphere IBL sample.
                float3 backDiffDir = sampleCosineHemisphere(rng.next2(), backN);
                float3 backEnv     = sceneEnvRadiance(backDiffDir, fc.envIntensity, fc.furnaceMode);
                transmission += tr * backEnv;

                transmission /= PI;
            }
        }

        // 6. Vellus hair sheen. Bounded rim term living *above* the skin
        // surface — added outside the (1-F) envelope because it's not a
        // subsurface effect. Direct: BRDF·NdotL·L_radiance. IBL: reuse the
        // cos-hemisphere sample so pdf cancels out, leaving PI·BRDF·L_env.
        // Skipped in furnace mode so the energy probe stays clean.
        float3 hairDirect = float3(0.0);
        float3 hairEnv    = float3(0.0);
        if (fc.furnaceMode == 0u && skinParams.hairDensity > 1e-3)
        {
            float3 hairBRDF = vellusHairBRDF(
                N, V, float3(1.0), skinParams.hairDensity, skinParams.hairTilt
            );
            if (useDirect)
            {
                float NdotLh = max(dot(N, analyticLightDir), 0.0);
                hairDirect   = hairBRDF * NdotLh * analyticLightRad;
            }
            hairEnv = PI * hairBRDF * envIrradiance;
        }

        sample = directSpec + envSpec + hairDirect + hairEnv
               + (directSSS + envSSS + bdptSSS + volume + transmission) * (1.0 - F);
    }
    else
    {
        // Primary-ray miss → show the environment directly.
        sample = sceneEnvRadiance(ray.direction, fc.envIntensity, fc.furnaceMode);
    }

    // Progressive accumulation — running mean in linear HDR.
    uint n = fc.accumFrame;
    float3 accum;
    if (n == 0)
    {
        accum = sample;
    }
    else
    {
        float3 prev = accumBuffer[pixel].rgb;
        accum = (prev * float(n) + sample) / float(n + 1);
    }
    accumBuffer[pixel] = float4(accum, 1.0);

    // Tonemap and gamma for display
    float3 display = linearToSRGB(acesFilmic(accum));

    // Furnace / energy-conservation probe. Under a unit-white environment
    // (every direction carries L = 1 per channel), a passive surface must
    // re-emit at most 1.0 outbound radiance per channel. Anything above that
    // means the BRDF/BSSRDF is creating energy — tint those pixels pink so
    // the violation is impossible to miss. Tolerance absorbs MC variance in
    // the first few accumulated samples.
    if (fc.furnaceMode != 0u && hit.hit)
    {
        float maxChan = max(accum.r, max(accum.g, accum.b));
        if (maxChan > 1.02)
        {
            display = float3(1.0, 0.35, 0.65);
        }
    }

    // HUD overlay: R8 mask packs (background shade | text) via value encoding —
    //   0.00      : no HUD
    //   (0, 0.75] : dim background panel with alpha = value
    //   > 0.75    : text ink (white)
    float hud = hudMask[pixel];
    if (hud > 0.75)
    {
        display = float3(1.0);
    }
    else if (hud > 0.0)
    {
        display = lerp(display, float3(0.02, 0.02, 0.03), hud);
    }

    outputBuffer[pixel] = float4(display, 1.0);
}
