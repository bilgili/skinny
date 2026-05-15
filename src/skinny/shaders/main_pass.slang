// Main render pass — dispatches camera rays through the unified material
// evaluator and composites the final display.
//
// References:
//   [1] Veach, Guibas, "Bidirectional Estimators for Light Transport",
//       EGSR 1995.
//   [2] Lafortune, Willems, "Bi-Directional Path Tracing",
//       Compugraphics 1993.
//   [3] Veach, "Robust Monte Carlo Methods for Light Transport Simulation",
//       PhD thesis, Stanford 1997.  (MIS, §9; BDPT, §10)
//   [4] Narkowicz, "ACES Filmic Tone Mapping Curve" (approximation fit),
//       blog post 2016; original ACES matrices by the Academy S-2014-003.

import common;
import bindings;
import interfaces;
import scene_trace;
import environment;
import integrators.path;
import integrators.bdpt;
import cameras.pinhole;
import cameras.thick_lens;
import generated_materials;
// FlatMaterial + loadFlatMaterial used by the BXDF visualizer tool branch.
import materials.flat.flat_material;
// Skin BSDF / BSSRDF helpers consumed by the BXDF visualizer for skin.
import materials.skin.skin_shading;
import materials.skin.skin_bssrdf;

// ── Tonemapping ─────────────────────────────────────────────────

/// ACES filmic tone-mapping curve (Narkowicz rational approximation) [4].
///
///   f(x) = (x · (a·x + b)) / (x · (c·x + d) + e)
///
/// where  a = 2.51,  b = 0.03,  c = 2.43,  d = 0.59,  e = 0.14
/// Maps linear HDR radiance → display [0, 1]. Applied per channel;
/// result is clamped to [0, 1] before sRGB gamma encoding.
float3 acesFilmic(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

/// Simplified sRGB gamma encoding (power-law approximation, γ = 2.2).
///
///   f(c) = max(c, 0)^(1/2.2)
///
/// The IEC 61966-2-1 standard uses a piecewise form with a linear toe;
/// the power-law approximation is accurate to < 1% for c > 0.003 and
/// avoids a branch in the shader.
float3 linearToSRGB(float3 linear)
{
    return pow(max(linear, float3(0.0)), float3(1.0 / 2.2));
}

// ── Per-pixel frame body, monomorphized on the camera model ─────

void runFrame<TC : ICamera>(TC cam, uint2 pixel, inout RNG rng)
{
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);
    HitInfo hit = traceScene(fc, ray);

    // BXDF visualizer pick. When the CPU has armed a pick and this pixel
    // matches the requested one, snapshot the hit into toolBuffer. Cheap:
    // one buffer write per frame at one pixel. Pick output sits at
    // slots 8..11 (slots 0..7 are reserved for the BXDF eval header).
    if (fc.pickArmed != 0u && pixel.x == fc.pickPixel.x && pixel.y == fc.pickPixel.y)
    {
        float matIdF = asfloat(hit.materialId);
        float hitFlagF = hit.hit ? 1.0 : 0.0;
        toolBuffer[8]  = float4(hit.position, hit.t);
        toolBuffer[9]  = float4(hit.normal, matIdF);
        toolBuffer[10] = float4(hit.tangent, hitFlagF);
        toolBuffer[11] = float4(hit.uv, hit.hasTangent ? 1.0 : 0.0, 0.0);
    }

    float3 sample;

    bool useBdpt = (fc.integratorType == INTEGRATOR_BDPT)
                && hit.hit
                && ((materialTypes[hit.materialId] & 0xFFu) == MATERIAL_TYPE_FLAT);

    if (useBdpt)
    {
        BDPTIntegrator<TC> bdpt;
        bdpt.cam = cam;
        sample = bdpt.estimateRadiance(ray, hit, rng);
    }
    else if (hit.hit)
    {
        PathTracer integrator;
        sample = integrator.estimateRadiance(ray, hit, rng);
    }
    else
    {
        sample = sceneEnvRadiance(ray.direction, fc.envIntensity, fc.furnaceMode);
    }

    // Apply the realistic-camera throughput weight (cos⁴θ / Z² · A_rear,
    // PBRT eq. 6.10). For pinhole rays this factor is 1.0; for vignetted
    // / TIR'd lens rays it is 0.0 and the sample blacks out.
    sample *= lensWeight;

    // Sanitize NaN / Inf / negative components without nuking healthy
    // channels: a tiny float-noise negative in green should not zero out
    // a valid red contribution.
    bool3 bad = isnan(sample) | isinf(sample);
    sample = select(bad, float3(0.0), sample);
    sample = max(sample, float3(0.0));

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

    // BDPT light-tracer (s = 1) composite. lightSplatBuffer accumulates
    // fixed-point Q22.10 radiance over all frames (atomic-added by
    // bdpt.slang's splatLightVertex); divide by frames-so-far to get the
    // running mean of the light-tracer estimator and add to the eye-side
    // accumulation. For non-BDPT modes the buffer stays zeroed (cleared on
    // accumulation reset) so the contribution is zero.
    uint base = (pixel.y * fc.width + pixel.x) * 3u;
    uint sR = lightSplatBuffer[base + 0u];
    uint sG = lightSplatBuffer[base + 1u];
    uint sB = lightSplatBuffer[base + 2u];
    float invScale = 1.0 / 1024.0;
    float3 splatSum = float3(sR, sG, sB) * invScale;
    float frames = float(fc.accumFrame + 1u);
    float3 splatMean = splatSum / frames;
    accum = accum + splatMean;

    float3 display = linearToSRGB(acesFilmic(accum));

    // Focal-plane visualiser: an infinite world-space plane (origin
    // `focusPlaneOrigin`, normal `focusPlaneNormal`). Intersect the
    // primary ray with the plane; if the intersection sits in front of
    // the camera and in front of the scene hit, alpha-blend a
    // translucent blue tint over `display`. The plane fades out
    // grazing-on so it reads as a soft sheet rather than a hard mask.
    if (fc.focusOverlay != 0u)
    {
        float denom = dot(ray.direction, fc.focusPlaneNormal);
        if (abs(denom) > 1e-5)
        {
            float tPlane = dot(fc.focusPlaneOrigin - ray.origin,
                               fc.focusPlaneNormal) / denom;
            float sceneT = hit.hit ? hit.t : 1e30;
            if (tPlane > 1e-3 && tPlane < sceneT)
            {
                // Grazing-angle falloff: dimmer when looking edge-on so the
                // plane reads volumetrically without a hard horizon line.
                float facing = clamp(abs(denom), 0.0, 1.0);
                float alpha = 0.32 * facing;
                float3 planeColor = float3(0.20, 0.55, 1.00);
                display = lerp(display, planeColor, alpha);
            }
        }
    }

    if (hit.hit && effectiveFurnaceMode(hit.materialId) != 0u)
    {
        float maxChan = max(accum.r, max(accum.g, accum.b));
        if (maxChan > 1.02)
        {
            display = float3(1.0, 0.35, 0.65);
        }
    }

    float hud = hudMask[pixel];
    if (hud > 0.75)
    {
        display = float3(1.0);
    }
    else if (hud > 0.0)
    {
        display = lerp(display, float3(0.02, 0.02, 0.03), hud);
    }

    // Gizmo overlay: cheap CPU→GPU 2D line list. For each segment we
    // measure the perpendicular distance from the current pixel to the
    // segment and blend the segment colour in with a 1-pixel anti-alias
    // falloff. Cap the loop at numGizmoSegments to skip the work entirely
    // when no gizmo is active.
    float2 px = float2(float(pixel.x) + 0.5, float(pixel.y) + 0.5);
    for (uint gi = 0u; gi < fc.numGizmoSegments; ++gi)
    {
        GizmoSegment seg = gizmoSegments[gi];
        float2 ab = seg.b - seg.a;
        float ab2 = max(dot(ab, ab), 1e-6);
        float t = clamp(dot(px - seg.a, ab) / ab2, 0.0, 1.0);
        float2 closest = seg.a + ab * t;
        float d = length(px - closest);
        float aa = clamp(seg.width + 1.0 - d, 0.0, 1.0);
        display = lerp(display, seg.color, aa);
    }

    outputBuffer[pixel] = float4(display, 1.0);
}

// ── Entry point ─────────────────────────────────────────────────

// ── BXDF visualizer eval ────────────────────────────────────────
//
// When `toolBuffer[0].x == TOOL_MODE_BXDF` the dispatch is hijacked: each
// thread maps to one (theta, phi) cell, evaluates the picked material's
// BSDF response in tangent space, and writes the result at slot
// `16 + j*n_theta + i`. Threads outside the requested grid early-return.
// The viewport image is *not* updated on tool frames — the CPU
// disarms (sets toolMode back to 0) after one frame so subsequent
// frames render normally.
void bxdfEvalThread(uint2 cell)
{
    uint lockMode = asuint(toolBuffer[0].y);
    uint nTheta   = asuint(toolBuffer[0].z);
    uint nPhi     = asuint(toolBuffer[0].w);
    if (cell.x >= nTheta || cell.y >= nPhi)
        return;

    uint matId = asuint(toolBuffer[1].x);
    float3 dLocked = toolBuffer[2].xyz;
    float3 P = toolBuffer[3].xyz;
    float3 N = toolBuffer[4].xyz;
    float3 T = toolBuffer[5].xyz;
    float2 UV = toolBuffer[6].xy;

    // Upper-hemisphere parameterisation. Theta ∈ (0, π/2), phi ∈ [0, 2π).
    float theta = (float(cell.x) + 0.5) / float(nTheta) * (0.5 * PI);
    float phi   = float(cell.y) / float(nPhi) * (2.0 * PI);
    float st = sin(theta), ct = cos(theta);
    float3 dSwept = float3(st * cos(phi), st * sin(phi), ct);

    float3 wo, wi;
    if (lockMode == 0u) { wi = dLocked; wo = dSwept; }
    else                { wo = dLocked; wi = dSwept; }

    // Synthetic HitInfo so the material loader can pull all the per-hit
    // bindings (StdSurfaceParams, MaterialX graph params, textures)
    // exactly the way the main bounce path does.
    HitInfo h;
    h.t              = 0.0;
    h.backT          = -1.0;
    h.position       = P;
    h.positionObject = P;
    h.normal         = N;
    h.tangent        = T;
    h.uv             = UV;
    h.hit            = true;
    h.hasTangent     = true;
    h.materialId     = matId;

    uint matType = materialTypes[matId] & 0xFFu;
    float3 response = float3(0.0);
    if (matType == MATERIAL_TYPE_FLAT)
    {
        FlatMaterial mat = loadFlatMaterial(h);
        BSDFEval ev = mat.evaluate(wo, wi);
        response = ev.response;
    }
    else if (matType == MATERIAL_TYPE_SKIN)
    {
        // Skin path renders via a self-integrating BSSRDF + surface
        // GGX lobe in the main path. For the visualizer we re-pack a
        // closed-form approximation: a Lambertian diffuse driven by
        // the layer-weighted single-scatter albedo (skin colour) with
        // Fresnel transmission at both interfaces, plus the same
        // surface specular term `evaluateSkinSpecular` uses in the
        // main path. Spatial BSSRDF diffusion (xi → xo distance) is
        // intentionally suppressed here — that's the `BSSRDF` mode.
        MtlxSkinParams mtlx = mtlxSkin[matId];
        float4 noInk = float4(0.0, 0.0, 0.0, 0.0);
        SkinLayerStack stack = buildSkinLayerStackFromMtlx(mtlx, noInk);

        float3 sigmaT_e = stack.epi.sigmaA + stack.epi.sigmaS * (1.0 - stack.epi.g);
        float3 sigmaT_d = stack.der.sigmaA + stack.der.sigmaS * (1.0 - stack.der.g);
        float3 sigmaT_s = stack.sub.sigmaA + stack.sub.sigmaS * (1.0 - stack.sub.g);
        float3 tau_e = sigmaT_e * stack.epi.thickness;
        float3 tau_d = sigmaT_d * stack.der.thickness;
        float3 tau_s = sigmaT_s * stack.sub.thickness;
        float3 totalTau = max(tau_e + tau_d + tau_s, float3(1e-6));
        float3 skinAlbedo = (tau_e * layerAlbedo(stack.epi)
                           + tau_d * layerAlbedo(stack.der)
                           + tau_s * layerAlbedo(stack.sub)) / totalTau;

        float n_wo = max(wo.z, 0.0);
        float n_wi = max(wi.z, 0.0);
        if (n_wi > 0.0 && n_wo > 0.0)
        {
            float Fi = fresnelSchlick(n_wi, mtlx.layer_top_ior);
            float Fo = fresnelSchlick(n_wo, mtlx.layer_top_ior);
            float3 diff = skinAlbedo * (1.0 / PI)
                        * (1.0 - Fi) * (1.0 - Fo);
            float3 Nz = float3(0.0, 0.0, 1.0);
            float3 spec = evaluateSkinSpecular(
                Nz, wo, wi, mtlx.skin_bsdf_roughness, mtlx.skin_bsdf_ior
            );
            response = (diff + spec) * n_wi;
        }
    }

    uint outIdx = 16u + cell.y * nTheta + cell.x;
    toolBuffer[outIdx] = float4(response, 1.0);
}

// BSSRDF eval (skin only). Reads xi (entrance) from header slot 7 and the
// exit / shading frame from slots 3..6 as in the BXDF path. Evaluates the
// separable diffusion BSSRDF
//   S(xi, wi → xo, wo) ≈ (1/π) · (1 − F(wi, η)) · R(r, d) · ρ · (1 − F(wo, η))
// with R = Burley diffusion profile, ρ = layer-weighted single-scatter
// albedo, η = layer_top_ior, r = ||xi − xo|| in world units. Each thread
// sweeps one wo on the upper hemisphere; the surface specular lobe is
// added so the visualisation shows both the diffuse subsurface response
// and the specular highlight at the exit point.
void bssrdfEvalThread(uint2 cell)
{
    uint lockMode = asuint(toolBuffer[0].y);
    uint nTheta   = asuint(toolBuffer[0].z);
    uint nPhi     = asuint(toolBuffer[0].w);
    if (cell.x >= nTheta || cell.y >= nPhi)
        return;

    uint matId = asuint(toolBuffer[1].x);
    if ((materialTypes[matId] & 0xFFu) != MATERIAL_TYPE_SKIN)
    {
        // BSSRDF is only meaningful for skin in this build; surface
        // non-skin materials with black so the user notices.
        uint outIdx = 16u + cell.y * nTheta + cell.x;
        toolBuffer[outIdx] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float3 dLocked = toolBuffer[2].xyz;
    float3 P  = toolBuffer[3].xyz;  // xo (exit, world)
    float3 xi = toolBuffer[7].xyz;  // entrance (world)
    // mmPerUnit folds the diffusion profile (which uses mm⁻¹ σ values)
    // into world units; the renderer pushes scene mm_per_unit into fc.
    float r_world = length(P - xi);
    float r_mm = r_world * fc.mmPerUnit;

    float theta = (float(cell.x) + 0.5) / float(nTheta) * (0.5 * PI);
    float phi   = float(cell.y) / float(nPhi) * (2.0 * PI);
    float st = sin(theta), ct = cos(theta);
    float3 dSwept = float3(st * cos(phi), st * sin(phi), ct);

    float3 wo, wi;
    if (lockMode == 0u) { wi = dLocked; wo = dSwept; }
    else                { wo = dLocked; wi = dSwept; }

    MtlxSkinParams mtlx = mtlxSkin[matId];
    float4 noInk = float4(0.0, 0.0, 0.0, 0.0);
    SkinLayerStack stack = buildSkinLayerStackFromMtlx(mtlx, noInk);

    float3 sigmaT_e = stack.epi.sigmaA + stack.epi.sigmaS * (1.0 - stack.epi.g);
    float3 sigmaT_d = stack.der.sigmaA + stack.der.sigmaS * (1.0 - stack.der.g);
    float3 sigmaT_s = stack.sub.sigmaA + stack.sub.sigmaS * (1.0 - stack.sub.g);
    float3 tau_e = sigmaT_e * stack.epi.thickness;
    float3 tau_d = sigmaT_d * stack.der.thickness;
    float3 tau_s = sigmaT_s * stack.sub.thickness;
    float3 totalTau = max(tau_e + tau_d + tau_s, float3(1e-6));
    float3 skinAlbedo = (tau_e * layerAlbedo(stack.epi)
                       + tau_d * layerAlbedo(stack.der)
                       + tau_s * layerAlbedo(stack.sub)) / totalTau;

    // Per-channel effective mean free path d = 1/σ_t. Use the dermis as
    // the dominant diffusion medium (where most of the lateral spread
    // happens); a fancier model would weight a multi-layer profile.
    float3 d_dermis = 1.0 / max(sigmaT_d, float3(1e-6));
    float3 R = burleyDiffusionProfile(max(r_mm, 1e-3), d_dermis);

    float3 response = float3(0.0);
    float n_wo = max(wo.z, 0.0);
    float n_wi = max(wi.z, 0.0);
    if (n_wi > 0.0 && n_wo > 0.0)
    {
        float Fi = fresnelSchlick(n_wi, mtlx.layer_top_ior);
        float Fo = fresnelSchlick(n_wo, mtlx.layer_top_ior);
        float3 diff = (1.0 - Fi) * (1.0 - Fo) * skinAlbedo * R / PI;
        float3 Nz = float3(0.0, 0.0, 1.0);
        float3 spec = evaluateSkinSpecular(
            Nz, wo, wi, mtlx.skin_bsdf_roughness, mtlx.skin_bsdf_ior
        );
        response = (diff + spec) * n_wi;
    }

    uint outIdx = 16u + cell.y * nTheta + cell.x;
    toolBuffer[outIdx] = float4(response, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void mainImage(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;

    // BXDF visualizer tool branch. When the host has armed an eval the
    // dispatch grid threads compute one lobe cell each; the regular
    // render path is skipped entirely for the frame.
    uint toolMode = asuint(toolBuffer[0].x);
    if (toolMode == TOOL_MODE_BXDF)
    {
        bxdfEvalThread(pixel);
        return;
    }
    if (toolMode == TOOL_MODE_BSSRDF)
    {
        bssrdfEvalThread(pixel);
        return;
    }

    if (pixel.x >= fc.width || pixel.y >= fc.height)
        return;

    RNG rng = createRNG(pixel, fc.frameIndex);

    // Vignette-debug mode: sample N lens rays at this pixel, count how
    // many produce a non-zero throughput, and output the success rate
    // as grayscale. Bypasses the integrator entirely so the cost is
    // bounded. Skips when no lens is active (success rate ≡ 1).
    if (fc.lensVignetteDebug != 0u && fc.numLensElements > 0u)
    {
        const uint N = 16u;
        uint hits = 0u;
        for (uint s = 0u; s < N; ++s)
        {
            float w;
            generateLensRay(fc, pixel, rng, w);
            if (w > 0.0) hits += 1u;
        }
        float rate = float(hits) / float(N);
        float3 col = lerp(float3(0.6, 0.0, 0.0),    // red = vignetted
                          float3(0.0, 0.9, 0.0),    // green = passing
                          rate);
        outputBuffer[pixel] = float4(col, 1.0);
        return;
    }

    // Tag-switch on camera model. Concrete struct instantiated at the
    // call site so estimators monomorphize against the chosen ICamera.
    if (fc.numLensElements > 0u)
    {
        ThickLensCamera cam;
        runFrame<ThickLensCamera>(cam, pixel, rng);
    }
    else
    {
        PinholeCamera cam;
        runFrame<PinholeCamera>(cam, pixel, rng);
    }
}
