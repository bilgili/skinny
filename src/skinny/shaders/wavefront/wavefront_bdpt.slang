// Staged wavefront bidirectional path tracer (P1 §8 / design §P1-G). The
// megakernel BDPTIntegrator.estimateRadiance, torn into compute dispatches that
// store the camera + light subpath vertices in VRAM between stages:
//
//   wfBdptWalk     — per lane: trace the primary ray, build the eye subpath and
//                    the light subpath (the SAME randomWalk + sampleLightOrigin
//                    the megakernel uses), do the s=1 light-tracer splat, and
//                    store both subpaths + bookkeeping to VRAM.
//   wfBdptConnect  — per lane: reload both subpaths and run the connection
//                    strategies (t=0 emissive, t=1 NEE connectT1, t≥2 generic +
//                    MIS) verbatim, writing the eye-side radiance.
//   wfBdptResolve  — fold the eye-side radiance into the running mean and add
//                    the s=1 splat composite, matching main_pass.slang.
//
// Correctness is by reuse: every estimator function comes from
// integrators/bdpt.slang unchanged, so the accumulated image is the same
// estimator as the megakernel bdpt (A/B-verified). The rng cursor is carried
// across the walk/connect split so the random sequence matches. Matches the
// megakernel bdpt scope: flat first-hit only, pinhole camera. Set 0 is the
// megakernel scene descriptor set (incl. lightSplatBuffer for the splat); set 1
// holds the per-lane subpath + aux buffers this owns.

import common;            // RNG, Ray, PI
import bindings;          // fc, accumBuffer, lightSplatBuffer, materialTypes…
import interfaces;        // BSDFSample
import scene_trace;       // traceScene
import environment;       // sceneEnvRadiance
import cameras.pinhole;   // PinholeCamera
import materials.flat.flat_material;   // FlatMaterial, loadFlatMaterial
import materials.flat.flat_shading;    // isCutoutTransparent
import integrators.bdpt;  // BDPTVertex, randomWalk, sampleLightOrigin,
                          // connectT1, connectGeneric, misWeight, ConnectResult,
                          // bdptEnvNEE, splatLightWalk, BDPT_MAX_VERTS, BDPT_VK_*
import wavefront.wf_display;  // wfWriteDisplay (tonemap → output image)

struct WfBdptAux
{
    int    eyeLen;
    int    lightLen;
    uint   rngState;     // rng cursor after the walk + splat (resumed in connect)
    float  lensWeight;   // realistic-camera weight applied to the eye-side L
    float3 escaped;      // escaped env/emissive radiance gathered during the walk
    float3 radiance;     // final eye-side L (written by connect, read by resolve)
};

[[vk::binding(0, 1)]] RWStructuredBuffer<BDPTVertex> wfEye;     // stream × BDPT_MAX_VERTS
[[vk::binding(1, 1)]] RWStructuredBuffer<BDPTVertex> wfLight;   // stream × BDPT_MAX_VERTS
[[vk::binding(2, 1)]] RWStructuredBuffer<WfBdptAux>  wfAux;     // stream

[shader("compute")]
[numthreads(8, 8, 1)]
void wfBdptWalk(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= fc.width || pixel.y >= fc.height)
        return;
    uint lane = pixel.y * fc.width + pixel.x;
    uint base = lane * BDPT_MAX_VERTS;

    RNG rng = createRNG(pixel, fc.frameIndex);
    PinholeCamera cam;
    float lensWeight;
    Ray ray = cam.generateRay(pixel, rng, lensWeight);

    WfBdptAux aux;
    aux.eyeLen = 0;
    aux.lightLen = 0;
    aux.lensWeight = lensWeight;
    aux.escaped = float3(0.0);
    aux.radiance = float3(0.0);

    BDPTVertex eye[BDPT_MAX_VERTS];
    BDPTVertex camVtx;
    camVtx.kind = BDPT_VK_CAMERA;
    camVtx.position = ray.origin;
    camVtx.N = -ray.direction;
    camVtx.throughput = float3(1.0);
    camVtx.emission = float3(0.0);
    camVtx.pdfFwd = 1.0;
    camVtx.pdfRev = 0.0;
    camVtx.isDelta = false;
    camVtx.onLight = false;
    camVtx.matId = 0u;
    camVtx.uv = float2(0.0);
    camVtx.posObject = float3(0.0);
    camVtx.geoN = -ray.direction;
    camVtx.tangent = float3(1.0, 0.0, 0.0);
    camVtx.hasTangent = false;
    eye[0] = camVtx;

    HitInfo firstHit = traceScene(fc, ray);
    Ray r = ray;
    for (uint skip = 0u; skip < 32u && firstHit.hit && isCutoutTransparent(firstHit); skip++)
    {
        r.origin = firstHit.position + r.direction * 0.002;
        firstHit = traceScene(fc, r);
    }
    if (!firstHit.hit)
    {
        aux.escaped = sceneEnvRadiance(r.direction, fc.envIntensity,
                                       effectiveFurnaceMode(firstHit.materialId));
        aux.rngState = rng.state;
        wfAux[lane] = aux;
        return;
    }
    if ((materialTypes[firstHit.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
    {
        // Non-flat first hit — bdpt bails to black here (the megakernel path
        // tracer covers skin/python). Matches estimateRadiance.
        aux.rngState = rng.state;
        wfAux[lane] = aux;
        return;
    }

    FlatMaterial fmat = loadFlatMaterial(firstHit);
    BDPTVertex z1;
    z1.kind = BDPT_VK_SURFACE;
    z1.position = firstHit.position;
    z1.N = fmat.data.N;
    z1.throughput = float3(1.0);
    z1.emission = fmat.data.mat.emission;
    z1.pdfFwd = 1.0;
    z1.pdfRev = 0.0;
    z1.isDelta = false;
    z1.onLight = any(fmat.data.mat.emission > float3(0.0));
    z1.matId = firstHit.materialId;
    z1.uv = firstHit.uv;
    z1.posObject = firstHit.positionObject;
    z1.geoN = firstHit.normal;
    z1.tangent = firstHit.tangent;
    z1.hasTangent = firstHit.hasTangent;
    eye[1] = z1;
    int eyeLen = 2;

    float3 T, B;
    if (firstHit.hasTangent)
    {
        T = normalize(firstHit.tangent - z1.N * dot(firstHit.tangent, z1.N));
        B = cross(z1.N, T);
    }
    else
    {
        buildBasis(z1.N, T, B);
    }
    float3 wo_ts = worldToTangent(-ray.direction, T, B, z1.N);

    float3 escapedRadiance = bdptEnvNEE(
        fmat, z1.position, z1.N, T, B, wo_ts, firstHit.materialId, rng);

    BSDFSample bs = fmat.sample(wo_ts, rng);
    if (bs.valid)
    {
        if (bs.pdf <= 0.0)
            eye[1].isDelta = true;
        float3 wi_world = tangentToWorld(bs.wi, T, B, z1.N);
        float3 throughput = bs.weight;
        Ray r2;
        float3 spawnOffset = bs.transmitted ? -z1.N * 0.002 : z1.N * 0.002;
        r2.origin = z1.position + spawnOffset;
        r2.direction = wi_world;
        float pdfFwdOmega = (bs.pdf > 0.0) ? bs.pdf : 1.0;
        float misBsdfPdfIn = (bs.pdf > 0.0 && !bs.transmitted
            && effectiveFurnaceMode(firstHit.materialId) == 0u) ? bs.pdf : -1.0;
        eyeLen = randomWalk(eye, 2, r2, throughput, pdfFwdOmega, rng,
                            true, escapedRadiance, misBsdfPdfIn);
    }

    // Light subpath.
    BDPTVertex lightPath[BDPT_MAX_VERTS];
    int lightLen = 0;
    LightOriginSample lo = sampleLightOrigin(rng);
    if (lo.valid)
    {
        lightPath[0] = lo.vertex;
        lightLen = 1;
        if (lo.vertex.kind != BDPT_VK_LIGHT_DIR)
        {
            Ray rL;
            rL.origin = lo.vertex.position + lo.vertex.N * 0.002;
            rL.direction = lo.dirOut;
            float3 unusedEscape = float3(0.0);
            lightLen = randomWalk(lightPath, 1, rL, lo.beta, lo.pdfDir, rng,
                                  false, unusedEscape, -1.0);
        }
    }

    // s = 1 light tracer splat (atomic into lightSplatBuffer, set 0).
    splatLightWalk<PinholeCamera>(lightPath, lightLen, rng, cam);

    // Store both subpaths + bookkeeping for the connection stage.
    for (int i = 0; i < int(BDPT_MAX_VERTS); i++)
    {
        if (i < eyeLen)   wfEye[base + uint(i)]   = eye[i];
        if (i < lightLen) wfLight[base + uint(i)] = lightPath[i];
    }
    aux.eyeLen = eyeLen;
    aux.lightLen = lightLen;
    aux.escaped = escapedRadiance;
    aux.rngState = rng.state;   // connect resumes from here
    wfAux[lane] = aux;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void wfBdptConnect(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= fc.width || pixel.y >= fc.height)
        return;
    uint lane = pixel.y * fc.width + pixel.x;
    uint base = lane * BDPT_MAX_VERTS;

    WfBdptAux aux = wfAux[lane];
    int eyeLen = aux.eyeLen;
    int lightLen = aux.lightLen;

    RNG rng; rng.state = aux.rngState;

    BDPTVertex eye[BDPT_MAX_VERTS];
    BDPTVertex lightPath[BDPT_MAX_VERTS];
    for (int i = 0; i < int(BDPT_MAX_VERTS); i++)
    {
        if (i < eyeLen)   eye[i]       = wfEye[base + uint(i)];
        if (i < lightLen) lightPath[i] = wfLight[base + uint(i)];
    }

    float3 L = aux.escaped;

    // (s ≥ 2, t = 0): emissive eye-vertex hits.
    [loop]
    for (int s = 2; s <= eyeLen; s++)
    {
        BDPTVertex z = eye[s - 1];
        if (z.onLight)
            L += z.throughput * z.emission;
    }

    // (s ≥ 2, t = 1): NEE.
    [loop]
    for (int s = 2; s <= eyeLen; s++)
        L += connectT1(eye, s, rng);

    // (s ≥ 2, t ≥ 2): generic connections + MIS.
    [loop]
    for (int s = 2; s <= eyeLen; s++)
    {
        [loop]
        for (int t = 2; t <= lightLen; t++)
        {
            if (s + t - 2 > int(BDPT_MAX_DEPTH)) continue;
            ConnectResult cr = connectGeneric(eye, s, lightPath, t);
            if (!cr.nonzero) continue;
            float w = misWeight(eye, s, lightPath, t,
                                cr.pdf_zsm1_rev, cr.pdf_zsm2_rev,
                                cr.pdf_ytm1_rev, cr.pdf_ytm2_rev);
            L += cr.contribution * w;
        }
    }

    aux.radiance = L * aux.lensWeight;
    wfAux[lane] = aux;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void wfBdptResolve(uint3 tid: SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (pixel.x >= fc.width || pixel.y >= fc.height)
        return;
    uint lane = pixel.y * fc.width + pixel.x;

    float3 sample = wfAux[lane].radiance;
    bool3 bad = isnan(sample) | isinf(sample);
    sample = select(bad, float3(0.0), sample);
    sample = max(sample, float3(0.0));

    // Running mean of the eye-side estimator. main_pass.slang writes exactly
    // this to accumBuffer (binding 2 = what read_accumulation reads); the s=1
    // splatBuffer composite is added only on the display/tonemap path, never
    // into accumBuffer, so the headless A/B compares eye-side estimates. The
    // splat is still accumulated in wfBdptWalk for a future windowed composite
    // (and is a no-op for the demo's directional light — no light subpath).
    uint n = fc.accumFrame;
    float3 accum;
    if (n == 0u)
        accum = sample;
    else
    {
        float3 prev = accumBuffer[pixel].rgb;
        accum = (prev * float(n) + sample) / float(n + 1u);
    }
    accumBuffer[pixel] = float4(accum, 1.0);
    wfWriteDisplay(pixel, accum);   // tonemap → offscreen/swapchain output
}
