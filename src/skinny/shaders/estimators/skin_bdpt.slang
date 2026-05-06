// Stored light-subpath BDPT estimator (term §7). Active only when
// fc.integratorMode == 3. Body lifted verbatim from the original
// storedLightSubpathBDPT in skin_material.slang.

import common;
import bindings;
import scene_trace;
import environment;
import skin_bssrdf;
import skin_shading;

float3 skinBDPTEstimator(
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

    float3 Le = sceneEnvRadiance(wiEnv, fc.envIntensity, effectiveFurnaceMode(cameraHit.materialId));
    float geometry = cosCamera * cosLightOut / dist2;
    return Le * fLight * fCamera * geometry * invPdfArea;
}
