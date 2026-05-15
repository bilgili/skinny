// Material preview compute entry — renders a single primitive (sphere /
// cube / plane) shaded with the user-picked MaterialX material, used by
// the Material Graph Editor window.
//
// Reuses the main descriptor set 0 (so `bindings.slang` symbols are
// already populated by the renderer) and adds its own output image on
// set 1 binding 0. All preview-specific parameters arrive via a single
// push-constants block — no UBO updates required per dispatch.
//
// Keep the PreviewPushConsts layout in sync with
// `skinny.vk_compute.PreviewPipeline._PUSH_FMT`.

import common;
import bindings;
import mtlx_std_surface;
import mtlx_closures;
import environment;
import generated_materials;

[[vk::binding(0, 1)]] RWTexture2D<float4> previewOutput;

struct PreviewPushConsts
{
    uint  matId;
    uint  graphId;
    uint  primKind;   // 0 sphere, 1 cube, 2 plane
    uint  size;       // square output dimension (pixels)
    float yaw;        // orbit, radians
    float pitch;      // orbit, radians
    float distance;   // camera distance from origin
    float fovTan;     // tan(half-fov), governs framing
};
[[vk::push_constant]] ConstantBuffer<PreviewPushConsts> pc;

static const float PREVIEW_HIT_EPS = 1e-3;

bool intersectSphere(float3 O, float3 D,
                     out float t, out float3 N, out float3 T, out float2 UV)
{
    t = 0.0;
    N = float3(0, 1, 0);
    T = float3(1, 0, 0);
    UV = float2(0, 0);
    float r = 1.0;
    float b = dot(O, D);
    float c = dot(O, O) - r * r;
    float disc = b * b - c;
    if (disc < 0.0) return false;
    float sd = sqrt(disc);
    float t0 = -b - sd;
    float t1 = -b + sd;
    t = (t0 > PREVIEW_HIT_EPS) ? t0 : t1;
    if (t < PREVIEW_HIT_EPS) return false;
    float3 P = O + t * D;
    N = normalize(P);
    float3 up = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    T = normalize(cross(up, N));
    UV.x = atan2(N.z, N.x) / (2.0 * PI) + 0.5;
    UV.y = asin(clamp(N.y, -1.0, 1.0)) / PI + 0.5;
    return true;
}

bool intersectCube(float3 O, float3 D,
                   out float t, out float3 N, out float3 T, out float2 UV)
{
    t = 0.0;
    N = float3(0, 1, 0);
    T = float3(1, 0, 0);
    UV = float2(0, 0);
    float3 inv = 1.0 / D;
    float3 tA = (-1.0 - O) * inv;
    float3 tB = ( 1.0 - O) * inv;
    float3 tmin3 = min(tA, tB);
    float3 tmax3 = max(tA, tB);
    float tN = max(max(tmin3.x, tmin3.y), tmin3.z);
    float tF = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (tN > tF || tF < PREVIEW_HIT_EPS) return false;
    t = (tN > PREVIEW_HIT_EPS) ? tN : tF;
    float3 P = O + t * D;
    float3 absP = abs(P);
    if (absP.x >= absP.y && absP.x >= absP.z)
    {
        float s = sign(P.x);
        N = float3(s, 0, 0);
        T = float3(0, 0, s);
        UV = (float2(-P.z * s, P.y) + 1.0) * 0.5;
    }
    else if (absP.y >= absP.z)
    {
        float s = sign(P.y);
        N = float3(0, s, 0);
        T = float3(1, 0, 0);
        UV = (float2(P.x, -P.z * s) + 1.0) * 0.5;
    }
    else
    {
        float s = sign(P.z);
        N = float3(0, 0, s);
        T = float3(-s, 0, 0);
        UV = (float2(P.x * s, P.y) + 1.0) * 0.5;
    }
    return true;
}

bool intersectPlane(float3 O, float3 D,
                    out float t, out float3 N, out float3 T, out float2 UV)
{
    t = 0.0;
    N = float3(0, 1, 0);
    T = float3(1, 0, 0);
    UV = float2(0, 0);
    if (abs(D.y) < 1e-5) return false;
    t = -O.y / D.y;
    if (t < PREVIEW_HIT_EPS) return false;
    float3 P = O + t * D;
    // Bounded 4x4 quad so the preview shows the material rather than
    // smearing to the horizon. Outside the bound, fall through to env.
    if (abs(P.x) > 2.0 || abs(P.z) > 2.0) return false;
    N = float3(0, 1, 0);
    T = float3(1, 0, 0);
    UV = (P.xz + 2.0) * 0.25;
    return true;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void previewMain(uint3 tid : SV_DispatchThreadID)
{
    uint S = pc.size;
    if (tid.x >= S || tid.y >= S) return;

    // Orbit camera around origin. yaw rotates around world +Y; pitch
    // tilts vertically.
    float cy = cos(pc.yaw);
    float sy = sin(pc.yaw);
    float cp = cos(pc.pitch);
    float spp = sin(pc.pitch);
    float3 camPos = float3(sy * cp, spp, cy * cp) * pc.distance;
    float3 forward = normalize(-camPos);
    float3 worldUp = abs(forward.y) < 0.999 ? float3(0, 1, 0) : float3(0, 0, 1);
    float3 right = normalize(cross(forward, worldUp));
    float3 up = cross(right, forward);

    float2 ndc = (float2(float(tid.x), float(tid.y)) + 0.5) / float(S) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float3 D = normalize(forward + right * ndc.x * pc.fovTan + up * ndc.y * pc.fovTan);
    float3 O = camPos;

    float t = 0.0;
    float3 N = float3(0, 1, 0);
    float3 T = float3(1, 0, 0);
    float2 UV = float2(0, 0);
    bool hit = false;
    if (pc.primKind == 0u)
        hit = intersectSphere(O, D, t, N, T, UV);
    else if (pc.primKind == 1u)
        hit = intersectCube(O, D, t, N, T, UV);
    else
        hit = intersectPlane(O, D, t, N, T, UV);

    float3 color;
    if (!hit)
    {
        color = sampleEnvironment(D, fc.envIntensity);
    }
    else
    {
        float3 P = O + t * D;
        StdSurfaceParams sp = stdSurfaceParams[pc.matId];
        // graphId 0/1 reserved (skin / flat) — only graph IDs ≥ 2 invoke
        // the per-pixel MaterialX nodegraph dispatch.
        if (pc.graphId >= 2u)
            evalSceneGraph(pc.graphId, pc.matId, P, N, T, UV, sp);

        // Re-orthogonalise tangent against the (possibly graph-perturbed)
        // normal so evalStdSurfaceBSDF's frame stays valid.
        T = normalize(T - N * dot(T, N));
        if (dot(T, T) < 1e-6) T = abs(N.y) < 0.999 ? normalize(cross(float3(0,1,0), N))
                                                   : float3(1, 0, 0);

        float3 V = -D;
        // `lightDirection` in the shared UBO points TOWARD the light source
        // (see materials/skin/skin_material.slang:129 and
        // integrators/bdpt.slang:128 for the convention).
        float3 L = normalize(lightDirection);
        ClosureData cd = makeClosureData(
            CLOSURE_TYPE_REFLECTION, L, V, N, P, 1.0);
        BSDF brdf = evalStdSurfaceBSDF(cd, sp, N, T);
        // bsdf.response already includes the NdotL cosine.
        float3 direct = brdf.response * lightRadiance;

        // Wrap-around key from camera direction so materials never go
        // pitch-black when the scene's analytical light points away from
        // the orbit camera. Independent of `lightRadiance` so the preview
        // remains readable on scenes that turn the key light off.
        float3 Lkey = normalize(V + float3(0.3, 0.6, 0.0));
        ClosureData cdKey = makeClosureData(
            CLOSURE_TYPE_REFLECTION, Lkey, V, N, P, 1.0);
        BSDF brdfKey = evalStdSurfaceBSDF(cdKey, sp, N, T);
        float3 keyFill = brdfKey.response * float3(0.6, 0.6, 0.6);

        // Cheap ambient term: env at the surface normal modulated by the
        // diffuse tint so non-graph materials still pick up colour-bleed
        // from the IBL when scene env intensity is non-zero.
        float3 envN = sampleEnvironment(N, max(fc.envIntensity, 0.0));
        float3 baseTint = max(sp.base_color, 0.0) * max(sp.base, 0.0);
        float3 ambient = baseTint * (1.0 - sp.metalness) * envN;

        color = direct + keyFill + ambient + evalStdSurfaceEmission(sp);
    }

    // Simple Reinhard tonemap + gamma 2.2. Matches what the editor's
    // PIL conversion expects (already-encoded sRGB byte stream).
    float3 mapped = color / (1.0 + color);
    mapped = pow(max(mapped, float3(0.0)), float3(1.0 / 2.2));
    previewOutput[uint2(tid.x, tid.y)] = float4(mapped, 1.0);
}
