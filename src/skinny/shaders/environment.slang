// HDR environment map — equirectangular lookup for image-based lighting.

import common;

[[vk::binding(4)]] Sampler2D<float4> envMap;

// Direction → equirectangular UV. Convention: +y up, phi around y from +z.
float2 directionToEquirectUV(float3 dir)
{
    float3 d = normalize(dir);
    float phi   = atan2(d.x, d.z);                    // [-π, π]
    float theta = acos(clamp(d.y, -1.0, 1.0));        // [0, π], 0 at +y (zenith)
    return float2(phi / (2.0 * PI) + 0.5, theta / PI);
}

float3 sampleEnvironment(float3 dir, float intensity)
{
    float2 uv = directionToEquirectUV(dir);
    return envMap.SampleLevel(uv, 0.0).rgb * intensity;
}

// Unit-white fallback when furnace / energy-conservation mode is active.
// Every direction carries L = 1 per channel, which is what lets the furnace
// probe check outgoing vs. incoming radiance.
float3 sceneEnvRadiance(float3 dir, float intensity, uint furnaceMode)
{
    if (furnaceMode != 0u) return float3(1.0);
    return sampleEnvironment(dir, intensity);
}
