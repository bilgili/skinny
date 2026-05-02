// Test harness for environment mapping functions.
// We re-implement directionToEquirectUV here to avoid importing
// environment.slang which requires a texture binding.

import common;

float2 test_directionToEquirectUV(float3 dir)
{
    float3 d = normalize(dir);
    float phi   = atan2(d.x, d.z);
    float theta = acos(clamp(d.y, -1.0, 1.0));
    return float2(phi / (2.0 * PI) + 0.5, theta / PI);
}
