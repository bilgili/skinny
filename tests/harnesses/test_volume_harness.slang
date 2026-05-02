// Test harness for HG phase function (inlined to avoid environment.slang binding).

import common;

static const float PI_VAL = 3.14159265358979;

float test_henyeyGreenstein(float cosTheta, float g)
{
    float g2  = g * g;
    float num = 1.0 - g2;
    float den = pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
    return num / (4.0 * PI_VAL * den);
}

float3 test_sampleHenyeyGreenstein(float3 forward, float g, float2 u)
{
    float cosTheta;
    if (abs(g) > 1e-3)
    {
        float t = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
        cosTheta = (1.0 + g * g - t * t) / (2.0 * g);
    }
    else
    {
        cosTheta = 1.0 - 2.0 * u.x;
    }
    cosTheta = clamp(cosTheta, -1.0, 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi      = 2.0 * PI_VAL * u.y;

    float3 tangent   = abs(forward.x) > 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 bitangent = normalize(cross(forward, tangent));
    tangent          = cross(bitangent, forward);

    return normalize(
        tangent   * cos(phi) * sinTheta +
        bitangent * sin(phi) * sinTheta +
        forward * cosTheta
    );
}
