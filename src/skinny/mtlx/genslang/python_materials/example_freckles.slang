public float3 freckle_albedo_tint(float3 base_albedo, float3 freckle_color, float mask)
{
    return sp.lerp(base_albedo, freckle_color, mask);
}

public float freckle_density(float3 position, float scale, float seed, float density)
{
    var p = (position * scale);
    var h = sp.fract((sp.sin((sp.dot(p, float3(127.1, 311.7, 74.7)) + seed)) * 43758.5453));
    return sp.step((1.0 - density), h);
}
