// Debug viewport line shader.
//
// One vertex stream of (position, color) line-list primitives plus a
// per-frame UBO holding the debug camera's view*proj. Used by the
// DebugViewport graphics pipeline to draw frustums, AABBs, lens rings,
// camera glyphs, ground grid, and mesh wireframes.

struct CameraUBO
{
    float4x4 viewProj;
};

[[vk::binding(0, 0)]] ConstantBuffer<CameraUBO> cam;

struct VSInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float4 color    : COLOR0;
};

struct VSOutput
{
    float4 sv_position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR0;
};

[shader("vertex")]
VSOutput vs_main(VSInput input)
{
    VSOutput o;
    // Sentinel: color.a > 1.5 marks a screen-space NDC vertex used for
    // the HUD text overlay. Bypass viewProj and emit position.xy as NDC
    // directly with z=0 (in front of everything when depth test runs).
    if (input.color.a > 1.5)
    {
        o.sv_position = float4(input.position.xy, 0.0, 1.0);
        o.color = float4(input.color.rgb, 1.0);
    }
    else
    {
        o.sv_position = mul(cam.viewProj, float4(input.position, 1.0));
        o.color = input.color;
    }
    return o;
}

struct FSOutput
{
    float4 color : SV_Target0;
};

[shader("fragment")]
FSOutput fs_main(VSOutput input)
{
    FSOutput o;
    o.color = input.color;
    return o;
}
