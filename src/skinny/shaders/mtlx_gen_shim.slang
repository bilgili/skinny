// Shim shared by MaterialX-generated graph modules.
// Replaces MaterialXGenSlang's prelude `struct SamplerTexture2D { Texture2D tex;
// SamplerState sampler; }` with a uint-slot wrapper that resolves against
// skinny's bindless flatMaterialTextures[] array (binding 14). Per-graph
// modules `import mtlx_gen_shim;` and use `SamplerTexture2D` for `filename`
// uniforms. The Renderer packs each graph's filename inputs as the texture's
// bindless slot index.

import bindings;  // flatMaterialTextures: Sampler2D<float4>[128]

// SENTINEL slot — set by Python's TexturePool.add_or_get when a texture
// fails to load (missing file, decode error, bindless array full). The
// shim treats SENTINEL as transparent black so the path tracer doesn't
// sample an unbound bindless descriptor (Vulkan validation error +
// undefined sample). Must match TexturePool.SENTINEL in renderer.py.
static const uint MTLX_GEN_TEX_SENTINEL = 0xFFFFFFFFu;

public struct SamplerTexture2D
{
    uint slot;

    public int get_width(int mipLevel = 0)
    {
        if (slot == MTLX_GEN_TEX_SENTINEL) return 0;
        uint w, h, levels;
        flatMaterialTextures[NonUniformResourceIndex(slot)].GetDimensions(mipLevel, w, h, levels);
        return int(w);
    }

    public int get_height(int mipLevel = 0)
    {
        if (slot == MTLX_GEN_TEX_SENTINEL) return 0;
        uint w, h, levels;
        flatMaterialTextures[NonUniformResourceIndex(slot)].GetDimensions(mipLevel, w, h, levels);
        return int(h);
    }

    public int get_num_mip_levels()
    {
        if (slot == MTLX_GEN_TEX_SENTINEL) return 0;
        uint w, h, levels;
        flatMaterialTextures[NonUniformResourceIndex(slot)].GetDimensions(0, w, h, levels);
        return int(levels);
    }
};

public float4 texture(SamplerTexture2D tex, float2 uv)
{
    if (tex.slot == MTLX_GEN_TEX_SENTINEL) return float4(0.0);
    // SampleLevel(0) instead of Sample(): plain `Sample` uses implicit
    // ddx/ddy from quad neighbors, which in a compute shader needs the
    // VK_KHR_compute_shader_derivatives extension. skinny's compute
    // pipeline doesn't request it; path-tracer texture lookups don't
    // have a meaningful pixel footprint anyway, so mip 0 is fine.
    return flatMaterialTextures[NonUniformResourceIndex(tex.slot)].SampleLevel(uv, 0.0);
}

public float4 textureLod(SamplerTexture2D tex, float2 uv, float lod)
{
    if (tex.slot == MTLX_GEN_TEX_SENTINEL) return float4(0.0);
    return flatMaterialTextures[NonUniformResourceIndex(tex.slot)].SampleLevel(uv, lod);
}

public float4 textureGrad(SamplerTexture2D tex, float2 uv, float2 ddx_, float2 ddy_)
{
    if (tex.slot == MTLX_GEN_TEX_SENTINEL) return float4(0.0);
    return flatMaterialTextures[NonUniformResourceIndex(tex.slot)].SampleGrad(uv, ddx_, ddy_);
}

public int2 textureSize(SamplerTexture2D tex, int mipLevel)
{
    return int2(tex.get_width(mipLevel), tex.get_height(mipLevel));
}
