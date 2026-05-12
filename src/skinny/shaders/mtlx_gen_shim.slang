// Shim shared by MaterialX-generated graph modules.
// Replaces MaterialXGenSlang's prelude `struct SamplerTexture2D { Texture2D tex;
// SamplerState sampler; }` with a uint-slot wrapper that resolves against
// skinny's bindless flatMaterialTextures[] array (binding 14). Per-graph
// modules `import mtlx_gen_shim;` and use `SamplerTexture2D` for `filename`
// uniforms. The Renderer packs each graph's filename inputs as the texture's
// bindless slot index.

import bindings;  // flatMaterialTextures: Sampler2D<float4>[128]

public struct SamplerTexture2D
{
    uint slot;

    public int get_width(int mipLevel = 0)
    {
        uint w, h, levels;
        flatMaterialTextures[NonUniformResourceIndex(slot)].GetDimensions(mipLevel, w, h, levels);
        return int(w);
    }

    public int get_height(int mipLevel = 0)
    {
        uint w, h, levels;
        flatMaterialTextures[NonUniformResourceIndex(slot)].GetDimensions(mipLevel, w, h, levels);
        return int(h);
    }

    public int get_num_mip_levels()
    {
        uint w, h, levels;
        flatMaterialTextures[NonUniformResourceIndex(slot)].GetDimensions(0, w, h, levels);
        return int(levels);
    }
};

public float4 texture(SamplerTexture2D tex, float2 uv)
{
    return flatMaterialTextures[NonUniformResourceIndex(tex.slot)].Sample(uv);
}

public float4 textureLod(SamplerTexture2D tex, float2 uv, float lod)
{
    return flatMaterialTextures[NonUniformResourceIndex(tex.slot)].SampleLevel(uv, lod);
}

public float4 textureGrad(SamplerTexture2D tex, float2 uv, float2 ddx_, float2 ddy_)
{
    return flatMaterialTextures[NonUniformResourceIndex(tex.slot)].SampleGrad(uv, ddx_, ddy_);
}

public int2 textureSize(SamplerTexture2D tex, int mipLevel)
{
    return int2(tex.get_width(mipLevel), tex.get_height(mipLevel));
}
