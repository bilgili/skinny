// GPU linear-blend skinning (UsdSkel).
//
// One thread per skinned vertex. Reads the rest vertex (position + normal,
// with u/v passed through unchanged), four joint influences, and the per-joint
// skinning matrices, and writes the deformed vertex into the shared mesh
// vertex buffer at `outOffset`.
//
// Matrix convention matches the rest of skinny: matrices are uploaded from
// numpy (row-vector "stored" form) and applied as `mul(M, v)`; the
// numpy-row → GPU-column reinterpretation makes this equivalent to the CPU
// `p @ M` used in usd_loader.lbs_points (validated in tests/test_usd_skel.py).
//
// Deformed points come out in the authored-points space, so the instance's
// existing TLAS transform places them — no world fold here (see design D2).

struct MeshVertex
{
    float3 position;
    float  u;
    float3 normal;
    float  v;
};

[[vk::binding(0)]] StructuredBuffer<MeshVertex>   restVertices;
[[vk::binding(1)]] StructuredBuffer<int4>         jointIndices;
[[vk::binding(2)]] StructuredBuffer<float4>       jointWeights;
[[vk::binding(3)]] StructuredBuffer<float4x4>     jointMatrices;
[[vk::binding(4)]] RWStructuredBuffer<MeshVertex> outVertices;

struct SkinPush
{
    uint vertexCount;   // number of skinned vertices in this mesh
    uint outOffset;     // start index of this mesh's BLAS in outVertices
};
[[vk::push_constant]] SkinPush pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void skinMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.vertexCount)
        return;

    MeshVertex rv = restVertices[i];
    int4   ji = jointIndices[i];
    float4 jw = jointWeights[i];

    float3 p = float3(0.0, 0.0, 0.0);
    float3 n = float3(0.0, 0.0, 0.0);
    [unroll]
    for (int k = 0; k < 4; ++k)
    {
        float w = jw[k];
        if (w == 0.0)
            continue;
        float4x4 m = jointMatrices[ji[k]];
        p += w * mul(m, float4(rv.position, 1.0)).xyz;
        n += w * mul(m, float4(rv.normal,   0.0)).xyz;
    }

    MeshVertex outv;
    outv.position = p;
    outv.u        = rv.u;
    outv.normal   = (dot(n, n) > 1e-12) ? normalize(n) : rv.normal;
    outv.v        = rv.v;
    outVertices[pc.outOffset + i] = outv;
}
