// GPU BVH refit over skinned (deformed) geometry.
//
// Keeps the bind-pose tree topology; recomputes node AABBs from the current
// (deformed) vertex positions. Single-threaded reverse-array-order pass: the
// depth-first preorder build guarantees a parent's index precedes both
// children, so iterating nodes high→low refits every child before its parent.
// Leaf nodes recompute from their triangles; inner nodes union their two
// children. Correct and simple; an atomic/parallel pass is a later scale tweak.
//
// Node encoding matches mesh.py::_pack_bvh / mesh_head.slang::BvhNode:
//   leftOrCount  < 0 : leaf, (-leftOrCount) triangles starting at rightOrFirst
//                ≥ 0 : inner, left-child index
//   rightOrFirst     : leaf → first triangle index; inner → right-child index
// Indices/children are BLAS-local; the push-constant offsets locate this
// mesh's slice of the shared buffers.

struct MeshVertex
{
    float3 position;
    float  u;
    float3 normal;
    float  v;
};

struct BvhNode
{
    float3 aabbMin;
    int    leftOrCount;
    float3 aabbMax;
    int    rightOrFirst;
};

[[vk::binding(0)]] StructuredBuffer<MeshVertex>   meshVertices;   // deformed
[[vk::binding(1)]] StructuredBuffer<uint>         meshIndices;
[[vk::binding(2)]] RWStructuredBuffer<BvhNode>    bvhNodes;

struct RefitPush
{
    uint nodeCount;     // BVH nodes in this mesh
    uint nodeOffset;    // start index of this mesh's BVH in bvhNodes
    uint indexOffset;   // start index of this mesh's tris in meshIndices (in uints)
    uint vertexOffset;  // start index of this mesh's verts in meshVertices
};
[[vk::push_constant]] RefitPush pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void refitMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0)
        return;

    // Reverse array order: children (higher index) are refit before parents.
    for (int local = int(pc.nodeCount) - 1; local >= 0; --local)
    {
        uint ni = pc.nodeOffset + uint(local);
        BvhNode node = bvhNodes[ni];

        float3 lo = float3( 1e30,  1e30,  1e30);
        float3 hi = float3(-1e30, -1e30, -1e30);

        if (node.leftOrCount < 0)
        {
            // Leaf: union the deformed triangle vertices.
            int count = -node.leftOrCount;
            uint first = uint(node.rightOrFirst);
            for (int t = 0; t < count; ++t)
            {
                uint triBase = pc.indexOffset + (first + uint(t)) * 3u;
                [unroll]
                for (int c = 0; c < 3; ++c)
                {
                    uint vi = pc.vertexOffset + meshIndices[triBase + uint(c)];
                    float3 p = meshVertices[vi].position;
                    lo = min(lo, p);
                    hi = max(hi, p);
                }
            }
        }
        else
        {
            // Inner: union the two child AABBs (already refit this pass).
            BvhNode lc = bvhNodes[pc.nodeOffset + uint(node.leftOrCount)];
            BvhNode rc = bvhNodes[pc.nodeOffset + uint(node.rightOrFirst)];
            lo = min(lc.aabbMin, rc.aabbMin);
            hi = max(lc.aabbMax, rc.aabbMax);
        }

        node.aabbMin = lo;
        node.aabbMax = hi;
        bvhNodes[ni] = node;
    }
}
