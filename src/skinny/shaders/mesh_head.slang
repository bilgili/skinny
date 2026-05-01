// Polygonal head — traversal of a pre-built linear BVH over triangles.
//
// The GPU-Gems-3 skin chapter uses a laser-scanned head mesh. Here we
// accept any OBJ placed in `heads/`; the CPU builds a median-split BVH
// and uploads three storage buffers: vertices, triangle indices, and
// 32-byte BVH nodes in depth-first order.
//
// Node encoding (matches `mesh.py::_pack_bvh`):
//   float3 aabbMin
//   int    leftOrCount   // <0 : leaf, -count triangles starting at rightOrFirst.
//                        //  ≥0: inner, left-child index.
//   float3 aabbMax
//   int    rightOrFirst  // leaf: first triangle index;
//                        // inner: right-child index.

import common;

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

[[vk::binding(5)]] StructuredBuffer<MeshVertex> meshVertices;
[[vk::binding(6)]] StructuredBuffer<uint>       meshIndices;
[[vk::binding(7)]] StructuredBuffer<BvhNode>    bvhNodes;

// ── TLAS instance record ───────────────────────────────────────
//
// One entry per renderable mesh instance. With a single instance carrying
// an identity transform and offsets all zero, this layer is mathematically
// a no-op — the ray traverses the same BLAS as before. The indirection is
// what lets Phase B-3 follow-ups (and Phase D's USD ingestion) attach many
// meshes to one scene.
//
// Note: the BLAS offsets are *not yet wired into the BLAS traversal*; the
// existing code reads from index/vertex/bvh buffers starting at zero. They
// are stored here so the multi-BLAS expansion in a follow-up only needs to
// touch findNearestHit, not the upload or layout side.
struct Instance
{
    float4x4 worldFromLocal;       // model→world
    float4x4 localFromWorld;       // world→model (precomputed inverse)
    uint     blasNodeOffset;       // bvhNodes[] starting index for this instance's BLAS
    uint     blasIndexOffset;      // meshIndices[] starting index
    uint     blasVertexOffset;     // meshVertices[] starting index
    uint     materialId;           // index into the active material set
};
[[vk::binding(12)]] StructuredBuffer<Instance> instances;

// ── Ray / AABB slab test ────────────────────────────────────────

bool intersectAabb(Ray ray, float3 invDir, float3 aabbMin, float3 aabbMax, float tMax, out float tNear)
{
    float3 t0 = (aabbMin - ray.origin) * invDir;
    float3 t1 = (aabbMax - ray.origin) * invDir;
    float3 tmin3 = min(t0, t1);
    float3 tmax3 = max(t0, t1);
    float  tmin  = max(max(tmin3.x, tmin3.y), tmin3.z);
    float  tmax  = min(min(tmax3.x, tmax3.y), tmax3.z);
    tNear = max(tmin, 0.0);
    return tmax >= tNear && tmin <= tMax;
}

// ── Möller-Trumbore ray/triangle intersection ──────────────────

bool intersectTriangle(
    Ray ray,
    float3 v0, float3 v1, float3 v2,
    out float tHit,
    out float2 bary
)
{
    tHit = 0.0;
    bary = float2(0.0);
    const float EPS = 1e-7;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pvec = cross(ray.direction, e2);
    float  det = dot(e1, pvec);
    if (abs(det) < EPS)
        return false;
    float  invDet = 1.0 / det;
    float3 tvec   = ray.origin - v0;
    float  u      = dot(tvec, pvec) * invDet;
    if (u < 0.0 || u > 1.0)
        return false;
    float3 qvec   = cross(tvec, e1);
    float  v      = dot(ray.direction, qvec) * invDet;
    if (v < 0.0 || u + v > 1.0)
        return false;
    float t = dot(e2, qvec) * invDet;
    if (t < 1e-4)
        return false;
    tHit = t;
    bary = float2(u, v);
    return true;
}

// ── BVH helpers ────────────────────────────────────────────────

static const int BVH_STACK_SIZE = 64;
static const float RAY_MAX_T    = 1e20;

// Find the nearest triangle hit with t > tMinExcl in the BLAS that lives
// at the given offsets in the unified vertex/index/bvh buffers. Returns
// t and triangle index (the GLOBAL triIdx*3 into meshIndices, so the
// caller can look up the hit's vertices directly), or -1 if nothing hits.
float findNearestHit(
    Ray ray, float tMinExcl,
    uint blasNodeOffset, uint blasIndexOffset, uint blasVertexOffset,
    out int outTri, out float2 outBary)
{
    outTri  = -1;
    outBary = float2(0.0);
    float closestT = RAY_MAX_T;

    float3 invDir = float3(1.0) / ray.direction;

    int stack[BVH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0)
    {
        int nodeIndex = stack[--sp];
        BvhNode node  = bvhNodes[uint(nodeIndex) + blasNodeOffset];

        float tBox;
        if (!intersectAabb(ray, invDir, node.aabbMin, node.aabbMax, closestT, tBox))
            continue;

        if (node.leftOrCount < 0)
        {
            int count = -node.leftOrCount;
            int first = node.rightOrFirst;
            for (int k = 0; k < count; k++)
            {
                // Local triangle index within this BLAS, then global into
                // the unified meshIndices buffer.
                uint triLocal  = uint(first + k);
                uint triIdx    = (triLocal + blasIndexOffset) * 3u;
                uint i0 = meshIndices[triIdx + 0u] + blasVertexOffset;
                uint i1 = meshIndices[triIdx + 1u] + blasVertexOffset;
                uint i2 = meshIndices[triIdx + 2u] + blasVertexOffset;

                float tHit;
                float2 bary;
                if (intersectTriangle(ray,
                        meshVertices[i0].position,
                        meshVertices[i1].position,
                        meshVertices[i2].position,
                        tHit, bary)
                    && tHit > tMinExcl && tHit < closestT)
                {
                    closestT = tHit;
                    outTri   = int(triIdx);
                    outBary  = bary;
                }
            }
        }
        else
        {
            int L = node.leftOrCount;
            int R = node.rightOrFirst;
            BvhNode nl = bvhNodes[uint(L) + blasNodeOffset];
            BvhNode nr = bvhNodes[uint(R) + blasNodeOffset];
            float tL, tR;
            bool hitL = intersectAabb(ray, invDir, nl.aabbMin, nl.aabbMax, closestT, tL);
            bool hitR = intersectAabb(ray, invDir, nr.aabbMin, nr.aabbMax, closestT, tR);

            if (hitL && hitR)
            {
                if (tL < tR)
                {
                    if (sp + 2 <= BVH_STACK_SIZE) { stack[sp++] = R; stack[sp++] = L; }
                }
                else
                {
                    if (sp + 2 <= BVH_STACK_SIZE) { stack[sp++] = L; stack[sp++] = R; }
                }
            }
            else if (hitL && sp < BVH_STACK_SIZE) { stack[sp++] = L; }
            else if (hitR && sp < BVH_STACK_SIZE) { stack[sp++] = R; }
        }
    }

    return closestT;
}

// ── BLAS traversal — local-space ray against one mesh ────────────
//
// Same algorithm as the legacy entry point: front-hit + back-hit lookup,
// barycentric interpolation, asset-or-fallback UVs, tangent build. The
// ray is expected in BLAS-local space; the caller transforms it.

HitInfo marchHeadBLAS(
    Ray ray,
    uint blasNodeOffset,
    uint blasIndexOffset,
    uint blasVertexOffset)
{
    HitInfo info;
    info.hit        = false;
    info.t          = 0.0;
    info.backT      = -1.0;
    info.tangent    = float3(1, 0, 0);
    info.hasTangent = false;
    info.materialId = 0u;

    int    closestTri;
    float2 closestBary;
    float  closestT = findNearestHit(
        ray, 0.0,
        blasNodeOffset, blasIndexOffset, blasVertexOffset,
        closestTri, closestBary
    );

    if (closestTri < 0)
        return info;

    // Second traversal: find the back-face exit (next hit beyond front).
    int    backTri;
    float2 backBary;
    float  backT = findNearestHit(
        ray, closestT + 1e-4,
        blasNodeOffset, blasIndexOffset, blasVertexOffset,
        backTri, backBary
    );
    if (backTri >= 0)
        info.backT = backT;

    // closestTri is the global meshIndices offset (in uints) for the hit
    // triangle's first vertex; vertices are local indices that need the
    // BLAS's vertex offset added before reading from meshVertices.
    uint i0 = meshIndices[uint(closestTri) + 0u] + blasVertexOffset;
    uint i1 = meshIndices[uint(closestTri) + 1u] + blasVertexOffset;
    uint i2 = meshIndices[uint(closestTri) + 2u] + blasVertexOffset;
    MeshVertex a = meshVertices[i0];
    MeshVertex b = meshVertices[i1];
    MeshVertex c = meshVertices[i2];

    float u = closestBary.x;
    float v = closestBary.y;
    float w = 1.0 - u - v;

    float3 position = a.position * w + b.position * u + c.position * v;
    float3 normal   = normalize(a.normal * w + b.normal * u + c.normal * v);

    info.hit            = true;
    info.t              = closestT;
    info.position       = position;
    info.positionObject = position;
    info.normal         = normal;

    // Interpolate asset-space UVs. If the OBJ had no `vt` data the loader
    // left them all at zero — detect that and fall back to the spherical UV
    // so existing un-uv-mapped meshes still look sane in tattoo mode.
    float2 uvA = float2(a.u, a.v);
    float2 uvB = float2(b.u, b.v);
    float2 uvC = float2(c.u, c.v);
    float2 uvHit = uvA * w + uvB * u + uvC * v;
    bool   hasAssetUV = any(uvA != 0.0) || any(uvB != 0.0) || any(uvC != 0.0);

    if (hasAssetUV)
    {
        info.uv = uvHit;
        // Build a tangent aligned with +U using the triangle's edge+UV deltas.
        // The classic textbook formula — det of the 2x2 UV matrix flips sign
        // on mirrored UVs so we just rely on the sign through unchanged.
        float3 e1  = b.position - a.position;
        float3 e2  = c.position - a.position;
        float2 du1 = uvB - uvA;
        float2 du2 = uvC - uvA;
        float  det = du1.x * du2.y - du2.x * du1.y;
        float3 T;
        if (abs(det) > 1e-8)
        {
            float invDet = 1.0 / det;
            T = (e1 * du2.y - e2 * du1.y) * invDet;
        }
        else
        {
            // Degenerate UVs (zero-area in texture space). Pick any stable
            // orthogonal axis so downstream code doesn't get a NaN.
            T = abs(normal.x) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
        }
        // Orthonormalise against the shading normal.
        T = T - normal * dot(T, normal);
        float tlen = length(T);
        info.tangent    = tlen > 1e-6 ? T / tlen : float3(1, 0, 0);
        info.hasTangent = true;
    }
    else
    {
        float3 lp = normalize(position - float3(0.0, 0.05, 0.0));
        info.uv = float2(
            0.5 + atan2(lp.z, lp.x) / (2.0 * PI),
            0.5 - asin(clamp(lp.y, -1.0, 1.0)) / PI
        );
        info.tangent    = float3(1, 0, 0);
        info.hasTangent = false;
    }
    return info;
}

// ── TLAS broad-phase: iterate instances, transform rays into BLAS-
//    local space, keep the nearest hit, transform results back to
//    world space ───────────────────────────────────────────────────
//
// With a single instance carrying an identity transform, the broad-phase
// loop is mathematically a no-op pass-through to the BLAS traversal — the
// rendered output is byte-identical to pre-TLAS.
//
// Phase B-3 follow-up: when fc.numInstances > 1, the sort by t-world
// across instances naturally handles overlapping BLAS bounds. Future BLAS
// concatenation reads BLAS offsets from inst.blas{Node,Index,Vertex}Offset.

HitInfo marchHeadMesh(Ray rayWorld, uint count)
{
    HitInfo best;
    best.hit            = false;
    best.t              = 1e20;
    best.backT          = -1.0;
    best.positionObject = float3(0.0);
    best.tangent        = float3(1, 0, 0);
    best.hasTangent     = false;
    best.materialId     = 0u;

    for (uint i = 0u; i < count; i++)
    {
        Instance inst = instances[i];

        // World→local ray transform. Direction is treated as a vector (w=0)
        // so it doesn't pick up the translation; for non-rigid (scaled)
        // transforms, t-values would need rescaling — we recompute t in
        // world space below to stay scale-correct.
        Ray rayLocal;
        rayLocal.origin    = mul(inst.localFromWorld, float4(rayWorld.origin, 1.0)).xyz;
        rayLocal.direction = mul(inst.localFromWorld, float4(rayWorld.direction, 0.0)).xyz;

        HitInfo h = marchHeadBLAS(
            rayLocal,
            inst.blasNodeOffset,
            inst.blasIndexOffset,
            inst.blasVertexOffset
        );
        if (!h.hit)
            continue;

        // Local→world transforms. Position uses worldFromLocal (w=1).
        // Normal uses (localFromWorld)^T to handle non-uniform scale; the
        // upper 3x3 of localFromWorld^T is the inverse-transpose of
        // worldFromLocal's upper 3x3.
        float3 posWorld     = mul(inst.worldFromLocal, float4(h.position, 1.0)).xyz;
        float3 normalWorld  = normalize(mul(transpose(inst.localFromWorld), float4(h.normal, 0.0)).xyz);
        float3 tangentWorld = h.hasTangent
            ? normalize(mul(inst.worldFromLocal, float4(h.tangent, 0.0)).xyz)
            : h.tangent;

        // Recompute t along the world-space ray. With unit |direction|
        // and identity transform this equals h.t.
        float tWorld = dot(posWorld - rayWorld.origin, rayWorld.direction);

        if (tWorld < best.t)
        {
            best                = h;
            best.t              = tWorld;
            best.position       = posWorld;
            best.positionObject = h.position;
            best.normal         = normalWorld;
            best.tangent        = tangentWorld;
            best.materialId     = inst.materialId;
            // backT scales with the same direction-projection convention.
            if (h.backT > 0.0)
            {
                float3 backPosLocal = rayLocal.origin + rayLocal.direction * h.backT;
                float3 backPosWorld = mul(inst.worldFromLocal, float4(backPosLocal, 1.0)).xyz;
                best.backT = dot(backPosWorld - rayWorld.origin, rayWorld.direction);
            }
        }
    }

    return best;
}
