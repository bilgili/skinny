// Camera Debug software rasteriser (change metal-tool-dock-render P2).
//
// The native Metal backend has no graphics pipeline, so the debug viewport's
// line/triangle vertex streams are scan-converted in a compute kernel. The math
// mirrors `skinny.debug_raster_ref` byte-for-byte so the kernel is checkable on
// the host without a GPU (task 3.6) and diffable against the GPU output (3.8).
//
// Vertex stream: 7 floats per vertex — `x y z r g b a` (debug_viewport
// _VERTEX_FLOATS). Lines are a vertex list (2/line); triangles a vertex list
// (3/tri). `a > 1.5` marks a screen-space HUD sentinel whose `xy` is already NDC
// (bypasses viewProj, depth 0 = on top).
//
// Pipeline (design D4, host serialises the passes via wait-idle):
//   clearImage + clearDepth → depthLines (atomic_min) → colorLines (write where
//   this pixel owns the winning depth) → blendTris (alpha-blend, depth-test, no
//   depth write; ordered after opaque lines).
//
// Output: `colorOut` = width*height packed-RGBA8 (`r | g<<8 | b<<16 | a<<24`),
// read back little-endian as an r,g,b,a byte stream. `depthOut` = width*height
// packed uint depth (nearest = smallest). Entry names are NOT `main` (Slang's
// Metal target reserves it).
//
// Watchdog: lines = one thread/line, per-line DDA capped at gMaxSteps; tris = one
// thread per (triangle, screen-row) so each thread walks at most one scanline
// (<= width iterations). No single dispatch has an unbounded per-thread loop.

static const uint VERTEX_FLOATS = 7u;
static const float DEPTH_SCALE = 16777215.0;  // 0xFFFFFF — 24-bit depth (low byte = line tag)

// viewProj as four row-major math rows (clip = viewProj * [x,y,z,1]); explicit
// rows avoid any Slang matrix-layout ambiguity between backends.
uniform float4 gVP0;
uniform float4 gVP1;
uniform float4 gVP2;
uniform float4 gVP3;
uniform uint   gWidth;
uniform uint   gHeight;
uniform uint   gCount;         // line count (line kernels) or tri count (blendTris)
uniform uint   gMaxSteps;      // per-line DDA cap (GPU watchdog bound)
uniform uint4  gClearPacked;   // .x = packed RGBA8 background

StructuredBuffer<float> lineVerts;
StructuredBuffer<float> triVerts;
RWStructuredBuffer<uint> colorOut;
RWStructuredBuffer<uint> depthOut;

uint packRGBA(float4 c)
{
    float4 v = clamp(c, 0.0, 1.0) * 255.0 + 0.5;
    return (uint)v.x | ((uint)v.y << 8) | ((uint)v.z << 16) | ((uint)v.w << 24);
}

float4 unpackRGBA(uint p)
{
    return float4(float(p & 0xFFu), float((p >> 8) & 0xFFu),
                  float((p >> 16) & 0xFFu), float((p >> 24) & 0xFFu)) / 255.0;
}

// 24-bit depth (near→0). The full key is `packDepth(d) << 8 | (lineTag & 0xFF)`,
// so atomic_min orders by depth then by the smallest line index at equal depth
// (deterministic tie-break ≈ Vulkan VK_COMPARE_OP_LESS "earlier draw wins").
uint packDepth(float d01)
{
    return (uint)(clamp(d01, 0.0, 1.0) * DEPTH_SCALE);
}

// Project vertex `vi` of `verts` → screen `pxy` + `d01`; false when behind eye.
bool projectVertex(StructuredBuffer<float> verts, uint vi,
                   out float2 pxy, out float d01)
{
    uint base = vi * VERTEX_FLOATS;
    float3 pos = float3(verts[base + 0], verts[base + 1], verts[base + 2]);
    float alpha = verts[base + 6];
    float ndcx, ndcy;
    if (alpha > 1.5)
    {
        ndcx = pos.x; ndcy = pos.y; d01 = 0.0;
    }
    else
    {
        float4 posH = float4(pos, 1.0);
        float4 clip = float4(dot(gVP0, posH), dot(gVP1, posH),
                             dot(gVP2, posH), dot(gVP3, posH));
        if (clip.w <= 1e-6) { pxy = float2(0.0, 0.0); d01 = 0.0; return false; }
        ndcx = clip.x / clip.w;
        ndcy = clip.y / clip.w;
        d01 = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
    }
    pxy = float2((ndcx * 0.5 + 0.5) * float(gWidth),
                 (0.5 - ndcy * 0.5) * float(gHeight));
    return true;
}

float4 loadColor(StructuredBuffer<float> verts, uint vi)
{
    uint base = vi * VERTEX_FLOATS;
    return float4(verts[base + 3], verts[base + 4], verts[base + 5], verts[base + 6]);
}

bool inBounds(int x, int y)
{
    return x >= 0 && y >= 0 && x < int(gWidth) && y < int(gHeight);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void clearImage(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gWidth || tid.y >= gHeight) return;
    colorOut[tid.y * gWidth + tid.x] = gClearPacked.x;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void clearDepth(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gWidth || tid.y >= gHeight) return;
    depthOut[tid.y * gWidth + tid.x] = 0xFFFFFFFFu;
}

// Pass 1: deposit line depth via atomic_min (opaque, depth-writing).
[shader("compute")]
[numthreads(64, 1, 1)]
void depthLines(uint3 tid : SV_DispatchThreadID)
{
    uint li = tid.x;
    if (li >= gCount) return;
    float2 pa, pb; float da, db;
    if (!projectVertex(lineVerts, li * 2u, pa, da)) return;
    if (!projectVertex(lineVerts, li * 2u + 1u, pb, db)) return;

    float dx = pb.x - pa.x, dy = pb.y - pa.y, dd = db - da;
    int steps = int(ceil(max(abs(dx), abs(dy))));
    if (steps < 0) steps = 0;
    if (uint(steps) > gMaxSteps) steps = int(gMaxSteps);
    float inv = steps > 0 ? 1.0 / float(steps) : 0.0;
    for (int i = 0; i <= steps; ++i)
    {
        float t = float(i) * inv;
        int x = int(floor(pa.x + dx * t + 0.5));
        int y = int(floor(pa.y + dy * t + 0.5));
        if (!inBounds(x, y)) continue;
        uint idx = uint(y) * gWidth + uint(x);
        uint key = (packDepth(da + dd * t) << 8) | (li & 0xFFu);
        uint prev;
        InterlockedMin(depthOut[idx], key, prev);
    }
}

// Pass 2: write line colour where this pixel owns the winning depth.
[shader("compute")]
[numthreads(64, 1, 1)]
void colorLines(uint3 tid : SV_DispatchThreadID)
{
    uint li = tid.x;
    if (li >= gCount) return;
    float2 pa, pb; float da, db;
    if (!projectVertex(lineVerts, li * 2u, pa, da)) return;
    if (!projectVertex(lineVerts, li * 2u + 1u, pb, db)) return;

    uint packedCol = packRGBA(loadColor(lineVerts, li * 2u));
    float dx = pb.x - pa.x, dy = pb.y - pa.y, dd = db - da;
    int steps = int(ceil(max(abs(dx), abs(dy))));
    if (steps < 0) steps = 0;
    if (uint(steps) > gMaxSteps) steps = int(gMaxSteps);
    float inv = steps > 0 ? 1.0 / float(steps) : 0.0;
    for (int i = 0; i <= steps; ++i)
    {
        float t = float(i) * inv;
        int x = int(floor(pa.x + dx * t + 0.5));
        int y = int(floor(pa.y + dy * t + 0.5));
        if (!inBounds(x, y)) continue;
        uint idx = uint(y) * gWidth + uint(x);
        uint key = (packDepth(da + dd * t) << 8) | (li & 0xFFu);
        if (depthOut[idx] == key)
            colorOut[idx] = packedCol;
    }
}

// Blend pass: one thread per (triangle, screen-row). Alpha-blend over the opaque
// scene, depth-tested against the line depth, no depth write.
[shader("compute")]
[numthreads(8, 8, 1)]
void blendTris(uint3 tid : SV_DispatchThreadID)
{
    uint ti = tid.x;
    int row = int(tid.y);
    if (ti >= gCount || row >= int(gHeight)) return;

    float2 p0, p1, p2; float z0, z1, z2;
    if (!projectVertex(triVerts, ti * 3u, p0, z0)) return;
    if (!projectVertex(triVerts, ti * 3u + 1u, p1, z1)) return;
    if (!projectVertex(triVerts, ti * 3u + 2u, p2, z2)) return;

    float minYf = min(min(p0.y, p1.y), p2.y);
    float maxYf = max(max(p0.y, p1.y), p2.y);
    if (float(row) + 0.5 < floor(minYf) || float(row) > ceil(maxYf)) return;

    float area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
    if (abs(area) < 1e-9) return;
    float invArea = 1.0 / area;

    int minX = max(int(floor(min(min(p0.x, p1.x), p2.x))), 0);
    int maxX = min(int(ceil(max(max(p0.x, p1.x), p2.x))), int(gWidth) - 1);

    float4 col = loadColor(triVerts, ti * 3u);
    float3 src = clamp(col.rgb, 0.0, 1.0);
    float alpha = clamp(col.a, 0.0, 1.0);
    float py = float(row) + 0.5;

    for (int x = minX; x <= maxX; ++x)
    {
        float px = float(x) + 0.5;
        float w0 = ((p1.x - px) * (p2.y - py) - (p2.x - px) * (p1.y - py)) * invArea;
        float w1 = ((p2.x - px) * (p0.y - py) - (p0.x - px) * (p2.y - py)) * invArea;
        float w2 = 1.0 - w0 - w1;
        if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) continue;
        float d01 = w0 * z0 + w1 * z1 + w2 * z2;
        uint idx = uint(row) * gWidth + uint(x);
        // Strict depth test against the opaque line depth (high 24 bits),
        // occluding on equal depth — Vulkan VK_COMPARE_OP_LESS.
        if (packDepth(d01) >= (depthOut[idx] >> 8)) continue;
        float3 dst = unpackRGBA(colorOut[idx]).rgb;
        float3 outc = clamp(src * alpha + dst * (1.0 - alpha), 0.0, 1.0);
        colorOut[idx] = packRGBA(float4(outc, 1.0));
    }
}
