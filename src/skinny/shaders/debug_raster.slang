// Camera Debug software rasteriser (change metal-tool-dock-render P2).
//
// The native Metal backend has no graphics pipeline, so the debug viewport's
// line/triangle vertex streams are scan-converted in a compute kernel. This is
// the phase-1 slice: a DDA line rasteriser with NO depth buffer yet (opaque
// last-writer-wins). Triangles, depth, alpha-blend, and HUD text land in later
// phases. The math mirrors `skinny.debug_raster_ref` byte-for-byte so the kernel
// is checkable on the host without a GPU (task 3.6) and diffable against the GPU
// output (task 3.8).
//
// Vertex stream: 7 floats per vertex — `x y z r g b a` (debug_viewport
// _VERTEX_FLOATS). Lines are a vertex list, 2 per line. `a > 1.5` marks a
// screen-space HUD sentinel whose `xy` is already NDC (bypasses viewProj).
//
// Output: `colorOut` is a width*height packed-RGBA8 buffer (`r | g<<8 | b<<16 |
// a<<24`); the host reads it back little-endian as an r,g,b,a byte stream.
//
// Entry names are NOT `main` (Slang's Metal target reserves it).

static const uint VERTEX_FLOATS = 7u;

// viewProj as four row-major math rows (clip = viewProj * [x,y,z,1]); passing
// explicit rows avoids any Slang matrix-layout ambiguity between backends.
uniform float4 gVP0;
uniform float4 gVP1;
uniform float4 gVP2;
uniform float4 gVP3;
uniform uint   gWidth;
uniform uint   gHeight;
uniform uint   gLineCount;
uniform uint   gMaxSteps;      // per-line step cap (GPU watchdog bound)
uniform uint4  gClearPacked;   // .x = packed RGBA8 background

StructuredBuffer<float> lineVerts;
RWStructuredBuffer<uint> colorOut;

uint packRGBA(float4 c)
{
    float4 v = clamp(c, 0.0, 1.0) * 255.0 + 0.5;
    uint r = (uint)v.x;
    uint g = (uint)v.y;
    uint b = (uint)v.z;
    uint a = (uint)v.w;
    return r | (g << 8) | (b << 16) | (a << 24);
}

float3 loadPos(uint vi)
{
    uint base = vi * VERTEX_FLOATS;
    return float3(lineVerts[base + 0], lineVerts[base + 1], lineVerts[base + 2]);
}

float4 loadColor(uint vi)
{
    uint base = vi * VERTEX_FLOATS;
    return float4(lineVerts[base + 3], lineVerts[base + 4],
                  lineVerts[base + 5], lineVerts[base + 6]);
}

// Project one vertex → (px, py) float pixel coords; `visible` false when behind
// the eye. HUD sentinel (a > 1.5) treats xy as NDC directly.
bool projectVertex(uint vi, out float2 pxy)
{
    float3 pos = loadPos(vi);
    float alpha = lineVerts[vi * VERTEX_FLOATS + 6];
    float ndcx, ndcy;
    if (alpha > 1.5)
    {
        ndcx = pos.x;
        ndcy = pos.y;
    }
    else
    {
        float4 posH = float4(pos, 1.0);
        float4 clip = float4(dot(gVP0, posH), dot(gVP1, posH),
                             dot(gVP2, posH), dot(gVP3, posH));
        if (clip.w <= 1e-6)
        {
            pxy = float2(0.0, 0.0);
            return false;
        }
        ndcx = clip.x / clip.w;
        ndcy = clip.y / clip.w;
    }
    pxy = float2((ndcx * 0.5 + 0.5) * float(gWidth),
                 (0.5 - ndcy * 0.5) * float(gHeight));
    return true;
}

void plot(int x, int y, uint packed)
{
    if (x < 0 || y < 0 || x >= int(gWidth) || y >= int(gHeight)) return;
    colorOut[uint(y) * gWidth + uint(x)] = packed;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void clearImage(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gWidth || tid.y >= gHeight) return;
    colorOut[tid.y * gWidth + tid.x] = gClearPacked.x;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterLines(uint3 tid : SV_DispatchThreadID)
{
    uint li = tid.x;
    if (li >= gLineCount) return;

    uint va = li * 2u;
    uint vb = va + 1u;
    float2 pa, pb;
    if (!projectVertex(va, pa) || !projectVertex(vb, pb)) return;

    uint packed = packRGBA(loadColor(va));

    float dx = pb.x - pa.x;
    float dy = pb.y - pa.y;
    float adx = abs(dx);
    float ady = abs(dy);
    int steps = int(ceil(max(adx, ady)));
    if (steps <= 0)
    {
        plot(int(floor(pa.x + 0.5)), int(floor(pa.y + 0.5)), packed);
        return;
    }
    if (uint(steps) > gMaxSteps) steps = int(gMaxSteps);
    float inv = 1.0 / float(steps);
    for (int i = 0; i <= steps; ++i)
    {
        float t = float(i) * inv;
        int x = int(floor(pa.x + dx * t + 0.5));
        int y = int(floor(pa.y + dy * t + 0.5));
        plot(x, y, packed);
    }
}
