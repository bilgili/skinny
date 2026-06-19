// Stochastic Progressive Photon Mapping — wavefront stages (change
// photon-mapping-sppm, PM-1). One SPPM pass == one progressive-accumulation
// frame: eye -> grid build (count -> scan -> scatter) -> photon -> update.
// Authored against blueprint.md. Entries are `wfSppm*` (Slang's Metal target
// reserves `main`). All dispatches have host-known counts (num_pixels, numCells,
// photonsEmitted) -> plain dispatch, no indirect dispatch / Metal readback.
//
// This file currently carries the GRID stages (count / scan / scatter); the
// eye, photon and update stages are authored next against the same buffer set.
//
// SPPM buffers are TYPED structured buffers, not the ByteAddressBuffer fold the
// blueprint first proposed: the gpu-portability + integration review confirmed a
// SPPM kernel references ~15/31 Metal buffer slots (it never compiles the neural
// weights), so the 4 plain buffers fit with headroom and stay type-safe.

import common;                     // FrameConstants fields via fc, pcgHash, buildBasis, RNG
import bindings;                   // fc, scene globals (flatMaterials, materialTypes, lights, textures)
import interfaces;                 // BSDFSample, HitInfo
import scene_trace;                // traceScene
import materials.flat.flat_shading;   // FlatHitMat, isCutoutTransparent
import materials.flat.flat_material;  // FlatMaterial, loadFlatMaterial
import scene_lights;               // sampleEmissiveTriangle, intersectSphereLights
import environment;                // sceneEnvRadianceCell, effectiveFurnaceMode
import nee;                        // allLightsNEE
import lights.sphere_light;
import lights.emissive_triangle_light;
import lights.directional_light;
import cameras.pinhole;            // PinholeCamera
import integrators.sppm_state;     // VisiblePoint, SppmAccum, VP_ACTIVE, grid helpers

// Per-tile push constant — same 12-byte {streamBase, _, streamSize} layout the
// path pass uses (a separate pass binds its own; no collision). The middle word
// is the path pass's shadeSlot, unused by SPPM.
struct SppmTilePC { uint streamBase; uint shadeSlot; uint streamSize; };
[[vk::push_constant]] ConstantBuffer<SppmTilePC> sppmTile;

static const uint SPPM_MAX_BOUNCES = 8u;  // eye specular chain depth before the diffuse VP

// set-1 pass-private SPPM buffers (fresh namespace; no collision with the path
// pass's set-1 layout — a distinct pass binds its own set 1).
[[vk::binding(0, 1)]] RWStructuredBuffer<VisiblePoint> sppmVisiblePoints;  // [num_pixels]
[[vk::binding(1, 1)]] RWStructuredBuffer<SppmAccum>    sppmAccum;          // [num_pixels]
// One uint buffer, four contiguous sub-ranges over numCells = pow2(2*num_pixels):
//   gridCount  [0       .. nc)       hits per cell (atomic in count)
//   gridOffset [nc      .. 2nc)      exclusive prefix sum of gridCount
//   gridCursor [2nc     .. 3nc)      scatter write cursor (atomic in scatter)
//   sortedIdx  [3nc     .. 3nc+np)   pixel indices bucketed by cell
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> sppmGrid;
[[vk::binding(3, 1)]] RWStructuredBuffer<uint> sppmScanScratch;            // [ceil(nc/256)] block sums

static const uint SPPM_SCAN_BLOCK = 256u;

uint sppmNumPixels() { return fc.width * fc.height; }

// Sub-range bases into sppmGrid.
uint sppmGridCountBase()  { return 0u; }
uint sppmGridOffsetBase(uint nc) { return nc; }
uint sppmGridCursorBase(uint nc) { return 2u * nc; }
uint sppmSortedIdxBase(uint nc)  { return 3u * nc; }

// ── Stage 1: eye pass ───────────────────────────────────────────────────────
// Store the active visible point, keeping the persistent estimator (tau/radius/n)
// — only the geometry + embedded BSDF + this-pass direct are rewritten. The host
// zeroes the VP buffer at accumFrame==0, so `n == 0` reliably means "never
// updated" → (re)initialize the radius until photons start shrinking it.
void sppmStoreVisiblePoint(uint idx, float3 pos, float3 ns, float3 wo,
                           float3 beta, float3 ld, FlatMaterial mat)
{
    VisiblePoint vp = sppmVisiblePoints[idx];   // read to preserve persistent fields
    vp.pos = pos; vp.ns = ns; vp.wo = wo; vp.beta = beta; vp.ld = ld;
    FlatHitMat m = mat.data.mat;
    vp.albedo = m.albedo; vp.F0 = mat.F0; vp.coatColor = m.coatColor;
    vp.roughness = m.roughness; vp.metallic = m.metallic; vp.specular = m.specular;
    vp.ior = m.ior; vp.opacity = m.opacity; vp.coat = m.coat;
    vp.coatRoughness = m.coatRoughness; vp.coatIOR = m.coatIOR;
    if (vp.n == 0.0)                              // first activation → init estimator
    {
        vp.radius = fc.sppmInitialRadius;
        vp.tau = float3(0.0);
    }
    vp.flags = VP_ACTIVE;
    sppmVisiblePoints[idx] = vp;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void wfSppmEye(uint3 tid: SV_DispatchThreadID)
{
    let i = tid.x;
    if (i >= sppmTile.streamSize)
        return;
    let lane = sppmTile.streamBase + i;
    if (lane >= sppmNumPixels())
        return;
    let pixel = uint2(lane % fc.width, lane / fc.width);

    // Drop a stale active point: a pixel that escapes this pass must not leave a
    // prior pass's visible point in the grid.
    sppmVisiblePoints[lane].flags = 0u;

    RNG rng = createRNG(pixel, fc.frameIndex);
    PinholeCamera cam;
    float lensWeight;
    Ray r = cam.generateRay(pixel, rng, lensWeight);
    float3 throughput = float3(lensWeight);
    float3 ld = float3(0.0);
    bool spawnedBySpecular = false;

    HitInfo h = traceScene(fc, r);
    for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
    {
        r.origin = h.position + r.direction * 0.002;
        h = traceScene(fc, r);
    }

    for (uint bounce = 0u; bounce < SPPM_MAX_BOUNCES; bounce++)
    {
        if (!h.hit)
        {
            // Direct-visible environment (through the specular chain). Taken at
            // full weight: SPPM's photon term is the indirect/caustic complement.
            ld += throughput * sceneEnvRadianceCell(
                r.direction, fc.envIntensity, effectiveFurnaceMode(h.materialId));
            break;
        }

        // PM-1 is flat-material-only. Glass/mirror are FLAT (delta lobe); skin /
        // python / debug receivers terminate inactive.
        if ((materialTypes[h.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
            break;

        FlatMaterial mat = loadFlatMaterial(h);
        float3 N = mat.data.N;
        float3 T, B;
        if (h.hasTangent)
        {
            T = normalize(h.tangent - N * dot(h.tangent, N));
            B = cross(N, T);
        }
        else
        {
            buildBasis(N, T, B);
        }
        let V = -r.direction;
        let wo = worldToTangent(V, T, B, N);

        // Per-pass direct at this vertex (NEE), accumulated into ld.
        ld += throughput * allLightsNEE(mat, h, N, T, B, wo, false, rng);

        BSDFSample bs = mat.sample(wo, rng);

        // Direct-visible self-emission, gated like the path tracer (delta chain /
        // no emissive NEE ⇒ full weight).
        if (bounce == 0u || fc.numEmissiveTriangles == 0u || spawnedBySpecular)
            ld += throughput * bs.emission;

        if (!bs.valid)
            break;

        if (bs.pdf > 0.0)
        {
            // Non-delta flat surface ⇒ this is the visible point.
            sppmStoreVisiblePoint(lane, h.position, N, V, throughput, ld, mat);
            return;
        }

        // Delta/specular lobe ⇒ caustic carrier: keep tracing.
        bs.wi = tangentToWorld(bs.wi, T, B, N);
        throughput *= bs.weight;
        spawnedBySpecular = true;
        let spawnOffset = bs.transmitted ? -h.normal * 0.002 : h.normal * 0.002;
        r.origin = h.position + spawnOffset;
        r.direction = bs.wi;
        h = traceScene(fc, r);
        for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
        {
            r.origin = h.position + r.direction * 0.002;
            h = traceScene(fc, r);
        }
    }

    // No visible point stored (escaped / non-flat / dead): keep the direct term
    // so the update stage still composites it; VP stays inactive (flags == 0).
    sppmVisiblePoints[lane].ld = ld;
}

// ── Stage 2: per-cell count ────────────────────────────────────────────────
[shader("compute")]
[numthreads(64, 1, 1)]
void wfSppmGridCount(uint3 tid: SV_DispatchThreadID)
{
    let i = tid.x;
    let np = sppmNumPixels();
    if (i >= np)
        return;
    VisiblePoint vp = sppmVisiblePoints[i];
    if ((vp.flags & VP_ACTIVE) == 0u)
        return;  // escaped / dead pixels are not in the grid
    let nc = sppmGridCellCount(np);
    let cell = sppmCellHash(sppmCellCoord(vp.pos, fc.sceneBoundsMin, fc.sppmCellSize), nc);
    uint old;
    InterlockedAdd(sppmGrid[sppmGridCountBase() + cell], 1u, old);
}

// ── Stage 3a: per-block exclusive prefix scan (count -> offset) ─────────────
// Hillis-Steele inclusive scan in shared memory, shifted to exclusive; the
// block total goes to sppmScanScratch for the block-sum scan below.
groupshared uint gScan[SPPM_SCAN_BLOCK];

[shader("compute")]
[numthreads(SPPM_SCAN_BLOCK, 1, 1)]
void wfSppmGridScanBlock(uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    let np = sppmNumPixels();
    let nc = sppmGridCellCount(np);
    let li = gtid.x;
    let gi = gid.x * SPPM_SCAN_BLOCK + li;

    let v = (gi < nc) ? sppmGrid[sppmGridCountBase() + gi] : 0u;
    gScan[li] = v;
    GroupMemoryBarrierWithGroupSync();

    // Inclusive scan: read all at offset, sync, write all, sync.
    [unroll]
    for (uint off = 1u; off < SPPM_SCAN_BLOCK; off <<= 1)
    {
        let add = (li >= off) ? gScan[li - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gScan[li] += add;
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi < nc)
        sppmGrid[sppmGridOffsetBase(nc) + gi] = gScan[li] - v;  // inclusive -> exclusive
    if (li == SPPM_SCAN_BLOCK - 1u)
        sppmScanScratch[gid.x] = gScan[li];                     // block total (inclusive last)
}

// ── Stage 3b: exclusive scan of the block sums (single workgroup, one thread) ─
// numBlocks = ceil(nc/256) is at most ~num_pixels/128 (a few thousand), so a
// single sequential pass is cheap and avoids a second multi-level scan.
[shader("compute")]
[numthreads(1, 1, 1)]
void wfSppmGridScanBlockSums(uint3 tid: SV_DispatchThreadID)
{
    let nc = sppmGridCellCount(sppmNumPixels());
    let numBlocks = (nc + SPPM_SCAN_BLOCK - 1u) / SPPM_SCAN_BLOCK;
    uint acc = 0u;
    [loop]
    for (uint b = 0u; b < numBlocks; b++)
    {
        let t = sppmScanScratch[b];
        sppmScanScratch[b] = acc;  // exclusive
        acc += t;
    }
}

// ── Stage 3c: add each block's base back into its offsets ───────────────────
[shader("compute")]
[numthreads(SPPM_SCAN_BLOCK, 1, 1)]
void wfSppmGridScanAdd(uint3 tid: SV_DispatchThreadID, uint3 gid: SV_GroupID)
{
    let nc = sppmGridCellCount(sppmNumPixels());
    let gi = tid.x;
    if (gi >= nc)
        return;
    sppmGrid[sppmGridOffsetBase(nc) + gi] += sppmScanScratch[gid.x];
}

// ── Stage 4: scatter pixel indices into their cell buckets ──────────────────
[shader("compute")]
[numthreads(64, 1, 1)]
void wfSppmGridScatter(uint3 tid: SV_DispatchThreadID)
{
    let i = tid.x;
    let np = sppmNumPixels();
    if (i >= np)
        return;
    VisiblePoint vp = sppmVisiblePoints[i];
    if ((vp.flags & VP_ACTIVE) == 0u)
        return;
    let nc = sppmGridCellCount(np);
    let cell = sppmCellHash(sppmCellCoord(vp.pos, fc.sceneBoundsMin, fc.sppmCellSize), nc);
    uint slot;
    InterlockedAdd(sppmGrid[sppmGridCursorBase(nc) + cell], 1u, slot);
    let dst = sppmGrid[sppmGridOffsetBase(nc) + cell] + slot;
    sppmGrid[sppmSortedIdxBase(nc) + dst] = i;
}
