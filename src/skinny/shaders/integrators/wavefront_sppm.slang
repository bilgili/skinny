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
import wavefront.wf_display;       // wfWriteDisplay (exposure + tonemap + present)
import integrators.sppm_state;     // VisiblePoint, SppmAccum, VP_ACTIVE, grid helpers
import integrators.path;           // PathTracer — non-flat receiver fallback

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

// Glossy / near-specular continuation predicate (change sppm-glossy-final-gather).
// True => follow the BSDF-sampled lobe one bounce like a delta carrier instead of
// storing a visible point. Metallic guard: BSDFSample carries no lobe id, so gate on
// metalness to keep dielectric diffuse samples on the gather side. Threshold 0 => PM-1.
bool sppmGlossyContinue(FlatMaterial mat, BSDFSample bs)
{
    let m = mat.data.mat;
    return (bs.pdf > 0.0) && (m.roughness < fc.sppmGlossyContinueRoughness) && (m.metallic >= 0.5);
}

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
    vp.transmissionColor = m.transmissionColor;
    vp.specularColor = m.specularColor;
    vp.diffuseRoughness = m.diffuseRoughness;
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
    // Was the current ray spawned by a delta/perfectly-specular carrier (pdf<=0)?
    // A delta lobe has no NEE partner ⇒ an emissive triangle it lands on is taken
    // at full weight; a non-delta glossy-continue carrier ran emissive NEE, so its
    // emissive hit is MIS-weighted instead. Mirrors path.slang (delta-only here,
    // not "any continue"). False for the primary ray.
    bool spawnedBySpecular = false;
    // BSDF pdf of the glossy-continue bounce that spawned the current ray, for
    // MIS against the env NEE that same vertex ran (change sppm-glossy-final-gather).
    // -1 ⇒ no env-NEE counterpart (primary ray, delta carrier, transmitted, or
    // furnace) ⇒ take the escaped env at full weight, mirroring path.slang.
    float misBsdfPdf = -1.0;
    // Companion for the emissive-triangle hit MIS (distinct from misBsdfPdf, which
    // is additionally gated on env NEE). -1 ⇒ no NEE partner ⇒ emission full weight.
    float prevBsdfPdf = -1.0;

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
            // Direct-visible environment (through the specular chain). A delta
            // carrier has no NEE partner ⇒ full weight (SPPM's photon term is the
            // indirect/caustic complement). A glossy-continue vertex DID run env
            // NEE, so its BSDF-sampled env is the MIS complement — weight it by
            // powerHeuristic(bsdfPdf, envPdf), exactly as path.slang does on its
            // env miss. Prevents double-counting the env at a glossy metal whose
            // reflection escapes straight to the environment (conductor_infinite).
            float3 envR = sceneEnvRadianceCell(
                r.direction, fc.envIntensity, effectiveFurnaceMode(h.materialId));
            float w = (misBsdfPdf > 0.0)
                ? powerHeuristic(misBsdfPdf, envPdf(r.direction)) : 1.0;
            ld += throughput * envR * w;
            break;
        }

        // PM-1 is flat-material-only. Glass/mirror are FLAT (delta lobe). A
        // non-flat receiver stores no photon visible point — there is no surface
        // BRDF density to gather — so it is shaded by the path fallback for EVERY
        // non-flat type (parity with the megakernel's non-flat→path routing):
        // path-integrate from this hit (SKINNY_WAVEFRONT → subsurfaceRadiance3D
        // for subsurface) weighted by the accumulated specular-chain throughput,
        // add it to the pixel's direct term, then terminate the eye walk with no
        // VP stored. Terminal types (SUBSURFACE / SKIN) evaluate one vertex; the
        // non-terminal ones (VOLUME / PYTHON) run the full multi-bounce path,
        // kept within the macOS GPU watchdog by the host bounding the Metal eye
        // submit per tile (record_sppm_loop's `bound_heavy_eye`, change
        // wavefront-nonflat-tiled-fallback).
        if ((materialTypes[h.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
        {
            PathTracer integrator;
            ld += throughput * integrator.estimateRadiance(r, h, rng);
            break;
        }

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

        // Direct-visible self-emission, MIS-gated like the path tracer. Bounce 0,
        // no emissive triangles, or a delta-spawned ray ⇒ no NEE partner ⇒ full
        // weight. A non-delta (glossy-continued) spawn ran emissive NEE at that
        // vertex, so add only the BSDF-sampling MIS complement — pdfLightSA is
        // reconstructed without the tri index (the per-tri area cancels in the
        // power-weighted area·lum CDF), per emissive-triangle-bsdf-hit-mis.
        if (bounce == 0u || fc.numEmissiveTriangles == 0u || spawnedBySpecular)
        {
            ld += throughput * bs.emission;
        }
        else if (any(bs.emission > 0.0))
        {
            float cosLight = max(dot(h.normal, -r.direction), 0.0);
            float lum = dot(bs.emission, float3(0.2126, 0.7152, 0.0722));
            float pdfLightSA = (cosLight > 1e-6 && fc.emissiveTotalPower > 0.0)
                ? lum * (h.t * h.t) / (fc.emissiveTotalPower * cosLight)
                : 1e30;   // grazing / degenerate ⇒ NEE owns it ⇒ wBSDF → 0
            ld += throughput * bs.emission * powerHeuristic(prevBsdfPdf, pdfLightSA);
        }

        if (!bs.valid)
            break;

        let glossyContinue = sppmGlossyContinue(mat, bs);
        if (bs.pdf > 0.0 && !glossyContinue)
        {
            // Non-delta, non-glossy-continued surface ⇒ this is the visible point.
            sppmStoreVisiblePoint(lane, h.position, N, V, throughput, ld, mat);
            return;
        }

        // Delta OR glossy-continue ⇒ caustic carrier: keep tracing so the visible
        // point lands on the next non-glossy surface and the reflection is rebuilt.
        bs.wi = tangentToWorld(bs.wi, T, B, N);
        throughput *= bs.weight;
        // Carry this bounce's MIS companions to the next iteration, mirroring
        // path.slang. A delta carrier (bs.pdf<=0) has no NEE partner ⇒ full-weight
        // env miss AND full-weight emissive hit next iteration. A non-delta
        // glossy-continue ran env + emissive NEE here, so both its env miss
        // (misBsdfPdf → envPdf companion) and any emissive-triangle hit
        // (prevBsdfPdf → pdfLightSA companion) are MIS-weighted, never
        // double-counted. Furnace / transmission suppress the env companion.
        bool envNeeActive = (effectiveFurnaceMode(h.materialId) == 0u);
        misBsdfPdf = (envNeeActive && !bs.transmitted && bs.pdf > 0.0) ? bs.pdf : -1.0;
        spawnedBySpecular = (bs.pdf <= 0.0);
        prevBsdfPdf = spawnedBySpecular ? -1.0 : bs.pdf;
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

// ── Stage 5: photon pass (emit → trace → deposit) ───────────────────────────
static const uint SPPM_PHOTON_MAX_DEPTH = 16u;

// z-up cosine-weighted hemisphere direction.
float3 sppmCosineHemisphere(float2 u)
{
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

// Rebuild the flat material stored in a visible point (no texture refetch / graph
// re-run / double normal-map). Emission is omitted — it is direct, not BRDF.
FlatMaterial sppmLoadMaterial(VisiblePoint vp)
{
    FlatHitMat m;
    m.albedo = vp.albedo;
    m.roughness = vp.roughness;
    m.metallic = vp.metallic;
    m.specular = vp.specular;
    m.ior = vp.ior;
    m.opacity = vp.opacity;
    m.emission = float3(0.0);
    m.coat = vp.coat;
    m.coatRoughness = vp.coatRoughness;
    m.coatIOR = vp.coatIOR;
    m.coatColor = vp.coatColor;
    // Stage-2 rich inputs: stored in the VP like every other FlatHitMat field
    // the lobe model reads — an unset field here feeds undefined values into
    // evaluate() at photon deposit (zeroed photon flux scene-wide when
    // flat-lobes-rich-inputs landed after PM-1 without a VP slot).
    m.transmissionColor = vp.transmissionColor;
    m.specularColor = vp.specularColor;
    m.diffuseRoughness = vp.diffuseRoughness;
    FlatMaterial mat;
    mat.data.mat = m;
    mat.data.N = vp.ns;
    mat.F0 = vp.F0;
    return mat;
}

// Sample one photon from the scene lights. Group selection is uniform over the
// present light groups; emissive triangles are power-weighted within the group
// (pSel, folded into LightSample.pdfArea). beta carries the per-photon flux
// (the 1/N_emitted normalization is applied in the update stage). For a diffuse
// area emitter the emission cosine cancels the cosine-pdf, so beta = Le·π/selPdf.
bool sppmEmitPhoton(inout RNG rng, out Ray ray, out float3 beta)
{
    ray.origin = float3(0.0);
    ray.direction = float3(0.0, 0.0, 1.0);
    beta = float3(0.0);

    let hasE = (fc.numEmissiveTriangles > 0u) ? 1u : 0u;
    let hasS = (fc.numSphereLights > 0u) ? 1u : 0u;
    let hasD = (fc.numDistantLights > 0u) ? 1u : 0u;
    let G = hasE + hasS + hasD;
    if (G == 0u)
        return false;
    let gsel = 1.0 / float(G);

    // Map a uniform draw to one of the present groups (order: emissive, sphere, distant).
    uint pick = min(uint(rng.next() * float(G)), G - 1u);
    uint gi = 0u;
    uint chosen = 99u;
    if (hasE == 1u) { if (gi == pick) chosen = 0u; gi++; }
    if (chosen == 99u && hasS == 1u) { if (gi == pick) chosen = 1u; gi++; }
    if (chosen == 99u && hasD == 1u) { if (gi == pick) chosen = 2u; gi++; }

    if (chosen == 0u)
    {
        let idx = sampleEmissiveTriangle(rng.next(), fc.numEmissiveTriangles);
        EmissiveTriangleLightImpl el =
            loadEmissiveTriangleLightImpl(emissiveTriangles[idx], fc.numEmissiveTriangles);
        LightSample ls = el.samplePoint(float3(0.0), rng.next2());
        if (!ls.valid || ls.pdfArea <= 0.0)
            return false;
        let selA = gsel * ls.pdfArea;             // area-measure selection pdf
        float3 t, b; buildBasis(ls.normal, t, b);
        let d = sppmCosineHemisphere(rng.next2());
        ray.direction = normalize(t * d.x + b * d.y + ls.normal * d.z);
        ray.origin = ls.point + ls.normal * 0.002;
        beta = ls.radiance * PI / max(selA, 1e-20);
        return true;
    }
    if (chosen == 1u)
    {
        let i = min(uint(rng.next() * float(fc.numSphereLights)), fc.numSphereLights - 1u);
        SphereLightImpl sl = loadSphereLightImpl(sphereLights[i]);
        LightSample ls = sl.samplePoint(float3(0.0), rng.next2());
        if (!ls.valid || ls.pdfArea <= 0.0)
            return false;
        let selA = gsel * (1.0 / float(fc.numSphereLights)) * ls.pdfArea;
        float3 t, b; buildBasis(ls.normal, t, b);
        let d = sppmCosineHemisphere(rng.next2());
        ray.direction = normalize(t * d.x + b * d.y + ls.normal * d.z);
        ray.origin = ls.point + ls.normal * 0.002;
        beta = ls.radiance * PI / max(selA, 1e-20);
        return true;
    }
    // Distant light: a parallel beam from a bbox-covering disk (pdfDir is a delta,
    // no cosine). beta = L·A_disk / selPdf.
    let i = min(uint(rng.next() * float(fc.numDistantLights)), fc.numDistantLights - 1u);
    DirectionalLightImpl dl = loadDirectionalLightImpl(distantLights[i]);
    let center = fc.sceneBoundsMin + 0.5 * fc.sceneBoundsExtent;
    let R = max(0.5 * length(fc.sceneBoundsExtent), 1e-4);
    float3 t, b; buildBasis(dl.dir, t, b);
    let u2 = rng.next2();
    let rr = R * sqrt(u2.x);
    let ph = 2.0 * PI * u2.y;
    ray.origin = center + dl.dir * R + t * (rr * cos(ph)) + b * (rr * sin(ph));
    ray.direction = -dl.dir;
    let selPdf = gsel * (1.0 / float(fc.numDistantLights));
    beta = dl.rad * (PI * R * R) / max(selPdf, 1e-20);
    return true;
}

// Deposit a photon's flux into every active visible point within its radius,
// using EACH visible point's own BSDF (bare f_r, no cosine — the photon-map
// density estimate). The 3x3x3 neighbour cells are de-duplicated so a hash
// collision can't double-deposit.
void sppmDepositPhoton(float3 beta, float3 photonPos, float3 photonInDir)
{
    let nc = sppmGridCellCount(sppmNumPixels());
    let base = sppmCellCoord(photonPos, fc.sceneBoundsMin, fc.sppmCellSize);
    uint seen[27];
    uint nSeen = 0u;
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++)
    {
        let cell = sppmCellHash(base + int3(dx, dy, dz), nc);
        bool dup = false;
        for (uint s = 0u; s < nSeen; s++)
            if (seen[s] == cell) { dup = true; break; }
        if (dup)
            continue;
        seen[nSeen++] = cell;

        let start = sppmGrid[sppmGridOffsetBase(nc) + cell];
        let cnt = sppmGrid[sppmGridCountBase() + cell];
        for (uint k = 0u; k < cnt; k++)
        {
            let j = sppmGrid[sppmSortedIdxBase(nc) + start + k];
            VisiblePoint vp = sppmVisiblePoints[j];
            if ((vp.flags & VP_ACTIVE) == 0u)
                continue;
            let dp = photonPos - vp.pos;
            if (dot(dp, dp) > vp.radius * vp.radius)
                continue;
            let toLight = -photonInDir;            // toward the photon's origin
            if (dot(vp.ns, toLight) <= 0.0)
                continue;
            float3 t, b; buildBasis(vp.ns, t, b);
            let woVP = worldToTangent(vp.wo, t, b, vp.ns);
            let wiVP = worldToTangent(toLight, t, b, vp.ns);
            if (wiVP.z <= 1e-4)
                continue;
            FlatMaterial mat = sppmLoadMaterial(vp);
            let f = mat.evaluate(woVP, wiVP).response / max(wiVP.z, 1e-4);  // bare f_r
            float3 phi = beta * f;
            if (any(phi != phi) || any(isinf(phi)))
                continue;
            phi = max(phi, float3(0.0));
            uint old;
            InterlockedAdd(sppmAccum[j].phiR, uint(phi.x * SPPM_FLUX_FIXED_SCALE), old);
            InterlockedAdd(sppmAccum[j].phiG, uint(phi.y * SPPM_FLUX_FIXED_SCALE), old);
            InterlockedAdd(sppmAccum[j].phiB, uint(phi.z * SPPM_FLUX_FIXED_SCALE), old);
            InterlockedAdd(sppmAccum[j].m, 1u, old);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void wfSppmPhotonTrace(uint3 tid: SV_DispatchThreadID)
{
    let pid = tid.x;
    if (pid >= fc.sppmPhotonsEmitted)
        return;
    RNG rng = createRNG(uint2(pid & 0xFFFFu, pid >> 16), fc.frameIndex);

    Ray r;
    float3 beta;
    if (!sppmEmitPhoton(rng, r, beta))
        return;

    HitInfo h = traceScene(fc, r);
    for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
    {
        r.origin = h.position + r.direction * 0.002;
        h = traceScene(fc, r);
    }

    uint depth = 0u;  // prior scatter count (specular or diffuse)
    for (uint bounce = 0u; bounce < SPPM_PHOTON_MAX_DEPTH; bounce++)
    {
        if (!h.hit)
            break;
        if ((materialTypes[h.materialId] & 0xFFu) != MATERIAL_TYPE_FLAT)
            break;  // flat-only photon transport (PM-1)

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

        BSDFSample bs = mat.sample(wo, rng);
        if (!bs.valid)
            break;
        // A glossy-continued vertex is treated like a specular vertex by the photon
        // stage (no deposit), keeping the eye VP placement and photon deposit consistent
        // and the direct(NEE)/indirect(photon) split disjoint (change sppm-glossy-final-gather).
        let specular = (bs.pdf <= 0.0) || sppmGlossyContinue(mat, bs);

        // Deposit at a non-specular vertex only after >= 1 prior bounce, so the
        // deposited term is purely indirect/caustic and disjoint from the NEE
        // direct the eye stage already accumulated (depth==0 = light->VP direct).
        if (!specular && depth >= 1u)
            sppmDepositPhoton(beta, h.position, r.direction);

        bs.wi = tangentToWorld(bs.wi, T, B, N);
        beta *= bs.weight;
        depth++;
        if (depth > 1u)
        {
            let p = clamp(max(max(beta.x, beta.y), beta.z), 0.05, 0.95);
            if (rng.next() > p)
                break;
            beta /= p;
        }
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
}

// ── Stage 6: update / resolve ───────────────────────────────────────────────
// Apply the SPPM radius/flux reduction, resolve this pass's radiance estimate
// (direct from the eye stage + photon-map indirect), composite into the
// progressive-accumulation film exactly like wfPathResolve, then clear the
// per-pass accumulator for the next pass.
[shader("compute")]
[numthreads(64, 1, 1)]
void wfSppmUpdate(uint3 tid: SV_DispatchThreadID)
{
    let i = tid.x;
    if (i >= sppmTile.streamSize)
        return;
    let lane = sppmTile.streamBase + i;
    if (lane >= sppmNumPixels())
        return;
    let pixel = uint2(lane % fc.width, lane / fc.width);

    VisiblePoint vp = sppmVisiblePoints[lane];
    SppmAccum acc = sppmAccum[lane];

    float3 lIndirect = float3(0.0);
    if ((vp.flags & VP_ACTIVE) != 0u)
    {
        // This pass's photon-map density estimate at the radius the photons
        // actually gathered within (the radius BEFORE this pass's reduction):
        //   L_indirect = Φ_thispass / (N_emitted · π · r²).
        // The progressive averaging is done by the accumulation film's
        // running mean over passes (each pass is an independent estimate at a
        // shrinking radius), NOT by accumulating τ here — accumulating τ AND
        // film-averaging would double-count. τ/N still advance the radius for
        // the next pass (bias → 0 as r shrinks).
        let phi = sppmDecodeFlux(acc);
        let rGather = vp.radius;
        let denom = max(float(fc.sppmPhotonsEmitted), 1.0) * PI * rGather * rGather;
        lIndirect = phi / max(denom, 1e-20);
        sppmUpdate(vp, phi, acc.m, 2.0 / 3.0);  // advance radius / N for the next pass
        sppmVisiblePoints[lane] = vp;
    }

    float3 sample = vp.ld + lIndirect;
    if (any(sample != sample) || any(isinf(sample)))
        sample = float3(0.0);
    sample = max(sample, float3(0.0));
    // Film per-sample radiance clamp (change film-maxcomponent-clamp); no-op
    // when fc.filmMaxComponent == 0. Matches main_pass.slang / wavefront_path.
    sample = clampSampleRadiance(sample, fc.filmMaxComponent);

    let n = fc.accumFrame;
    float3 accum;
    if (n == 0u)
        accum = sample;
    else
    {
        let prev = accumBuffer[pixel].rgb;
        accum = (prev * float(n) + sample) / float(n + 1u);
    }
    accumBuffer[pixel] = float4(accum, 1.0);
    wfWriteDisplay(pixel, accum);

    // Reset the per-pass photon accumulator for the next pass.
    SppmAccum z;
    z.phiR = 0u; z.phiG = 0u; z.phiB = 0u; z.m = 0u;
    sppmAccum[lane] = z;
}
