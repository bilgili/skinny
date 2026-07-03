// Participating-medium density seam (pbrt-subsurface-volumetric).
//
// The interior random walk reads a medium ONLY through `densityAt` and
// `mediumMajorant`, dispatched by `Medium.kind` (the runtime switch idiom — like
// flatSampleLobe — because a scene mixes media kinds per hit). This is the
// additive-extension contract the subsurface spec promised, now exercised by
// TWO kinds: MEDIUM_HOMOGENEOUS (the dielectric-bounded subsurface interior /
// a homogeneous free-standing interior) and MEDIUM_NANOVDB (a normalized
// density grid in the `volumeDensity` 3D texture, binding 26 — change
// nanovdb-volume-rendering). The transport loop, equations, RR, and integrator
// wiring are untouched, exactly as designed.
//
// `resolveMedium` is the handle indirection: today the handle is a material id
// and the medium is read inline from FlatMaterialParams (binding 13 — no new
// SSBO, to respect the Metal 31-buffer cap, per design.md), including the
// folded world→uvw rows for the grid kind.

import common;     // Medium, FlatMaterialParams
import bindings;   // MEDIUM_HOMOGENEOUS / MEDIUM_NANOVDB / MEDIUM_CLOUD, flatMaterials, volumeDensity
import materials.subsurface.cloud_noise;  // pbrt Perlin fBm cloudDensity (MEDIUM_CLOUD)

// slang-rhi's Metal backend cannot bind a combined `Sampler3D`, so on the Metal
// target the density grid is a plain `Texture3D` + a discrete `SamplerState`
// (bindings 26 + 44, same split as the discrete 2D maps — see environment.slang).
// Macros do not cross `import`, so the expansion lives here beside its one user.
#if defined(SKINNY_METAL)
#define VOLUME_SAMPLE_LEVEL(UVW) volumeDensity.SampleLevel(volumeDensitySampler, UVW, 0.0)
#else
#define VOLUME_SAMPLE_LEVEL(UVW) volumeDensity.SampleLevel(UVW, 0.0)
#endif

// Local density multiplier in [0,1] applied to the base extinction σ_t. The whole
// transport is written against this so the grid lookup is just a new case.
float densityAt(Medium m, float3 pWorld)
{
    switch (m.kind)
    {
    case MEDIUM_NANOVDB:
    {
        // One folded affine world → [0,1]³ texture coords (rows packed CPU-side).
        float3 uvw = float3(
            dot(m.worldToUvw0.xyz, pWorld) + m.worldToUvw0.w,
            dot(m.worldToUvw1.xyz, pWorld) + m.worldToUvw1.w,
            dot(m.worldToUvw2.xyz, pWorld) + m.worldToUvw2.w);
        // Explicit outside-grid zero: the sampler clamps to edge, and without
        // this guard a boundary texel's density would smear outward to every
        // point between the grid AABB and the bounding shape.
        if (any(uvw < float3(0.0)) || any(uvw > float3(1.0)))
            return 0.0;
        return VOLUME_SAMPLE_LEVEL(uvw);
    }
    case MEDIUM_CLOUD:
    {
        // Same folded affine, but the rows map world → medium-local [0,1]³
        // (importer-folded medium CTM). pbrt clips cloud transport to the
        // medium bounds (default unit cube) in SampleRay; the outside-zero
        // here is that clip — and it also kills the altitude floor term
        // (2·max(0, 0.5−y)) below the cube, which is unbounded otherwise.
        float3 pLocal = float3(
            dot(m.worldToUvw0.xyz, pWorld) + m.worldToUvw0.w,
            dot(m.worldToUvw1.xyz, pWorld) + m.worldToUvw1.w,
            dot(m.worldToUvw2.xyz, pWorld) + m.worldToUvw2.w);
        if (any(pLocal < float3(0.0)) || any(pLocal > float3(1.0)))
            return 0.0;
        // pbrt CloudMedium::Density (cloud_noise.slang) — already in [0,1].
        return cloudDensity(pLocal, m.cloudParams.x, m.cloudParams.y, m.cloudParams.z);
    }
    case MEDIUM_HOMOGENEOUS:
        return 1.0;
    default:
        return 1.0;
    }
}

// Majorant extinction σ̄_t ≥ σ_t(x) over the segment [a,b], for the null-collision
// accept probability σ_t/σ̄_t. Homogeneous ⇒ the constant σ_t.
float3 mediumMajorant(Medium m, float3 a, float3 b)
{
    switch (m.kind)
    {
    case MEDIUM_NANOVDB:
        // Global majorant. The grid's value max is folded into the packed σ on
        // the CPU (texels are normalized to [0,1]), so densityAt(x) ≤ 1 and the
        // packed σ_t IS the exact global majorant — no per-segment max needed
        // for correctness (macrocell/DDA local majorants are a noted follow-up
        // for convergence, not unbiasedness).
        return m.sigmaA + m.sigmaS;
    case MEDIUM_CLOUD:
        // pbrt CloudMedium::Density clamps to [0,1], so the packed σ_t is the
        // exact global majorant — the same argument (and pbrt's own
        // SampleRay homogeneous-majorant choice) as the grid case.
        return m.sigmaA + m.sigmaS;
    case MEDIUM_HOMOGENEOUS:
        return m.sigmaA + m.sigmaS;
    default:
        return m.sigmaA + m.sigmaS;
    }
}

// Handle → Medium. Handle = material id (material-attached medium: the
// subsurface interior or the free-standing interface boundary's interior).
Medium resolveMedium(uint materialId)
{
    FlatMaterialParams p = flatMaterials[materialId];
    Medium m;
    m.sigmaA      = p.mediumSigmaA;
    m.sigmaS      = p.mediumSigmaS;
    m.g           = p.mediumG;
    m.eta         = p.ior;            // boundary IOR reuses the flat `ior` slot
    m.kind        = p.mediumKind;
    m.worldToUvw0 = p.worldToUvw0;    // folded grid/medium transform (identity if none)
    m.worldToUvw1 = p.worldToUvw1;
    m.worldToUvw2 = p.worldToUvw2;
    m.cloudParams = p.cloudParams;    // density/wispiness/frequency (MEDIUM_CLOUD)
    return m;
}
