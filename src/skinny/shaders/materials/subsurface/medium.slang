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

// World → medium support box: the folded affine (rows packed CPU-side) mapping
// a world point to the medium's [0,1]³ box — uvw texture coords for the nanovdb
// grid, medium-local coords for the procedural cloud (both share the row
// convention and the unit-box support). Returns false (and leaves `uvw`
// computed) when the point is outside the box, where every box-supported kind
// is exactly vacuum. `clipSegmentToGrid` (volume_walk.slang) clips segments to
// the same box through the same rows.
bool mediumBoxPoint(Medium m, float3 pWorld, out float3 uvw)
{
    uvw = float3(
        dot(m.worldToUvw0.xyz, pWorld) + m.worldToUvw0.w,
        dot(m.worldToUvw1.xyz, pWorld) + m.worldToUvw1.w,
        dot(m.worldToUvw2.xyz, pWorld) + m.worldToUvw2.w);
    return !(any(uvw < float3(0.0)) || any(uvw > float3(1.0)));
}

// Local density multiplier in [0,1] applied to the base extinction σ_t. The whole
// transport is written against this so the grid lookup is just a new case.
float densityAt(Medium m, float3 pWorld)
{
    switch (m.kind)
    {
    case MEDIUM_NANOVDB:
    {
        // One folded affine world → [0,1]³ texture coords. The explicit
        // outside-box zero matters: the sampler clamps to edge, so without it a
        // boundary texel's density would smear across the empty margin between
        // the grid AABB and the bounding shape.
        float3 uvw;
        if (!mediumBoxPoint(m, pWorld, uvw))
            return 0.0;
        return VOLUME_SAMPLE_LEVEL(uvw);
    }
    case MEDIUM_CLOUD:
    {
        // Same folded affine, rows map world → medium-local [0,1]³ (folded
        // medium CTM). pbrt clips cloud transport to the medium bounds (default
        // unit cube) in SampleRay; this outside-zero is that clip — and it also
        // kills the altitude floor term (2·max(0, 0.5−y)) below the cube, which
        // is otherwise unbounded.
        float3 pLocal;
        if (!mediumBoxPoint(m, pWorld, pLocal))
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
// accept probability σ_t/σ̄_t. Every current kind uses the same GLOBAL majorant =
// the packed σ_t: homogeneous is constant σ_t; the nanovdb grid normalizes its
// texels to [0,1] (value max folded into the packed σ on the CPU) so densityAt ≤
// 1; and pbrt's CloudMedium::Density clamps to [0,1]. So σ_t bounds σ_t·density
// exactly for all of them — no per-segment max needed for unbiasedness
// (macrocell/DDA local majorants are a convergence-only follow-up). pbrt itself
// takes this homogeneous-majorant choice in both grid and cloud `SampleRay`.
float3 mediumMajorant(Medium m, float3 a, float3 b)
{
    return m.sigmaA + m.sigmaS;
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
