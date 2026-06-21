// Participating-medium density seam (pbrt-subsurface-volumetric).
//
// The interior random walk reads a medium ONLY through `densityAt` and
// `mediumMajorant`, dispatched by `Medium.kind` (the runtime switch idiom — like
// flatSampleLobe — because a scene mixes media kinds per hit). This is the
// additive-extension contract: adding a heterogeneous source (NanoVDB / Disney
// cloud) is one enum value + the two `case` bodies + ingestion — the transport
// loop, equations, RR, and integrator wiring are untouched.
//
// `resolveMedium` is the handle indirection: today the handle is a material id and
// the homogeneous medium is read inline from FlatMaterialParams (binding 13 — no
// new SSBO, to respect the Metal 31-buffer cap, per design.md). A future
// free-standing `MediumInterface` adds a dedicated registry behind this same
// function.

import common;     // Medium, FlatMaterialParams
import bindings;   // MEDIUM_HOMOGENEOUS / MEDIUM_NANOVDB, flatMaterials

// Local density multiplier in [0,1] applied to the base extinction σ_t. The whole
// transport is written against this so a grid lookup slots in as a new case.
float densityAt(Medium m, float3 pWorld)
{
    switch (m.kind)
    {
    case MEDIUM_HOMOGENEOUS:
        return 1.0;
    // case MEDIUM_NANOVDB:  return sampleDensityGrid(m.gridHandle, pWorld); // later
    default:
        return 1.0;
    }
}

// Majorant extinction σ̄_t ≥ σ_t(x) over the segment [a,b], for the null-collision
// accept probability σ_t/σ̄_t. Homogeneous ⇒ the constant σ_t. A grid returns its
// (optionally per-segment) max so thin clouds stay efficient.
float3 mediumMajorant(Medium m, float3 a, float3 b)
{
    switch (m.kind)
    {
    case MEDIUM_HOMOGENEOUS:
        return m.sigmaA + m.sigmaS;
    default:
        return m.sigmaA + m.sigmaS;
    }
}

// Handle → Medium. Handle = material id (material-attached interior). The
// free-standing-media follow-up discriminates other sources here, behind the same
// signature, without touching the walk.
Medium resolveMedium(uint materialId)
{
    FlatMaterialParams p = flatMaterials[materialId];
    Medium m;
    m.sigmaA     = p.mediumSigmaA;
    m.sigmaS     = p.mediumSigmaS;
    m.g          = p.mediumG;
    m.eta        = p.ior;            // boundary IOR reuses the flat `ior` slot
    m.kind       = p.mediumKind;
    m.gridHandle = 0u;               // reserved (NanoVDB)
    return m;
}
