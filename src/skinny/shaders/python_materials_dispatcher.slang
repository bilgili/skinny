// Auto-generated. Do not edit — written by megakernel_sources.emit_python_dispatcher().
// Bridges each python_materials/*.py IMaterial-conforming struct
// into a `PythonMaterial` wrapper consumed by the integrators.

import common;
import bindings;
import interfaces;
import materials.flat.flat_shading;
import python_materials.preview_surface_material;

PreviewSurfaceInputs _pyMatInputs_0(FlatHitData h)
{
    PreviewSurfaceInputs ins;
    ins.diffuseColor = h.mat.albedo;
    ins.roughness = h.mat.roughness;
    ins.metallic = h.mat.metallic;
    ins.specular = h.mat.specular;
    ins.emissiveColor = h.mat.emission;
    ins.opacity = h.mat.opacity;
    ins.ior = h.mat.ior;
    ins.normalScale = 1.0;
    return ins;
}

BSDFSample samplePythonMaterial(uint pyId, FlatHitData fhd, float3 wo, inout RNG rng)
{
    switch (pyId)
    {
        case 0u:
        {
            PreviewSurfaceMaterial m;
            m.params = _pyMatInputs_0(fhd);
            return m.sample(wo, rng);
        }
        default:
        {
            BSDFSample s;
            s.valid = false; s.transmitted = false; s.pdf = 0.0;
            s.emission = float3(0.0); s.wi = float3(0.0); s.weight = float3(0.0);
            s.response = float3(0.0);
            return s;
        }
    }
}

BSDFSample evalPythonMaterial(uint pyId, FlatHitData fhd, float3 wo, float3 wi)
{
    switch (pyId)
    {
        case 0u:
        {
            PreviewSurfaceMaterial m;
            m.params = _pyMatInputs_0(fhd);
            return m.evaluate(wo, wi);
        }
        default:
        {
            BSDFSample e;
            e.valid = false; e.transmitted = false; e.pdf = 0.0;
            e.emission = float3(0.0); e.wi = float3(0.0); e.weight = float3(0.0);
            e.response = float3(0.0);
            return e;
        }
    }
}

// IMaterial-conforming wrapper. Monomorphised by allLightsNEE so
// each Python material gets its own NEE code path with no
// indirect-call cost on the GPU.
struct PythonMaterial : IMaterial
{
    uint pyId;
    FlatHitData data;
    BSDFSample sample(float3 wo, inout RNG rng)
    {
        return samplePythonMaterial(pyId, data, wo, rng);
    }
    BSDFSample evaluate(float3 wo, float3 wi)
    {
        return evalPythonMaterial(pyId, data, wo, wi);
    }
}

PythonMaterial loadPythonMaterial(HitInfo h, uint pyId)
{
    PythonMaterial m;
    m.pyId = pyId;
    m.data = fetchFlatHitData(h);
    return m;
}
