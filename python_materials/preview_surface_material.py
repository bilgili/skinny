# ruff: noqa: F821
"""Python-authored surface material — UsdPreviewSurface-shaped, IMaterial-conforming.

Analogous to a MaterialX `<surfacematerial>` authored in XML, but written as
pure Python and transpiled to Slang by SlangPile. Structure mirrors MaterialX
plus the renderer's `IMaterial` interface seam:

    MaterialX                      | This module
    -------------------------------|------------------------------------------
    `<nodedef>` inputs             | `PreviewSurfaceInputs` (param struct)
    `<nodegraph>` helper nodes     | `checker_pattern`, `perturb_normal`
    `<surface>` shader             | `PreviewSurfaceMaterial : IMaterial`
    `<surfacematerial>` binding    | `assets/cornell_box_python_material.usda`

The `PreviewSurfaceMaterial` struct conforms to `IMaterial`
([src/skinny/shaders/interfaces.slang](src/skinny/shaders/interfaces.slang))
and exposes the `sample` / `evaluate` pair the renderer's path integrator
calls in tangent space (N = +Z). BSDF is plain Lambertian over
`params.diffuseColor`; opacity / ior / specular / metallic are forwarded for
parity with UsdPreviewSurface but only `diffuseColor` and `emissiveColor`
participate in this minimal implementation. Extend the methods to layer GGX
specular on top once it's needed.

Codegen runs automatically at compute-pipeline startup via
`vk_compute.ComputePipeline._run_codegen`, which transpiles every `.py` in
this folder through `skinny.slangpile.build_module`. Output:

    src/skinny/mtlx/genslang/python_materials/preview_surface_material.slang

Default values match the `UsdPreviewSurface` inputs declared in
`assets/cornell_box_python_material.usda`.
"""

from skinny import slangpile as sp

_common = sp.slang_import("common")
_interfaces = sp.slang_import("interfaces")

PI = sp.extern(name="PI", module="common", args=[], returns=sp.float32)
sampleCosineHemisphereTS = sp.extern(
    name="sampleCosineHemisphereTS",
    module="common",
    args=[sp.float32x2],
    returns=sp.float32x3,
)
cosineHemispherePdfTS = sp.extern(
    name="cosineHemispherePdfTS",
    module="common",
    args=[sp.float32x3],
    returns=sp.float32,
)

RNG = sp.extern_type("RNG")
BSDFSample = sp.extern_type("BSDFSample")


@sp.struct
class PreviewSurfaceInputs:
    diffuseColor: sp.float32x3
    roughness: sp.float32
    metallic: sp.float32
    specular: sp.float32
    emissiveColor: sp.float32x3
    opacity: sp.float32
    ior: sp.float32
    normalScale: sp.float32


@sp.shader
def checker_pattern(uv: sp.float32x2, scale: sp.float32) -> sp.float32:
    """Procedural UV checker — returns 0.0 or 1.0."""
    su = sp.floor(uv.x * scale)
    sv = sp.floor(uv.y * scale)
    return sp.fmod(su + sv, 2.0)


@sp.shader
def perturb_normal(N: sp.float32x3, scale: sp.float32) -> sp.float32x3:
    """Stand-in for a normal-map lookup. Returns the geometric normal blended
    toward (0,0,1) by `scale`. Replace with a sampled tangent-space normal
    once texture bindings are wired."""
    return sp.normalize(sp.lerp(sp.float32x3(0.0, 0.0, 1.0), N, scale))


@sp.struct(conforms_to="IMaterial")
class PreviewSurfaceMaterial:
    params: PreviewSurfaceInputs

    def sample(self, wo: sp.float32x3, rng: sp.inout(RNG)) -> BSDFSample:
        s: BSDFSample
        u: sp.float32x2 = rng.next2()
        wi: sp.float32x3 = sampleCosineHemisphereTS(u)
        s.wi = wi
        s.weight = self.params.diffuseColor
        s.pdf = cosineHemispherePdfTS(wi)
        s.emission = self.params.emissiveColor
        s.valid = wi.z > 0.0
        s.transmitted = False
        return s

    def evaluate(self, wo: sp.float32x3, wi: sp.float32x3) -> BSDFSample:
        e: BSDFSample
        NdotL: sp.float32 = max(wi.z, 0.0)
        e.response = self.params.diffuseColor * (NdotL / PI)
        e.pdf = NdotL / PI
        e.wi = sp.float32x3(0.0, 0.0, 0.0)
        e.weight = sp.float32x3(0.0, 0.0, 0.0)
        e.emission = sp.float32x3(0.0, 0.0, 0.0)
        e.valid = False
        e.transmitted = False
        return e
