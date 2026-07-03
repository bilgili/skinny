"""StdSurfaceParams (binding 19) MSL-relocation invariants.

``StructuredBuffer<StdSurfaceParams>`` is packed scalar/std430 (``pack_std_surface_params``,
float3 = 12 B) but Slang reads it MSL-padded on Metal (float3 → 16 B, element
stride ≈400 B): every field after ``base_color`` shifts (metalness reads
specular, specular reads specular_roughness, coat → 0, …). ``pack_std_surface_params_msl``
relocates each field into the reflected MSL layout — the same repack the skin
params (``_pack_mtlx_skin_array_msl``) and per-graph SSBOs already get.

FORWARD-LOOKING: binding 19 is read only by ``preview_pass.slang`` (the
BXDF/std_surface visualiser), a Vulkan-only ``PreviewPipeline`` where the scalar
layout is correct; the Metal megakernel dead-strips binding 19
(``loadStdSurfaceParams`` is uncalled). So this relocation is inert until a Metal
pipeline references ``stdSurfaceParams`` (a Metal preview/raster port) — it is
*not* the path-traced wood ~10 % brightness (that BSDF reads the float4-wrapped,
MSL-safe FlatMaterialParams at binding 13). This file pins the relocation correct
now so the future port is a no-op.

Two halves:

* **Host invariant (always, no GPU):** ``_STD_SURFACE_FIELDS`` covers the whole
  256 B scalar record with no gap/overlap, and ``pack_std_surface_params_msl``
  relocates fields to the offsets a given layout dictates.

* **Metal round-trip (guarded by a real Metal device, NOT the megakernel gate —
  this builds only a tiny struct-probe kernel, no MTLCompilerService spike):**
  relocate real wood constants, upload to ``StructuredBuffer<StdSurfaceParams>``,
  read fields back on the GPU, and assert they match the scalar (Vulkan-correct)
  values. The ``[1]`` index exercises the per-element MSL stride.

Run the Metal half (gpu-marked — kept out of the default sweep, one Metal
process at a time; the raw probe device is torn down by the ``metal_probe_device``
fixture):

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    PYTHONPATH=$PWD/src <repo>/bin/python3.13 -m pytest \
        tests/test_metal_std_surface_layout.py -m gpu -q
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

# skinny.renderer imports `vulkan` unconditionally; skip cleanly without the SDK.
try:
    from skinny.renderer import (
        STD_SURFACE_STRIDE,
        _STD_SURFACE_FIELDS,
        pack_std_surface_params,
        pack_std_surface_params_msl,
    )
except OSError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}",
                allow_module_level=True)

# StdSurfaceParams struct verbatim from mtlx_std_surface.slang — kept inline so
# the Metal round-trip compiles a tiny kernel (no megakernel / closures import).
_STD_SURFACE_STRUCT = """
struct StdSurfaceParams {
    float3 base_color; float base; float diffuse_roughness; float metalness;
    float specular; float specular_roughness; float3 specular_color;
    float specular_IOR; float specular_anisotropy; float specular_rotation;
    float transmission; float transmission_depth; float3 transmission_color;
    float transmission_scatter_anisotropy; float3 transmission_scatter;
    float transmission_dispersion; float transmission_extra_roughness;
    float subsurface; float subsurface_scale; float subsurface_anisotropy;
    float3 subsurface_color; float _pad0; float3 subsurface_radius; float sheen;
    float3 sheen_color; float sheen_roughness; float coat; float coat_roughness;
    float coat_anisotropy; float coat_rotation; float coat_IOR;
    float coat_affect_color; float coat_affect_roughness; float _pad1;
    float3 coat_color; float thin_film_thickness; float thin_film_IOR;
    float emission; float3 emission_color; float _pad2; float3 opacity;
    uint thin_walled; float _pad3; float _pad4;
};
"""


class _WoodStub:
    """Wood non-graph constants (standard_surface_wood_tiled.mtlx + packer
    defaults); base_color / specular_roughness are graph-driven at runtime, but
    the constant fields are what the binding-19 SSBO must carry correctly."""
    parameter_overrides = {
        "base": 1.0, "metalness": 0.0, "specular": 0.4, "specular_roughness": 0.5,
        "specular_color": (1.0, 1.0, 1.0), "coat": 0.1, "coat_roughness": 0.2,
        "coat_IOR": 1.5, "base_color": (0.8, 0.8, 0.8),
    }


def test_std_surface_scalar_fields_cover_256():
    """`_STD_SURFACE_FIELDS` is the relocation table; it must tile the entire
    scalar record exactly (no gap/overlap) or the MSL relocation drops bytes."""
    total = sum(n for _, n in _STD_SURFACE_FIELDS) * 4
    assert total == STD_SURFACE_STRIDE, (
        f"_STD_SURFACE_FIELDS covers {total} B, expected {STD_SURFACE_STRIDE}")


def test_std_surface_msl_relocation_places_fields():
    """`pack_std_surface_params_msl` moves each field from its scalar offset to the
    layout-dictated MSL offset. Exercised with the real reflected Metal layout
    (float3 padded to 16 B), independent of any GPU."""
    # Reflected Metal MSL offsets (from slang reflection of the struct above).
    layout = {
        "base_color": (0, 16), "base": (16, 4), "diffuse_roughness": (20, 4),
        "metalness": (24, 4), "specular": (28, 4), "specular_roughness": (32, 4),
        "specular_color": (48, 16), "coat": (260, 4), "coat_roughness": (264, 4),
        "coat_IOR": (276, 4),
    }
    stride = 400
    scalar = pack_std_surface_params(_WoodStub())
    msl = pack_std_surface_params_msl(scalar, layout, stride)
    assert len(msl) == stride
    rd = lambda off: struct.unpack_from("<f", msl, off)[0]  # noqa: E731
    assert rd(16) == pytest.approx(1.0)    # base
    assert rd(24) == pytest.approx(0.0)    # metalness
    assert rd(28) == pytest.approx(0.4)    # specular
    assert rd(32) == pytest.approx(0.5)    # specular_roughness
    assert rd(260) == pytest.approx(0.1)   # coat
    assert rd(264) == pytest.approx(0.2)   # coat_roughness
    assert rd(276) == pytest.approx(1.5)   # coat_IOR
    assert rd(0) == pytest.approx(0.8)     # base_color.x (float3 aligned at 0)


@pytest.mark.gpu
def test_metal_reads_relocated_std_surface_params(metal_probe_device):
    """On a real Metal device, the relocated record reads back field-for-field
    correct through `StructuredBuffer<StdSurfaceParams>` (the `[1]` index exercises
    the per-element MSL stride) — proves the relocation is layout-correct for a
    future Metal pipeline that reads binding 19. Cheap: one tiny probe kernel."""
    spy = pytest.importorskip("slangpy")

    kernel_src = _STD_SURFACE_STRUCT + """
StructuredBuffer<StdSurfaceParams> stdSurfaceParams;
RWStructuredBuffer<float> outBuf;
[shader("compute")][numthreads(1,1,1)]
void computeMain(uint3 t : SV_DispatchThreadID) {
    StdSurfaceParams p = stdSurfaceParams[1];
    outBuf[0]=p.base; outBuf[1]=p.metalness; outBuf[2]=p.specular;
    outBuf[3]=p.specular_roughness; outBuf[4]=p.coat; outBuf[5]=p.coat_roughness;
    outBuf[6]=p.coat_IOR; outBuf[7]=p.base_color.x;
}
"""
    dev = metal_probe_device
    opts = spy.SlangCompilerOptions()
    opts.matrix_layout = spy.SlangMatrixLayout.column_major
    sess = dev.create_slang_session(compiler_options=opts)
    mod = sess.load_module_from_source("ss_probe", kernel_src, "ss_probe.slang")
    prog = sess.link_program([mod], [mod.entry_point("computeMain")])
    par = {p.name: p for p in prog.layout.parameters}["stdSurfaceParams"]
    etl = par.type_layout.element_type_layout
    layout = {f.name: (int(f.offset), int(f.type_layout.size)) for f in etl.fields}
    stride = int(getattr(etl, "stride", 0) or etl.size)
    assert stride > STD_SURFACE_STRIDE, (
        f"expected MSL stride > {STD_SURFACE_STRIDE} (float3 padding), got {stride}")

    scalar = pack_std_surface_params(_WoodStub())
    msl = pack_std_surface_params_msl(scalar, layout, stride)
    data = (b"\x00" * stride) + msl  # slot 0 zeroed, wood in slot 1
    buf = dev.create_buffer(
        element_count=2, struct_size=stride,
        usage=spy.BufferUsage.shader_resource,
        data=np.frombuffer(data, np.uint8).copy(),
        memory_type=spy.MemoryType.device_local, label="ss")
    out = dev.create_buffer(
        element_count=8, struct_size=4,
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
        memory_type=spy.MemoryType.device_local, label="o")
    kernel = dev.create_compute_kernel(prog)
    kernel.dispatch(thread_count=[1, 1, 1],
                    vars={"stdSurfaceParams": buf, "outBuf": out})
    dev.wait_for_idle()
    g = out.to_numpy().view(np.float32)[:8]
    expect = [1.0, 0.0, 0.4, 0.5, 0.1, 0.2, 1.5, 0.8]
    names = ["base", "metalness", "specular", "specular_roughness",
             "coat", "coat_roughness", "coat_IOR", "base_color.x"]
    for name, e, got in zip(names, expect, g):
        assert got == pytest.approx(e, abs=1e-4), (
            f"Metal misread {name}: got {got}, expected {e}")
