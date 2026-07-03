"""FlatMaterialParams (binding 13) Metal-stride pin — nanovdb-volume-rendering
+ pbrt-cloud-procedural-medium.

The 192 → 240 B growth (three appended float4 worldToUvw rows) and the
240 → 256 B growth (one appended float4 of cloud density/wispiness/frequency)
must keep the
scalar CPU pack byte-compatible with what Metal reads through
``StructuredBuffer<FlatMaterialParams>``. Unlike StdSurfaceParams (loose
float3s → MSL repack), FlatMaterialParams is deliberately float4-wrapped, so
the MSL element stride must EQUAL the scalar stride (240) and every field must
reflect at its documented scalar offset — no relocation pass exists for this
buffer, so this test failing means the struct gained a padding-sensitive field
and the packer/struct must be fixed together.

Mirrors tests/test_metal_std_surface_layout.py's Metal round-trip: a tiny
struct-probe kernel (no megakernel import, no MTLCompilerService spike) reads a
packed volume-material record back field-for-field, exercising the ``[1]``
index for the per-element stride. gpu-marked: needs a real Metal device.

Run:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    PYTHONPATH=$PWD/src <repo>/bin/python3.13 -m pytest \
        tests/test_metal_flat_material_layout.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

try:
    from skinny.renderer import FLAT_MATERIAL_STRIDE, pack_flat_material
except OSError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}",
                allow_module_level=True)

pytestmark = pytest.mark.gpu

# FlatMaterialParams fields verbatim from common.slang (properties dropped —
# they don't affect layout). Kept inline so the probe compiles standalone.
_FLAT_MATERIAL_STRUCT = """
struct FlatMaterialParams {
    float4 _diffuseColorRoughness;
    float  metallic;
    float  specular;
    float  opacity;
    uint   diffuseTextureIdx;
    uint   roughnessTextureIdx;
    uint   metallicTextureIdx;
    uint   normalTextureIdx;
    uint   emissiveTextureIdx;
    float4 _emissiveColorIor;
    float  coat;
    float  coatRoughness;
    float  coatIOR;
    uint   opacityTextureIdx;
    float4 _coatColorOpacityThreshold;
    float4 _normalScaleChannelMask;
    float4 _normalBiasPad;
    float4 _transmissionColorDiffuseRough;
    float4 _specularColorPad;
    float4 _mediumSigmaA_g;
    float4 _mediumSigmaS_kind;
    float4 _worldToUvw0;
    float4 _worldToUvw1;
    float4 _worldToUvw2;
    float4 _cloudDensityWispinessFrequency;
};
"""


def test_metal_flat_material_stride_and_volume_fields(metal_probe_device):
    """MSL stride == scalar stride (256) and the appended worldToUvw rows +
    medium + cloud fields read back exactly from the scalar-packed record."""
    spy = pytest.importorskip("slangpy")

    kernel_src = _FLAT_MATERIAL_STRUCT + """
StructuredBuffer<FlatMaterialParams> flatMaterials;
RWStructuredBuffer<float> outBuf;
[shader("compute")][numthreads(1,1,1)]
void computeMain(uint3 t : SV_DispatchThreadID) {
    FlatMaterialParams p = flatMaterials[1];
    outBuf[0] = p._mediumSigmaA_g.x;   outBuf[1] = p._mediumSigmaA_g.w;
    outBuf[2] = p._mediumSigmaS_kind.x;
    outBuf[3] = float(asuint(p._mediumSigmaS_kind.w));    // mediumKind as float
    outBuf[4] = p._worldToUvw0.x;  outBuf[5] = p._worldToUvw0.w;
    outBuf[6] = p._worldToUvw1.y;  outBuf[7] = p._worldToUvw2.z;
    outBuf[8] = p._emissiveColorIor.w;                    // ior slot (eta = 1)
    outBuf[9] = p._diffuseColorRoughness.x;
    outBuf[10] = p._cloudDensityWispinessFrequency.x;
    outBuf[11] = p._cloudDensityWispinessFrequency.z;
}
"""
    dev = metal_probe_device
    opts = spy.SlangCompilerOptions()
    opts.matrix_layout = spy.SlangMatrixLayout.column_major
    sess = dev.create_slang_session(compiler_options=opts)
    mod = sess.load_module_from_source("fm_probe", kernel_src, "fm_probe.slang")
    prog = sess.link_program([mod], [mod.entry_point("computeMain")])
    par = {p.name: p for p in prog.layout.parameters}["flatMaterials"]
    etl = par.type_layout.element_type_layout
    stride = int(getattr(etl, "stride", 0) or etl.size)
    assert stride == FLAT_MATERIAL_STRIDE == 256, (
        f"FlatMaterialParams must stay float4-wrapped: MSL stride {stride} != "
        f"scalar {FLAT_MATERIAL_STRIDE} — a padding-sensitive field was added")
    offsets = {f.name: int(f.offset) for f in etl.fields}
    assert offsets["_worldToUvw0"] == 192
    assert offsets["_worldToUvw1"] == 208
    assert offsets["_worldToUvw2"] == 224
    assert offsets["_cloudDensityWispinessFrequency"] == 240

    rows = np.array([[0.5, 0.0, 0.0, 0.25],
                     [0.0, 0.5, 0.0, 0.75],
                     [0.0, 0.0, 0.5, -0.5]], np.float32)
    mat = SimpleNamespace(parameter_overrides={
        "volume_interface": True,
        "volume_sigma_a": (0.125, 0.0, 0.0),
        "volume_sigma_s": (2.0, 1.0, 0.5),
        "volume_g": 0.877,
        "volume_grid_asset": "/x/cloud.nvdb",
        "cloud_density": 2.0,        # ignored: not a volume_cloud material
        "cloud_frequency": 5.0,
        "diffuseColor": (0.25, 0.25, 0.25),
    })
    rec = pack_flat_material(
        mat, volume_world_to_uvw=rows, volume_value_max=2.0, mm_per_unit=1000.0)
    assert len(rec) == 256
    data = (b"\x00" * 256) + rec  # slot 0 zeroed, volume material in slot 1

    buf = dev.create_buffer(
        element_count=2, struct_size=256,
        usage=spy.BufferUsage.shader_resource,
        data=np.frombuffer(data, np.uint8).copy(),
        memory_type=spy.MemoryType.device_local, label="fm")
    out = dev.create_buffer(
        element_count=12, struct_size=4,
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
        memory_type=spy.MemoryType.device_local, label="o")
    kernel = dev.create_compute_kernel(prog)
    kernel.dispatch(thread_count=[1, 1, 1],
                    vars={"flatMaterials": buf, "outBuf": out})
    dev.wait_for_idle()
    g = out.to_numpy().view(np.float32)[:12]

    fold = 2.0 / 1000.0  # value_max / mm_per_unit
    assert g[0] == pytest.approx(0.125 * fold, rel=1e-4)   # σ_a.x folded
    assert g[1] == pytest.approx(0.877, rel=1e-4)          # g
    assert g[2] == pytest.approx(2.0 * fold, rel=1e-4)     # σ_s.x folded
    assert g[3] == pytest.approx(1.0)                      # mediumKind = NANOVDB
    assert g[4] == pytest.approx(0.5)                      # worldToUvw rows
    assert g[5] == pytest.approx(0.25)
    assert g[6] == pytest.approx(0.5)
    assert g[7] == pytest.approx(0.5)
    assert g[8] == pytest.approx(1.0)                      # index-matched eta
    assert g[9] == pytest.approx(0.25)                     # untouched prefix
    assert g[10] == pytest.approx(0.0)                     # cloud lanes zero for grid
    assert g[11] == pytest.approx(0.0)
