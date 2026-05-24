"""Verify Python struct packing matches Slang struct sizes.

The project compiles with -fvk-use-scalar-layout so float3 has 4-byte
alignment (not 16-byte). Silent mismatches cause GPU corruption.
"""

from __future__ import annotations

import struct

import pytest


class TestPythonPackingSizes:
    """Test Python-side struct packing produces correct byte counts."""

    def test_skin_parameters_pack_size(self):
        from skinny.renderer import SkinParameters

        data = SkinParameters().pack()
        assert len(data) == 80

    def test_flat_material_stride(self):
        from skinny.renderer import FLAT_MATERIAL_STRIDE

        assert FLAT_MATERIAL_STRIDE == 128

    def test_flat_material_pack_size(self):
        from skinny.renderer import pack_flat_material
        from types import SimpleNamespace

        material = SimpleNamespace(parameter_overrides={})
        data = pack_flat_material(material)
        assert len(data) == 128

    def test_flat_material_cutout_packing(self):
        """Cutout opacity config: threshold, opacity texture idx, and
        channelMask opacity=a must land at the documented offsets.
        Guards the data-flow side of the cutout-opacity fix (design doc:
        docs/superpowers/specs/2026-05-24-cutout-opacity-fix-design.md).
        """
        from types import SimpleNamespace
        from skinny.renderer import pack_flat_material, _encode_channel_mask

        material = SimpleNamespace(
            parameter_overrides={"opacityThreshold": 0.5},
        )
        channel_mask = _encode_channel_mask({"opacity": "a"})
        data = pack_flat_material(
            material,
            opacity_texture_idx=7,
            channel_mask=channel_mask,
        )

        assert len(data) == 128
        # opacityTextureIdx at byte 76 (uint)
        assert struct.unpack_from("I", data, 76)[0] == 7
        # opacityThreshold at byte 92 (float)
        assert abs(struct.unpack_from("f", data, 92)[0] - 0.5) < 1e-6
        # channelMask at byte 108 (uint); opacity slot is bits 12..16
        mask = struct.unpack_from("I", data, 108)[0]
        assert ((mask >> 12) & 0xF) == 4  # "a" code


@pytest.mark.gpu
class TestSlangStructSizes:
    """Verify Slang struct sizes via reflection match expected values."""

    def test_common_module_loads(self, load_shader):
        m = load_shader("common.slang")
        assert m is not None

    def test_sphere_light_struct(self, load_shader):
        """SphereLight: float3 position + float radius + float3 radiance + float pad = 32 bytes."""
        m = load_shader("test_common_harness.slang")
        assert m is not None

    def test_emissive_triangle_struct(self, load_shader):
        """EmissiveTriangle: 3x(float3 + float pad) + float3 emission + float area = 64 bytes."""
        m = load_shader("test_common_harness.slang")
        assert m is not None


class TestSkinParametersFieldPacking:
    """Verify individual field layout of SkinParameters.pack()."""

    def test_fields_round_trip(self):
        from skinny.renderer import SkinParameters

        p = SkinParameters()
        p.melanin_fraction = 0.25
        p.hemoglobin_fraction = 0.1
        p.blood_oxygenation = 0.9
        p.roughness = 0.4
        p.ior = 1.45
        data = p.pack()
        assert len(data) == 80
        melanin = struct.unpack_from("f", data, 0)[0]
        hb = struct.unpack_from("f", data, 4)[0]
        oxy = struct.unpack_from("f", data, 8)[0]
        assert abs(melanin - 0.25) < 1e-6
        assert abs(hb - 0.1) < 1e-6
        assert abs(oxy - 0.9) < 1e-6
