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
        # 192 → 240 (nanovdb-volume-rendering): three float4 worldToUvw rows
        # appended at 192/208/224; 240 → 256 (pbrt-cloud-procedural-medium):
        # one float4 (cloud density/wispiness/frequency + pad) appended at 240.
        # The 0..240 prefix layout is unchanged.
        from skinny.renderer import FLAT_MATERIAL_STRIDE

        assert FLAT_MATERIAL_STRIDE == 256

    def test_flat_material_pack_size(self):
        from skinny.renderer import pack_flat_material
        from types import SimpleNamespace

        material = SimpleNamespace(parameter_overrides={})
        data = pack_flat_material(material)
        assert len(data) == 256

    def test_openpbr_names_map_to_std_surface(self):
        """OpenPBR shader-input names (`transmission_weight`, `base_metalness`,
        `specular_ior`, …) must reach the standard_surface packer, which reads
        Autodesk names (`transmission`, `metalness`, `specular_IOR`). The
        materialxusd exporter authors OpenPBR names; without the alias map
        every weight silently falls back to its default (transmission→0 ⇒
        opaque glass, base_metalness→0 ⇒ metals render as dielectric).
        """
        from types import SimpleNamespace
        from skinny.renderer import pack_std_surface_params
        from skinny.usd_loader import _store_shader_override

        overrides: dict[str, object] = {}
        # Mimic what _extract_material records for the Glass_OPBR material.
        _store_shader_override(overrides, "transmission_weight", 1.0)
        _store_shader_override(overrides, "base_metalness", 0.0)
        _store_shader_override(overrides, "base_color", (1.0, 1.0, 1.0))
        _store_shader_override(overrides, "specular_ior", 1.52)

        data = pack_std_surface_params(SimpleNamespace(parameter_overrides=overrides))
        transmission = struct.unpack_from("f", data, 56)[0]
        metalness = struct.unpack_from("f", data, 20)[0]
        specular_ior = struct.unpack_from("f", data, 44)[0]

        assert abs(transmission - 1.0) < 1e-6, "transmission_weight must map to transmission"
        assert abs(metalness - 0.0) < 1e-6
        assert abs(specular_ior - 1.52) < 1e-4, "specular_ior must map to specular_IOR"

    def test_transmission_lowers_opacity(self):
        """A transmissive standard_surface/OpenPBR material must lower `opacity`
        so the flat path's refraction branch (`if (m.opacity < 1.0)`) fires.
        Without this the surface stays opaque regardless of `transmission`
        (glass/oil rendered solid). Guards the native-USD parse path, which —
        unlike the .mtlx fallback — previously never bridged the two.
        """
        from skinny.usd_loader import (
            _store_shader_override,
            _derive_opacity_from_transmission,
        )

        # OpenPBR glass: transmission_weight=1 → transmission=1 → opacity=0.
        o: dict[str, object] = {}
        _store_shader_override(o, "transmission_weight", 1.0)
        _derive_opacity_from_transmission(o)
        assert abs(float(o["opacity"]) - 0.0) < 1e-6

        # Partial transmission halves opacity.
        o2: dict[str, object] = {}
        _store_shader_override(o2, "transmission_weight", 0.25)
        _derive_opacity_from_transmission(o2)
        assert abs(float(o2["opacity"]) - 0.75) < 1e-6

        # Opaque material is untouched (no opacity key invented).
        o3: dict[str, object] = {"transmission": 0.0}
        _derive_opacity_from_transmission(o3)
        assert "opacity" not in o3

        # An explicitly authored opacity wins over the derived value.
        o4: dict[str, object] = {"transmission": 1.0, "opacity": 0.3}
        _derive_opacity_from_transmission(o4)
        assert abs(float(o4["opacity"]) - 0.3) < 1e-6

    def test_transmission_opacity_gate_ignores_default_opaque(self):
        """A `standard_surface`/OpenPBR shader ALWAYS authors `opacity` (its
        MaterialX default is the fully-opaque `(1, 1, 1)`). That default must NOT
        block the `transmission → opacity` refraction bridge — only a genuine
        cutout (`opacity < 1`) should win over transmission.

        Regression: the MaterialX `standard_surface` glass in
        `assets/glass_caustics_test.usda` (`transmission = 1`, default
        `opacity = (1, 1, 1)`) rendered OPAQUE because the bridge skipped on the
        mere presence of the default-opaque authored opacity.
        """
        from skinny.usd_loader import _derive_opacity_from_transmission

        # Default-opaque SCALAR opacity (1.0) must not block transmission.
        scal: dict[str, object] = {"transmission": 1.0, "opacity": 1.0}
        _derive_opacity_from_transmission(scal)
        assert abs(float(scal["opacity"]) - 0.0) < 1e-6, (
            "default-opaque scalar opacity must not block the transmission bridge"
        )

        # Default-opaque PER-CHANNEL opacity (1, 1, 1) — the standard_surface
        # default — must not block transmission (the glass_caustics_test bug).
        col: dict[str, object] = {"transmission": 1.0, "opacity": (1.0, 1.0, 1.0)}
        _derive_opacity_from_transmission(col)
        assert float(col["opacity"]) < 1.0, (
            "default-opaque (1,1,1) opacity must not block the transmission bridge"
        )

        # Partial transmission with a default-opaque opacity still bridges.
        part: dict[str, object] = {"transmission": 0.25, "opacity": (1.0, 1.0, 1.0)}
        _derive_opacity_from_transmission(part)
        assert abs(float(part["opacity"]) - 0.75) < 1e-6

        # A GENUINE cutout (some channel < 1) is preserved over transmission.
        cutout: dict[str, object] = {"transmission": 1.0, "opacity": (1.0, 1.0, 0.5)}
        _derive_opacity_from_transmission(cutout)
        assert cutout["opacity"] == (1.0, 1.0, 0.5), (
            "a genuine cutout opacity (channel < 1) must win over transmission"
        )

    def test_subsurface_opacity_gate_requires_medium(self):
        """Only a genuine volumetric subsurface medium (pbrt `Material
        "subsurface"`, carrying σ_a/σ_s and routed to MATERIAL_TYPE_SUBSURFACE)
        may open the flat refraction gate via `opacity = 0`.

        A plain Autodesk standard_surface `subsurface` *weight* (e.g. the
        three_materials_demo marble) is a diffuse-SSS shading term, NOT a
        transmissive boundary — forcing opacity=0 there turns it into clear
        glass that refracts the environment (the "marble totally broken" bug).
        The gate must mirror renderer._material_is_subsurface (σ_a/σ_s present).
        """
        from skinny.usd_loader import _derive_opacity_from_subsurface

        # Plain standard_surface subsurface weight, NO medium → opacity untouched.
        marble: dict[str, object] = {
            "subsurface": 0.4,
            "subsurface_color": (0.45, 0.45, 0.55),
            "diffuseColor": (0.45, 0.45, 0.55),
        }
        _derive_opacity_from_subsurface(marble)
        assert "opacity" not in marble, (
            "a standard_surface subsurface weight with no medium must stay opaque"
        )

        # Genuine pbrt subsurface (σ_a/σ_s present) → opacity 0 (refraction gate).
        sss: dict[str, object] = {
            "subsurface": 1.0,
            "subsurface_sigma_a": (0.032, 0.17, 0.48),
            "subsurface_sigma_s": (0.74, 0.88, 1.01),
        }
        _derive_opacity_from_subsurface(sss)
        assert float(sss["opacity"]) == 0.0

        # σ present but all-zero is not a medium → opacity untouched.
        zero_sigma: dict[str, object] = {
            "subsurface": 1.0,
            "subsurface_sigma_a": (0.0, 0.0, 0.0),
            "subsurface_sigma_s": (0.0, 0.0, 0.0),
        }
        _derive_opacity_from_subsurface(zero_sigma)
        assert "opacity" not in zero_sigma

        # Explicitly authored opacity always wins, even with a medium.
        authored: dict[str, object] = {
            "subsurface": 1.0,
            "subsurface_sigma_s": (0.74, 0.88, 1.01),
            "opacity": 0.3,
        }
        _derive_opacity_from_subsurface(authored)
        assert abs(float(authored["opacity"]) - 0.3) < 1e-6

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

        assert len(data) == 256
        # opacityTextureIdx at byte 76 (uint)
        assert struct.unpack_from("I", data, 76)[0] == 7
        # opacityThreshold at byte 92 (float)
        assert abs(struct.unpack_from("f", data, 92)[0] - 0.5) < 1e-6
        # channelMask at byte 108 (uint); opacity slot is bits 12..16
        mask = struct.unpack_from("I", data, 108)[0]
        assert ((mask >> 12) & 0xF) == 4  # "a" code

    def test_flat_material_rich_inputs_packing(self):
        """Stage-2 rich inputs (flat-lobes-rich-inputs): transmissionColor@128,
        diffuseRoughness@140, specularColor@144 — and the back-compat defaults
        (transmissionColor ← diffuseColor, specularColor = white,
        diffuseRoughness = 0) when the overrides are absent.
        """
        from types import SimpleNamespace
        from skinny.renderer import pack_flat_material

        # Explicit rich inputs land at the documented offsets.
        material = SimpleNamespace(
            parameter_overrides={
                "diffuseColor": (0.1, 0.2, 0.3),
                "transmission_color": (0.4, 0.5, 0.6),
                "diffuse_roughness": 0.7,
                "specular_color": (0.8, 0.9, 1.0),
            }
        )
        data = pack_flat_material(material)
        assert struct.unpack_from("fff", data, 128) == pytest.approx((0.4, 0.5, 0.6))
        assert struct.unpack_from("f", data, 140)[0] == pytest.approx(0.7)
        assert struct.unpack_from("fff", data, 144) == pytest.approx((0.8, 0.9, 1.0))

        # Defaults: transmissionColor falls back to diffuseColor, specularColor
        # is white, diffuseRoughness is 0 — so existing renders are unchanged.
        bare = SimpleNamespace(parameter_overrides={"diffuseColor": (0.1, 0.2, 0.3)})
        d2 = pack_flat_material(bare)
        assert struct.unpack_from("fff", d2, 128) == pytest.approx((0.1, 0.2, 0.3))
        assert struct.unpack_from("f", d2, 140)[0] == pytest.approx(0.0)
        assert struct.unpack_from("fff", d2, 144) == pytest.approx((1.0, 1.0, 1.0))

    def test_flat_material_subsurface_medium_packing(self):
        """Subsurface medium (pbrt-subsurface-volumetric): σ_a@160, g@172,
        σ_s@176, mediumKind@188 (uint, MEDIUM_HOMOGENEOUS=0). Zero for a
        non-subsurface material so the inline pack is inert."""
        from types import SimpleNamespace
        from skinny.renderer import pack_flat_material

        mat = SimpleNamespace(parameter_overrides={
            "subsurface_sigma_a": (0.032, 0.17, 0.48),
            "subsurface_sigma_s": (0.74, 0.88, 1.01),
            "subsurface_g": 0.0,
            "subsurface_eta": 1.33,
            "ior": 1.33,
        })
        data = pack_flat_material(mat)
        assert struct.unpack_from("fff", data, 160) == pytest.approx((0.032, 0.17, 0.48))
        assert struct.unpack_from("f", data, 172)[0] == pytest.approx(0.0)
        assert struct.unpack_from("fff", data, 176) == pytest.approx((0.74, 0.88, 1.01))
        assert struct.unpack_from("I", data, 188)[0] == 0  # MEDIUM_HOMOGENEOUS

        # Non-subsurface material: medium is zero (inert).
        bare = SimpleNamespace(parameter_overrides={"diffuseColor": (0.5, 0.5, 0.5)})
        d2 = pack_flat_material(bare)
        assert struct.unpack_from("fff", d2, 160) == pytest.approx((0.0, 0.0, 0.0))
        assert struct.unpack_from("fff", d2, 176) == pytest.approx((0.0, 0.0, 0.0))


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
