"""Gates for the material field table (change flat-material-field-table).

The two material records are owned field by field, not just by stride. The
tests here are what makes that ownership real:

* **Transposition gate** — a permanent name→byte-offset golden. Swapping two
  same-typed fields moves both offsets and fails, where the size-equality
  assert that used to guard these records passes. A negative control performs
  that swap and asserts the gate catches it.
* **Byte identity** — the field-table packers emit exactly what the
  60-argument positional ``struct.pack`` emitted.
* **One vocabulary** — the three dialect alias tables are projections; the
  "keep in sync with pack_flat_material" comments they used to carry are
  assertions here.
* **One derivation step** — the override merge is ordered before the
  derivations, and each derivation runs once.

Hostless: these packers left ``renderer.py`` in ``renderer-pure-core-extraction``
precisely so this gate runs on a Metal-only host, where the same bytes are
uploaded. Import from the OWNING modules — importing ``skinny.renderer`` drags
in ``vulkan`` and would make the whole file skip.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from skinny import material_pack, slang_layout

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "material_field_offsets.json").read_text())

RECORDS = ("FlatMaterialParams", "StdSurfaceParams")


class _Mat:
    """Minimal stand-in for ``scene.Material`` — the packers read only these."""

    def __init__(self, overrides=None, *, name="m", mtlx_target_name=None,
                 python_module=None):
        self.parameter_overrides = dict(overrides or {})
        self.name = name
        self.mtlx_target_name = mtlx_target_name
        self.python_module = python_module


# ── Transposition gate ───────────────────────────────────────────────


@pytest.mark.parametrize("record", RECORDS)
@pytest.mark.parametrize("dialect", ("scalar", "msl"))
def test_field_offsets_match_the_permanent_golden(record, dialect):
    want = GOLDEN[f"{record}.{dialect}"]
    got = slang_layout.material_field_offsets(record, msl=dialect == "msl")
    assert got == want["offsets"]
    stride = (slang_layout.msl_stride if dialect == "msl"
              else slang_layout.scalar_stride)(record)
    assert stride == want["stride"]


@pytest.mark.parametrize("record", RECORDS)
def test_fields_tile_the_stride_with_no_gap_or_overlap(record):
    slang_layout.check_material_record(record)


def test_transposing_two_same_typed_fields_fails_the_gate(monkeypatch):
    """Negative control. ``roughness`` and ``metallic`` are both floats and both
    four bytes, so the old size-equality assert could not tell them apart —
    exactly the failure this gate exists for."""
    fields = list(slang_layout.FLAT_MATERIAL_FIELDS)
    i = next(n for n, f in enumerate(fields) if f.name == "roughness")
    j = next(n for n, f in enumerate(fields) if f.name == "metallic")
    a, b = fields[i], fields[j]
    swapped = list(fields)
    swapped[i] = slang_layout.MaterialField(a.name, b.row, b.lane, a.kind,
                                            a.default, key=a.key)
    swapped[j] = slang_layout.MaterialField(b.name, a.row, a.lane, b.kind,
                                            b.default, key=b.key)
    monkeypatch.setattr(slang_layout, "FLAT_MATERIAL_FIELDS", tuple(swapped))

    # The record still tiles its stride perfectly — the size-only check passes.
    slang_layout.check_material_record("FlatMaterialParams")
    # The offsets, however, have moved.
    got = slang_layout.material_field_offsets("FlatMaterialParams")
    assert got != GOLDEN["FlatMaterialParams.scalar"]["offsets"]
    assert got["roughness"] == 16 and got["metallic"] == 12


def test_a_field_naming_a_row_the_shader_lacks_is_refused(monkeypatch):
    bad = (*slang_layout.FLAT_MATERIAL_FIELDS[:-1],
           slang_layout.MaterialField("_cloudPad", "_noSuchRow", 3, "float", 0.0))
    monkeypatch.setattr(slang_layout, "FLAT_MATERIAL_FIELDS", bad)
    with pytest.raises(slang_layout.SlangLayoutError, match="_noSuchRow"):
        slang_layout.material_field_offsets("FlatMaterialParams")


def test_a_field_overrunning_its_row_is_refused(monkeypatch):
    bad = (*slang_layout.FLAT_MATERIAL_FIELDS[:-1],
           slang_layout.MaterialField("_cloudPad", "_cloudDensityWispinessFrequency",
                                      3, "color3", (0.0, 0.0, 0.0)))
    monkeypatch.setattr(slang_layout, "FLAT_MATERIAL_FIELDS", bad)
    with pytest.raises(slang_layout.SlangLayoutError, match="lanes"):
        slang_layout.material_field_offsets("FlatMaterialParams")


# ── Name-keyed packing ───────────────────────────────────────────────


def test_packing_an_unknown_field_name_raises():
    with pytest.raises(slang_layout.SlangLayoutError, match="rooughness"):
        slang_layout.pack_material_record("FlatMaterialParams", {"rooughness": 1.0})


@pytest.mark.parametrize("record", RECORDS)
def test_omitted_fields_take_the_table_default(record):
    rec = slang_layout.pack_material_record(record, {})
    offsets = slang_layout.material_field_offsets(record)
    for f in slang_layout.material_fields(record):
        off = offsets[f.name]
        if f.kind == "uint":
            got = struct.unpack_from("<I", rec, off)[0]
            assert got == int(f.default) & 0xFFFFFFFF, f.name
        elif f.kind == "float":
            assert struct.unpack_from("<f", rec, off)[0] == pytest.approx(
                float(f.default)), f.name
        else:
            n = f.lanes
            got = struct.unpack_from(f"<{n}f", rec, off)
            assert got == pytest.approx(tuple(float(c) for c in f.default)), f.name


def test_every_field_lands_where_the_table_says():
    """Write one distinguishable value per field, then read each back at its
    golden offset. Catches a packer that agrees with the table by accident."""
    values, expect = {}, {}
    for n, f in enumerate(slang_layout.FLAT_MATERIAL_FIELDS):
        if f.kind == "uint":
            v = 0x1000 + n
        elif f.kind == "float":
            v = 100.0 + n
        else:
            v = tuple(100.0 + n + 0.25 * k for k in range(f.lanes))
        values[f.name] = v
        expect[f.name] = v
    rec = slang_layout.pack_material_record("FlatMaterialParams", values)
    assert len(rec) == slang_layout.scalar_stride("FlatMaterialParams")
    offsets = GOLDEN["FlatMaterialParams.scalar"]["offsets"]
    for f in slang_layout.FLAT_MATERIAL_FIELDS:
        off = offsets[f.name]
        if f.kind == "uint":
            assert struct.unpack_from("<I", rec, off)[0] == expect[f.name], f.name
        elif f.kind == "float":
            assert struct.unpack_from("<f", rec, off)[0] == pytest.approx(
                expect[f.name]), f.name
        else:
            got = struct.unpack_from(f"<{f.lanes}f", rec, off)
            assert got == pytest.approx(expect[f.name]), f.name


# ── Byte identity with the pre-change positional packers ─────────────
#
# The reference bytes below are the pre-change `struct.pack` output, held as the
# literal format+argument spelling the old packers used. If the table ever
# disagrees with them, the refactor stopped being pure.


def _legacy_flat(**f) -> bytes:
    return struct.pack(
        "fff f f f f I I I I I fff f  f f f I  fff f  fff I fff f  fff f  fff I"
        "  fff f fff I ffff ffff ffff ffff",
        *f["diffuse"], f["roughness"], f["metallic"], f["specular"], f["opacity"],
        f["diffuse_texture_idx"], f["roughness_texture_idx"],
        f["metallic_texture_idx"], f["normal_texture_idx"],
        f["emissive_texture_idx"],
        *f["emissive"], f["ior"],
        f["coat"], f["coat_roughness"], f["coat_ior"], f["opacity_texture_idx"],
        *f["coat_color"], f["opacity_threshold"],
        *f["normal_scale"], f["channel_mask"],
        *f["normal_bias"], f["glass_cauchy_b"],
        *f["transmission_color"], f["diffuse_roughness"],
        *f["specular_color"], f["conductor_metal_id"],
        *f["medium_sigma_a"], f["medium_g"],
        *f["medium_sigma_s"], f["medium_kind"],
        *f["w2u"][0], *f["w2u"][1], *f["w2u"][2],
        f["cloud_density"], f["cloud_wispiness"], f["cloud_frequency"], 0.0,
    )


def test_flat_bytes_match_the_pre_change_positional_pack():
    mat = _Mat({
        "diffuseColor": (0.1, 0.2, 0.3), "roughness": 0.25, "metallic": 0.75,
        "specular": 0.4, "opacity": 0.6, "emissiveColor": (1.0, 2.0, 3.0),
        "ior": 1.7, "coat": 0.5, "coat_roughness": 0.15, "coat_IOR": 1.4,
        "coat_color": (0.9, 0.8, 0.7), "opacityThreshold": 0.33,
        "transmission_color": (0.11, 0.22, 0.33),
        "specular_color": (0.4, 0.5, 0.6), "diffuse_roughness": 0.8,
    })
    got = material_pack.pack_flat_material(
        mat, 1, 2, 3, 4, 5, 6, normal_scale=(1.0, 2.0, 3.0),
        normal_bias=(-0.5, -0.25, 0.0), channel_mask=0x54321)
    want = _legacy_flat(
        diffuse=(0.1, 0.2, 0.3), roughness=0.25, metallic=0.75, specular=0.4,
        opacity=0.6, diffuse_texture_idx=1, roughness_texture_idx=2,
        metallic_texture_idx=3, normal_texture_idx=4, emissive_texture_idx=5,
        emissive=(1.0, 2.0, 3.0), ior=1.7, coat=0.5, coat_roughness=0.15,
        coat_ior=1.4, opacity_texture_idx=6, coat_color=(0.9, 0.8, 0.7),
        opacity_threshold=0.33, normal_scale=(1.0, 2.0, 3.0), channel_mask=0x54321,
        normal_bias=(-0.5, -0.25, 0.0), glass_cauchy_b=0.0,
        transmission_color=(0.11, 0.22, 0.33), diffuse_roughness=0.8,
        specular_color=(0.4, 0.5, 0.6), conductor_metal_id=0,
        medium_sigma_a=(0.0, 0.0, 0.0), medium_g=0.0,
        medium_sigma_s=(0.0, 0.0, 0.0), medium_kind=0,
        w2u=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
        cloud_density=0.0, cloud_wispiness=0.0, cloud_frequency=0.0)
    assert got == want


def test_absent_transmission_color_still_falls_back_to_the_diffuse_albedo():
    """The table's own default for that field is inert; the packer supplies the
    albedo, which is what keeps the delta-transmission weight unchanged."""
    mat = _Mat({"diffuseColor": (0.25, 0.5, 0.75)})
    rec = material_pack.pack_flat_material(mat)
    off = GOLDEN["FlatMaterialParams.scalar"]["offsets"]["transmissionColor"]
    assert struct.unpack_from("<3f", rec, off) == pytest.approx((0.25, 0.5, 0.75))


def test_std_surface_bytes_match_the_pre_change_positional_pack():
    mat = _Mat({
        "base_color": (0.3, 0.4, 0.5), "base": 0.9, "metalness": 0.8,
        "specular_roughness": 0.35, "coat": 0.45, "coat_IOR": 1.55,
        "thin_walled": 1, "opacity": (0.4, 0.5, 0.6), "emission": 2.0,
    })
    got = material_pack.pack_std_surface_params(mat)
    want = struct.pack(
        "ffffffff" "ffffffff" "ffffffff" "ffffffff"
        "ffffffff" "ffffffff" "ffffffff" "fffffIff",
        0.3, 0.4, 0.5, 0.9,            # base_color, base
        0.0, 0.8, 1.0, 0.35,           # diffuse_roughness, metalness, specular, spec_rough
        1.0, 1.0, 1.0, 1.5,            # specular_color, specular_IOR
        0.0, 0.0, 0.0, 0.0,            # spec_aniso, spec_rot, transmission, trans_depth
        1.0, 1.0, 1.0, 0.0,            # transmission_color, scatter_aniso
        0.0, 0.0, 0.0, 0.0,            # transmission_scatter, dispersion
        0.0, 0.0, 1.0, 0.0,            # extra_rough, subsurface, ss_scale, ss_aniso
        1.0, 1.0, 1.0, 0.0,            # subsurface_color, _pad0
        1.0, 1.0, 1.0, 0.0,            # subsurface_radius, sheen
        1.0, 1.0, 1.0, 0.3,            # sheen_color, sheen_roughness
        0.45, 0.1, 0.0, 0.0,           # coat, coat_roughness, coat_aniso, coat_rot
        1.55, 0.0, 0.0, 0.0,           # coat_IOR, affect_color, affect_rough, _pad1
        1.0, 1.0, 1.0, 0.0,            # coat_color, thin_film_thickness
        1.5, 2.0, 1.0, 1.0,            # thin_film_IOR, emission, emission_color.rg
        1.0, 0.0, 0.4, 0.5, 0.6,       # emission_color.b, _pad2, opacity
        1, 0.0, 0.0,                   # thin_walled, _pad3, _pad4
    )
    assert got == want


def test_std_surface_msl_relocation_reads_the_same_table():
    scalar = material_pack.pack_std_surface_params(_Mat({"base_color": (1, 2, 3),
                                                        "metalness": 0.5}))
    msl_layout = slang_layout.msl_layout("StdSurfaceParams")
    relocated = material_pack.pack_std_surface_params_msl(
        scalar, msl_layout.offsets, msl_layout.stride)
    assert len(relocated) == msl_layout.stride
    msl_off = slang_layout.material_field_offsets("StdSurfaceParams", msl=True)
    assert struct.unpack_from("<3f", relocated, msl_off["base_color"]) == \
        pytest.approx((1.0, 2.0, 3.0))
    assert struct.unpack_from("<f", relocated, msl_off["metalness"])[0] == \
        pytest.approx(0.5)


# ── One vocabulary ───────────────────────────────────────────────────


def test_alias_tables_are_projections_not_restatements():
    """The assertion the two "Keep in sync with pack_flat_material" comments
    used to be."""
    from skinny import mtlx_synthesis, usd_loader

    table = slang_layout.std_surface_to_flat()
    assert usd_loader._STD_SURFACE_TO_FLAT == table
    assert mtlx_synthesis._STD_SURFACE_TO_FLAT_PACK == table
    assert mtlx_synthesis._PREVIEW_SURFACE_FLAT_KEYS == \
        slang_layout.PREVIEW_SURFACE_FLAT_KEYS


def test_the_alias_table_still_holds_the_twelve_historical_entries():
    """Recorded verdict for task 4.1. The two tables disagreed by 7 entries; all
    7 are IDENTITY mappings, benign only because `_store_shader_override` writes
    the raw name unconditionally. The union is the projection."""
    assert slang_layout.std_surface_to_flat() == {
        "base_color": "diffuseColor",
        "specular_roughness": "roughness",
        "metalness": "metallic",
        "specular_IOR": "ior",
        "specular": "specular",
        "specular_color": "specular_color",
        "transmission_color": "transmission_color",
        "diffuse_roughness": "diffuse_roughness",
        "coat": "coat",
        "coat_roughness": "coat_roughness",
        "coat_color": "coat_color",
        "coat_IOR": "coat_IOR",
    }


def test_std_surface_opacity_is_not_aliased_onto_the_flat_float():
    """The one std-surface key the kind guard withholds: `color3` there, `float`
    in the flat record. Aliasing it would advertise an edit the packer's
    `_override_float` discards."""
    assert "opacity" in slang_layout.STD_SURFACE_OVERRIDE_KEYS
    assert "opacity" in slang_layout.FLAT_OVERRIDE_KEYS
    assert "opacity" not in slang_layout.std_surface_to_flat()


def test_alias_targets_and_preview_keys_are_all_real_flat_keys():
    table = slang_layout.std_surface_to_flat()
    assert set(table.values()) <= slang_layout.FLAT_OVERRIDE_KEYS
    assert set(table) <= slang_layout.STD_SURFACE_OVERRIDE_KEYS
    assert slang_layout.PREVIEW_SURFACE_FLAT_KEYS <= slang_layout.FLAT_OVERRIDE_KEYS


def test_the_vocabulary_matches_the_golden():
    v = GOLDEN["vocabulary"]
    assert sorted(slang_layout.FLAT_OVERRIDE_KEYS) == v["flat"]
    assert sorted(slang_layout.STD_SURFACE_OVERRIDE_KEYS) == v["std_surface"]
    assert sorted(slang_layout.INTAKE_ONLY_KEYS) == v["intake_only"]
    assert sorted(slang_layout.RENDERER_OVERRIDE_KEYS) == v["renderer"]
    assert sorted(slang_layout.PREVIEW_SURFACE_INPUT_KEYS) == v["preview_surface_inputs"]
    assert sorted(slang_layout.OPENPBR_ONLY_KEYS) == v["openpbr_only"]
    assert slang_layout.OPENPBR_TO_STD_SURFACE == v["openpbr_to_std_surface"]
    # The whole vocabulary, pinned: growing it is a deliberate edit, because
    # every addition widens what packing will silently accept.
    assert sorted(slang_layout.MATERIAL_OVERRIDE_KEYS) == v["all"]


def test_every_flat_field_key_is_in_the_flat_vocabulary():
    for f in slang_layout.FLAT_MATERIAL_FIELDS:
        if f.key:
            assert f.key in slang_layout.FLAT_OVERRIDE_KEYS, f.name


def test_an_unknown_key_on_a_table_owned_material_is_refused():
    mat = _Mat({"diffuseColor": (1, 1, 1), "roughnes": 0.5}, name="typo")
    with pytest.raises(material_pack.UnknownOverrideKey, match="roughnes"):
        material_pack.pack_flat_material(mat)


def test_an_unknown_key_on_a_data_driven_material_only_warns(caplog):
    """A MaterialX-targeted material may legitimately carry the referenced
    document's own input names, so refusing there would fail a valid scene."""
    mat = _Mat({"layer_top_melanin": 0.4}, name="skin",
               mtlx_target_name="M_skinny_skin_default")
    with caplog.at_level("WARNING"):
        material_pack.pack_flat_material(mat)
    assert "layer_top_melanin" in caplog.text


def test_python_materials_are_also_data_driven():
    mat = _Mat({"my_slangpile_input": 1.0}, python_module="python_materials.foo")
    material_pack.pack_flat_material(mat)  # must not raise


@pytest.mark.parametrize("key", sorted(slang_layout.MATERIAL_OVERRIDE_KEYS))
def test_no_vocabulary_key_is_refused(key):
    material_pack.pack_flat_material(_Mat({key: 0.0}))


# The gate that closes the hole the corpus survey could not see. Surveying
# SCENES only reaches the authoring paths some scene happens to exercise; the
# inline-preview authoring path and the plugin-present OpenPBR intake author
# names no corpus scene carries, and enforcement turned each of those into a
# crash. Enumerate the AUTHORING SITES instead — a new key at any of them fails
# here, hostlessly, instead of at someone's next render.

def test_every_authored_preview_input_is_in_the_vocabulary():
    from skinny import usd_material_edit

    for name, _type, _default in usd_material_edit._PREVIEW_INPUTS:
        assert name in slang_layout.MATERIAL_OVERRIDE_KEYS, name
        # and it must actually pack, not merely be listed
        material_pack.pack_flat_material(_Mat({name: 0.0}))


def test_every_openpbr_spelling_is_in_the_vocabulary():
    """`_store_shader_override` keeps the raw OpenPBR name AND writes the folded
    standard_surface one, so both spellings reach a table-owned material."""
    for raw, std in slang_layout.OPENPBR_TO_STD_SURFACE.items():
        assert raw in slang_layout.MATERIAL_OVERRIDE_KEYS, raw
        assert std in slang_layout.MATERIAL_OVERRIDE_KEYS, std


def test_every_advertised_editable_is_in_the_vocabulary():
    from skinny import mtlx_synthesis, scene_graph

    for name in mtlx_synthesis._PREVIEW_AUTHORABLE:
        assert name in slang_layout.MATERIAL_OVERRIDE_KEYS, name
    for name in mtlx_synthesis._MATERIAL_FLOAT_RANGES:
        assert name in slang_layout.MATERIAL_OVERRIDE_KEYS, name
    for name in scene_graph._MATERIAL_FLOAT_RANGES:
        assert name in slang_layout.MATERIAL_OVERRIDE_KEYS, name


def test_the_openpbr_table_is_a_projection():
    from skinny import usd_loader

    assert usd_loader._OPENPBR_TO_STD_SURFACE is slang_layout.OPENPBR_TO_STD_SURFACE


def test_an_inline_authored_preview_material_packs():
    """Regression for the codex P1: `author_preview_material` writes
    `specularColor` on every inline UsdPreviewSurface, and that material is
    table-owned (no mtlx target), so a missing vocabulary entry crashed the
    upload rather than rendering."""
    mat = _Mat({"diffuseColor": (0.8, 0.8, 0.8), "emissiveColor": (0.0, 0.0, 0.0),
                "specularColor": (0.0, 0.0, 0.0), "roughness": 0.5,
                "metallic": 0.0, "clearcoat": 0.0, "clearcoatRoughness": 0.01,
                "opacity": 1.0, "ior": 1.5}, name="Preview")
    assert len(material_pack.pack_flat_material(mat)) == \
        slang_layout.scalar_stride("FlatMaterialParams")


def test_an_openpbr_material_read_without_the_mtlx_hint_packs():
    """The plugin-present intake stores every OpenPBR input under its raw name,
    and such a material has no `mtlx_target_name` — so it is table-owned."""
    ov = {raw: 0.5 for raw in slang_layout.OPENPBR_TO_STD_SURFACE}
    ov.update({std: 0.5 for std in slang_layout.OPENPBR_TO_STD_SURFACE.values()})
    ov["geometry_opacity"] = 1.0
    assert len(material_pack.pack_flat_material(_Mat(ov, name="openpbr"))) == \
        slang_layout.scalar_stride("FlatMaterialParams")


# ── One derivation step ──────────────────────────────────────────────


def test_derivations_run_once_and_in_order():
    """The transmission bridge must author `opacity` BEFORE the subsurface
    bridge looks for it, or a transmissive subsurface material gets 0 instead of
    1 - transmission."""
    from skinny import usd_loader

    overrides = {"transmission": 0.25, "subsurface": 1.0,
                 "subsurface_sigma_a": (1.0, 1.0, 1.0),
                 "subsurface_sigma_s": (1.0, 1.0, 1.0),
                 "clearcoat": 0.5, "clearcoatRoughness": 0.2}
    usd_loader._apply_override_derivations(overrides)
    assert overrides["opacity"] == pytest.approx(0.75)
    assert overrides["coat"] == pytest.approx(0.5)
    assert overrides["coat_roughness"] == pytest.approx(0.2)


def test_applying_the_derivations_twice_changes_nothing():
    """What the removed re-run relied on. It holds — but the ordering, not the
    idempotence, is now what guarantees the result."""
    from skinny import usd_loader

    base = {"transmission": 0.25, "subsurface": 1.0,
            "subsurface_sigma_a": (1.0, 0.0, 0.0), "clearcoat": 0.5}
    once = dict(base)
    usd_loader._apply_override_derivations(once)
    twice = dict(once)
    usd_loader._apply_override_derivations(twice)
    assert once == twice


def test_the_subsurface_rederive_is_gone_from_the_merge_seam():
    """`_merge_prim_overrides` used to call `_derive_opacity_from_subsurface`
    directly, because `_load_mtlx_materials` had already derived once, too
    early. Both sites go through the one ordered step now."""
    import ast
    import inspect

    from skinny import usd_loader

    for fn in (usd_loader._merge_prim_overrides, usd_loader._extract_material):
        tree = ast.parse(inspect.getsource(fn).lstrip())
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "_derive_opacity_from_subsurface" not in called, fn.__name__
        assert "_derive_opacity_from_transmission" not in called, fn.__name__
        assert "_apply_override_derivations" in called, fn.__name__
