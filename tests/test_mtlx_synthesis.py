"""Hostless tests for skinny.mtlx_synthesis (GPU-free, no renderer import).

Run with the py3.13 venv that has PyMaterialXGenSlang:

    PYTHONPATH=src ./bin/python3.13 -m pytest tests/test_mtlx_synthesis.py -q

These exercise the real MaterialX Slang generator dry-run — no GPU, no Vulkan.
The per-node gen test is the standing gate for the node whitelist (design D4).
"""

from __future__ import annotations

import MaterialX as mx
import pytest

import skinny.mtlx_synthesis as m
from skinny.mtlx_synthesis import MaterialSpecError


# ─── Form dispatch ────────────────────────────────────────────────────

def test_preset_form():
    norm = m.validate_spec({"preset": "marble_solid"})
    assert norm["form"] == "preset"
    assert norm["path"].endswith("standard_surface_marble_solid.mtlx")


def test_preview_form():
    norm = m.validate_spec({"model": "preview", "params": {"roughness": 0.5}})
    assert norm["form"] == "preview"
    assert norm["params"]["roughness"] == 0.5


def test_standard_surface_form():
    norm = m.validate_spec({"model": "standard_surface", "params": {"base": 0.8}})
    assert norm["form"] == "standard_surface"
    assert norm["graph"] is None


def test_no_form_rejected():
    with pytest.raises(MaterialSpecError):
        m.validate_spec({"params": {}})


def test_mixed_forms_rejected():
    with pytest.raises(MaterialSpecError) as e:
        m.validate_spec({"preset": "marble_solid", "model": "preview"})
    msg = str(e.value)
    assert "preset" in msg and "model" in msg


def test_graph_on_preview_rejected():
    with pytest.raises(MaterialSpecError) as e:
        m.validate_spec({"model": "preview", "graph": {"nodes": {}}})
    assert "standard_surface" in str(e.value)


def test_unknown_model_rejected():
    with pytest.raises(MaterialSpecError):
        m.validate_spec({"model": "toon", "params": {}})


# ─── Preset catalog + path safety ─────────────────────────────────────

def test_catalog_lists_curated():
    presets = m.list_presets()
    for expected in ("marble_solid", "wood_tiled", "brass_tiled", "chrome",
                     "glass", "jade"):
        assert expected in presets, expected
        assert presets[expected].endswith(".mtlx")


def test_unknown_preset_lists_names():
    with pytest.raises(MaterialSpecError) as e:
        m.resolve_preset("does_not_exist")
    # error lists available names
    assert "marble_solid" in str(e.value)


def test_path_shaped_preset_not_resolved_as_path():
    # A path-shaped client string must miss the catalog, never be joined onto
    # a filesystem path (design D3).
    with pytest.raises(MaterialSpecError) as e:
        m.resolve_preset("../../../etc/foo")
    assert "unknown preset" in str(e.value)


# ─── Param validation ─────────────────────────────────────────────────

def test_color3_coercion():
    norm = m.validate_spec(
        {"model": "standard_surface", "params": {"base_color": [0.7, 0.2, 0.1]}}
    )
    assert norm["params"]["base_color"] == [0.7, 0.2, 0.1]


def test_float_bounds_enforced():
    with pytest.raises(MaterialSpecError):
        # specular_roughness range is 0.04..1.0
        m.validate_spec(
            {"model": "standard_surface", "params": {"specular_roughness": 5.0}}
        )


def test_non_finite_rejected():
    with pytest.raises(MaterialSpecError):
        m.validate_spec(
            {"model": "standard_surface", "params": {"specular_IOR": float("nan")}}
        )


def test_unknown_param_rejected():
    with pytest.raises(MaterialSpecError):
        m.validate_spec(
            {"model": "standard_surface", "params": {"not_a_real_input": 1.0}}
        )


# ─── Graph validation ─────────────────────────────────────────────────

def test_whitelisted_graph_accepted():
    spec = {
        "model": "standard_surface",
        "graph": {
            "nodes": {
                "pos": {"type": "position"},
                "n": {"type": "fractal3d"},
                "mx": {"type": "mix"},
            },
            "connections": [],
        },
    }
    norm = m.validate_spec(spec)
    assert norm["form"] == "standard_surface"


def test_unsupported_node_lists_full_set():
    spec = {
        "model": "standard_surface",
        "graph": {"nodes": {"v": {"type": "voronoi"}}, "connections": []},
    }
    with pytest.raises(MaterialSpecError) as e:
        m.validate_spec(spec)
    msg = str(e.value)
    # error contains the full supported set
    for node in m.NODE_WHITELIST:
        assert node in msg, node


def test_dangling_connection_named():
    spec = {
        "model": "standard_surface",
        "graph": {
            "nodes": {"pos": {"type": "position"}},
            "connections": [["ghost.out", "base_color"]],
        },
    }
    with pytest.raises(MaterialSpecError) as e:
        m.validate_spec(spec)
    assert "ghost" in str(e.value)


def test_checker_not_whitelisted():
    # `checker` was dropped: this MaterialX build ships `checkerboard`.
    assert "checker" not in m.NODE_WHITELIST
    assert "checker" not in m.TEMPLATES


# ─── Template expansion + bounds ──────────────────────────────────────

def test_noise_template_expansion():
    norm = m.validate_spec(
        {"template": "noise",
         "params": {"colorA": [0.9, 0.9, 0.85], "colorB": [0.2, 0.2, 0.25], "octaves": 4}}
    )
    assert norm["form"] == "standard_surface"
    assert norm["_promote_all"] is True
    assert norm["graph"]["nodes"]  # a real graph was produced


def test_negative_octaves_rejected():
    with pytest.raises(MaterialSpecError):
        m.expand_template("noise", {"octaves": -3})


def test_template_unknown_param_rejected():
    with pytest.raises(MaterialSpecError):
        m.expand_template("noise", {"not_a_param": 1.0})


def test_unknown_template_rejected():
    with pytest.raises(MaterialSpecError):
        m.expand_template("plaid", {})


# ─── Salting / naming contract ────────────────────────────────────────

def test_surfacematerial_name_equals_prim_name():
    doc = m.build_document(
        m.validate_spec({"model": "standard_surface", "params": {"base": 1.0}}),
        "Speckle",
    )
    assert m._find_surfacematerial(doc) == "Speckle"


def test_two_same_spec_materials_distinct_element_names():
    spec = {"template": "noise", "params": {}}
    r1 = m.synthesize(spec, "NoiseA")
    r2 = m.synthesize(spec, "NoiseB")
    d1 = mx.createDocument(); mx.readFromXmlString(d1, r1.document_xml)
    d2 = mx.createDocument(); mx.readFromXmlString(d2, r2.document_xml)
    names1 = {c.getName() for c in d1.getChildren()}
    names2 = {c.getName() for c in d2.getChildren()}
    # surfacematerial + shader + nodegraph names all salted, so no overlap
    assert names1.isdisjoint(names2), (names1 & names2)
    assert "NoiseA" in names1 and "NoiseB" in names2


# ─── Mapping derivation ───────────────────────────────────────────────

def test_template_params_mapped():
    r = m.synthesize({"template": "noise", "params": {}}, "N")
    for key in ("colorA", "colorB", "scale", "octaves", "lacunarity", "diminish"):
        assert key in r.mapping, key
        assert r.mapping[key]  # at least one uniform


def test_unexposed_constant_absent_from_mapping():
    spec = {
        "model": "standard_surface",
        "graph": {
            "nodes": {
                "pos": {"type": "position", "output": "vector3"},
                "f": {"type": "fractal3d", "output": "color3",
                      "position": {"_connect": "pos"},
                      "octaves": 3},  # NOT exposed
            },
            "connections": [["f.out", "base_color"]],
        },
    }
    r = m.synthesize(spec, "Const1")
    assert "octaves" not in r.mapping
    assert r.mapping == {}


def test_shattered_input_maps_to_all_uniforms():
    # One interface input `k` feeds two node inputs; both generated uniforms
    # must appear in its mapping entry (design D5, M1).
    spec = {
        "model": "standard_surface",
        "graph": {
            "nodes": {
                "base": {"type": "multiply", "output": "color3",
                         "in1": [0.5, 0.5, 0.5],
                         "in2": {"expose": True, "value": 2.0, "name": "k"}},
                "top": {"type": "add", "output": "color3",
                        "in1": {"_connect": "base"},
                        "in2": {"expose": True, "value": 0.1, "name": "k"}},
            },
            "connections": [["top.out", "base_color"]],
        },
    }
    r = m.synthesize(spec, "Shatter1")
    assert "k" in r.mapping
    assert len(r.mapping["k"]) == 2, r.mapping["k"]
    assert set(r.mapping["k"]) == {"base_in2", "top_in2"}


# ─── Per-node generator dry-run (standing whitelist gate) ─────────────

def _minimal_node_doc(category: str, material_name: str) -> mx.Document:
    """A minimal std_surface document exercising one whitelisted node.

    The node's output is bridged to base_color (color3) with a `convert` node
    when it cannot itself produce color3. This proves the node compiles through
    the generator — the gate the design requires for every whitelist entry.
    """
    # (output_type, {input: value|('node', category)}) per node.
    cases = {
        "position":   ("vector3", {}),
        "texcoord":   ("vector2", {}),
        "fractal3d":  ("color3", {"position": ("node", "position", "vector3"),
                                  "octaves": 3}),
        "noise2d":    ("color3", {"texcoord": ("node", "texcoord", "vector2")}),
        "noise3d":    ("color3", {"position": ("node", "position", "vector3")}),
        "mix":        ("color3", {"fg": [0.1, 0.1, 0.1], "bg": [0.8, 0.8, 0.8],
                                  "mix": 0.5}),
        "multiply":   ("color3", {"in1": [0.5, 0.5, 0.5], "in2": 2.0}),
        "add":        ("color3", {"in1": [0.1, 0.1, 0.1], "in2": 0.2}),
        "subtract":   ("color3", {"in1": [0.8, 0.8, 0.8], "in2": 0.2}),
        "sin":        ("float",  {"in": 0.5}),
        "power":      ("color3", {"in1": [0.5, 0.5, 0.5], "in2": 2.0}),
        "dotproduct": ("float",  {"in1": ("node", "position", "vector3"),
                                  "in2": ("node", "position", "vector3")}),
        "ramplr":     ("color3", {"valuel": [0.0, 0.0, 0.0], "valuer": [1.0, 1.0, 1.0]}),
        "ramptb":     ("color3", {"valuet": [0.0, 0.0, 0.0], "valueb": [1.0, 1.0, 1.0]}),
    }
    otype, inputs = cases[category]
    doc = mx.createDocument()
    ng = doc.addNodeGraph(f"NG_{material_name}")
    shader = doc.addNode("standard_surface", f"SR_{material_name}", "surfaceshader")
    smat = doc.addNode("surfacematerial", material_name, "material")
    smat.addInput("surfaceshader", "surfaceshader").setNodeName(f"SR_{material_name}")

    node = ng.addNode(category, f"n_{category}", otype)
    for iname, val in inputs.items():
        if isinstance(val, tuple) and val and val[0] == "node":
            _kind, subcat, subtype = val
            ng.addNode(subcat, f"n_{category}_{iname}", subtype)
            node.addInput(iname, subtype).setNodeName(f"n_{category}_{iname}")
        else:
            itype = m._infer_const_type(iname, val)
            m._set_typed_value(node.addInput(iname, itype), itype, val)

    feed = node.getName()
    if otype != "color3":
        conv = ng.addNode("convert", f"cv_{category}", "color3")
        conv.addInput("in", otype).setNodeName(feed)
        feed = conv.getName()
    out = ng.addOutput("out", "color3")
    out.setNodeName(feed)
    bc = shader.addInput("base_color", "color3")
    bc.setAttribute("nodegraph", ng.getName())
    bc.setAttribute("output", "out")
    return doc


@pytest.mark.parametrize("category", m.NODE_WHITELIST)
def test_every_whitelisted_node_generates(category):
    doc = _minimal_node_doc(category, f"Probe_{category}")
    cm, frag = m._gen_dry_run(doc, f"Probe_{category}")
    assert cm.pixel_source
    assert frag is not None, f"{category} produced no compute fragment"


# ─── Session file lifecycle ───────────────────────────────────────────

def test_one_file_per_material(tmp_path):
    store = m.SessionMaterialStore(str(tmp_path))
    r1 = m.synthesize({"template": "noise", "params": {}}, "MatA")
    r2 = m.synthesize({"template": "noise", "params": {}}, "MatB")
    p1 = store.write_document("MatA", r1.document_xml, r1.mapping)
    p2 = store.write_document("MatB", r2.document_xml, r2.mapping)
    assert p1 != p2
    assert (tmp_path / "MatA.mtlx").exists()
    assert (tmp_path / "MatB.mtlx").exists()


def test_sidecar_persists_mapping(tmp_path):
    store = m.SessionMaterialStore(str(tmp_path))
    r = m.synthesize({"template": "noise", "params": {}}, "MatC")
    store.write_document("MatC", r.document_xml, r.mapping)
    assert store.read_mapping("MatC") == r.mapping


def test_rollback_removes_session_file(tmp_path):
    store = m.SessionMaterialStore(str(tmp_path))
    r = m.synthesize({"template": "noise", "params": {}}, "MatD")
    store.write_document("MatD", r.document_xml, r.mapping)
    assert (tmp_path / "MatD.mtlx").exists()
    store.delete("MatD")
    assert not (tmp_path / "MatD.mtlx").exists()
    assert not (tmp_path / "MatD.json").exists()


# ─── Preset editable-input reflection + mtime cache ───────────────────

def test_preset_editable_inputs_graph():
    inputs = m.list_preset_inputs("marble_solid")
    assert inputs  # marble is a graph material — has editable uniforms
    # gen uniform names (writable keys), e.g. color_mix_bg
    assert any("color_mix" in i or "scale" in i for i in inputs)


def test_preset_editable_inputs_constant():
    # chrome is a constant-shader preset — editable keys are std_surface params
    inputs = m.list_preset_inputs("chrome")
    assert inputs


def test_preset_inputs_mtime_cache():
    m._PRESET_INPUT_CACHE.clear()
    first = m.list_preset_inputs("marble_solid")
    path = m.resolve_preset("marble_solid")
    assert path in m._PRESET_INPUT_CACHE
    second = m.list_preset_inputs("marble_solid")
    assert first == second  # served from cache, same result


# ─── Editable-input descriptors (design D5, finding #2) ───────────────

def test_descriptors_carry_type_default_and_declared_range():
    """A template's promoted inputs surface with their DECLARED type/default/
    range — octaves is an int accepting 1..8, scale reaches its 64 maximum, and
    colorA keeps its authored default — not a bare 0..1 float 0.0 (finding #2)."""
    r = m.synthesize({"template": "noise", "params": {}}, "N")
    d = r.descriptors
    assert d["octaves"]["type"] == "int"
    assert d["octaves"]["default"] == 4
    assert d["octaves"]["range"] == [1, 8]
    assert d["octaves"]["uniforms"] == r.mapping["octaves"]
    assert d["scale"]["type"] == "float"
    assert d["scale"]["range"] == [0.01, 64.0]
    assert d["colorA"]["type"] == "color3"
    assert d["colorA"]["default"] == [0.9, 0.9, 0.85]


def test_raw_graph_expose_range_from_material_ranges():
    """A raw-graph `expose: true` float whose name matches a known material
    range advertises that range; otherwise it is unbounded (finding #2)."""
    spec = {
        "model": "standard_surface",
        "graph": {
            "nodes": {
                "m": {"type": "multiply", "output": "color3",
                      "in1": [0.5, 0.5, 0.5],
                      "in2": {"expose": True, "value": 0.5, "name": "roughness"}},
            },
            "connections": [["m.out", "base_color"]],
        },
    }
    r = m.synthesize(spec, "RG")
    assert r.descriptors["roughness"]["type"] == "float"
    assert r.descriptors["roughness"]["range"] == [0.04, 1.0]
    assert r.descriptors["roughness"]["default"] == 0.5


def test_sidecar_persists_descriptors_roundtrip(tmp_path):
    store = m.SessionMaterialStore(str(tmp_path))
    r = m.synthesize({"template": "noise", "params": {}}, "D")
    store.write_document("D", r.document_xml, r.descriptors)
    assert store.read_mapping("D") == r.descriptors


def test_write_document_refuses_overwrite(tmp_path):
    """A second write to the same name is refused unless explicit (finding #5)."""
    store = m.SessionMaterialStore(str(tmp_path))
    store.write_document("X", "<materialx/>", None)
    with pytest.raises(MaterialSpecError):
        store.write_document("X", "<other/>", None)
    # explicit overwrite is allowed
    store.write_document("X", "<other/>", None, overwrite=True)
    assert (tmp_path / "X.mtlx").read_text() == "<other/>"


# ─── Preset identity descriptors (design D3, finding #3) ──────────────

def test_preset_identity_descriptors_identity_mapped():
    """A curated preset's descriptors map each writable key to itself so the
    scene graph surfaces exactly the advertised keys (finding #3)."""
    d = m.identity_descriptors_for_file(m.resolve_preset("marble_solid"))
    assert d
    for name, desc in d.items():
        assert desc["uniforms"] == [name]
        # Full type lattice (finding #2): a reflected vector uniform keeps its
        # vector kind instead of collapsing to a scalar float.
        assert desc["type"] in ("float", "color3", "int", "bool", "vector2", "vector3")
    # marble's add_xyz_in2 is a vector3 uniform — it must NOT collapse to float.
    assert d["add_xyz_in2"]["type"] == "vector3"
    # list_preset_inputs is exactly the descriptor keys
    assert set(m.list_preset_inputs("marble_solid")) == set(d)


# ─── Descriptor type lattice (finding #2) ─────────────────────────────

def test_descriptor_kind_covers_full_type_lattice():
    """bool/vector kinds must not collapse to a scalar float (finding #2)."""
    assert m._descriptor_kind("boolean") == "bool"
    assert m._descriptor_kind("bool") == "bool"
    assert m._descriptor_kind("vector2") == "vector2"
    assert m._descriptor_kind("vector3") == "vector3"
    assert m._descriptor_kind("vector4") == "vector3"
    assert m._descriptor_kind("color3") == "color3"
    assert m._descriptor_kind("color4") == "color3"
    assert m._descriptor_kind("integer") == "int"
    assert m._descriptor_kind("float") == "float"
    assert m._descriptor_kind(None) == "float"


def test_vector3_override_packs_as_sequence_not_zeroed():
    """A vector3 edit must reach the packer as a 3-sequence; the scalar a
    collapsed 'float' descriptor produced fills zeros instead (finding #2)."""
    import struct

    from skinny.materialx_runtime import UniformField, pack_uniform_block
    fld = UniformField(name="add_xyz_in2", type_name="vector3", offset=0, size=12)
    packed = pack_uniform_block([fld], {"add_xyz_in2": [0.25, 0.5, 0.75]})
    assert struct.unpack_from("<3f", packed, 0) == (0.25, 0.5, 0.75)
    # a scalar float override (the pre-fix descriptor shape) cannot fill 3 comps
    zeroed = pack_uniform_block([fld], {"add_xyz_in2": 0.5})
    assert struct.unpack_from("<3f", zeroed, 0) == (0.0, 0.0, 0.0)


# ─── Constant vs graph preset advertised keys (finding #3) ────────────

def test_constant_preset_advertises_flat_pack_keys():
    """A constant-shader preset (chrome) advertises the FlatMaterialParams keys
    pack_flat_material actually reads (diffuseColor/metallic/roughness) — NOT the
    std_surface input names (base_color/metalness), which are dual-authored but
    the ACTIVE path-tracing packer never reads, so those edits were no-ops (B)."""
    inputs = m.list_preset_inputs("chrome")
    assert inputs
    assert not any(i.startswith("SR_") for i in inputs)
    # the writable keys are the flat-pack (UsdPreviewSurface-style) names
    assert {"diffuseColor", "metallic", "roughness", "specular"} <= set(inputs)
    # the std_surface input names themselves are NOT advertised (they are no-ops)
    assert "base_color" not in inputs
    assert "metalness" not in inputs
    assert "specular_roughness" not in inputs
    # a folded input with no packer route is never advertised
    assert "base" not in inputs
    # constant preset → loader must NOT attach these as logical_inputs
    assert m.preset_is_graph(m.resolve_preset("chrome")) is False


def test_constant_preset_advertised_keys_change_flat_pack_bytes():
    """Discriminating gate (finding B): setting each key chrome advertises via
    parameter_overrides MUST change pack_flat_material's output — proving every
    advertised key has a real route into the active path-tracing pack, not the
    inert binding-19 std_surface pack."""
    from skinny.renderer import pack_flat_material
    from skinny.scene import Material

    base = pack_flat_material(Material(name="c", parameter_overrides={}))
    # sample values that differ from the flat-pack defaults so the bytes move
    probes = {
        "diffuseColor": (0.11, 0.22, 0.33),
        "roughness": 0.37,
        "metallic": 1.0,
        "specular": 0.13,
        "specular_color": (0.2, 0.4, 0.6),
    }
    advertised = set(m.list_preset_inputs("chrome"))
    unprobed = advertised - set(probes)
    assert not unprobed, f"advertised chrome keys with no byte-gate probe: {unprobed}"
    for key in advertised:
        packed = pack_flat_material(
            Material(name="c", parameter_overrides={key: probes[key]})
        )
        assert packed != base, f"advertised key {key!r} did not move the flat pack"


def test_graph_preset_advertises_reflection_uniforms():
    """A graph preset (marble) keeps its gen reflection uniforms as writable
    keys and is flagged as a graph preset (#3)."""
    inputs = set(m.list_preset_inputs("marble_solid"))
    assert m.preset_is_graph(m.resolve_preset("marble_solid")) is True
    assert "add_xyz_in2" in inputs  # a reflection uniform, not a canonical key
    assert not (inputs <= set(m.model_param_schema("standard_surface")))


# ─── Preview schema == authored inputs (design D4, finding #9) ────────

def test_preview_schema_advertised_equals_authorable():
    from skinny import usd_material_edit as ume
    advertised = set(m.model_param_schema("preview"))
    authored = {name for name, _t, _d in ume._PREVIEW_INPUTS}
    assert advertised == authored


def test_preview_rejects_non_authorable_input():
    # `normal` reflects on the full UsdPreviewSurface schema but the inline
    # author path drops it — discovery == authoring, so it is refused.
    with pytest.raises(MaterialSpecError):
        m.validate_spec({"model": "preview", "params": {"normal": [0.0, 0.0, 1.0]}})


# ─── Graph finite-check + connection suffix (design D4, finding #10) ──

def test_graph_non_finite_node_param_rejected():
    spec = {
        "model": "standard_surface",
        "graph": {
            "nodes": {"m": {"type": "multiply", "output": "color3",
                            "in1": [0.5, 0.5, 0.5], "in2": float("inf")}},
            "connections": [["m.out", "base_color"]],
        },
    }
    with pytest.raises(MaterialSpecError):
        m.validate_spec(spec)


def test_graph_bad_connection_output_suffix_rejected():
    spec = {
        "model": "standard_surface",
        "graph": {
            "nodes": {"pos": {"type": "position"}},
            "connections": [["pos.DOES_NOT_EXIST", "base_color"]],
        },
    }
    with pytest.raises(MaterialSpecError) as e:
        m.validate_spec(spec)
    assert "DOES_NOT_EXIST" in str(e.value)
