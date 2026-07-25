"""Hostless drift gates for the derived byte-layout authority
(change reflection-owned-byte-layouts).

GPU-free. Every mirrored Slang struct's derived scalar/MSL stride is pinned to a
golden value here, so a field reorder/retype/insert in a ``.slang`` source — or a
derivation change in ``slang_layout`` — fails under plain ``pytest`` instead of
garbling GPU output. The goldens are the independent leg: they are NOT derived,
so a parser change and a shader change moving together still trips a
human-visible failure (the parity-manifest discipline).
"""

from __future__ import annotations

import pytest

from skinny import slang_layout as sl
from skinny.slang_layout import SlangLayoutError

# (struct, kwargs, scalar stride, MSL stride) — axis labelled per entry below.
_SCALAR_GOLDENS = [
    # (struct, kwargs, golden) — RGB/spectral axis
    ("WavefrontPathState", {}, 68),                      # RGB
    ("WavefrontPathState", {"spectral": True}, 108),     # spectral
    ("RecVertex", {}, 76),                               # layout-invariant struct
    ("VisiblePoint", {}, 180),                           # RGB
    ("VisiblePoint", {"spectral": True}, 192),           # spectral
    ("SppmAccum", {}, 16),                               # RGB
    ("SppmAccum", {"spectral": True}, 20),               # spectral
    ("BDPTVertex", {}, 120),                             # RGB
    ("BDPTVertex", {"spectral": True}, 128),             # spectral
    ("WfBdptAux", {}, 92),                               # RGB
    ("WfBdptAux", {"spectral": True}, 136),              # spectral
    ("MltPrimarySample", {}, 16),
    ("MltChainMeta", {}, 32),
    ("MltRecord", {}, 16),
    ("FlatMaterialParams", {}, 256),
    ("StdSurfaceParams", {}, 256),
]

# (struct, kwargs, golden MSL stride) — scalar/MSL axis.
_MSL_GOLDENS = [
    ("WavefrontPathState", {}, 96),
    ("WavefrontPathState", {"spectral": True}, 128),
    ("RecVertex", {}, 112),
    ("VisiblePoint", {}, 240),
    ("SppmAccum", {}, 16),                # all-uint struct: layout-invariant
    ("SppmAccum", {"spectral": True}, 20),
    ("MltPrimarySample", {}, 16),
    ("MltChainMeta", {}, 32),
    ("MltRecord", {}, 16),
]


@pytest.mark.parametrize("struct,kwargs,golden", _SCALAR_GOLDENS)
def test_scalar_stride_matches_golden(struct, kwargs, golden):
    assert sl.scalar_stride(struct, **kwargs) == golden


@pytest.mark.parametrize("struct,kwargs,golden", _MSL_GOLDENS)
def test_msl_stride_matches_golden(struct, kwargs, golden):
    assert sl.msl_stride(struct, metal=False, **kwargs) == golden


@pytest.mark.parametrize("struct,kwargs,_golden", _SCALAR_GOLDENS)
def test_scalar_layout_has_no_gaps_or_overlaps(struct, kwargs, _golden):
    """Scalar layout packs tightly: every field starts where the previous ended."""
    layout = sl.scalar_layout(struct, **kwargs)
    off = 0
    for name, size in layout.entries:
        assert layout.offset(name) == off, (struct, name)
        off += size
    assert off == layout.stride


@pytest.mark.parametrize("struct,kwargs,_golden", _MSL_GOLDENS)
def test_msl_layout_fields_are_ordered_and_in_bounds(struct, kwargs, _golden):
    """MSL layout may pad between fields, but never overlaps or overruns."""
    layout = sl.msl_layout(struct, metal=False, **kwargs)
    end = 0
    for name, size in layout.entries:
        off = layout.offset(name)
        assert off >= end, (struct, name, off, end)
        end = off + size
    assert end <= layout.stride


def test_msl_is_never_smaller_than_scalar():
    for struct, kwargs, _g in _MSL_GOLDENS:
        assert (sl.msl_stride(struct, metal=False, **kwargs)
                >= sl.scalar_stride(struct, **kwargs)), struct


# ── FrameConstants host scalar blob (design D1 blob rule) ────────────


def test_fc_blob_lengths_are_pinned():
    assert sl.fc_blob_size() == 568
    assert sl.fc_blob_size(mlt=True) == 600
    # The MLT tail is exactly 8 uint/float fields.
    assert sl.fc_blob_size(mlt=True) - sl.fc_blob_size() == 32


def test_fc_blob_puts_tile_origin_y_last_and_mlt_sigma_at_564():
    base = sl.fc_scalar_blob()
    mlt = sl.fc_scalar_blob(mlt=True)
    assert base[-1] == ("tileOriginY", 4)
    assert mlt[-1] == ("tileOriginY", 4)
    assert sl.fc_tile_origin_y_offset() == 564
    # `mltSigma` must land where the Vulkan MLT SPIR-V expects it: immediately
    # after sppmGroupPmfEnv, i.e. the offset the base blob's filler word occupies.
    off = 0
    for name, size in mlt:
        if name == "mltSigma":
            break
        off += size
    else:  # pragma: no cover - defensive
        pytest.fail("mltSigma missing from the MLT blob")
    assert off == 564
    # The MLT tail sits BEFORE the trailing tileOriginY word.
    names = [n for n, _ in mlt]
    assert names.index("mltSeed") < names.index("tileOriginY")
    # Base blob is the MLT blob minus the tail, in order.
    assert [n for n, _ in base if n != "tileOriginY"] == [
        n for n, _ in mlt if n != "tileOriginY" and not n.startswith("mlt")]


def test_fc_blob_field_order_golden():
    """Golden field-order lock: the derived (name, size) sequence. A shader-side
    reorder fails here and forces the human to update this golden AND the
    ``_pack_uniforms`` body together."""
    assert sl.fc_scalar_blob() == (
        ("camera.viewInverse", 64), ("camera.projInverse", 64),
        ("camera.view", 64), ("camera.proj", 64),
        ("camera.position", 12), ("camera.fov", 4),
        ("frameIndex", 4), ("accumFrame", 4), ("time", 4), ("width", 4),
        ("height", 4), ("numDistantLights", 4), ("useMesh", 4),
        ("tattooDensity", 4), ("envIntensity", 4), ("furnaceMode", 4),
        ("mmPerUnit", 4), ("detailFlags", 4), ("normalMapStrength", 4),
        ("displacementScaleMM", 4), ("numInstances", 4), ("numSphereLights", 4),
        ("numEmissiveTriangles", 4), ("integratorType", 4),
        ("numGizmoSegments", 4), ("numLensElements", 4), ("filmDistance", 4),
        ("rearZ", 4), ("rearAperture", 4), ("frontZ", 4), ("filmHalfH", 4),
        ("emissiveTotalPower", 4), ("numPupilBounds", 4),
        ("filmDiagRadiusW", 4), ("focusOverlay", 4), ("focusPlaneOrigin", 12),
        ("focusPlaneNormal", 12), ("zoomMin", 8), ("zoomMax", 8),
        ("lensVignetteDebug", 4), ("pickPixel", 8), ("pickArmed", 4),
        ("exposure", 4), ("tonemapMode", 4), ("proposalMask", 4),
        ("reuseMode", 4), ("proposalAlpha", 16), ("flatLobeSamplers", 4),
        ("sceneBoundsMin", 12), ("sceneBoundsExtent", 12),
        ("neuralNetworkVersion", 4), ("recordMode", 4), ("cameraMirror", 4),
        ("sppmInitialRadius", 4), ("sppmCellSize", 4), ("sppmGridRes", 12),
        ("sppmPhotonsEmitted", 4), ("sppmGlossyContinueRoughness", 4),
        ("filmMaxComponent", 4), ("sppmGroupPmfE", 4), ("sppmGroupPmfS", 4),
        ("sppmGroupPmfD", 4), ("sppmGroupPmfEnv", 4), ("tileOriginY", 4),
    )


def test_fc_mlt_tail_golden():
    mlt = sl.fc_scalar_blob(mlt=True)
    assert mlt[-9:] == (
        ("mltSigma", 4), ("mltLargeStepProb", 4), ("mltB", 4),
        ("mltMppActual", 4), ("mltNumChains", 4), ("mltChainBase", 4),
        ("mltMaxDepth", 4), ("mltSeed", 4), ("tileOriginY", 4),
    )


def test_fc_without_metal_define_has_no_tile_origin_y():
    """The Vulkan-compiled struct genuinely lacks the field — that is why the
    trailing host word is filler there."""
    names = [n for _t, n in sl.struct_fields("FrameConstants", metal=False)]
    assert "tileOriginY" not in names
    names_metal = [n for _t, n in sl.struct_fields("FrameConstants", metal=True)]
    assert "tileOriginY" in names_metal


# ── Variant resolution ───────────────────────────────────────────────


def test_spectral_variant_retypes_spectrum_and_appends_wavelengths():
    rgb = sl.struct_fields("WavefrontPathState")
    spec = sl.struct_fields("WavefrontPathState", spectral=True)
    assert ("float3", "throughput") in rgb
    assert ("float4", "throughput") in spec
    assert ("SampledWavelengths", "sw") in spec
    assert ("SampledWavelengths", "sw") not in rgb


def test_mlt_tail_only_present_under_mlt_define():
    base = [n for _t, n in sl.struct_fields("FrameConstants", metal=True)]
    mlt = [n for _t, n in sl.struct_fields("FrameConstants", metal=True, mlt=True)]
    assert "mltSigma" not in base
    assert "mltSigma" in mlt


def test_nested_struct_fields_are_flattened_with_dotted_names():
    layout = sl.scalar_layout("FrameConstants", metal=True)
    # Slang's reflection records the parent as well as the flattened children.
    assert layout.offset("camera") == 0
    assert layout.offset("camera.viewInverse") == 0
    assert layout.offset("camera.fov") == 268
    assert [n for n, _ in layout.entries][:2] == [
        "camera.viewInverse", "camera.projInverse"]


# ── Fail loudly, never guess ─────────────────────────────────────────


def test_unknown_field_type_raises():
    src = "struct Weird {\n float3 a;\n quaternion q;\n};"
    with pytest.raises(SlangLayoutError, match="unknown field type"):
        sl.parse_struct_fields(src, "Weird")


def test_unresolvable_gate_raises():
    src = ("struct Weird {\n float a;\n#if defined(SKINNY_FUTURE)\n"
           " float b;\n#endif\n};")
    with pytest.raises(SlangLayoutError, match="unresolvable preprocessor gate"):
        sl.parse_struct_fields(src, "Weird")


def test_unsupported_directive_raises():
    src = ("struct Weird {\n float a;\n#ifdef SKINNY_MLT\n float b;\n#endif\n};")
    with pytest.raises(SlangLayoutError, match="unsupported preprocessor directive"):
        sl.parse_struct_fields(src, "Weird")


def test_unrecognised_declaration_raises():
    src = "struct Weird { float a, b; };"
    with pytest.raises(SlangLayoutError, match="unrecognised declaration"):
        sl.parse_struct_fields(src, "Weird")


def test_unbalanced_gate_raises():
    src = "struct Weird {\n#if defined(SKINNY_MLT)\n float a;\n};"
    with pytest.raises(SlangLayoutError, match="unbalanced"):
        sl.parse_struct_fields(src, "Weird")


def test_missing_struct_raises():
    with pytest.raises(SlangLayoutError, match="not found"):
        sl.parse_struct_fields("struct Other { float a; };", "Weird")


def test_unregistered_struct_raises():
    with pytest.raises(SlangLayoutError, match="not registered"):
        sl.struct_fields("NotAStruct")
