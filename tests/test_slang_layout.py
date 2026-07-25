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


# Golden DECLARED FIELDS per owned struct — (slang type, field name) in
# declaration order, hand-transcribed from the `.slang` sources. Types are
# pinned, not just names/sizes: a stride pin cannot see a same-size field swap,
# and neither a stride nor an offset nor Slang's reflected field *size* can see
# a same-width RETYPE (`float`→`uint`, `float3`→`uint3`) that changes how the
# GPU interprets bytes the host packs. The `wavefront_layout` / `test_sppm_state`
# locks now compare derived lists against derived lists, so this is the
# independent leg. Update it consciously, together with the shader.
_DECLARED_FIELD_GOLDENS = {
    ("FlatMaterialParams", ()): [('float4', '_diffuseColorRoughness'), ('float', 'metallic'), ('float', 'specular'), ('float', 'opacity'), ('uint', 'diffuseTextureIdx'), ('uint', 'roughnessTextureIdx'), ('uint', 'metallicTextureIdx'), ('uint', 'normalTextureIdx'), ('uint', 'emissiveTextureIdx'), ('float4', '_emissiveColorIor'), ('float', 'coat'), ('float', 'coatRoughness'), ('float', 'coatIOR'), ('uint', 'opacityTextureIdx'), ('float4', '_coatColorOpacityThreshold'), ('float4', '_normalScaleChannelMask'), ('float4', '_normalBiasPad'), ('float4', '_transmissionColorDiffuseRough'), ('float4', '_specularColorPad'), ('float4', '_mediumSigmaA_g'), ('float4', '_mediumSigmaS_kind'), ('float4', '_worldToUvw0'), ('float4', '_worldToUvw1'), ('float4', '_worldToUvw2'), ('float4', '_cloudDensityWispinessFrequency')],
    ("StdSurfaceParams", ()): [('float3', 'base_color'), ('float', 'base'), ('float', 'diffuse_roughness'), ('float', 'metalness'), ('float', 'specular'), ('float', 'specular_roughness'), ('float3', 'specular_color'), ('float', 'specular_IOR'), ('float', 'specular_anisotropy'), ('float', 'specular_rotation'), ('float', 'transmission'), ('float', 'transmission_depth'), ('float3', 'transmission_color'), ('float', 'transmission_scatter_anisotropy'), ('float3', 'transmission_scatter'), ('float', 'transmission_dispersion'), ('float', 'transmission_extra_roughness'), ('float', 'subsurface'), ('float', 'subsurface_scale'), ('float', 'subsurface_anisotropy'), ('float3', 'subsurface_color'), ('float', '_pad0'), ('float3', 'subsurface_radius'), ('float', 'sheen'), ('float3', 'sheen_color'), ('float', 'sheen_roughness'), ('float', 'coat'), ('float', 'coat_roughness'), ('float', 'coat_anisotropy'), ('float', 'coat_rotation'), ('float', 'coat_IOR'), ('float', 'coat_affect_color'), ('float', 'coat_affect_roughness'), ('float', '_pad1'), ('float3', 'coat_color'), ('float', 'thin_film_thickness'), ('float', 'thin_film_IOR'), ('float', 'emission'), ('float3', 'emission_color'), ('float', '_pad2'), ('float3', 'opacity'), ('uint', 'thin_walled'), ('float', '_pad3'), ('float', '_pad4')],
    ("WavefrontPathState", ()): [('float3', 'rayOrigin'), ('float3', 'rayDir'), ('float3', 'throughput'), ('float3', 'radiance'), ('uint', 'pixelIndex'), ('uint', 'rngState'), ('uint', 'depth'), ('uint', 'flags'), ('float', 'bsdfPdf')],
    ("WavefrontPathState", ('spectral',)): [('float3', 'rayOrigin'), ('float3', 'rayDir'), ('float4', 'throughput'), ('float4', 'radiance'), ('uint', 'pixelIndex'), ('uint', 'rngState'), ('uint', 'depth'), ('uint', 'flags'), ('float', 'bsdfPdf'), ('SampledWavelengths', 'sw')],
    ("RecVertex", ()): [('float3', 'pos'), ('float3', 'normal'), ('float3', 'wo'), ('float3', 'wiLocal'), ('float3', 'L_k'), ('float3', 'beta_in'), ('uint', 'depth')],
    ("VisiblePoint", ()): [('float3', 'pos'), ('float3', 'ns'), ('float3', 'wo'), ('float3', 'beta'), ('float3', 'ld'), ('float3', 'albedo'), ('float3', 'F0'), ('float3', 'coatColor'), ('float', 'roughness'), ('float', 'metallic'), ('float', 'specular'), ('float', 'ior'), ('float', 'opacity'), ('float', 'coat'), ('float', 'coatRoughness'), ('float', 'coatIOR'), ('float3', 'transmissionColor'), ('float3', 'specularColor'), ('float', 'diffuseRoughness'), ('float3', 'tau'), ('uint', 'flags'), ('float', 'radius'), ('float', 'n')],
    ("VisiblePoint", ('spectral',)): [('float3', 'pos'), ('float3', 'ns'), ('float3', 'wo'), ('float4', 'beta'), ('float4', 'ld'), ('float3', 'albedo'), ('float3', 'F0'), ('float3', 'coatColor'), ('float', 'roughness'), ('float', 'metallic'), ('float', 'specular'), ('float', 'ior'), ('float', 'opacity'), ('float', 'coat'), ('float', 'coatRoughness'), ('float', 'coatIOR'), ('float3', 'transmissionColor'), ('float3', 'specularColor'), ('float', 'diffuseRoughness'), ('uint', 'conductorMetalId'), ('float3', 'tau'), ('uint', 'flags'), ('float', 'radius'), ('float', 'n')],
    ("SppmAccum", ()): [('uint', 'phiR'), ('uint', 'phiG'), ('uint', 'phiB'), ('uint', 'm')],
    ("SppmAccum", ('spectral',)): [('uint', 'phiR'), ('uint', 'phiG'), ('uint', 'phiB'), ('uint', 'phiW'), ('uint', 'm')],
    ("BDPTVertex", ()): [('uint', 'kind'), ('float3', 'position'), ('float3', 'N'), ('float3', 'throughput'), ('float3', 'emission'), ('float', 'pdfFwd'), ('float', 'pdfRev'), ('bool', 'isDelta'), ('bool', 'onLight'), ('uint', 'matId'), ('float2', 'uv'), ('float3', 'posObject'), ('float3', 'geoN'), ('float3', 'tangent'), ('bool', 'hasTangent')],
    ("BDPTVertex", ('spectral',)): [('uint', 'kind'), ('float3', 'position'), ('float3', 'N'), ('float4', 'throughput'), ('float4', 'emission'), ('float', 'pdfFwd'), ('float', 'pdfRev'), ('bool', 'isDelta'), ('bool', 'onLight'), ('uint', 'matId'), ('float2', 'uv'), ('float3', 'posObject'), ('float3', 'geoN'), ('float3', 'tangent'), ('bool', 'hasTangent')],
    ("WfBdptAux", ()): [('int', 'eyeLen'), ('int', 'lightLen'), ('uint', 'rngState'), ('float', 'lensWeight'), ('uint', 'pixel'), ('float3', 'escaped'), ('float3', 'radiance'), ('float3', 'ewRayO'), ('float3', 'ewRayD'), ('float3', 'ewThroughput'), ('float', 'ewPdfFwdOmega'), ('float', 'ewMisBsdfPdf'), ('uint', 'ewFlags')],
    ("WfBdptAux", ('spectral',)): [('int', 'eyeLen'), ('int', 'lightLen'), ('uint', 'rngState'), ('float', 'lensWeight'), ('uint', 'pixel'), ('float4', 'escaped'), ('float4', 'radiance'), ('float3', 'ewRayO'), ('float3', 'ewRayD'), ('float4', 'ewThroughput'), ('float', 'ewPdfFwdOmega'), ('float', 'ewMisBsdfPdf'), ('uint', 'ewFlags'), ('SampledWavelengths', 'sw')],
    ("MltPrimarySample", ()): [('float', 'value'), ('float', 'valueBackup'), ('uint', 'lastMod'), ('uint', 'modBackup')],
    ("MltChainMeta", ()): [('uint', 'rngState'), ('uint', 'currentIteration'), ('uint', 'lastLargeStepIteration'), ('uint', 'seedIndex'), ('float', 'cCurrent'), ('uint', 'nRecords'), ('uint', 'pad0'), ('uint', 'pad1')],
    ("MltRecord", ()): [('uint', 'pixel'), ('float', 'r'), ('float', 'g'), ('float', 'b')],
    ("Camera", ()): [('float4x4', 'viewInverse'), ('float4x4', 'projInverse'), ('float4x4', 'view'), ('float4x4', 'proj'), ('float3', 'position'), ('float', 'fov')],
    ("SampledWavelengths", ()): [('float4', 'lambda'), ('float4', 'pdf')],
}


@pytest.mark.parametrize("key", list(_DECLARED_FIELD_GOLDENS),
                         ids=lambda k: f"{k[0]}{'_spectral' if k[1] else ''}")
def test_declared_fields_match_golden(key):
    """Catches a same-size field swap AND a same-width retype."""
    struct, flags = key
    fields = sl.struct_fields(struct, spectral="spectral" in flags)
    assert fields == _DECLARED_FIELD_GOLDENS[key]


# FrameConstants declared fields (SKINNY_METAL + SKINNY_MLT), typed, in FULL —
# the partial head/tail lock left the middle of the struct open to the same
# same-width retype hole (`exposure` float→uint) the typed goldens exist to
# close. The base variant is this list minus the `mlt*` tail.
_FC_DECLARED_MLT = [
    ("Camera", "camera"), ("uint", "frameIndex"), ("uint", "accumFrame"),
    ("float", "time"), ("uint", "width"), ("uint", "height"),
    ("uint", "numDistantLights"), ("uint", "useMesh"),
    ("float", "tattooDensity"), ("float", "envIntensity"),
    ("uint", "furnaceMode"), ("float", "mmPerUnit"), ("uint", "detailFlags"),
    ("float", "normalMapStrength"), ("float", "displacementScaleMM"),
    ("uint", "numInstances"), ("uint", "numSphereLights"),
    ("uint", "numEmissiveTriangles"), ("uint", "integratorType"),
    ("uint", "numGizmoSegments"), ("uint", "numLensElements"),
    ("float", "filmDistance"), ("float", "rearZ"), ("float", "rearAperture"),
    ("float", "frontZ"), ("float", "filmHalfH"),
    ("float", "emissiveTotalPower"), ("uint", "numPupilBounds"),
    ("float", "filmDiagRadiusW"), ("uint", "focusOverlay"),
    ("float3", "focusPlaneOrigin"), ("float3", "focusPlaneNormal"),
    ("float2", "zoomMin"), ("float2", "zoomMax"),
    ("uint", "lensVignetteDebug"), ("uint2", "pickPixel"),
    ("uint", "pickArmed"), ("float", "exposure"), ("uint", "tonemapMode"),
    ("uint", "proposalMask"), ("uint", "reuseMode"),
    ("float4", "proposalAlpha"), ("uint", "flatLobeSamplers"),
    ("float3", "sceneBoundsMin"), ("float3", "sceneBoundsExtent"),
    ("uint", "neuralNetworkVersion"), ("uint", "recordMode"),
    ("uint", "cameraMirror"), ("float", "sppmInitialRadius"),
    ("float", "sppmCellSize"), ("uint3", "sppmGridRes"),
    ("uint", "sppmPhotonsEmitted"), ("float", "sppmGlossyContinueRoughness"),
    ("float", "filmMaxComponent"), ("float", "sppmGroupPmfE"),
    ("float", "sppmGroupPmfS"), ("float", "sppmGroupPmfD"),
    ("float", "sppmGroupPmfEnv"), ("uint", "tileOriginY"),
    ("float", "mltSigma"), ("float", "mltLargeStepProb"), ("float", "mltB"),
    ("float", "mltMppActual"), ("uint", "mltNumChains"),
    ("uint", "mltChainBase"), ("uint", "mltMaxDepth"), ("uint", "mltSeed"),
]


def test_frame_constants_declared_fields_are_typed_locked():
    """Full typed lock — every FrameConstants field, not just head and tail."""
    assert sl.struct_fields("FrameConstants", metal=True, mlt=True) == \
        _FC_DECLARED_MLT
    assert sl.struct_fields("FrameConstants", metal=True) == [
        f for f in _FC_DECLARED_MLT if not f[1].startswith("mlt")]
    # The embedded Camera is locked by _DECLARED_FIELD_GOLDENS above.
    assert ("Camera", "camera") in _FC_DECLARED_MLT


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


def test_attributed_field_declaration_raises():
    """An attributed field is neither dropped nor silently un-attributed
    (codex pre-merge findings). Skipping it would lose a field and shift every
    offset after it; honouring declaration order while erasing e.g.
    `[[vk::offset(16)]]` would emit a confidently WRONG offset. So: raise."""
    src = "struct Weird {\n float3 a;\n [[vk::offset(16)]] float b;\n};"
    with pytest.raises(SlangLayoutError, match="attributed field declaration"):
        sl.parse_struct_fields(src, "Weird")


def test_attribute_on_its_own_line_still_attributes_the_next_field():
    """An attribute may sit on its own line above the declaration it applies to
    — that split form must raise exactly like the same-line form, or a
    `[[vk::offset(…)]]` above its field is silently lost (codex round-3)."""
    split = "struct Weird {\n float3 a;\n [[vk::offset(16)]]\n float b;\n};"
    with pytest.raises(SlangLayoutError, match="attributed field declaration"):
        sl.parse_struct_fields(split, "Weird")


def test_initialized_field_is_not_mistaken_for_a_method():
    """`float b = float(0);` is a FIELD. A bare "line contains `(` ⇒ method"
    test drops it silently and shifts every offset after it (codex round-4)."""
    src = "struct Weird {\n float3 a;\n float b = float(0);\n float c;\n};"
    assert sl.parse_struct_fields(src, "Weird") == [
        ("float3", "a"), ("float", "b"), ("float", "c")]
    # …and an attributed initialized field still raises.
    attributed = "struct Weird {\n [[vk::offset(4)]] float b = float(0);\n};"
    with pytest.raises(SlangLayoutError, match="attributed field declaration"):
        sl.parse_struct_fields(attributed, "Weird")


def test_methods_are_still_skipped():
    """Real methods (definition or declaration) carry no layout."""
    defn = "struct Weird {\n float a;\n float area(float x) { return x; }\n};"
    assert sl.parse_struct_fields(defn, "Weird") == [("float", "a")]
    decl = "struct Weird {\n float a;\n void f(int x);\n};"
    assert sl.parse_struct_fields(decl, "Weird") == [("float", "a")]


@pytest.mark.parametrize("body", [
    # brace on the following line
    " float area(float x)\n {\n     float local = float(0);\n"
    "     return max(local, x);\n }\n",
    # brace on the signature line
    " float area(float x) {\n     float local = 1.0;\n     return local;\n }\n",
    # nested block inside the method body
    " float area(float x) {\n     if (x > 0) { float inner = 2.0;"
    " return inner; }\n     return 0;\n }\n",
    # multiline property accessor
    " property float scaled\n {\n     get { float t = 2.0; return a * t; }\n }\n",
], ids=["brace-next-line", "brace-same-line", "nested-block", "property"])
def test_locals_inside_multiline_bodies_are_not_fields(body):
    """A local declaration inside a method or property body is NOT a struct
    field. Classifying lines independently made `float local = float(0);` a
    phantom field that shifted every offset after it (codex round-5)."""
    src = "struct Weird {\n float a;\n" + body + " float b;\n};"
    assert sl.parse_struct_fields(src, "Weird") == [("float", "a"),
                                                    ("float", "b")]


def test_multi_declarator_raises_in_both_forms():
    """`float x = float(0), y = float(0);` must not yield only `x` — dropping
    `y` shifts every offset after it (codex round-6). The bare form already
    failed to match; both are now loud."""
    init = "struct Weird {\n float x = float(0), y = float(0);\n};"
    with pytest.raises(SlangLayoutError, match="multiple declarators"):
        sl.parse_struct_fields(init, "Weird")
    bare = "struct Weird {\n float x, y;\n};"
    with pytest.raises(SlangLayoutError, match="unrecognised declaration"):
        sl.parse_struct_fields(bare, "Weird")
    # Mixed form: parens would otherwise send it to the method skip.
    mixed = "struct Weird {\n float x, y = float(0);\n};"
    with pytest.raises(SlangLayoutError, match="multiple declarators"):
        sl.parse_struct_fields(mixed, "Weird")


def test_commas_inside_an_initializer_call_are_not_multi_declarators():
    """`float3 v = float3(0, 1, 2);` is ONE declarator, and a method's parameter
    commas must not be mistaken for declarators either."""
    src = "struct Weird {\n float3 v = float3(0, 1, 2);\n float b;\n};"
    assert sl.parse_struct_fields(src, "Weird") == [("float3", "v"),
                                                    ("float", "b")]
    method = ("struct Weird {\n float a;\n void f(int x, int y);\n"
              " float g(int x, int y) { return 0; }\n float b;\n};")
    assert sl.parse_struct_fields(method, "Weird") == [("float", "a"),
                                                       ("float", "b")]


def test_array_member_raises_rather_than_guessing():
    """Array members are not supported; they must fail loudly, not silently."""
    with pytest.raises(SlangLayoutError, match="unrecognised declaration"):
        sl.parse_struct_fields("struct Weird {\n float v[4];\n};", "Weird")


def test_attribute_above_a_method_is_skipped():
    """`[mutating]` above a METHOD is not a field attribute — methods are not
    layout, so the struct still parses."""
    src = ("struct Weird {\n float a;\n [mutating]\n"
           " void bump(float d) { a += d; }\n};")
    assert sl.parse_struct_fields(src, "Weird") == [("float", "a")]


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
