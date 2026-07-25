"""Hostless gate for the shader build-variant matrix (change
shader-variant-key-module).

Two layers:

* **Goldens** — the flag tuples / defines dicts / ``.spv`` filename tags
  transcribed verbatim from the **pre-refactor** tree (survey below names each
  source site). These are *permanent* fixtures, not migration scaffolding: they
  anchor the module to recorded reality instead of letting the agreement test
  be purely self-referential against the module's own table. A deliberate
  variant change edits them; drift fails here first.
* **Agreement** — for every family valid on both targets, the Metal define set
  minus ``METAL_ONLY_DEFINES`` must equal the Vulkan define set, modulo the
  module's explicit recorded-asymmetry table.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from skinny.sampling.neural_weights import (
    Encoding,
    NeuralBuildConfig,
    NeuralPrecision,
)
from skinny.shader_variants import (
    METAL_ONLY_DEFINES,
    RECORDED_ASYMMETRIES,
    Family,
    ShaderVariantKey,
    Target,
    slangc_flags,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "skinny"

# Stand-in include paths — the goldens pin flag *order*, not real directories.
S, M = "/S", "/M"
INC1 = (S,)          # vk_wavefront._slang_flags: shader_dir only
INC2 = (S, M)        # every other Vulkan site: shader_dir + mtlx/genslang

# A representative non-default neural config: cache_tag `L6B24H96_E1_fp16-compute`
# is the slug carried by tagged `.spv` files already on disk.
NEURAL = NeuralBuildConfig(encoding=Encoding.E1,
                           precision=NeuralPrecision.FP16_COMPUTE)
NEURAL_TOKENS = ("-D", "NF_WT=half", "-D", "NF_CT=half", "-D", "NF_ENCODING=1")


# ── 1.1 goldens: Vulkan flag tuples ──────────────────────────────────
# Transcribed from the pre-refactor tree. `-fvk-use-scalar-layout` sits in a
# different place at each of the three sites and is hashed positionally into
# the `build/spv_cache` blake2b key, so its position is load-bearing.

def _mega(entry, *tail):
    # vk_compute.ComputePipeline._compile_slang — base BEFORE the scalar-layout
    # flag, SKINNY_SPECTRAL appended AFTER it.
    return ("-target", "spirv", "-entry", entry, "-stage", "compute",
            "-I", S, "-I", M, "-D", "SKINNY_COMPUTE_PIPELINE=1",
            "-fvk-use-scalar-layout", *tail)


def _full(entry, *defines):
    # vk_wavefront._compile_full_spv — ALL defines before the scalar-layout flag.
    return ("-target", "spirv", "-entry", entry, "-stage", "compute",
            "-I", S, "-I", M, "-D", "SKINNY_COMPUTE_PIPELINE=1",
            "-D", "SKINNY_WAVEFRONT=1", *defines, "-fvk-use-scalar-layout")


GOLDEN_VULKAN_FLAGS: dict[str, tuple[tuple[str, ...], ShaderVariantKey, str, tuple[str, ...]]] = {
    "megakernel_rgb": (
        _mega("mainImage"),
        ShaderVariantKey(Target.VULKAN, Family.MEGAKERNEL), "mainImage", INC2),
    "megakernel_spectral": (
        _mega("mainImage", "-D", "SKINNY_SPECTRAL=1"),
        ShaderVariantKey(Target.VULKAN, Family.MEGAKERNEL, spectral=True),
        "mainImage", INC2),
    "preview": (
        # vk_compute.PreviewPipeline._compile_slang — same shape, no spectral axis.
        ("-target", "spirv", "-entry", "previewMain", "-stage", "compute",
         "-I", S, "-I", M, "-D", "SKINNY_COMPUTE_PIPELINE=1",
         "-fvk-use-scalar-layout"),
        ShaderVariantKey(Target.VULKAN, Family.PREVIEW), "previewMain", INC2),
    "wavefront_foundation": (
        # vk_wavefront._slang_flags — scalar-layout flag BEFORE the define,
        # and a single include path.
        ("-target", "spirv", "-entry", "computeMain", "-stage", "compute",
         "-I", S, "-fvk-use-scalar-layout", "-D", "SKINNY_WAVEFRONT=1"),
        ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT_FOUNDATION),
        "computeMain", INC1),
    "wavefront_rgb": (
        _full("wfPathShade"),
        ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT), "wfPathShade", INC2),
    "wavefront_spectral": (
        _full("wfPathShade", "-D", "SKINNY_SPECTRAL=1"),
        ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT, spectral=True),
        "wfPathShade", INC2),
    "wavefront_neural": (
        _full("wfNeuralProposal", *NEURAL_TOKENS),
        ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT, neural=NEURAL),
        "wfNeuralProposal", INC2),
    "wavefront_neural_spectral": (
        _full("wfPathShade", "-D", "SKINNY_SPECTRAL=1", *NEURAL_TOKENS),
        ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT, spectral=True, neural=NEURAL),
        "wfPathShade", INC2),
    "mlt_rgb": (
        _full("wfMltMutate", "-D", "SKINNY_MLT=1"),
        ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT, mlt=True),
        "wfMltMutate", INC2),
    "mlt_spectral": (
        _full("wfMltMutate", "-D", "SKINNY_SPECTRAL=1", "-D", "SKINNY_MLT=1"),
        ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT, mlt=True, spectral=True),
        "wfMltMutate", INC2),
}


# ── 1.1 goldens: Metal defines dicts ─────────────────────────────────
# Transcribed from metal_compute.py (3 literals) and metal_wavefront.py
# (session base + 5 per-pass assemblies).

_MW_BASE = {"SKINNY_COMPUTE_PIPELINE": "1", "SKINNY_METAL": "1",
            "SKINNY_WAVEFRONT": "1"}
_NF = {"NF_WT": "half", "NF_CT": "half", "NF_ENCODING": "1"}

GOLDEN_METAL_DEFINES: dict[str, tuple[dict[str, str], ShaderVariantKey]] = {
    # metal_compute.ComputePipeline._build
    "megakernel_rgb": (
        {"SKINNY_COMPUTE_PIPELINE": "1", "SKINNY_METAL": "1"},
        ShaderVariantKey(Target.METAL, Family.MEGAKERNEL)),
    "megakernel_spectral": (
        {"SKINNY_COMPUTE_PIPELINE": "1", "SKINNY_METAL": "1", "SKINNY_SPECTRAL": "1"},
        ShaderVariantKey(Target.METAL, Family.MEGAKERNEL, spectral=True)),
    # metal_compute.PreviewPipelineMetal._build
    "preview": (
        {"SKINNY_COMPUTE_PIPELINE": "1", "SKINNY_METAL": "1"},
        ShaderVariantKey(Target.METAL, Family.PREVIEW)),
    # metal_compute.DebugRasterMetal.__init__ — bare SKINNY_METAL
    "debug_raster": (
        {"SKINNY_METAL": "1"},
        ShaderVariantKey(Target.METAL, Family.DEBUG_RASTER)),
    # metal_wavefront._metal_slang_session base
    "wavefront_base": (
        dict(_MW_BASE), ShaderVariantKey(Target.METAL, Family.WAVEFRONT)),
    # MetalNeuralProposalPass — neural defines + SKINNY_METAL_NEURAL
    "wavefront_neural_pass": (
        {**_MW_BASE, **_NF, "SKINNY_METAL_NEURAL": "1"},
        ShaderVariantKey(Target.METAL, Family.WAVEFRONT, neural=NEURAL,
                         metal_neural=True)),
    # MetalWavefrontPathPass with neural + records + spectral armed
    "wavefront_path_all": (
        {**_MW_BASE, **_NF, "SKINNY_METAL_NEURAL": "1",
         "SKINNY_METAL_RECORDS": "1", "SKINNY_SPECTRAL": "1"},
        ShaderVariantKey(Target.METAL, Family.WAVEFRONT, neural=NEURAL,
                         metal_neural=True, metal_records=True, spectral=True)),
    # MetalWavefrontSppmPass — neural defines, no Metal gates (see the
    # recorded asymmetry: the Vulkan SPPM compile passes none)
    "wavefront_sppm_neural": (
        {**_MW_BASE, **_NF},
        ShaderVariantKey(Target.METAL, Family.WAVEFRONT, neural=NEURAL)),
    # MetalWavefrontBdptPass — plain define set, spectral only
    "wavefront_bdpt_spectral": (
        {**_MW_BASE, "SKINNY_SPECTRAL": "1"},
        ShaderVariantKey(Target.METAL, Family.WAVEFRONT, spectral=True)),
    # MetalWavefrontMltPass
    "wavefront_mlt_spectral": (
        {**_MW_BASE, "SKINNY_MLT": "1", "SKINNY_SPECTRAL": "1"},
        ShaderVariantKey(Target.METAL, Family.WAVEFRONT, mlt=True, spectral=True)),
}


# ── 1.1 goldens: `.spv` filename tags ────────────────────────────────
# f"{out_name}{tag}{spectral_suffix}.spv" in the pre-refactor
# vk_wavefront._compile_full_spv, with `tag` = the neural slug or "_mlt".

GOLDEN_CACHE_TOKENS: dict[str, tuple[str, ShaderVariantKey]] = {
    "default": ("", ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT)),
    "mlt": ("_mlt", ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT, mlt=True)),
    "spectral": ("_spectral",
                 ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT, spectral=True)),
    "mlt_spectral": ("_mlt_spectral",
                     ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT,
                                      mlt=True, spectral=True)),
    "neural": ("_L6B24H96_E1_fp16-compute",
               ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT, neural=NEURAL)),
    "neural_spectral": ("_L6B24H96_E1_fp16-compute_spectral",
                        ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT,
                                         spectral=True, neural=NEURAL)),
    # A default NeuralBuildConfig emits zero NF_* flags → no slug at all.
    "neural_default": ("", ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT,
                                            neural=NeuralBuildConfig())),
    "megakernel_spectral": (
        "_spectral", ShaderVariantKey(Target.VULKAN, Family.MEGAKERNEL, spectral=True)),
}


# ── Goldens ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(GOLDEN_VULKAN_FLAGS))
def test_vulkan_flag_tuples_match_pre_refactor_goldens(name):
    golden, key, entry, inc = GOLDEN_VULKAN_FLAGS[name]
    assert slangc_flags(key, entry=entry, include_paths=inc) == golden


@pytest.mark.parametrize("name", sorted(GOLDEN_METAL_DEFINES))
def test_metal_defines_match_pre_refactor_goldens(name):
    golden, key = GOLDEN_METAL_DEFINES[name]
    assert key.session_defines() == golden


@pytest.mark.parametrize("name", sorted(GOLDEN_CACHE_TOKENS))
def test_cache_tokens_match_pre_refactor_goldens(name):
    golden, key = GOLDEN_CACHE_TOKENS[name]
    assert key.cache_token() == golden


# ── Cross-backend agreement ──────────────────────────────────────────

_BOTH_TARGET_FAMILIES = (Family.MEGAKERNEL, Family.WAVEFRONT, Family.PREVIEW)


def _axes_for(family):
    """Every axis combination legal for `family` (mlt is wavefront-only)."""
    for spectral in (False, True):
        for neural in (None, NeuralBuildConfig(), NEURAL):
            yield {"spectral": spectral, "neural": neural, "mlt": False}
            if family is Family.WAVEFRONT:
                yield {"spectral": spectral, "neural": neural, "mlt": True}


def _parsed(segments):
    out = {}
    for seg in segments:
        it = iter(seg)
        for tok in it:
            assert tok == "-D", f"unexpected non-define token {tok!r}"
            name, _, value = next(it).partition("=")
            out[name] = value or "1"
    return out


@pytest.mark.parametrize("family", _BOTH_TARGET_FAMILIES)
def test_backends_agree_modulo_metal_only_defines(family):
    """Same axes on both targets ⇒ same defines, up to METAL_ONLY_DEFINES."""
    for axes in _axes_for(family):
        vk = ShaderVariantKey(Target.VULKAN, family, **axes)
        mtl = ShaderVariantKey(Target.METAL, family, **axes)
        metal = {k: v for k, v in mtl.session_defines().items()
                 if k not in METAL_ONLY_DEFINES}
        assert metal == _parsed(vk.slangc_defines()), f"{family} {axes}"
        # Same key, both Vulkan forms: segments and dict cannot diverge.
        assert _parsed(vk.slangc_defines()) == vk.session_defines()


def test_sppm_neural_asymmetry_matches_the_recorded_entry():
    """The one recorded asymmetry: Metal SPPM compiles with the active neural
    NF_* defines; Vulkan SPPM passes none. Vacuous at the default config."""
    entry, = [a for a in RECORDED_ASYMMETRIES if a.name == "sppm-neural-defines"]
    assert entry.defines == "NF_*"
    metal = ShaderVariantKey(Target.METAL, Family.WAVEFRONT, neural=NEURAL)
    vulkan = ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT)  # no neural
    diff = {k: v for k, v in metal.session_defines().items()
            if k not in METAL_ONLY_DEFINES
            and k not in _parsed(vulkan.slangc_defines())}
    assert diff == _NF, "SPPM divergence drifted from the recorded NF_* shape"
    # Vacuous at the default config — the whole reason this is tolerable today.
    metal_default = ShaderVariantKey(Target.METAL, Family.WAVEFRONT,
                                     neural=NeuralBuildConfig())
    assert ({k: v for k, v in metal_default.session_defines().items()
             if k not in METAL_ONLY_DEFINES}
            == _parsed(vulkan.slangc_defines()))


# ── Recorded byte-identity guarantees ────────────────────────────────

def test_no_vulkan_key_emits_a_metal_only_define():
    for family in (Family.MEGAKERNEL, Family.WAVEFRONT,
                   Family.WAVEFRONT_FOUNDATION, Family.PREVIEW):
        for axes in _axes_for(family):
            if family is Family.WAVEFRONT_FOUNDATION and axes["mlt"]:
                continue
            key = ShaderVariantKey(Target.VULKAN, family, **axes)
            assert not (set(_parsed(key.slangc_defines())) & METAL_ONLY_DEFINES)


def test_default_neural_config_emits_no_flags_and_no_slug():
    key = ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT,
                           neural=NeuralBuildConfig())
    plain = ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT)
    assert key.slangc_defines() == plain.slangc_defines()
    assert key.cache_token() == ""


def test_axes_off_emit_no_define():
    key = ShaderVariantKey(Target.VULKAN, Family.WAVEFRONT)
    names = set(_parsed(key.slangc_defines()))
    assert "SKINNY_MLT" not in names and "SKINNY_SPECTRAL" not in names


def test_megakernel_never_carries_the_mlt_axis():
    with pytest.raises(ValueError, match="wavefront-only"):
        ShaderVariantKey(Target.VULKAN, Family.MEGAKERNEL, mlt=True)


# ── Validation ───────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs, match", [
    ({"target": Target.METAL, "family": Family.WAVEFRONT_FOUNDATION},
     "invalid \\(target, family\\)"),
    ({"target": Target.VULKAN, "family": Family.DEBUG_RASTER},
     "invalid \\(target, family\\)"),
    ({"target": Target.VULKAN, "family": Family.WAVEFRONT, "metal_neural": True},
     "METAL-only"),
    ({"target": Target.VULKAN, "family": Family.WAVEFRONT, "metal_records": True},
     "METAL-only"),
    ({"target": Target.METAL, "family": Family.PREVIEW, "mlt": True},
     "wavefront-only"),
])
def test_invalid_combinations_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ShaderVariantKey(**kwargs)


def test_slangc_flags_refuses_a_metal_key():
    key = ShaderVariantKey(Target.METAL, Family.MEGAKERNEL)
    with pytest.raises(ValueError, match="Vulkan-only"):
        slangc_flags(key, entry="mainImage", include_paths=(S, M))


# ── No hand-assembled variant defines remain outside the module ──────

_VARIANT_DEFINES = re.compile(
    r"SKINNY_(COMPUTE_PIPELINE|WAVEFRONT|SPECTRAL|MLT|METAL"
    r"|METAL_NEURAL|METAL_RECORDS)\s*(=1[\"']|[\"']\s*:)")


@pytest.mark.parametrize("module", [
    "vk_compute.py", "vk_wavefront.py", "metal_compute.py", "metal_wavefront.py",
])
def test_consumers_do_not_hand_assemble_variant_defines(module):
    """Emission of a variant define into a flag tuple (`"SKINNY_X=1"`) or a
    defines dict (`"SKINNY_X": "1"`) outside shader_variants.py."""
    text = (SRC / module).read_text(encoding="utf-8")
    hits = [line for line in text.splitlines()
            if _VARIANT_DEFINES.search(line) and not line.lstrip().startswith("#")]
    assert hits == [], f"{module} still emits variant defines: {hits}"
