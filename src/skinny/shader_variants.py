"""Owner of the shader build-variant matrix (change shader-variant-key-module).

A compiled kernel's identity is a point in a small product space — compile
target (Vulkan SPIR-V via ``slangc`` vs native Metal via an in-process SlangPy
session), pipeline family, spectral, MLT, the Metal-only argument-table gates,
and the neural build config. Before this module every backend re-derived its
own define list at its own compile site, so cross-backend agreement was upheld
only by convention.

:class:`ShaderVariantKey` is that point. From a key alone the module derives:

* :meth:`ShaderVariantKey.slangc_defines` — the Vulkan ``-D`` tokens as **named
  ordered segments** (``base`` / ``spectral`` / ``neural`` / ``mlt``);
* :func:`slangc_flags` — a full ``slangc`` flag tuple, splicing those segments
  at each family's **recorded historical position** relative to
  ``-fvk-use-scalar-layout`` (three distinct orders exist across the sites, and
  the flag tuple is hashed positionally into ``build/spv_cache``, so a single
  contiguous block could not reproduce the existing cache keys);
* :meth:`ShaderVariantKey.session_defines` — the SlangPy defines dict;
* :meth:`ShaderVariantKey.cache_token` — the ``.spv`` filename tag.

Both derivational forms come from one shared internal define table, so they
cannot diverge. The neural ``NF_*`` defines and cache slug are **not**
re-derived here: :class:`~skinny.sampling.neural_weights.NeuralBuildConfig`
stays their owner and this key composes it.

Hostless — no GPU/window imports, so the goldens/agreement test runs in the
default ``pytest`` sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from skinny.sampling.neural_weights import NeuralBuildConfig


class Target(Enum):
    """Compile target — flat ``-D`` token tuples for ``slangc`` (SPIR-V) vs a
    defines dict for the in-process SlangPy Metal session."""

    VULKAN = "vulkan"
    METAL = "metal"


class Family(Enum):
    """Pipeline family — the base define set a compile belongs to."""

    MEGAKERNEL = "megakernel"                      # SKINNY_COMPUTE_PIPELINE
    WAVEFRONT = "wavefront"                        # + SKINNY_WAVEFRONT
    WAVEFRONT_FOUNDATION = "wavefront_foundation"  # SKINNY_WAVEFRONT only
    PREVIEW = "preview"                            # SKINNY_COMPUTE_PIPELINE
    DEBUG_RASTER = "debug_raster"                  # bare SKINNY_METAL


#: ``(target, family)`` pairs that exist. ``WAVEFRONT_FOUNDATION`` is
#: Vulkan-only (Metal compiles every wavefront kernel through the full
#: CP+METAL+WAVEFRONT session); ``DEBUG_RASTER`` is Metal-only (the Vulkan
#: debug viewport is a graphics rasteriser, not this compute path).
VALID_TARGET_FAMILIES: frozenset[tuple[Target, Family]] = frozenset({
    (Target.VULKAN, Family.MEGAKERNEL),
    (Target.METAL, Family.MEGAKERNEL),
    (Target.VULKAN, Family.WAVEFRONT),
    (Target.METAL, Family.WAVEFRONT),
    (Target.VULKAN, Family.WAVEFRONT_FOUNDATION),
    (Target.VULKAN, Family.PREVIEW),
    (Target.METAL, Family.PREVIEW),
    (Target.METAL, Family.DEBUG_RASTER),
})

#: Defines that exist only on the Metal target. Deliberately have no Vulkan
#: counterpart — the cross-backend agreement test subtracts exactly this set.
METAL_ONLY_DEFINES: frozenset[str] = frozenset({
    "SKINNY_METAL", "SKINNY_METAL_NEURAL", "SKINNY_METAL_RECORDS",
})

# Base define names per family (target-independent part).
_FAMILY_BASE: dict[Family, tuple[str, ...]] = {
    Family.MEGAKERNEL: ("SKINNY_COMPUTE_PIPELINE",),
    Family.WAVEFRONT: ("SKINNY_COMPUTE_PIPELINE", "SKINNY_WAVEFRONT"),
    Family.WAVEFRONT_FOUNDATION: ("SKINNY_WAVEFRONT",),
    Family.PREVIEW: ("SKINNY_COMPUTE_PIPELINE",),
    Family.DEBUG_RASTER: (),
}


@dataclass(frozen=True)
class RecordedAsymmetry:
    """A define divergence between the two backends that exists **today** and
    is kept as-is by this pure refactor, recorded here rather than left as
    call-site folklore. Resolving one is a separate change."""

    name: str
    defines: str          # which define family diverges
    metal_site: str       # source location that emits them
    vulkan_site: str      # source location that does not
    note: str


#: The one asymmetry the survey found. Vacuous at the default
#: :class:`NeuralBuildConfig`, which emits zero ``NF_*`` flags.
RECORDED_ASYMMETRIES: tuple[RecordedAsymmetry, ...] = (
    RecordedAsymmetry(
        name="sppm-neural-defines",
        defines="NF_*",
        metal_site="metal_wavefront.MetalWavefrontSppmPass",
        vulkan_site="vk_wavefront.WavefrontSppmPass",
        note="The Metal SPPM pass compiles with the active NeuralBuildConfig's "
             "NF_* defines; the Vulkan SPPM compile passes none. A non-default "
             "neural config therefore makes the two SPPM define sets diverge. "
             "Aligning them (either direction) is a follow-up change.",
    ),
)


class DefineSegments(NamedTuple):
    """Vulkan ``-D`` tokens grouped so each compile site can splice them at its
    own recorded position relative to ``-fvk-use-scalar-layout``."""

    base: tuple[str, ...]
    spectral: tuple[str, ...]
    neural: tuple[str, ...]
    mlt: tuple[str, ...]


@dataclass(frozen=True)
class ShaderVariantKey:
    """One point in the shader build-variant matrix."""

    target: Target
    family: Family
    spectral: bool = False
    mlt: bool = False
    metal_neural: bool = False    # SKINNY_METAL_NEURAL (METAL only)
    metal_records: bool = False   # SKINNY_METAL_RECORDS (METAL only)
    neural: NeuralBuildConfig | None = None

    def __post_init__(self) -> None:
        if (self.target, self.family) not in VALID_TARGET_FAMILIES:
            raise ValueError(
                f"invalid (target, family): {self.target.name} + "
                f"{self.family.name} — see VALID_TARGET_FAMILIES "
                "(WAVEFRONT_FOUNDATION is Vulkan-only, DEBUG_RASTER Metal-only)")
        if self.target is not Target.METAL and (self.metal_neural or self.metal_records):
            raise ValueError(
                f"metal_neural/metal_records are METAL-only, got target="
                f"{self.target.name}")
        if self.mlt and self.family is not Family.WAVEFRONT:
            raise ValueError(
                f"mlt=True is wavefront-only, got family={self.family.name}")
        # An axis a family cannot carry must be REFUSED, never accepted and then
        # dropped at emission: `cache_token()` would name a variant the compile
        # does not build. The neural NF_* defines reach only the wavefront
        # full-tree kernels; the preview / foundation / debug-raster compiles
        # take no spectral variant either.
        if self.neural is not None and self.family is not Family.WAVEFRONT:
            raise ValueError(
                f"a neural build config is wavefront-only, got "
                f"family={self.family.name}")
        if self.spectral and self.family not in (Family.MEGAKERNEL, Family.WAVEFRONT):
            raise ValueError(
                f"spectral=True is megakernel/wavefront-only, got "
                f"family={self.family.name}")

    # ── Shared define table ───────────────────────────────────────

    def _neural_tokens(self) -> tuple[str, ...]:
        """The composed ``NeuralBuildConfig`` ``-D`` tokens (never re-derived
        here). Default config → empty tuple."""
        return self.neural.slang_defines() if self.neural is not None else ()

    def _segments(self) -> DefineSegments:
        base = [f"{name}=1" for name in _FAMILY_BASE[self.family]]
        if self.target is Target.METAL:
            # The Metal-only gates ride in `base`: segments exist to order the
            # Vulkan splice, and no Vulkan key ever carries them.
            base.append("SKINNY_METAL=1")
            if self.metal_neural:
                base.append("SKINNY_METAL_NEURAL=1")
            if self.metal_records:
                base.append("SKINNY_METAL_RECORDS=1")
        spectral = ["SKINNY_SPECTRAL=1"] if self.spectral else []
        neural = [tok for tok in self._neural_tokens() if tok != "-D"]
        mlt = ["SKINNY_MLT=1"] if self.mlt else []
        return DefineSegments(tuple(base), tuple(spectral), tuple(neural), tuple(mlt))

    # ── Derivations ───────────────────────────────────────────────

    def slangc_defines(self) -> DefineSegments:
        """Vulkan ``-D`` token segments. Each element is a flat ``("-D", "K=V")``
        pair sequence, ready to splice into a ``slangc`` flag tuple."""
        segs = self._segments()
        return DefineSegments(*(
            tuple(tok for kv in seg for tok in ("-D", kv)) for seg in segs))

    def session_defines(self) -> dict[str, str]:
        """SlangPy ``opts.defines`` dict. Returns a fresh complete dict —
        ``SlangCompilerOptions.defines`` is copy-on-read, so consumers must
        assign it in ONE statement (see ``metal_compute.ComputePipeline._build``)."""
        out: dict[str, str] = {}
        for seg in self._segments():
            for kv in seg:
                name, _, value = kv.partition("=")
                out[name] = value or "1"
        return out

    def cache_token(self) -> str:
        """The ``.spv`` filename tag — byte-identical to the pre-refactor
        ``f"{neural_tag}{'_mlt' if mlt}{'_spectral' if spectral}"`` derivation.
        The default key yields ``""``."""
        nf = self._neural_tokens()
        neural = f"_{self.neural.cache_tag}" if nf else ""
        return f"{neural}{'_mlt' if self.mlt else ''}{'_spectral' if self.spectral else ''}"


def slangc_flags(key: ShaderVariantKey, *, entry: str,
                 include_paths: "tuple[Path | str, ...]",
                 stage: str = "compute") -> tuple[str, ...]:
    """Full ``slangc`` flag tuple (minus ``-o <out>``) for a Vulkan key.

    The define segments are spliced at the family's **recorded historical
    position** relative to ``-fvk-use-scalar-layout``. Three distinct orders
    exist and every one is hashed positionally into the ``build/spv_cache``
    blake2b key, so these positions are byte-identity load-bearing:

    * megakernel / preview (``vk_compute``) — base **before** the scalar-layout
      flag, ``SKINNY_SPECTRAL`` **after** it;
    * wavefront foundation (``vk_wavefront._slang_flags``) — scalar-layout flag
      **before** the base define;
    * wavefront full-tree (``vk_wavefront._compile_full_spv``) — every segment
      **before** the scalar-layout flag.
    """
    if key.target is not Target.VULKAN:
        raise ValueError(f"slangc_flags is Vulkan-only, got target={key.target.name}")
    segs = key.slangc_defines()
    head = ("-target", "spirv", "-entry", entry, "-stage", stage)
    for path in include_paths:
        head = (*head, "-I", str(path))
    scalar = "-fvk-use-scalar-layout"
    # EVERY segment is emitted in each arm — a segment a family cannot carry is
    # already empty (`__post_init__` refuses those keys), so no axis can be
    # silently dropped into a compile that `cache_token()` names differently.
    if key.family in (Family.MEGAKERNEL, Family.PREVIEW):
        flags = (*head, *segs.base, scalar, *segs.spectral, *segs.neural, *segs.mlt)
    elif key.family is Family.WAVEFRONT_FOUNDATION:
        flags = (*head, scalar, *segs.base, *segs.spectral, *segs.neural, *segs.mlt)
    else:
        flags = (*head, *segs.base, *segs.spectral, *segs.neural, *segs.mlt, scalar)
    # Belt-and-braces: every define the key declares reached the flag tuple.
    emitted = {kv for tok, kv in zip(flags, flags[1:]) if tok == "-D"}
    declared = {kv for seg in segs for kv in seg if kv != "-D"}
    if declared - emitted:  # pragma: no cover - unreachable while the arms agree
        raise AssertionError(
            f"slangc_flags dropped defines {sorted(declared - emitted)} for {key}")
    return flags
