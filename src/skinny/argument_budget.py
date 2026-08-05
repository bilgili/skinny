"""Owner of the Metal argument-table limits and the per-variant global census.

Metal gives each compute kernel three fixed-size argument tables: **31** buffer
entries, **128** texture entries, **16** sampler entries. The buffer limit is the
Metal Shading Language ABI; no GPU family raises it. slang-rhi assigns a
program's globals **program-wide, in declaration order**, so a declared-but-dead
global still costs an entry — a global must be *compiled out*, not merely unused,
to give its slot back.

Before this module the budget had no owner. ``metal_wavefront`` learned a variant
was over-budget only from an opaque ``SLANG_FAIL`` at pipeline creation, on a
Metal host, for wavefront kernels only; the texture table had no signal at all
(an over-128 texture build produced a silent all-black frame). This module makes
the budget explicit:

* It is the **one home** for the three limits (spec ``metal-argument-table-budget``).
* :func:`census` derives, for a :class:`~skinny.shader_variants.ShaderVariantKey`,
  how many globals each argument table carries, **from the compiler itself** —
  ``slangc -reflection-json`` on the module under the variant's build gates. The
  reflection is the compiler's own module-scope argument assignment, so a global
  a ``#if`` compiles out is absent from the count, and the count matches what the
  Metal ABI charges.
* :data:`REGISTERED_VARIANTS` is the set a hostless test censuses; a variant over
  a limit fails that test before any GPU device is built.

**Union, not per-kernel.** ``slangc`` reflects a module's *global scope* — the
union of every global the module and its imports declare under the gates — not
the per-entry dead-stripped set that ``slang-rhi`` links at runtime. Empirically
every entry point of one module reflects the same union. So a module that
carries a heavy catch-all kernel (``wfPathShade``) reports that kernel's globals
even for a variant that never links it. The union is therefore an **upper
bound** on any one kernel: exact for a single-scope module (the megakernel), an
over-count for the multi-kernel wavefront modules. The exact per-kernel ``≤31``
is the GPU cross-check (``program.layout.parameters``, task 1.7). What the union
delivers hostlessly: the megakernel counts (exact), the bindless-capacity
derivation, the fold's buffer-count drop, and a baseline-diff on every variant.

**No GPU device.** ``slangc`` is a CPU compiler and the generated-material
modules are emitted by CPU codegen, so the census needs no GPU device — only
``slangc`` on ``PATH`` (the Vulkan SDK ships it) and the codegen dependency. On a
host without ``slangc`` the census is unavailable and the test asserts against
the checked-in baseline only.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from skinny.shader_variants import Family, ShaderVariantKey, Target

# ── The limits: exactly one home (spec: metal-argument-table-budget) ──────────
#
# Any other module that needs a limit reads it from here. A source comment is
# never the authority; the hostless grep gate (test_argument_budget) fails on a
# second declaration of any of these numbers.
METAL_BUFFER_LIMIT = 31
METAL_TEXTURE_LIMIT = 128
METAL_SAMPLER_LIMIT = 16

#: Vulkan's device limits sit far above these counts, so the census enforces only
#: Metal and reports Vulkan (design open question D5). The one Vulkan number that
#: matters is the bindless texture-array size, which equals the texture limit
#: because Vulkan sets ``PARTIALLY_BOUND`` and never reserves the unused tail.
VULKAN_TEXTURE_LIMIT = 128

#: The three argument tables. A census reports one count per table.
BUFFER = "buffer"
TEXTURE = "texture"
SAMPLER = "sampler"
TABLES = (BUFFER, TEXTURE, SAMPLER)


def table_limits(target: Target) -> dict[str, int | None]:
    """The per-table limit for ``target``. ``None`` means "far above any count —
    reported, not enforced" (every Vulkan table)."""
    if target is Target.METAL:
        return {BUFFER: METAL_BUFFER_LIMIT,
                TEXTURE: METAL_TEXTURE_LIMIT,
                SAMPLER: METAL_SAMPLER_LIMIT}
    return {BUFFER: None, TEXTURE: None, SAMPLER: None}


#: Name of the bindless flat-material texture pool. Singled out because the
#: derived bindless capacity is ``texture limit − every OTHER texture global``.
BINDLESS_POOL_NAME = "flatMaterialTextures"

#: The bindless flat-material texture pool capacity — ONE home for the number
#: ``metal_compute``, ``gpu_backend`` and the shader array size all restate.
#: A checked-in constant, not a live :func:`bindless_texture_capacity` call: the
#: runtime import of ``metal_compute`` must not trigger ``slangc``/codegen. The
#: hostless ``test_argument_budget`` ties each to ``bindless_texture_capacity``
#: from the compiler, so the constant cannot silently drift from the sources.
BINDLESS_TEXTURE_CAPACITY_METAL = 119
BINDLESS_TEXTURE_CAPACITY_VULKAN = 128


def hard_limit_enforced(key: ShaderVariantKey) -> bool:
    """Whether the hostless census enforces ``count ≤ limit`` for ``key``.

    Only where the ``slangc`` module union equals the per-kernel set — the
    single-scope families (megakernel, preview). The multi-kernel wavefront
    modules carry a heavy catch-all kernel whose globals inflate the union above
    any one kernel's true count, so their ``≤31`` is the GPU cross-check
    (``program.layout.parameters``, task 1.7), and the hostless gate on them is
    the baseline-diff, not a hard limit."""
    return key.target is Target.METAL and key.family in (
        Family.MEGAKERNEL, Family.PREVIEW)

# ── Reflection kind → argument table ─────────────────────────────────────────
#
# slangc's Metal reflection tags each global with a binding ``kind``. Device
# buffers (uniform blocks, ``constantBuffer``, and RW buffers alike) land in the
# buffer table; every texture (RO and RW) in the texture table; samplers in the
# sampler table. A kind outside this map raises rather than being silently
# dropped — under-reporting is the failure this module exists to prevent.
_BUFFER_KINDS = frozenset(
    {"uniform", "constantBuffer", "parameterBlock", "pushConstantBuffer"})
_TEXTURE_KINDS = frozenset({"shaderResource", "unorderedAccess"})
_SAMPLER_KINDS = frozenset({"samplerState"})


class ArgumentBudgetError(RuntimeError):
    """The census could not resolve something and refuses rather than
    under-report."""


# ── slangc + codegen: the compiler is the census engine ──────────────────────

def _shader_dir() -> Path:
    return Path(__file__).resolve().parent / "shaders"


def _genslang_dir() -> Path:
    return Path(__file__).resolve().parent / "mtlx" / "genslang"


def slangc_path() -> str | None:
    """``slangc`` on ``PATH``, else the one the Vulkan SDK ships (``VULKAN_SDK``).
    ``None`` when the census cannot run on this host."""
    import os
    import shutil
    found = shutil.which("slangc")
    if found:
        return found
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        cand = Path(sdk) / "bin" / "slangc"
        if cand.is_file():
            return str(cand)
    return None


_CODEGEN_DONE = False


def _ensure_codegen() -> None:
    """Emit the build-time generated modules (``generated_materials``, the python
    dispatcher, and the python-material Slang) into the shader tree, once, so
    ``slangc`` can resolve every ``import``. The outputs are gitignored and
    regenerated by the real build; emitting an empty-graph aggregator matches a
    scene with no MaterialX graphs (``graphParamsCombined`` then rides as the
    documented per-family supplement)."""
    global _CODEGEN_DONE
    if _CODEGEN_DONE:
        return
    from skinny import megakernel_sources as ms
    shd = _shader_dir()
    ms.run_codegen(shd)
    ms.emit_generated_materials(shd, [])
    ms.emit_python_dispatcher(shd)
    _CODEGEN_DONE = True


#: ``graphParamsCombined`` (one buffer) is declared only when a scene carries
#: MaterialX graphs, so the empty-graph codegen omits it. It is present in the
#: two families that ``import generated_materials`` and reach ``evalSceneGraph``.
#: Added as a documented worst-case supplement so the baseline reflects a
#: graph-bearing scene, not the smaller no-graph build.
_GRAPH_FAMILIES = frozenset({Family.MEGAKERNEL, Family.PREVIEW})

_ENTRY_RE = re.compile(
    r'\[shader\(\s*"compute"\s*\)\][^{;]*?\bvoid\s+([A-Za-z_]\w*)\s*\(', re.S)


def _module_file(entry_module: str) -> Path:
    rel = Path(*entry_module.split(".")).with_suffix(".slang")
    path = _shader_dir() / rel
    if not path.is_file():
        raise ArgumentBudgetError(f"no shader module {entry_module!r} at {path}")
    return path


def _first_entry(entry_module: str) -> str:
    """Any compute entry the module declares — the union reflection is the same
    for every entry, so the first is representative."""
    text = _module_file(entry_module).read_text()
    m = _ENTRY_RE.search(re.sub(r"//.*", "", text))
    if m is None:
        raise ArgumentBudgetError(
            f"{entry_module}: no compute entry point found to reflect")
    return m.group(1)


def _reflect(key: ShaderVariantKey, entry_module: str) -> list[dict]:
    """The reflected global parameters ``slangc`` assigns for ``entry_module``
    under ``key``'s build gates."""
    slangc = slangc_path()
    if slangc is None:
        raise ArgumentBudgetError(
            "slangc not found (PATH or VULKAN_SDK/bin) — the census cannot run "
            "on this host; assert against the checked-in baseline instead")
    _ensure_codegen()
    target = "metal" if key.target is Target.METAL else "spirv"
    defines = [f"-D{k}={v}" for k, v in key.session_defines().items()]
    entry = _first_entry(entry_module)
    with tempfile.TemporaryDirectory() as tmp:
        refl = Path(tmp) / "refl.json"
        out = Path(tmp) / ("out.metal" if target == "metal" else "out.spv")
        cmd = [slangc, str(_module_file(entry_module)),
               "-target", target, "-entry", entry, "-stage", "compute",
               *defines,
               "-I", str(_shader_dir()), "-I", str(_genslang_dir()),
               "-reflection-json", str(refl), "-o", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if not refl.is_file():
            raise ArgumentBudgetError(
                f"slangc reflection failed for {entry_module}:{entry} "
                f"({' '.join(defines)}):\n{proc.stderr[:2000]}")
        return json.loads(refl.read_text()).get("parameters", [])


def _table_sizes(params: list[dict]) -> dict[str, int]:
    """The argument-table size per table from reflected parameters. The buffer
    and sampler tables are contiguous scalar slots (one per param); the texture
    table's size is the top ``index + count`` (the bindless pool contributes its
    array count)."""
    sizes = {BUFFER: 0, TEXTURE: 0, SAMPLER: 0}
    for p in params:
        b = p.get("binding", {})
        kind = b.get("kind")
        count = b.get("count", 1) or 1
        index = b.get("index", 0)
        # `count` honoured on every table: the tree has no buffer/sampler arrays
        # today (so `count` is 1 there), but a future `StructuredBuffer<T> foo[N]`
        # must count N entries, not 1 — otherwise the census under-reports.
        if kind in _BUFFER_KINDS:
            sizes[BUFFER] += count
        elif kind in _TEXTURE_KINDS:
            sizes[TEXTURE] = max(sizes[TEXTURE], index + count)
        elif kind in _SAMPLER_KINDS:
            sizes[SAMPLER] += count
        else:
            raise ArgumentBudgetError(
                f"unclassified reflection kind {kind!r} on global "
                f"{p.get('name')!r} — teach argument_budget its argument table")
    return sizes


# ── The census ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Census:
    """The per-table global counts a variant's module compiles (the union)."""

    name: str
    key: ShaderVariantKey
    entry_module: str
    buffers: int
    textures: int
    samplers: int

    def count(self, table: str) -> int:
        return {BUFFER: self.buffers, TEXTURE: self.textures,
                SAMPLER: self.samplers}[table]


def census(variant: "RegisteredVariant") -> Census:
    """Census ``variant`` from the compiler: the module's reflected argument
    tables, plus the ``graphParamsCombined`` buffer for the graph-bearing
    families (absent from the empty-graph codegen)."""
    sizes = _table_sizes(_reflect(variant.key, variant.entry_module))
    if variant.key.family in _GRAPH_FAMILIES:
        sizes[BUFFER] += 1  # graphParamsCombined, present in a graph scene
    return Census(variant.name, variant.key, variant.entry_module,
                  sizes[BUFFER], sizes[TEXTURE], sizes[SAMPLER])


def bindless_texture_capacity(target: Target) -> int:
    """The bindless flat-material texture pool capacity, DERIVED: the texture
    limit minus every OTHER texture global the heaviest textured module (the
    megakernel) declares. Never a literal — the one home for the number the
    shader array size and the host both restate.

    Vulkan pays no reservation for the unused tail of the ``PARTIALLY_BOUND``
    bindless array, so its capacity is the array size = the limit. Only Metal,
    whose table binds every reserved slot, subtracts the other textures."""
    if target is Target.VULKAN:
        return VULKAN_TEXTURE_LIMIT
    params = _reflect(ShaderVariantKey(target=Target.METAL,
                                       family=Family.MEGAKERNEL), "main_pass")
    others = 0
    for p in params:
        b = p.get("binding", {})
        if b.get("kind") in _TEXTURE_KINDS and p.get("name") != BINDLESS_POOL_NAME:
            others += b.get("count", 1) or 1
    return METAL_TEXTURE_LIMIT - others


# ── The registered variants a hostless test censuses ──────────────────────────
#
# Each Metal program the renderer compiles: the megakernel, the wavefront
# path/bdpt/sppm/mlt integrators, and the preview, RGB and (where valid)
# spectral, plus the wavefront neural + records pressure points on the path
# module. Only Metal is enforced (Vulkan's tables are far above any count); a
# variant is identified by its module + key, and the census is per module
# because the slangc reflection is the module union.
@dataclass(frozen=True)
class RegisteredVariant:
    name: str
    key: ShaderVariantKey
    entry_module: str


def _variants() -> tuple[RegisteredVariant, ...]:
    out: list[RegisteredVariant] = []

    def add(name, family, module, **kw):
        out.append(RegisteredVariant(
            name, ShaderVariantKey(target=Target.METAL, family=family, **kw),
            module))

    add("megakernel", Family.MEGAKERNEL, "main_pass")
    add("megakernel.spectral", Family.MEGAKERNEL, "main_pass", spectral=True)
    add("wf_path", Family.WAVEFRONT, "wavefront.wavefront_path")
    add("wf_path.spectral", Family.WAVEFRONT, "wavefront.wavefront_path",
        spectral=True)
    add("wf_path.neural", Family.WAVEFRONT, "wavefront.wavefront_path",
        metal_neural=True)
    add("wf_path.records", Family.WAVEFRONT, "wavefront.wavefront_path",
        metal_records=True)
    add("wf_path.spectral.records", Family.WAVEFRONT, "wavefront.wavefront_path",
        spectral=True, metal_records=True)
    add("wf_bdpt", Family.WAVEFRONT, "wavefront.wavefront_bdpt")
    add("wf_bdpt.spectral", Family.WAVEFRONT, "wavefront.wavefront_bdpt",
        spectral=True)
    add("wf_sppm", Family.WAVEFRONT, "integrators.wavefront_sppm")
    add("wf_sppm.spectral", Family.WAVEFRONT, "integrators.wavefront_sppm",
        spectral=True)
    add("wf_mlt", Family.WAVEFRONT, "wavefront.wavefront_mlt", mlt=True)
    add("wf_mlt.spectral", Family.WAVEFRONT, "wavefront.wavefront_mlt",
        mlt=True, spectral=True)
    add("preview", Family.PREVIEW, "preview_pass")
    return tuple(out)


REGISTERED_VARIANTS: tuple[RegisteredVariant, ...] = _variants()


# ── Baseline: the recorded measurement, regenerated from the compiler ─────────

def baseline_path() -> Path:
    return (Path(__file__).resolve().parent.parent.parent
            / "tests" / "fixtures" / "argument_budget_baseline.json")


def compute_baseline() -> dict:
    """Census every registered variant from the compiler → the baseline dict.
    Needs ``slangc``; raises :class:`ArgumentBudgetError` otherwise."""
    variants = {}
    for v in REGISTERED_VARIANTS:
        c = census(v)
        variants[v.name] = {BUFFER: c.buffers, TEXTURE: c.textures,
                            SAMPLER: c.samplers}
    return {
        "_comment": "slangc module-union census per registered Metal variant. "
                    "Regenerate: python -c 'from skinny import argument_budget "
                    "as a; a.write_baseline()'. Wavefront counts are a union "
                    "upper bound (see argument_budget.hard_limit_enforced); the "
                    "per-kernel <=31 truth is the gpu cross-check.",
        "bindless_texture_capacity_metal":
            bindless_texture_capacity(Target.METAL),
        "variants": variants,
    }


def write_baseline() -> Path:
    path = baseline_path()
    path.write_text(json.dumps(compute_baseline(), indent=2, sort_keys=True)
                    + "\n")
    return path


def load_baseline() -> dict:
    return json.loads(baseline_path().read_text())
