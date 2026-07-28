"""Hostless gate: the MLT chain buffers' cross-backend identity is declared
once, and that declaration agrees with the shader (change
mlt-binding-declaration).

`wavefront_layout.MLT_CHAIN_BUFFERS` states, per buffer, its `mlt_buffer_sizes`
key, its Vulkan set-0 binding, and its Metal shader-global name. The shader
already states the last two together in one declaration:

    [[vk::binding(52)]] RWStructuredBuffer<MltPrimarySample> mltPrimarySamples;

The failure this gate exists for is a **transposition**, not an omission. An
omission fails loudly today — a missing `size_key` is a `KeyError` at pass
construction. A transposition (a valid binding paired with the wrong buffer, or
a valid Metal name paired with the wrong buffer) type-checks, allocates six
correctly-sized buffers, binds all six, and dispatches. One backend then reads
the seeds buffer where the shader expects current records. Nothing raises, and
the parity matrix charges the divergence to MLT's Markov correlation, because
MLT already carries a 0.15 self-consistency tolerance for exactly that reason.

Scope of the parse: the two sources that declare MLT globals — `common.slang`
(binding 52, inside its `#if defined(SKINNY_MLT)` block) and
`wavefront/wavefront_mlt.slang` (53–57, an MLT-only translation unit). Neither
declares any other single-argument `vk::binding` today; the shared scene
bindings live in `bindings.slang` (gated separately by
`test_vk_binding_layout.py`). The count assertion below therefore doubles as
the "a seventh chain buffer was added shader-side" check.

`parse_binding_declarations` is deliberately a module-level public helper: the
obvious follow-on extends the same comparison to the shared scene bindings
0–51, which `gpu_resources.DECLARATIONS` already states binding-plus-Metal-name
for.
"""

from __future__ import annotations

import re
from pathlib import Path

from skinny.wavefront_layout import (
    MLT_CHAIN_BUFFERS,
    MltChainBuffer,
    mlt_binding_numbers,
    mlt_buffer_sizes,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADERS = PROJECT_ROOT / "src" / "skinny" / "shaders"
MLT_SHADER_SOURCES = (
    SHADERS / "common.slang",
    SHADERS / "wavefront" / "wavefront_mlt.slang",
)

# Single-argument `[[vk::binding(N)]]` (= set 0) followed by a global's type and
# name. The two-argument form `vk::binding(N, 1)` is set 1 (pass-owned sets) and
# is deliberately not matched. `[^;{]*?` keeps the match inside one declaration:
# it cannot run past a `;` into the next statement or into a function body.
_DECL = re.compile(
    r"\[\[vk::binding\((\d+)\)\]\]\s*[^;{]*?(\w+)\s*;")

# The 1.1 baseline capture, at a fixed budget. Pinned so the move is provably
# byte-for-byte: same keys, same bindings, same names, same sizes.
CAPTURE_CHAINS = 16384
CAPTURE_BOOTSTRAP = 100000
CAPTURE = (
    # (size_key,             binding, metal_name,            bytes)
    ("mlt_primary_samples",   52, "mltPrimarySamples",   50331648),
    ("mlt_chain_meta",        53, "mltChainMeta",          524288),
    ("mlt_current_records",   54, "mltCurrentRecords",    2097152),
    ("mlt_bootstrap_weights", 55, "mltBootstrapWeights",   400000),
    ("mlt_chain_seeds",       56, "mltChainSeeds",          65536),
    ("mlt_proposal_records",  57, "mltProposalRecords",   2097152),
)


def parse_binding_declarations(path: Path) -> tuple[tuple[int, str], ...]:
    """``((binding, global_name), …)`` for every single-argument
    ``[[vk::binding(N)]] … <name>;`` in a Slang source, in declaration order.

    Pure text analysis — no ``slangc``, no preprocessor, no GPU. Deliberately
    build-flavour-independent: the MLT globals sit behind
    ``#if defined(SKINNY_MLT)``, and this gate compares *declarations*, not a
    compiled variant.
    """
    return tuple(
        (int(m.group(1)), m.group(2))
        for m in _DECL.finditer(path.read_text(encoding="utf-8"))
    )


def shader_mlt_declarations() -> tuple[tuple[int, str], ...]:
    """The MLT chain-buffer declarations as the shader states them."""
    out: list[tuple[int, str]] = []
    for src in MLT_SHADER_SOURCES:
        out.extend(parse_binding_declarations(src))
    return tuple(sorted(out))


def disagreements(table: tuple[MltChainBuffer, ...],
                  shader: tuple[tuple[int, str], ...]) -> list[str]:
    """Per-buffer pairing disagreements between a host table and the shader,
    each naming the offending buffer. Factored out so the negative self-test
    below can drive it with a deliberately transposed table."""
    by_binding = dict(shader)
    out = []
    for decl in table:
        actual = by_binding.get(decl.binding)
        if actual is None:
            out.append(
                f"{decl.key}: host declares binding {decl.binding}, which the "
                f"shader does not declare at all")
        elif actual != decl.metal_name:
            out.append(
                f"{decl.key}: host pairs binding {decl.binding} with "
                f"'{decl.metal_name}', shader pairs it with '{actual}'")
    return out


# ── The parse is not vacuous ────────────────────────────────────────────────

def test_parse_finds_the_expected_shape():
    """A regex gone stale would make every check below vacuously green."""
    common = parse_binding_declarations(SHADERS / "common.slang")
    mlt = parse_binding_declarations(SHADERS / "wavefront" / "wavefront_mlt.slang")
    assert common == ((52, "mltPrimarySamples"),), common
    assert [b for b, _ in mlt] == [53, 54, 55, 56, 57], mlt
    assert mlt[-1] == (57, "mltProposalRecords"), mlt


def test_parser_ignores_set1_and_bodies():
    """Two-argument (set 1) bindings are out of scope, and the match cannot run
    past a `;` into a following statement or a function body."""
    text = (
        "[[vk::binding(3, 1)]] RWStructuredBuffer<uint> passOwned;\n"
        "[[vk::binding(9)]] RWStructuredBuffer<float> sceneGlobal;\n"
        "void f() { uint x; }\n"
    )
    tmp = Path(__file__).parent / "_parser_probe.slang"
    tmp.write_text(text, encoding="utf-8")
    try:
        assert parse_binding_declarations(tmp) == ((9, "sceneGlobal"),)
    finally:
        tmp.unlink()


def test_shader_declares_exactly_as_many_as_the_table():
    """Asserted BEFORE any comparison, so a parse that matches nothing fails
    instead of silently reporting agreement — and so a chain buffer added to
    the shader without a host declaration fails on the count rather than being
    quietly skipped."""
    shader = shader_mlt_declarations()
    assert len(shader) == len(MLT_CHAIN_BUFFERS), (
        f"parsed {len(shader)} MLT binding declarations from "
        f"{[p.name for p in MLT_SHADER_SOURCES]} but "
        f"wavefront_layout.MLT_CHAIN_BUFFERS declares "
        f"{len(MLT_CHAIN_BUFFERS)}: {shader}")


# ── The declaration matches the shader, and the capture ─────────────────────

def test_table_agrees_with_the_shader():
    shader = shader_mlt_declarations()
    assert len(shader) == len(MLT_CHAIN_BUFFERS)
    bad = disagreements(MLT_CHAIN_BUFFERS, shader)
    assert not bad, "MLT binding declaration disagrees with the shader:\n" + "\n".join(bad)


def test_transposed_table_is_rejected():
    """Negative self-test (task 3.3): without it the gate is unproven. Swap the
    Metal names of 54 and 56 — every binding number, every name and every size
    is still individually present and valid, which is precisely why nothing
    else in the build catches it."""
    shader = shader_mlt_declarations()
    transposed = tuple(
        d._replace(metal_name={"mltCurrentRecords": "mltChainSeeds",
                               "mltChainSeeds": "mltCurrentRecords"}
                   .get(d.metal_name, d.metal_name))
        for d in MLT_CHAIN_BUFFERS
    )
    bad = disagreements(transposed, shader)
    assert len(bad) == 2, bad
    assert any("mlt_current_records" in m for m in bad), bad
    assert any("mlt_chain_seeds" in m for m in bad), bad


def test_table_matches_the_baseline_capture():
    """Task 2.2 — the move preserves the 1.1 capture entry for entry."""
    sizes = mlt_buffer_sizes(CAPTURE_CHAINS, CAPTURE_BOOTSTRAP)
    msl_sizes = mlt_buffer_sizes(CAPTURE_CHAINS, CAPTURE_BOOTSTRAP, msl=True)
    assert len(MLT_CHAIN_BUFFERS) == len(CAPTURE)
    for decl, (key, binding, name, nbytes) in zip(MLT_CHAIN_BUFFERS, CAPTURE, strict=True):
        assert (decl.key, decl.binding, decl.metal_name) == (key, binding, name)
        assert sizes[key] == nbytes, key
        assert msl_sizes[key] == nbytes, key  # all-scalar fields → strides agree


def test_size_keys_and_declaration_keys_are_the_same_set():
    """The table indexes `mlt_buffer_sizes`; a key present in one and not the
    other is a `KeyError` at pass construction on a GPU host only."""
    assert [d.key for d in MLT_CHAIN_BUFFERS] == list(
        mlt_buffer_sizes(64, 64).keys())


def test_binding_numbers_accessor():
    assert mlt_binding_numbers() == (52, 53, 54, 55, 56, 57)


# ── No consumer states the identity independently ───────────────────────────

CONSUMERS = (
    "src/skinny/gpu_resources.py",
    "src/skinny/vk_compute.py",
    "src/skinny/vk_wavefront.py",
    "src/skinny/metal_wavefront.py",
)


def test_no_consumer_carries_its_own_binding_table():
    """The requirement is structural, so it is gated structurally: a consumer
    may reference the declaration, never restate it. Catches a re-introduced
    literal tuple `(52, 53, 54, 55, 56, 57)` and any hardcoded Metal global."""
    literal = re.compile(r"52\s*,\s*53\s*,\s*54\s*,\s*55\s*,\s*56\s*,\s*57")
    names = [d.metal_name for d in MLT_CHAIN_BUFFERS]
    for rel in CONSUMERS:
        src = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith("#"))
        assert not literal.search(code), (
            f"{rel} restates the MLT binding numbers — derive them from "
            "wavefront_layout.MLT_CHAIN_BUFFERS instead")
        for name in names:
            assert name not in code, (
                f"{rel} hardcodes the Metal global '{name}' — it belongs to "
                "wavefront_layout.MLT_CHAIN_BUFFERS")


def test_gpu_resources_mlt_bindings_is_derived():
    """`MLT_BINDINGS` keeps its name and public shape (design D3); only its
    provenance changes.

    No `importorskip("vulkan")`: `gpu_resources` imports vulkan lazily inside
    its methods, so this is genuinely hostless — and `importorskip` catches
    only `ImportError` while the vulkan binding raises `OSError` when the SDK
    is off the dynamic-library path, so the guard would have failed the test
    rather than skipping it (the same trap recorded in
    `shader-variant-key-module`)."""
    from skinny.gpu_resources import MLT_BINDINGS

    assert MLT_BINDINGS == mlt_binding_numbers()
