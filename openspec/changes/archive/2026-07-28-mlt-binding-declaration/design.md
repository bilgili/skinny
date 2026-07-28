# Design: mlt-binding-declaration

## Context

Six buffers, five statements of their identity (see the proposal's table). The
shader is the only place that already binds the two halves together:

```slang
[[vk::binding(52)]] RWStructuredBuffer<MltPrimarySample> mltPrimarySamples;
```

— binding number and global name in one declaration. The host then re-derives
that pairing twice, independently, once per backend, keyed by a third thing
(the `mlt_buffer_sizes` key) that appears in neither.

The failure this admits is a *transposition*, not an omission. An omission
fails loudly: a missing `size_key` is a `KeyError` at pass construction, and a
missing binding number trips the existing count cross-check. A transposition
type-checks, allocates, binds, and dispatches — and produces a wrong image on
one backend only.

## Goals / Non-Goals

**Goals**
- One declaration per MLT chain buffer: size key, Vulkan binding, Metal name.
- A hostless gate that the declaration matches the shader's own pairing.
- Byte-identical behaviour: same buffers, sizes, bindings, names, write order.

**Non-Goals**
- A general pass-owned-resource framework (one implementation — see proposal).
- Runtime derivation of bindings from shader reflection.
- Changing `mlt_buffer_sizes` or the `msl=` stride mirror.

## Decisions

### D1 — The table lives with the sizes, not with the resource set

`wavefront_layout.mlt_buffer_sizes()` already owns the six `size_key`s and is
the module both passes import. Putting the declaration beside it means the key,
the size formula, the binding and the name are one block of text.

Rejected: `gpu_resources.py`. It owns the **renderer's** inventory, whose
lifetime is the renderer's; the MLT chain buffers are built and destroyed by
the pass, per accumulation reset. Folding them into `SceneResourceSet` would
put two different lifetimes in one set and force `DECLARATIONS` to carry
conditionally-absent, pass-owned entries — the accept-then-drop shape that
`shader-variant-key-module` documents as the thing to avoid. `gpu_resources`
stays a *consumer*: it needs the binding numbers only, for the creation-time
dummy writes and the pool count.

Rejected: a new `mlt_resources.py`. A six-row table plus two accessors does not
earn a module, and it would separate the sizes from the identity again.

### D2 — The shader is the gate, not the runtime source

The declaration is checked against the shader by parsing
`[[vk::binding(N)]] … <name>;` out of the MLT sources at test time. This gives
the strong property — a transposition on either backend fails the build —
without making startup depend on parsing shaders, and without the host needing
`slangc` reflection for something it can state in six lines.

The parse is deliberately narrow: only the six MLT globals, only the
`vk::binding` form, and the test fails loudly if it finds a different count than
the table declares (so a shader edit that adds a seventh buffer fails rather
than being silently ignored — the failure mode a lenient regex would create).

### D3 — `MLT_BINDINGS` stays, derived

`gpu_resources.MLT_BINDINGS` is referenced by `pool_sizes()` and `bind_vulkan()`
and is part of that module's tested surface. Keep the name, derive the value
from the new table. This keeps the diff off `gpu_resources`' public behaviour
and lets the existing `gpu_resources` tests stand unchanged.

### D4 — The existing cross-check test is deleted, not kept

`test_mlt_binding_numbers_agree_with_the_wavefront_pass` compares
`MLT_BINDINGS` with `WavefrontMltPass._BINDINGS`. Once both derive from one
table it compares a value with itself — a test that cannot fail is worse than
no test, because it reads as coverage. The shader-agreement gate replaces it.

## Risks / Trade-offs

- **Risk: the parse silently matches nothing** and the gate passes vacuously —
  the failure mode that made `ruff check src/` a no-op here and
  `importorskip("vulkan")` a vacuous skip in `shader-variant-key-module`. Gate:
  the test asserts the parsed count equals six *before* comparing, and a
  self-test feeds it a deliberately transposed table and asserts failure.
- **Risk: MLT globals are declared under `#if defined(SKINNY_MLT)`**, so a naive
  reader must not require them to be unconditionally present. The parse targets
  the declaration text, not a compiled variant, so the gate stays hostless and
  build-flavour-independent.
- **Trade-off: `wavefront_layout.py` grows a non-layout concern.** Binding
  identity is not byte layout. Accepted because the alternative separates the
  key from its size, and the module is already the single import both passes
  share for this buffer family.

## Migration Plan

1. Add the declaration table + accessors beside `mlt_buffer_sizes`; nothing
   consumes it yet.
2. Add the shader-agreement gate, and the negative self-test that proves it
   fails on a transposition.
3. Point `gpu_resources` at it (`MLT_BINDINGS` becomes derived).
4. Point both passes at it; delete their local `_BINDINGS`.
5. Delete the now-vacuous cross-check test; update the `docs/Architecture.md`
   binding-map rows 52–57 with a pointer to the host-side owner.
6. Gate: hostless suite, plus an MLT render on both backends at equal budget
   confirming the documented bit-identity still holds.

## Open Questions

- Should the shader-agreement gate extend to the other shared-scene-set
  bindings (0–51) rather than only 52–57? The same parse would cover them, and
  `gpu_resources.DECLARATIONS` already states binding-plus-Metal-name for all of
  them — so the check is nearly free and would gate the renderer inventory
  against the shader too. Deliberately out of scope here (it would widen the
  change from "one pass's table" to "every binding in the project"), but it is
  the obvious follow-on and the reason D2 builds the parser as a reusable
  helper rather than inline in one test.
