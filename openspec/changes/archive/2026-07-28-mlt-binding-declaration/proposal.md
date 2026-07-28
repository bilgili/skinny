# Change: mlt-binding-declaration

## Why

The identity of an MLT chain buffer — *which* buffer, at *which* Vulkan binding,
under *which* Metal shader-global name, at *which* size — is stated in five
places:

| Where | States |
|-------|--------|
| `shaders/common.slang`, `shaders/wavefront/wavefront_mlt.slang` | `[[vk::binding(52)]] … mltPrimarySamples` — binding number **and** global name, together. Authoritative. |
| `gpu_resources.MLT_BINDINGS` (`:433`) | the binding numbers `(52…57)`, for the creation-time dummy writes and the descriptor-pool count |
| `vk_wavefront.WavefrontMltPass._BINDINGS` (`:1128`) | `(binding, size_key)` pairs, for the real chain-buffer writes |
| `metal_wavefront.MetalWavefrontMltPass._BINDINGS` (`:1155`) | `(metal_global_name, size_key)` pairs, for bind-by-name |
| `wavefront_layout.mlt_buffer_sizes()` (`:296`) | the six `size_key`s and their byte sizes |

`renderer-gpu-resource-set` just removed exactly this shape for the renderer's
own inventory — one declaration per resource carrying allocation, binding on
both backends, and destruction. The MLT chain buffers are the same fact one
level down, in a **pass-owned** set, and they were left out of that change's
scope. It is the last place in the wavefront plumbing where a resource's
cross-backend identity is assembled from independent tables.

**The concrete ungated risk is narrower than "duplication", and worth naming
precisely.** A cross-check test already pins `MLT_BINDINGS` against the Vulkan
pass's binding numbers, so a *count* or *number* drift fails the build. What
nothing checks is the **pairing**: the Vulkan table maps `binding → size_key`
and the Metal table maps `global_name → size_key`, independently. Transpose two
rows in one table only — say `54 → mlt_chain_seeds` and `55 →
mlt_current_records` on Vulkan while Metal keeps the shader's pairing — and both
backends allocate six correctly-sized buffers, bind all six, and run. Vulkan
then reads the seeds buffer where the shader expects current records. Nothing
raises. The MLT image silently diverges between backends, breaking the
documented "bit-identical at equal budget, RGB and spectral" property, and the
parity matrix attributes it to Markov correlation because MLT already carries a
0.15 self-consistency tolerance for exactly that reason.

That is a bug the existing gates are structurally unable to catch, and the size
of the edit that introduces it is one transposed line.

## What Changes

- Add **one declaration table** for the MLT chain buffers — per buffer: the size
  key, the Vulkan binding number, and the Metal shader-global name — in the
  module that already owns their sizes (`wavefront_layout.py`, alongside
  `mlt_buffer_sizes`).
- `gpu_resources`, `vk_wavefront.WavefrontMltPass` and
  `metal_wavefront.MetalWavefrontMltPass` all consume that one table. The three
  local `_BINDINGS` / `MLT_BINDINGS` tuples go away; `MLT_BINDINGS` may remain
  as a derived re-export so `gpu_resources`' public surface is unchanged.
- Add a gate that the host declaration agrees with the **shader**, which already
  states binding and name together: parse the `[[vk::binding(N)]] … <name>`
  declarations out of the MLT shader sources and assert they match the table
  entry for entry. This is the check that makes a transposition impossible
  rather than merely unlikely, and it is hostless.
- Pure refactor: same buffers, same sizes, same binding numbers, same names,
  same write order. No shader change, no behaviour change.

## Capabilities

### Modified Capabilities

- `metropolis-light-transport`: the chain-state buffers' cross-backend binding
  identity gets a single declaration, checked against the shader.

## Impact

- New: a declaration table + its accessors in `src/skinny/wavefront_layout.py`;
  one hostless test module (or a section in `tests/test_mlt_host.py`).
- Modified: `src/skinny/gpu_resources.py` (consume, re-export),
  `src/skinny/vk_wavefront.py` and `src/skinny/metal_wavefront.py` (drop the
  local tables).
- Unchanged: every binding number, every Metal global name, every buffer size,
  the shaders, and `docs/Architecture.md`'s binding-map contents (its rows 52–57
  gain a pointer to the host-side owner).
- Removed: `tests/test_gpu_resources.py::test_mlt_binding_numbers_agree_with_
  the_wavefront_pass`, which becomes vacuous — it compares a table against
  itself once both derive from one source. The shader-agreement gate replaces
  it and is strictly stronger.

## Non-Goals

- Generalizing to the other wavefront passes. They were surveyed: only the MLT
  pass carries this shape (path / BDPT / SPPM / ReSTIR bind through pass-owned
  descriptor sets, not the shared scene set), so there is no second consumer to
  design for. Building a general pass-resource framework here would be an
  abstraction with one implementation.
- Deriving binding numbers from the shader at runtime. The gate *compares*
  host and shader; it does not make the shader the runtime source. Runtime
  derivation is `shader-byte-layouts` territory and would change startup cost.
- Touching the MLT chain-buffer sizes or `mlt_buffer_sizes`' `msl=` mirror.
