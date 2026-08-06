# Design: choice-table-wavefront-owners

## Context

`choice-table-owners` owns the host-side render axes. This change owns the
wavefront layer's two mirrors: the kernel entry-point names, and the pass
constants duplicated between the two backend pass modules.

## Goals / Non-Goals

**Goals**
- One owner for the wavefront kernel entry-point names.
- One home for the pass constants that must be equal across backends.
- A test that pins the per-backend constants and states each reason.

**Non-Goals**
- Changing any dispatched kernel name or constant value. The runtime behaviour
  is byte-identical.
- Merging the two backend pass modules. This change shares constants, not
  classes.
- Changing the wavefront stage order or the `WavefrontRecorder` protocol.

## Decisions

### D1 — Kernel names are module-level constants in `wavefront_driver.py`

The driver already owns the backend-neutral loop order and imports no GPU
package. It declares one named constant per kernel (`WF_PATH_GENERATE =
"wfPathGenerate"`, …). The driver dispatches through the constant; both backend
`entries` lists import the constants with `from skinny.wavefront_driver import
(…)`. A rename edits one line, and a removed name is an import-time failure in
every consumer. A golden test pins each constant to its historical string and
asserts no kernel-name string literal remains in the three modules.

Rejected: a dict keyed by short name — a wrong key is a runtime `KeyError`, not
the import-time failure the requirement asks for.

### D2 — Shared constants get one home; per-backend constants get a pin

The constants that must be equal — `MAX_BOUNCES`, `BDPT_MAX_VERTS`,
`EYE_BOUNCES`, `LIGHT_BOUNCES`, `WALK_MODES`, the two `STREAM_CAP` values, and
the ReSTIR `DEFAULT_CONFIG` — get one home the pass classes read. The constants
that are per-backend by design stay separate but a test pins the pair and states
the reason:

- the record-stack sizing formula (`rec_stack_elems = capacity × MAX_BOUNCES` on
  Vulkan vs the per-lane `rec_lane = MAX_BOUNCES + records_active` on Metal);
- `VERTEX_STRIDE` / `AUX_STRIDE` / `RESERVOIR_STRIDE` — the same value, but the
  Metal copies are a reflection fallback (the MSL stride is authoritative) while
  the Vulkan copies are the stride itself;
- the Metal rebuild-key elements that carry extra members.

Do not force equality on a value that is per-backend by design.

### D3 — Byte-identical, then GPU-smoked

The change moves declarations, never values. The hostless golden test proves the
strings and constant values are unchanged; the dual-backend wavefront GPU smoke
(one render each on Vulkan and native Metal) confirms every kernel still
dispatches. The GPU smoke is why this change lands on its own schedule — it must
serialise against any other Metal work (ZERO-SWAP, one guarded Metal process at
a time).

## Risks / Trade-offs

- **Risk: a moved constant silently changes a stride.** The golden test pins
  every value against the pre-change source, so a typo in the move fails
  hostlessly before the GPU smoke.
- **Risk: import cycle from the name table.** The names live in
  `wavefront_driver.py`, which already imports no GPU package; the backends
  already import the driver, so no new edge appears.

## Open Questions

- Should the shared constants live in `wavefront_driver.py` beside the names, or
  in a small `wavefront_layout`-adjacent module? Leaning `wavefront_driver`,
  since it is already the backend-neutral owner both backends import.
