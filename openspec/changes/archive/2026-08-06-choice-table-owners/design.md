# Design: choice-table-owners

## Context

`render_envelope.py` owns whether a combo is *valid*. Nothing owns what the
axes are *called*. The result is four copies of the integrator index, four of
the tonemap list, and a 17-entry placeholder table in the render-session proxy
whose entries are already wrong in shipped UI.

The wavefront layer has the same shape one level down: kernel entry-point names
as string literals in three files, and 14 pass constants duplicated verbatim
between the two backend pass modules with nothing pinning them equal.

## Goals / Non-Goals

**Goals**
- One owner per axis for values, labels and indices.
- Mirrors read the owner or are deleted.
- The already-drifted values are corrected as part of the change.

**Non-Goals**
- Changing the envelope predicate or any validity rule.
- Merging the two wavefront pass modules. This change gives them shared
  constants, not shared classes.
- Renaming any user-facing choice value. The CLI vocabulary is stable.

## Decisions

### D1 — The owner is a table, not a function

Each axis is a list of records: token (CLI value), index (the renderer's
integer), label (display). The CLI's `choices` is a projection; the renderer's
display list is a projection; the headless dict is a projection. Rejected:
extending `render_envelope` itself — it owns validity, and mixing display
labels into it would make a hostless predicate carry UI vocabulary.

### D2 — The proxy placeholders import the table

`render_session._default_choice_names` exists because the GUI thread needs
names before a renderer exists. That need is real; retyping the lists is not.
It imports the toolkit-free table. Same for `_default_values`, which is
currently a third defaults authority beside `params.py` and
`Renderer.__init__` — it reads the params registry instead.

### D3 — Drifted values are corrected, and the corrections are listed

Fixing them is the point, but each is a user-visible change and must be named:
the missing MLT integrator label, the `["Filmic"]` tonemap placeholder against
the real four-entry list, and the six divergent placeholder lists. Some of
these may have been deliberate stubs; check each before "fixing" it.

### D4 — Kernel names get a table; constants get a table or a pin (DEFERRED)

**Deferred to the follow-up change `choice-table-wavefront-owners`.** Entry-point
names are pure strings shared by three modules — a table, and a rename becomes an
import error. The 14 pass constants are subtler: some genuinely differ (the
record-stack sizing formula differs by design between backends, and the Metal
rebuild keys carry extra elements). So: shared where they must be equal, and a
test pinning the pair where they are separately maintained for a stated reason.
Do not force equality on constants that are legitimately per-backend. This half
is large mechanical churn across the two GPU pass modules and is gated by a
dual-backend wavefront GPU smoke, so per D5 it lands on its own schedule.

### D5 — One axis at a time

Every item here is independent. Land them separately; each is small, hostless,
and individually reviewable.

## Risks / Trade-offs

- **Risk: a "placeholder" was deliberate.** `["Off"]` for reuse modes may have
  been a considered stub for a pre-renderer GUI state. Check each of the six
  before changing it; record the verdict.
- **Risk: an import cycle from the table.** Keep it dependency-free — it is a
  data module, importable by anything, importing nothing from the renderer.
- **Trade-off: correcting labels changes UI text.** Small and correct; note in
  `CHANGELOG.md`.

## Open Questions

- Does the label belong with the token, or should labels live with the UI?
  Leaning: with the token — there is one label per token, and splitting them
  recreates the mirror this change removes.
- Should the kernel-name table live in `wavefront_driver.py` (which already
  owns the loop orders) or in its own module? Leaning: `wavefront_driver`,
  since it is already the backend-neutral owner and imports nothing GPU.
