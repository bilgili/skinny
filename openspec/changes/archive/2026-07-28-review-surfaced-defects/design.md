# Design: review-surfaced-defects

## Context

These defects were found while reviewing architecture proposals, not while
investigating a report. None has a user-filed ticket, and several are invisible
until someone looks: a frozen web stream reads as a hung render, a dead
animation transport reads as an unimplemented feature, a stale HUD in a
screenshot reads as a capture-timing quirk.

Three of them violate a requirement `qt-render-threading` already states. That
matters for how they are fixed: the requirement is not wrong, it is
under-specified in three places that the implementations found their way
through.

## Goals / Non-Goals

**Goals**
- Fix each defect at the point where all callers route through it.
- Close the specification loopholes so the same class cannot silently recur.
- Keep each defect independently landable.

**Non-Goals**
- Restructuring how front-ends drive the renderer. That is
  `renderer-command-interface`, proposal-only; this change fixes defects inside
  the current structure.
- Deleting the Panel dock's copied dispatcher as part of a UI-spec seam. That is
  `ui-spec-scene-properties`, also proposal-only; the deletion here is the
  narrow one — call the shared function, keep everything else.
- Settings-file erasure, owned by `session-settings-owner`.

## Decisions

### D1 — Marshal the mutation; do not merely guard the iteration

The `mtlx.*` crash could be silenced by copying the mapping before iterating it
in the accumulation-state provider. That treats the symptom: the write is still
unsynchronised, and the *next* provider to iterate a mutable renderer-owned
collection reintroduces it. The fix is to marshal the write, which the existing
requirement already demands.

The render-loop guard is added as well, but as a backstop with a different job:
a session must never sit marked running with a dead render thread, whatever
raises. Two changes, two reasons — not belt-and-braces on one.

### D2 — Sub-object writes are the loophole worth naming

The Qt animation transport is the instructive case: it obeys the letter of
"marshal writes of renderer state" — no top-level attribute is written — while
doing nothing at all. Any proxy that holds a local instance of a renderer-owned
object has this hazard, and the proxy holds several (`clock`, `film`,
`scene_graph`). Naming it in the requirement is what stops the next one.

Whether the proxy grows clock verbs or the setters are rewritten to post is an
implementation choice; the requirement constrains the outcome, not the shape.

### D3 — The missing resize callback is a fallback that should not exist silently

`_add_resolution` calls `renderer.resize` when no callback is supplied. That
fallback is what turns a missing wire into an unsynchronised call rather than an
error. Supplying the callback fixes the web front-end; the requirement addition
is what makes the next front-end supply it too. Consider whether the fallback
should raise instead of defaulting — decided during implementation, recorded in
tasks.

### D4 — Call the shared dispatcher; do not fix the copy

The Panel copy could be patched to handle `light_env`. It would then still be a
copy, still returning `None` where the shared function returns a reason, and
still free to drift. Deleting it in favour of the shared call fixes the reported
symptom and removes the mechanism. The two re-inlined fan-out guards go with it,
because the shared function checks fan-out first.

### D5 — HUD: fill where you copy

The asymmetry is that one path fills and both copy. Making the offscreen path
fill is the smaller change and matches what a caller expects from a screenshot.
The alternative — stop copying on the offscreen path — would silently drop the
HUD from headless output, which is a product decision, not a defect fix. If the
HUD is genuinely unwanted in headless output, that is a separate change with its
own rationale.

### D6 — Delete the vestigial rewrite rather than "fixing" it

The binding-1 rewrite is not wrong in a way that produces a wrong image today —
it writes the binding that is already bound. It is dead work plus two comments
that actively mislead about what the windowed path does. Deleting it is the
whole fix; nothing is meant to take its place.

### D7 — The three stale claims are one task each, no ceremony

A hand-typed stride that duplicates a derived value, a published function with
no callers, and a docstring contradicting its own function. Each is a small
edit. They earn a requirement only in the sense that the first one — the stride
— belongs to `reflection-owned-byte-layouts`' existing "derive, don't
hand-copy" discipline, so it is fixed the way that capability implies rather
than by updating the literal.

## Risks / Trade-offs

- **Risk: marshalling the parameter write changes web latency.** The queue
  coalesces last-write-wins, which is what a slider drag wants, and the value is
  only read at the next frame either way. Measure a drag before and after; do
  not assume.
- **Risk: supplying the resize callback changes web resize semantics.** The web
  session's existing `resize` method holds its lock across resize, encoder
  rebuild, stale-frame drain and the WebSocket notifications, in that order and
  for a stated reason. Route the callback to that method rather than to
  `renderer.resize`, or the ordering guarantee is lost.
- **Risk: the HUD appears in output that previously lacked it.** Headless
  renders that never set HUD text are unaffected — the overlay is empty and the
  renderer early-returns. Sessions that do set it will start seeing it in
  screenshots, which is the intent.
- **Risk: deleting the Panel dispatcher changes edit behaviour beyond
  `light_env`.** The shared function checks fan-out first and returns reason
  strings; both are behaviour changes, both desirable. Diff the two
  implementations per reference kind before deleting, and record any difference
  that is not an improvement.

## Open Questions

- Should `_add_resolution`'s direct-renderer fallback raise instead of
  defaulting? Leaning yes — a front-end that offers the control and forgets the
  callback should fail loudly — but it turns a silent defect into a startup
  error for any caller not yet updated, so it wants a sweep of the call sites
  first.
- Is the HUD wanted in headless output at all? This change assumes yes, because
  the path already copies it. If the answer is no, D5 inverts and the fix is to
  stop copying.
