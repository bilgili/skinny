# Design: session-settings-owner

## Context

`settings.py` is 177 lines and already has the right shape for one part of the
problem: `record_last_dir` loads, merges and writes. `save_settings` does not —
it writes whatever dict it is handed. The schema lives in two `main()`-adjacent
functions in two front-ends, neither of which is tested, and each of which
knows only its own keys.

The four front-ends split cleanly: `skinny` and `skinny-gui` persist;
`skinny-web` and `skinny-render` deliberately do not (they pass
`persisted=None` to `plan_bringup`). That asymmetry is intentional and stays.

## Goals / Non-Goals

**Goals**
- Stop mutual erasure.
- One declared schema; front-ends contribute, not re-author.
- One camera restore rule and one parameter-set rule.
- Hostless tests, which do not exist at all today.

**Non-Goals**
- Changing the settings file location or format.
- Making `skinny-web` or `skinny-render` persist. Their persistence-free
  behaviour is a recorded bring-up decision.
- Redesigning presets. `save_user_preset` / `delete_user_preset` are dead
  surface and are resolved here only because they sit in the same module.

## Decisions

### D1 — Merge on write, always

`save_settings` loads the current file, updates the keys it was given, and
writes. This is a one-line-shaped fix in the right place: it protects every
current and future writer, including ones that do not know the full schema.
Rejected: making each front-end write the full schema — that is the current
design, and it is what broke.

### D2 — Declared schema with contributed sections

The snapshot module declares the shared section (params, camera, gizmo mode,
backend, encoding, sppm glossy roughness) and accepts front-end sections
(`vulkan_window` and the neural/online-training keys for `skinny`; docks,
geometry, last dirs, section states for `skinny-gui`). Contributed sections are
opaque to the module — it does not need to understand a Qt geometry blob to
preserve it.

### D3 — Camera restore: keep the GLFW rule, drop the clamp

Two rules disagree. `skinny` raises `max_distance` to fit the restored
distance; `skinny-gui` clamps distance to 50 and ignores `max_distance`. The
Qt clamp silently destroys a legitimately persisted wide view, and
`_orbit_distance_cap` already exists as the cap authority. Choose the GLFW
rule. This is a deliberate behaviour choice, not a merge — it must be stated
in the change and the `CHANGELOG`, because Qt users' restored camera distance
may change.

### D4 — Snapshot all params, filter on restore

`skinny` snapshots `build_visible_params`, which drops fallback-light params
under an authored-lighting scene; `skinny-gui` snapshots `build_all_params`.
Dropping on *capture* loses data permanently. Capture everything; let
visibility affect what is shown and restored, not what is stored. This also
makes the two front-ends' files interchangeable, which is the point.

### D4b — `online_training` persists the intent, not the outcome

`skinny` persisted the flag *after* its prerequisite gate, so a session refused
for a missing prerequisite (megakernel, or no neural proposal) wrote
`online_training: false` and silently discarded what the user asked for. The
snapshot reads `renderer._online_training_requested`, which both front-ends set
to the pre-gate request. A refusal is a per-session condition, not a preference
change. The runtime gate still refuses the same way on the next launch, so
nothing runs that could not run before.

### D5 — Make the help text true

`cli_common` claims `--neural-trainer`, `--train-precision` and
`--online-training` are persisted on the interactive front-ends. With D1 and
D2, `skinny-gui` stops erasing them; whether it also *writes* them is a
smaller question — it can, since it has the renderer. Prefer implementing it,
so the documented behaviour holds on both front-ends.

### D6 — Delete the dead preset surface

`save_user_preset` and `delete_user_preset` have zero callers anywhere. By the
deletion test, removing them concentrates nothing — they are pure dead weight.
Delete, and update the stale `presets.py:31` docstring. If the Delete button
returns, the functions are eight lines.

## Risks / Trade-offs

- **Risk: D3 changes restored camera distance for Qt users.** Small, visible,
  and correct. Announce in `CHANGELOG.md`.
- **Risk: merge-on-write preserves a stale key forever.** Acceptable — the
  alternative is deleting keys the writer does not recognise, which is exactly
  today's bug. Add an explicit prune only if a key is ever renamed.
- **Risk: a corrupt settings file now fails a load *and* a save.**
  `load_settings` already returns `{}` on `JSONDecodeError`; merge-on-write
  inherits that, so a corrupt file is replaced rather than propagated.
- **Trade-off: front-end sections are opaque blobs.** That is deliberate — the
  module's job is preservation, not interpretation.
- **Accepted: the merge is an unlocked read-modify-write.** Two front-ends that
  exit in the same instant can still lose one update, and both write the same
  `settings.json.tmp` path, so one `os.replace` can fail (each caller already
  catches that). This is strictly better than the wholesale write it replaces —
  the loss window shrinks from "every key the other front-end owns" to
  "overlapping keys written in the same instant". A lock file is not worth the
  complexity for a two-process desktop app whose worst case is one session's
  window layout.
- **Accepted: a persisted flag does not reach the bring-up guards.** Each
  front-end resolves the persisted flags *after* `plan_bringup`, so a persisted
  `online_training: true` skips `validate_render_flags` (which refuses
  `--online-training` under `--integrator bdpt`/`mlt`) and is refused by the
  runtime gate instead. `skinny` behaved exactly this way before the change; the
  change makes `skinny-gui` match it. Feeding a persisted flag into the guard
  sequence belongs to `bringup.py`, which already takes `persisted=` for the
  integrator, backend and encoding — a follow-up, in that module.
- **Accepted: `settings.record_last_dir` still authors the `last_dirs` key.**
  It is the pre-existing write-through path, declared in `QT_KEYS` and merged
  like any other key. Moving its key semantics into the snapshot module would
  make every file dialog depend on the snapshot owner for no behaviour change.

## Open Questions

- Should the file gain a schema version? Not yet — merge-on-write makes
  additive changes safe, and there is nothing to migrate. Revisit on the first
  rename.
