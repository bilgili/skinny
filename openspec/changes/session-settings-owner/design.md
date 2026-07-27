# Design: session-settings-owner

## Context

`settings.py` is 177 lines and already has the right shape for one part of the
problem: `record_last_dir` loads, merges and writes. `save_settings` does not —
it writes whatever dict it is handed. The schema lives in two `main()`-adjacent
functions in two front-ends, neither of which is tested.

`record_last_dir` is called from the Qt UI (`ui/qt/backend.py:458`,
`ui/qt/windows/scene_graph.py:303,691,731`) and the Panel/web UI
(`ui/panel/backend.py:458`, `ui/panel/windows.py:123`) — never from `skinny`.
So `skinny-web` *does* mutate `settings.json`, even though it persists no
session snapshot. That interaction is benign under merge-on-write, but it is
why the `last_dirs` sub-dict clobber (D2) is real rather than theoretical.

## Goals / Non-Goals

**Goals**
- Stop erasure — including the two sub-dict cases a top-level merge misses.
- One camera restore rule, one captured parameter set.
- A truthful persistence contract.
- Hostless tests, which do not exist at all today.

**Non-Goals**
- Changing the settings file location or format.
- Making `skinny-web` or `skinny-render` persist a session snapshot. Their
  persistence-free bring-up is a recorded decision.
- A declared schema with pinned key sets. See D3.
- Preventing auto-frame from overwriting a restored camera. See Out of scope.

## Decisions

### D1 — Merge on write, replacing per key

`save_settings` loads the current file, updates the keys it was given, and
writes. This protects every current and future writer, including one that does
not know the full schema. Rejected: making each front-end write the full schema
— that is the current design, and it is what broke.

### D2 — Entry-wise merge for `params` and `last_dirs`

A top-level merge preserves a *key*, not its entries, and two keys hold
sub-dicts assembled from partial sources:

- `params` is built from `build_all_params`, whose dynamic half returns `[]`
  when no skin material is loaded (`params.py:304-306`). Exiting from a
  skin-free scene would still erase every persisted `mtlx.*` value.
- `last_dirs` is written from `last_dirs_snapshot()`, a process-local cache
  seeded once at first dialog use (`settings.py:70-83`).

So these two merge entry-wise; everything else replaces. `last_dirs_snapshot`
and its Qt call site (`ui/qt/app.py:546`) are then deleted — the helper exists
solely as the pre-merge workaround, which `tests/test_last_dirs.py:63-73`
documents. That test is retired with it.

### D3 — Move the three duplicated helpers; do **not** build a declared schema

What is genuinely duplicated is `_snapshot_camera` (20 lines, byte-identical),
the camera restore (~46 lines) and the gizmo restore (~8 lines) — ~74 lines.
The dict literals are *not* duplicated: they are disjoint key sets built from
different sources, 14 lines in `skinny` and almost entirely Qt-specific in
`skinny-gui`.

A declared schema with contributed opaque sections plus a key-set pinning test
would be a registry with two consumers and one implementation, guarding a
failure mode (an unwritten key) that D1 has already made harmless — while
adding a two-place edit for every new key. Move the three helpers; leave each
front-end writing its own dict.

### D4 — Camera restore by deletion, not by a third rule

`OrbitCamera.set_distance` (`renderer.py:1266-1279`) already implements clamp
to ≥0.5 and raise `max_distance` to fit. Both front-ends bypass it with direct
field assignment, and one of them additionally clamps to 50. The fix is to call
the method from both — two divergent rules deleted, none authored.

The Qt clamp guards nothing: the distance slider is declared `growable` and
re-reads live `max_distance` on each pull
(`scene_graph.py:1168-1177`, `ui/qt/windows/scene_graph.py:522-531`).

`_orbit_distance_cap` (`renderer.py:1143-1152`) is *not* the authority here —
it computes the initial ceiling from scene size at load time.

### D5 — Capture everything; add no restore-time filter

`skinny` captures `build_visible_params`, which drops fallback-light params
under authored lighting — permanent loss at capture time. Capture
`build_all_params` in both, i.e. drop the second argument at `app.py:686`.

No restore-time filtering is added, because it could never run: restore happens
before the async scene load, so `uses_default_lights` is still `True` and
`build_visible_params == build_all_params` (`params.py:340-345`). Nor is it
needed — fallback-light params are gated at the point of use
(`renderer.py:2662`, `:2665`, `:2681-2682`), which is already `skinny`'s
behaviour on the restore side (`app.py:595` passes `params=None`).

### D6 — Persistence is read **and** write, or neither

Writing the neural/online-training keys from `skinny-gui` without a restore
path would overwrite `skinny`'s persisted values with argparse defaults
(`auto` / `fp32` / `False`, `cli_common.py:663,677,690`) on every Qt exit —
and merge-on-write does not protect a key that *is* written. The Qt side has a
second hazard: `_neural_handoff_kind` / `_neural_trainer_kind` /
`_train_precision` live on the real renderer, not on `QtRendererProxy`
(`render_session.py:265-300`), so a `renderer.request` timeout would write
defaults.

So: implement both halves — restore mirroring `app.py:559-572`'s CLI/env >
persisted precedence, and a write sourced only from the settled
`renderer.request` future — or correct the help text. The help text is broader
than first stated: `cli_common.py:696` asserts `neural_handoff` transitively
(four keys), and `:569` / `:583` make the same claim for `--backend` and
`--encoding`, which is false for `skinny-web`. D6 must define what "interactive
front-ends" names either way.

Related: `app.py:645-648` downgrades `online_training_requested` to `False`
when the prereq gate refuses, and `:692` persists it. Decide explicitly whether
the persisted value is the user's *intent* or the gated result. Intent is
almost certainly right.

### D7 — Widen the load guard

`load_settings` catches `OSError, json.JSONDecodeError` (`settings.py:56`) but
not `UnicodeDecodeError`, which is what a non-UTF-8 file raises on read. Today
that only bites at startup — Qt guards it with a bare `except Exception`
(`ui/qt/app.py:212`), `app.py:506` does not. After D1 the same exception fires
on the **save** path, where `app.py:698` catches only `OSError`, turning an
exit into a traceback. Catch `(OSError, ValueError)`, which covers both.

### D8 — Delete the dead preset surface

`save_user_preset` and `delete_user_preset` have zero callers, and so do their
private helpers `_sanitize_filename` and `_SAFE_NAME_RE` (`settings.py:121`,
`:124`) — `load_user_presets` does not use them. Four deletions. Fix the stale
`presets.py:31-32` docstring.

## Risks / Trade-offs

- **Risk: merge-on-write preserves a stale key forever.** Acceptable — the
  alternative is deleting keys the writer does not recognise, which is today's
  bug. Add an explicit prune only if a key is ever renamed.
- **Risk: entry-wise merge for `params` keeps a value for a parameter that no
  longer exists.** Restore is `get`-with-default per parameter, so an orphan is
  inert.
- **Risk: D4 changes restored camera distance for Qt users.** Small, visible,
  correct — but see Out of scope: in practice a loaded scene reframes anyway.
- **Trade-off: D3 leaves two dict literals.** They are disjoint and
  front-end-specific; unifying them buys a guard that D1 makes redundant.

## Out of scope

Restored camera state is overwritten by auto-framing whenever a scene loads:
`_frame_camera_to_scene` runs on USD metadata arrival (`renderer.py:4926`), on
streaming completion (`:5002`) and on reload (`:5456-5458`), overwriting
`max_distance`, `distance`, `yaw`, `pitch` and `target`
(`renderer.py:2714-2727`, or `:2797-2810` with an authored camera) — and
restore runs before the async load in both front-ends (`app.py:596`,
`ui/qt/app.py:465-471`). A hostless test at `_apply_saved_camera` level passes
while the app still reframes. Suppressing auto-frame after a restore is a
separate change; the spec scenario here is scoped to the no-scene case so it
does not claim an end-to-end guarantee that does not hold.

## Open Questions

- Should the file gain a schema version? Not yet — merge-on-write makes
  additive changes safe, and there is nothing to migrate. Revisit on the first
  rename.
