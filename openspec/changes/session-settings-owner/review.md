# Design review — session-settings-owner

Adversarial review, 2026-07-27, against the tree at `8247148`.

**Note: this is the one change whose findings were folded in place**, before the
decision to keep the originals and append reviews. The proposal, design, spec
and tasks in this directory are the **post-review** versions. This file records
what the review found and what changed, so the delta is not lost.

**Verdict: survives; scope cut roughly in half; the bug is worse than stated.**

## Confirmed as originally claimed

- `save_settings` is a wholesale write, no merge — `settings.py:60-65`.
- The key sets and the 6-key intersection are exact. Nothing re-adds a lost key:
  every restore is `data.get(k)`-with-default, so an erased key silently becomes
  the default. Only `last_dirs` partially self-heals, one category at a time,
  when the user next opens a file dialog (`settings.py:99-111`).
- The camera divergence (`app.py:86-87` vs `ui/qt/app.py:635`).
- `save_user_preset` / `delete_user_preset`: zero callers in `src/` and `tests/`;
  `presets.py:31-32` docstring stale.

## MAJOR (all folded)

**M1 — A top-level merge does not protect the two keys that hold the state.**
`params` and `last_dirs` are sub-dicts written as single opaque values; a
shallow merge preserves the *key*, not the entries. `build_dynamic_params`
returns `[]` when no skin material is loaded (`params.py:304-306`), so exiting
`skinny-gui` on a skin-free scene still erases every persisted `mtlx.*` value.
`last_dirs_snapshot` is a process-local cache seeded once at first dialog use
(`settings.py:70-83`), so a concurrent `skinny-web` session is clobbered.
→ Folded: entry-wise merge for those two keys; delete `last_dirs_snapshot` and
`ui/qt/app.py:546`; retire `tests/test_last_dirs.py:63-73`, which documents the
workaround.

**M2 — Task 3.3 as originally worded re-created the bug in value form.**
"Persist the neural keys on `skinny-gui` too" specifies a *write*; `skinny-gui`
has no restore path (`ui/qt/app.py:686-715` passes argparse defaults `auto` /
`fp32` / `False` straight through). Writing without reading overwrites
`skinny`'s values on every Qt exit, and merge cannot protect a key that *is*
written. Also `_neural_handoff_kind` / `_neural_trainer_kind` /
`_train_precision` are not on `QtRendererProxy` (`render_session.py:265-300`),
so a timeout would write defaults. → Folded: read-and-write or neither.

**M3 — D3 named the wrong authority.** `_orbit_distance_cap`
(`renderer.py:1143-1152`) computes the *initial* ceiling at load time. The
actual authority is `OrbitCamera.set_distance` (`:1266-1279`), which already
implements the GLFW rule; both restores bypass it with field assignment. The Qt
clamp guards nothing — the distance slider is `growable` and re-reads live
`max_distance` on each pull (`scene_graph.py:1168-1177`,
`ui/qt/windows/scene_graph.py:522-531`). → Folded: the fix is a deletion of two
rules, not the authoring of a third.

**M4 — The "wide orbit distance survives" scenario is false end-to-end whenever
a scene loads.** `_frame_camera_to_scene` runs on USD metadata arrival
(`renderer.py:4926`), on streaming completion (`:5002`) and on reload
(`:5456-5458`), overwriting `max_distance`, `distance`, `yaw`, `pitch`, `target`
(`:2714-2727`, or `:2797-2810` with an authored camera) — and restore runs
before the async load in both front-ends (`app.py:596`, `ui/qt/app.py:465-471`).
→ Folded: the scenario is scoped to the no-scene case, and suppressing
auto-frame is recorded as out of scope.

**M5 — D4's "filter on restore" was a no-op.** Restore runs before the scene
loads, so `uses_default_lights` is still `True` and `build_visible_params ==
build_all_params` (`params.py:340-345`). Nor is a filter needed: fallback-light
params are gated at use (`renderer.py:2662`, `:2665`, `:2681-2682`). → Folded:
collapses to dropping the second argument at `app.py:686`.

**M6 — D2 was unearned; "~110 duplicated lines" was overstated.** The proposal
itself said D1 alone fixes the erasure. What is genuinely duplicated is three
helpers totalling ~74 lines. The dict literals are *not* duplicated — disjoint
key sets from different sources. A declared schema with pinned key sets would be
a registry with two consumers and one implementation, guarding a failure mode D1
has already made harmless. → Folded: move the three helpers; no schema, no
pinning test.

**M7 — The corrupt-file scenario was not free.** `load_settings` catches
`OSError, json.JSONDecodeError` (`settings.py:56`) but not `UnicodeDecodeError`.
Today that only bites at startup; after merge-on-write the same exception fires
on the **save** path, where `app.py:698` catches only `OSError` — turning an
exit into a traceback. → Folded: catch `(OSError, ValueError)`.

## MINOR (all folded)

- `record_last_dir` is called from the Qt UI (`ui/qt/backend.py:458`,
  `ui/qt/windows/scene_graph.py:303,691,731`) and the Panel/web UI
  (`ui/panel/backend.py:458`, `ui/panel/windows.py:123`) — never from `skinny`.
  So `skinny-web` *does* mutate `settings.json`, though it persists no session
  snapshot.
- The help-text claim is broader than stated: `cli_common.py:696` asserts
  `neural_handoff` transitively (four keys), and `:569` / `:583` make the same
  claim for `--backend` / `--encoding`, false for `skinny-web`.
- **A third erasure direction:** on a `renderer.request` timeout,
  `ui/qt/app.py:521-534` omits `params`, `camera` and `gizmo_mode` entirely.
  Merge fixes it. Related: delete the `out.setdefault(...)` fallbacks at
  `:548-552` — they persist proxy-held startup-resolved values over the correct
  live ones.
- `app.py:645-648` downgrades `online_training_requested` to `False` when the
  prereq gate refuses, and `:692` persists it — one launch without a neural
  proposal silently drops the opt-in. Persist the intent.
- D6 was incomplete: `_sanitize_filename` and `_SAFE_NAME_RE`
  (`settings.py:121`, `:124`) are used only by the two functions being deleted.
  Four deletions.
- Spec structure: the persistence scenario was filed under the camera/parameter
  requirement. → Folded: D5/D6 now has its own requirement.
