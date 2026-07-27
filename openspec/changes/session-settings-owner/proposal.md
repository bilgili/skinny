# Change: session-settings-owner

## Why

**This is a live data-loss bug, not only a shape problem.**

`settings.save_settings(data)` (`settings.py:60-65`) writes the dict
**wholesale** — `json.dump` over a fresh file, no merge with what is on disk.
Two front-ends call it with independently authored key sets:

- `skinny` (`app.py:683-696`) writes 11 keys: `backend`, `vulkan_window`,
  `params`, `camera`, `gizmo_mode`, `neural_handoff`, `neural_trainer`,
  `train_precision`, `online_training`, `encoding`, `sppm_glossy_roughness`.
- `skinny-gui` (`ui/qt/app.py:521-565`) writes 11 keys: `params`, `camera`,
  `gizmo_mode`, `encoding`, `sppm_glossy_roughness`, `open_docks`, `last_dirs`,
  `backend`, `section_states`, `qt_geometry`, `qt_dock_state`.

The intersection is 6. Every restore is `data.get(key)`-with-default, so an
erased key silently becomes the default — nothing re-adds it. **Closing
`skinny-gui` erases** `vulkan_window`, `neural_handoff`, `neural_trainer`,
`train_precision` and `online_training`; **closing `skinny` erases**
`open_docks`, `last_dirs`, `section_states`, `qt_geometry` and `qt_dock_state`.

There is a **third erasure direction**: on a `renderer.request` timeout,
`ui/qt/app.py:521-534` omits `params`, `camera` and `gizmo_mode` from the dict
entirely, and the wholesale write then wipes all three.

**A top-level merge is not sufficient.** Two keys hold sub-dicts written as
single opaque values, so preserving the key does not preserve its entries:

- `params` — `build_all_params` includes `build_dynamic_params`, which returns
  `[]` whenever no skin material is loaded (`params.py:304-306`). Open
  `skinny-gui` on a pbrt or USD scene with no skin material, exit, and every
  persisted dynamic `mtlx.*` parameter is gone.
- `last_dirs` — `ui/qt/app.py:546` writes `last_dirs_snapshot()`, a
  process-local cache seeded once from disk at first dialog use
  (`settings.py:70-83`). A concurrent `skinny-web` session's recorded directory
  is clobbered on Qt exit. `last_dirs_snapshot` exists *only* as a pre-merge
  workaround, documented as such by `tests/test_last_dirs.py:63-73`.

Related divergences from the same duplication:

- `_snapshot_camera` is byte-identical in both front-ends (20 lines each,
  `app.py:39-58` / `ui/qt/app.py:585-604`), but the restores are not:
  `app.py:86-87` does `max(0.5, …)` and raises `max_distance`, while
  `ui/qt/app.py:635` does `np.clip(…, 0.5, 50.0)` and never touches
  `max_distance`. Both **bypass `OrbitCamera.set_distance`**
  (`renderer.py:1266-1279`), which already implements the first rule.
- `app.py:686` snapshots `build_visible_params(renderer)` where
  `ui/qt/app.py:522` snapshots `build_all_params(renderer)` — a difference in
  what is *captured*, i.e. permanent loss, not a display filter.
- `cli_common.py` claims `--backend` (`:569`), `--encoding` (`:583`),
  `--neural-handoff`, `--neural-trainer`, `--train-precision` and
  `--online-training` (`:663-696`) are "persisted on the interactive
  front-ends". Only `skinny` implements the neural group — and `skinny-web` is
  an interactive front-end that persists nothing by design.
- `app.py:645-648` downgrades `online_training_requested` to `False` when the
  prereq gate refuses, and `:692` persists that `False` — one launch without a
  neural proposal silently drops the user's opt-in.
- `settings.save_user_preset` / `delete_user_preset` (`:153`, `:168`) have
  **zero callers** in `src/` or `tests/`, as do their private helpers
  `_sanitize_filename` and `_SAFE_NAME_RE` (`:121`, `:124`). The docstring at
  `presets.py:31-32` still refers to a removed Tk Delete button.

There is **no test anywhere** for the settings snapshot or restore.

## What Changes

- **Merge on write**, with entry-wise merge for `params` and `last_dirs` and
  replacement for every other key. Delete `last_dirs_snapshot` and its Qt call
  site — the workaround it embodies is what merge-on-write removes.
- Widen the load guard to `(OSError, ValueError)` so a non-UTF-8 settings file
  raises `UnicodeDecodeError` inside the guard rather than turning an exit into
  a traceback on the new save path.
- Move the three genuinely duplicated helpers — `_snapshot_camera`, the camera
  restore, the gizmo-mode restore (~74 lines) — into one module. Each front-end
  keeps writing its own dict; the key sets are disjoint and front-end-specific,
  and merge-on-write already makes an unwritten key harmless.
- One camera restore rule by **deletion**: both restores call the existing
  `OrbitCamera.set_distance`, removing the two divergent hand-rolled rules.
- Capture the full parameter set in `skinny` — drop the second argument at
  `app.py:686`. No restore-time filtering is added; fallback-light parameters
  are already inert under authored lighting at the point of use
  (`renderer.py:2662`, `:2665`, `:2681-2682`).
- Delete `ui/qt/app.py:548-552`'s `setdefault` fallbacks: on a timeout they
  persist proxy-held CLI-resolved values over the correct live ones, which
  merge-on-write would otherwise have preserved.
- Make the documented persistence true: either implement **read and write** for
  the neural/online-training keys on `skinny-gui`, or correct the help text.
  Write-only is not an option — it would overwrite `skinny`'s values with
  argparse defaults on every Qt exit.
- Delete the dead preset surface (`save_user_preset`, `delete_user_preset`,
  `_sanitize_filename`, `_SAFE_NAME_RE`) and fix the stale docstring.
- Add hostless tests for merge, both erasure directions, the timeout-omission
  direction, the sub-dict cases, and the corrupt-file case. No GPU needed.

## Capabilities

### New Capabilities

- `session-settings`: one owner for persisted session state — merge-on-write
  including sub-dict entries, one camera restore rule, one captured parameter
  set, and a truthful persistence contract, with hostless tests where there are
  none today.

## Impact

- Modified: `src/skinny/settings.py` (merge-on-write, widened guard, dead
  surface removed), `src/skinny/app.py`, `src/skinny/ui/qt/app.py`, a new small
  module for the three shared camera/gizmo helpers, and
  `src/skinny/cli_common.py` help text.
- **User-visible**: keys previously erased now survive. Note in `CHANGELOG.md` —
  users have been silently losing settings.
- Unchanged: settings file location and format, preset files, the
  `record_last_dir` path (which is called from the Qt and Panel UIs, not from
  `skinny`).
- Retire `tests/test_last_dirs.py:63-73`, which exists to document the
  wholesale-write workaround being removed.
- Independent of every other change in this set; land it first.

## Out of scope

Restored camera state is overwritten by auto-framing whenever a scene loads
(`renderer.py:4926`, `:5002`, `:5456-5458` → `_frame_camera_to_scene`), and
restore runs before the async load in both front-ends (`app.py:596`,
`ui/qt/app.py:465-471`). So persisted camera state only survives when no scene
loads. Suppressing auto-frame after a restore is a separate, deliberate change;
this one does not silently acquire it.
