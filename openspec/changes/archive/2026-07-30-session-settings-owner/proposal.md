# Change: session-settings-owner

## Why

**This is a live data-loss bug, not only a shape problem.**

`settings.save_settings(data)` (`settings.py:60`) writes the dict **wholesale**
— `json.dump` over a fresh file, no merge with what is on disk. Two front-ends
call it with independently authored key sets:

- `skinny` (`app.py:683-696`) writes 11 keys: `backend`, `vulkan_window`,
  `params`, `camera`, `gizmo_mode`, `neural_handoff`, `neural_trainer`,
  `train_precision`, `online_training`, `encoding`, `sppm_glossy_roughness`.
- `skinny-gui` (`ui/qt/app.py:517-566`) writes 11 keys: `params`, `camera`,
  `gizmo_mode`, `encoding`, `sppm_glossy_roughness`, `open_docks`, `last_dirs`,
  `backend`, `section_states`, `qt_geometry`, `qt_dock_state`.

The intersection is 6. So **closing `skinny-gui` erases** `vulkan_window`,
`neural_handoff`, `neural_trainer`, `train_precision` and `online_training`,
and **closing `skinny` erases** `open_docks`, `last_dirs`, `section_states`,
`qt_geometry` and `qt_dock_state`. `record_last_dir` (`settings.py:99`) is the
only writer that loads-then-merges, so the web path is the only safe one.

Related divergences from the same duplication:

- `_snapshot_camera` is byte-identical in both files (20 lines each), but the
  restores are not: `app.py:86-87` does `max(0.5, …)` and raises
  `max_distance`, while `ui/qt/app.py:635` does `np.clip(…, 0.5, 50.0)` and
  never touches `max_distance`. A persisted orbit distance above 50 restores
  correctly in one front-end and is silently clamped in the other.
- The snapshotted **parameter set** differs: `skinny` snapshots
  `build_visible_params(renderer)`, which filters out fallback-light params
  when the scene has authored lighting, so under an authored-USD-lighting
  scene it silently drops `env_index`, `env_intensity`, `direct_light_index`
  and `light_*`. `skinny-gui` snapshots `build_all_params(renderer)` and keeps
  them.
- `cli_common.py:672`, `:685`, `:697` document `--neural-trainer`,
  `--train-precision` and `--online-training` as "persisted on the interactive
  front-ends". Only `skinny` implements it; `skinny-gui` neither reads nor
  writes those keys — and erases them.
- `settings.save_user_preset` / `delete_user_preset` (`settings.py:153-177`)
  have **zero callers** in `src/` or `tests/`; the docstring at
  `presets.py:31` still refers to a removed Tk Delete button.

There is **no test anywhere** for the settings snapshot or restore, for either
writer.

## What Changes

- Add one module owning the session snapshot: capture and restore, with the
  schema declared once and front-ends contributing their own keys rather than
  re-authoring the whole dict.
- **Merge on write.** Keys the writing front-end does not own are preserved
  from disk. This alone fixes the mutual erasure.
- One camera restore rule, replacing the two that disagree; the resulting
  behaviour is chosen deliberately (see design) rather than inherited from
  whichever front-end wrote last.
- One decision about which parameter set is snapshotted, applied to both
  front-ends.
- Make the persistence documented in `cli_common` true: either `skinny-gui`
  persists the neural/online-training keys too, or the help text stops
  claiming it does.
- Delete `save_user_preset` / `delete_user_preset` — dead surface by the
  deletion test — or wire them, if the removed Delete button is coming back.
- Add hostless tests for capture, restore, merge-on-write, and the key sets of
  both front-ends. No GPU needed for any of it.

## Capabilities

### New Capabilities

- `session-settings`: one owner for the persisted session snapshot — declared
  schema, merge-on-write, one camera restore rule, one parameter-set rule, and
  hostless tests for both front-ends' key sets.

## Impact

- New: `src/skinny/session_snapshot.py` (or an expanded `settings.py`),
  hostless test module.
- Modified: `src/skinny/settings.py` (merge-on-write), `src/skinny/app.py`
  (~110 duplicated lines removed), `src/skinny/ui/qt/app.py`, and
  `src/skinny/cli_common.py` help text if the persistence claim changes.
- **User-visible**: an existing `~/.skinny/settings.json` keeps working; keys
  previously erased now survive. Call this out in `CHANGELOG.md` — users have
  been silently losing settings.
- Unchanged: the settings file location and format, preset files, the
  `last_dirs` merge path.
- Independent of every other change in this set; land it any time.
