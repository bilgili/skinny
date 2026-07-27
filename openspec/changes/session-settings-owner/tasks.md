# Tasks: session-settings-owner

## 1. Merge on write (the bug fix, land first)

- [ ] 1.1 `save_settings` loads, merges, writes. Preserve the atomic
      tmp-file + `os.replace` behaviour.
- [ ] 1.2 Hostless tests: both erasure directions from the spec; corrupt-file
      case.

## 2. Declared schema

- [ ] 2.1 Add the snapshot module: shared section + contributed sections
      preserved opaquely.
- [ ] 2.2 Move `_snapshot_camera` (identical in both front-ends) and the
      gizmo-mode restore (identical in both) into it.
- [ ] 2.3 Both front-ends contribute instead of authoring the dict; ~110
      duplicated lines removed.
- [ ] 2.4 Hostless test pinning each front-end's contributed key set against
      the declared schema.

## 3. Reconcile the divergences

- [ ] 3.1 One camera restore rule (GLFW rule: raise the cap, no clamp).
      Note the Qt behaviour change in `CHANGELOG.md`.
- [ ] 3.2 Capture the full parameter set in both front-ends; visibility
      filtering moves to display/restore.
- [ ] 3.3 Persist the neural/online-training keys on `skinny-gui` too, or
      correct the `cli_common.py:672,685,697` help text. Prefer persisting.

## 4. Dead surface

- [ ] 4.1 Delete `save_user_preset` / `delete_user_preset` (zero callers) and
      fix the stale `presets.py:31` docstring referring to a removed button.

## 5. Gates

- [ ] 5.1 `ruff check src/`; full hostless `pytest`.
- [ ] 5.2 Manual: launch `skinny`, exit; launch `skinny-gui`, exit; inspect
      `~/.skinny/settings.json` — all keys from both present.
- [ ] 5.3 Restore a pre-change settings file and confirm both front-ends start.
- [ ] 5.4 `CHANGELOG.md` entry — users have been silently losing settings.
- [ ] 5.5 `openspec validate session-settings-owner --strict`.
