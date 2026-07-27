# Tasks: session-settings-owner

## 1. Merge on write (the bug fix, land first)

- [ ] 1.1 `save_settings` loads, merges, writes. Preserve the atomic
      tmp-file + `os.replace`. **Entry-wise** merge for `params` and
      `last_dirs`; replace for every other key.
- [ ] 1.2 Widen the load guard to `(OSError, ValueError)` — covers
      `json.JSONDecodeError` and the currently-uncaught `UnicodeDecodeError`,
      which after 1.1 would fire on the save path and turn an exit into a
      traceback.
- [ ] 1.3 Delete `last_dirs_snapshot` (`settings.py:114`) and its Qt call site
      (`ui/qt/app.py:546`) — the helper exists only as the pre-merge
      workaround. Retire `tests/test_last_dirs.py:63-73`, which documents it.
- [ ] 1.4 Delete the `out.setdefault(...)` fallbacks at `ui/qt/app.py:548-552`:
      on a request timeout they persist proxy-held startup-resolved values over
      the correct live ones, which merge would otherwise have preserved.
- [ ] 1.5 Hostless tests: both erasure directions, the timeout-omission
      direction, both sub-dict cases, and the corrupt-file case (invalid JSON
      **and** invalid UTF-8).

## 2. Move the three duplicated helpers

- [ ] 2.1 Move `_snapshot_camera` (byte-identical, 20 lines), the camera
      restore (~46 lines) and the gizmo-mode restore (~8 lines) into one
      module. ~74 lines, not a schema.
- [ ] 2.2 Leave each front-end writing its own dict — the key sets are disjoint
      and front-end-specific, and 1.1 already makes an unwritten key harmless.
      **No declared schema, no key-set pinning test.**

## 3. Reconcile the divergences

- [ ] 3.1 Both camera restores call `OrbitCamera.set_distance`
      (`renderer.py:1266-1279`) instead of assigning fields — deletes two
      divergent rules, authors none. Note the Qt behaviour change in
      `CHANGELOG.md`.
- [ ] 3.2 Capture the full parameter set: drop the second argument at
      `app.py:686`. Add no restore-time filter.
- [ ] 3.3 Decide the persistence contract and implement one side fully:
      **either** `skinny-gui` gains both a restore (mirroring `app.py:559-572`'s
      CLI/env > persisted precedence) and a write sourced only from the settled
      `renderer.request` future — noting `_neural_handoff_kind` /
      `_neural_trainer_kind` / `_train_precision` are not on `QtRendererProxy` —
      **or** the help text is corrected. Write-only is not an option.
- [ ] 3.4 Fix the help text scope regardless: `cli_common.py:696` asserts
      `neural_handoff` transitively (four keys, not three), and `:569` /
      `:583` make the same claim for `--backend` / `--encoding`, which is false
      for `skinny-web`. Define what "interactive front-ends" names.
- [ ] 3.5 Persist the online-training **request**, not the gated result
      (`app.py:645-648` → `:692` currently persists `False` after a refusal).

## 4. Dead surface

- [ ] 4.1 Delete `save_user_preset`, `delete_user_preset`, and their
      now-unused private helpers `_sanitize_filename` and `_SAFE_NAME_RE`
      (`settings.py:121`, `:124`) — four deletions, zero callers. Fix the stale
      `presets.py:31-32` docstring.

## 5. Gates

- [ ] 5.1 `ruff check src/`; full hostless `pytest`.
- [ ] 5.2 Manual: launch `skinny`, exit; launch `skinny-gui`, exit; inspect
      `~/.skinny/settings.json` — all keys from both present, including dynamic
      params and last dirs.
- [ ] 5.3 Restore a pre-change settings file and confirm both front-ends start.
- [ ] 5.4 `CHANGELOG.md` entry — users have been silently losing settings.
- [ ] 5.5 `openspec validate session-settings-owner --strict`.

## Out of scope

Restored camera state is overwritten by auto-framing whenever a scene loads
(`renderer.py:4926`, `:5002`, `:5456-5458`), and restore runs before the async
load. Suppressing that is a separate change; 3.1's gate is scoped to the
no-scene case so it does not claim a guarantee that does not hold.
