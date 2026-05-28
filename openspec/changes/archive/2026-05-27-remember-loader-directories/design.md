## Context

File-open dialogs reach the OS picker through two paths:

1. The shared `FilePicker` spec node (`ui/spec.py`), rendered by both the Qt
   backend (`ui/qt/backend.py:438`, `QFileDialog.getOpenFileName`) and the
   Panel/web backend (`ui/panel/backend.py:369`, `FileSelector`). The node
   already has an unused `start_dir: Path | None`. The model/scene loader
   (`build_app_ui._add_scene_loader`) builds one of these but passes no
   `start_dir`.
2. Qt-only direct `QFileDialog` calls: the lens loader and HDR loader in
   `ui/qt/windows/scene_graph.py`, and the "File ▸ Open scene" menu in
   `ui/qt/app.py:340`.

Persistence today lives in `settings.py` (`load_settings`/`save_settings`,
flat dict written atomically to `~/.skinny/settings.json`). Only the Qt app
snapshots/restores it (`app.py` `_snapshot_session_state`/`_restore_session_state`
on `closeEvent`). The Panel front-end never calls `save_settings`, so it has no
shutdown-time persistence hook.

Two current defects: the lens/HDR dialogs anchor their start dir at
`Path(__file__).parents[N] / "lenses"|"hdrs"`, which resolves to `src/lenses`
and `src/hdrs` — directories that do not exist (real assets are at repo root).
The model loader supplies no start dir at all. Both leave the user at cwd.

## Goals / Non-Goals

**Goals:**
- Remember the last-used directory per loader category and reuse it across app
  restarts.
- Three categories: `model`, `ibl`, `lens`. Both model entry points share `model`.
- Sensible first-run defaults: `assets/`, `hdrs/`, `lenses/` at the true repo root.
- Work in both front-ends with one source of truth.
- Survive crashes and Panel's lack of a shutdown hook.

**Non-Goals:**
- Material-graph texture browse and screenshot save dialogs (out of scope).
- A general per-dialog history list (only the single most-recent dir per category).
- Packaging asset dirs for non-source installs — the app already assumes a
  source-tree layout.

## Decisions

### Central registry in `settings.py`
A small set of module-level helpers owns the `last_dirs` state:
- `get_last_dir(category) -> str` — lazily seeds an in-memory cache from
  `load_settings()`, returns the remembered dir **only if it still exists on
  disk**, else the category default, else `""`.
- `record_last_dir(category, directory)` — updates the cache, then read-modify-
  writes `settings.json` (load → set `last_dirs` → `save_settings`).
- `last_dirs_snapshot() -> dict` — returns the cache for inclusion in the Qt
  close-snapshot.

Defaults: `LAST_DIR_DEFAULTS = {"model": REPO_ROOT/"assets", "ibl": REPO_ROOT/"hdrs",
"lens": REPO_ROOT/"lenses"}`, with `REPO_ROOT = Path(__file__).resolve().parents[2]`
(correct from `src/skinny/settings.py`).

*Why settings.py:* it already owns the `settings.json` schema and atomic writes.
Keeping the schema knowledge in one module avoids spreading JSON layout across
the UI. *Alternative — new `ui/last_dirs.py`:* rejected; it would still depend on
`settings.py` for load/save and split ownership of the file format.

### Resolve start dir at click time, not build time
`FilePicker` widgets are built once at startup; a `start_dir` captured then would
go stale after the first pick. Both backends therefore call `get_last_dir(category)`
inside the click/open handler, and `record_last_dir(category, picked.parent)` only
on a successful pick.

### `category` field on `FilePicker`
Add `category: str | None = None` to the dataclass. When set, the backend wires
registry get/record; when `None`, behavior is unchanged (honor existing `start_dir`,
no recording). This keeps the node backend-agnostic and leaves non-categorized
pickers untouched.

### Write-through persistence
`record_last_dir` writes immediately rather than relying on a shutdown hook. This
is the only persistence path for the Panel front-end and makes the value crash-safe.
The Qt close-snapshot rewrites the whole settings dict, so it must include
`last_dirs` (via `last_dirs_snapshot()`) to avoid clobbering the write-through value.

### Fix the broken defaults as part of the work
The lens/HDR dialogs' `src/...` anchors are replaced by registry defaults at the
true repo root. This is a targeted fix in code being modified, not unrelated cleanup.

## Risks / Trade-offs

- [Write-through does a full settings.json read-modify-write on every pick] →
  Picks are rare, human-driven, single-threaded UI events; the file is tiny.
  Negligible cost.
- [Qt close-snapshot could still clobber `last_dirs` if a contributor forgets the
  snapshot field] → Covered by an explicit test asserting the snapshot includes
  `last_dirs` and that a record survives a subsequent full save.
- [Remembered dir deleted/moved between sessions] → `get_last_dir` checks
  existence and falls back to the category default.
- [Repo-root anchor breaks under a non-source (wheel) install where asset dirs
  aren't packaged] → Out of scope; the app already resolves `hdr_dir` the same
  way and assumes a source tree. `get_last_dir` returning a nonexistent default
  degrades gracefully to `""` (OS default dir).

## Open Questions

None — scope, persistence, HDR start-dir precedence, and registry location are
resolved.
