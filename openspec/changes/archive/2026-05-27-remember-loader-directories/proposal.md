## Why

Every file-open dialog (model/scene, IBL/HDR, lens) starts from a fixed or
broken location, so the user must re-navigate to their working folder on every
load. The model loader passes no start directory at all, and the lens/HDR
dialogs point at `src/lenses` and `src/hdrs` — paths that do not exist (the real
asset dirs are at repo root), so Qt silently falls back to the process cwd.

## What Changes

- Add a small persisted registry that remembers the last-used directory per
  loader category (`model`, `ibl`, `lens`) in `~/.skinny/settings.json`.
- File-open dialogs start from the remembered directory, resolved at click time;
  after a successful pick the directory is recorded.
- First-ever use (or a remembered dir that no longer exists) falls back to a
  type-specific default: `assets/` for models, `hdrs/` for IBL, `lenses/` for
  lenses — anchored at the true repo root (fixes the current broken defaults).
- The two model entry points — the sidebar "Load scene…" picker and the
  "File ▸ Open scene" menu — share one `model` memory.
- Persistence is write-through on each pick, so it survives crashes and the
  Panel/web front-end (which has no shutdown hook).
- Applies to both front-ends: the shared `FilePicker` spec node (Qt + Panel) and
  the Qt-only lens/HDR/Open-scene dialogs.

## Capabilities

### New Capabilities
- `loader-directory-memory`: persisting and recalling the last-used directory
  per file-loader category, with type-specific default fallbacks, shared across
  front-ends.

### Modified Capabilities
<!-- none: no existing specs -->

## Impact

- `src/skinny/settings.py` — new `last_dirs` schema key + registry helpers
  (`get_last_dir`, `record_last_dir`, `last_dirs_snapshot`) and repo-root-anchored
  default dirs.
- `src/skinny/ui/spec.py` — `FilePicker` gains a `category` field.
- `src/skinny/ui/qt/backend.py` + `src/skinny/ui/panel/backend.py` — file-picker
  builders resolve start dir from the registry and record on pick.
- `src/skinny/ui/build_app_ui.py` — scene loader passes `category="model"`.
- `src/skinny/ui/qt/windows/scene_graph.py` — lens + HDR dialogs use the
  registry; drop broken `src/...` defaults.
- `src/skinny/ui/qt/app.py` — "Open scene" menu uses the registry; close-snapshot
  includes `last_dirs` to avoid clobbering write-through state.
- New unit tests for the registry. No shader, GPU, or rendering changes.
