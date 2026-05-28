## 1. Registry in settings.py

- [x] 1.1 Add `REPO_ROOT` (= `Path(__file__).resolve().parents[2]`) and `LAST_DIR_DEFAULTS` mapping `model→assets`, `ibl→hdrs`, `lens→lenses`.
- [x] 1.2 Implement `get_last_dir(category) -> str`: lazily seed an in-memory cache from `load_settings()["last_dirs"]`; return remembered dir only if it exists on disk, else the category default if it exists, else `""`.
- [x] 1.3 Implement `record_last_dir(category, directory)`: update cache, then read-modify-write `settings.json` (load → set `last_dirs` → `save_settings`) preserving other keys.
- [x] 1.4 Implement `last_dirs_snapshot() -> dict` returning the current cache for the Qt close-snapshot.

## 2. Shared FilePicker wiring

- [x] 2.1 Add `category: str | None = None` to `FilePicker` in `ui/spec.py` (update docstring).
- [x] 2.2 Qt `_build_file_picker` (`ui/qt/backend.py`): when `category` set, resolve start dir from `get_last_dir(category)` inside the click handler; on successful pick call `record_last_dir(category, path.parent)`. Unchanged when `category` is `None`.
- [x] 2.3 Panel `_build_file_picker` (`ui/panel/backend.py`): same — start `FileSelector` at `get_last_dir(category)`, record parent on successful pick.
- [x] 2.4 `build_app_ui._add_scene_loader`: pass `category="model"` to the file picker.

## 3. Qt-only dialogs

- [x] 3.1 Lens loader (`ui/qt/windows/scene_graph.py:_add_lens_file`): start = `get_last_dir("lens")`; record on success; remove the `parents[4]/"lenses"` anchor.
- [x] 3.2 HDR loader (`ui/qt/windows/scene_graph.py:_add_texture_file`): start = `get_last_dir("ibl")`; record on success; remove the current-file-folder preference and the `parents[4]/"hdrs"` anchor.
- [x] 3.3 "Open scene" menu (`ui/qt/app.py:_on_menu_open_scene`): start = `get_last_dir("model")`; record on success.

## 4. Persistence integration

- [x] 4.1 In `ui/qt/app.py:_snapshot_session_state`, add `out["last_dirs"] = last_dirs_snapshot()` so the close-save does not clobber write-through state.

## 5. Tests

- [x] 5.1 Registry unit tests (no GPU): default fallback when unset; missing remembered dir falls back to default; `record_last_dir` → `get_last_dir` round-trip.
- [x] 5.2 Write-through merge test: `record_last_dir` preserves existing `settings.json` keys; `last_dirs` survives a subsequent full `save_settings(snapshot)`.
- [x] 5.3 `ruff check src/` clean; `pytest` green.
