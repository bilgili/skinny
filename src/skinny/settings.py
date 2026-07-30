"""Persistent on-disk settings + user presets, stored under ~/.skinny/.

Layout::

    ~/.skinny/
        settings.json          # window geometry + parameter snapshot
        presets/
            <name>.json        # one user-saved preset per file

`settings.json` is merged with the on-disk file and rewritten atomically on exit
via tmp-file + replace, so a front-end can only add or update its own keys.
Every field is optional — a missing file, missing key, or out-of-range value
falls back to the in-code default so a corrupted/partial settings file can never
brick startup. `skinny.session_snapshot` owns which keys exist.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from skinny.presets import Preset

SETTINGS_DIR = Path.home() / ".skinny"
PRESETS_DIR = SETTINGS_DIR / "presets"
MESH_CACHE_DIR = SETTINGS_DIR / "mesh_cache"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

# Default starting directories for file-open dialogs, per loader category.
# Anchored at the repository root (this file lives at src/skinny/settings.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
LAST_DIR_DEFAULTS: dict[str, Path] = {
    "model": REPO_ROOT / "assets",
    "ibl": REPO_ROOT / "hdrs",
    "lens": REPO_ROOT / "lenses",
}


def ensure_dirs() -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    MESH_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── settings.json ───────────────────────────────────────────────────

def load_settings() -> dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(data: dict[str, Any]) -> None:
    """Merge ``data`` into the settings already on disk, then rewrite atomically.

    Merge, never replace (change session-settings-owner): the two interactive
    front-ends own overlapping-but-different key sets, so a wholesale write let
    each erase the other's keys on exit — closing `skinny-gui` dropped
    `vulkan_window` and the neural keys, closing `skinny` dropped the Qt dock
    layout. A writer that does not know the full schema can now only add or
    update its own keys.

    Nested values are replaced whole, not deep-merged: the writer of a key owns
    that key's entire value. `load_settings` yields {} for a missing or corrupt
    file, so a corrupt file is replaced rather than propagated.
    """
    ensure_dirs()
    merged = load_settings()
    merged.update(data)
    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, sort_keys=True)
    os.replace(tmp, SETTINGS_FILE)


# ── last-used directories per file-loader category ──────────────────

_last_dirs_cache: dict[str, str] | None = None


def _last_dirs() -> dict[str, str]:
    """In-memory category→dir cache, lazily seeded from settings.json."""
    global _last_dirs_cache
    if _last_dirs_cache is None:
        raw = load_settings().get("last_dirs")
        _last_dirs_cache = (
            {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}
            if isinstance(raw, dict)
            else {}
        )
    return _last_dirs_cache


def get_last_dir(category: str) -> str:
    """Remembered directory for ``category`` if it still exists on disk, else
    the category default if it exists, else ``""``. Call at dialog-open time.
    """
    remembered = _last_dirs().get(category)
    if remembered and Path(remembered).is_dir():
        return remembered
    default = LAST_DIR_DEFAULTS.get(category)
    if default is not None and default.is_dir():
        return str(default)
    return ""


def record_last_dir(category: str, directory: str | Path) -> None:
    """Remember ``directory`` for ``category`` and write it through to disk,
    preserving every other settings key.
    """
    directory = str(directory)
    _last_dirs()[category] = directory
    data = load_settings()
    last = data.get("last_dirs")
    if not isinstance(last, dict):
        last = {}
    last[category] = directory
    data["last_dirs"] = last
    save_settings(data)


def last_dirs_snapshot() -> dict[str, str]:
    """Current category→dir map, for inclusion in a full settings snapshot."""
    return dict(_last_dirs())


# ── user presets ────────────────────────────────────────────────────

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\- ]+")


def _sanitize_filename(name: str) -> str:
    name = _SAFE_NAME_RE.sub("_", name).strip().strip(".")
    return name or "preset"


def load_user_presets() -> list[Preset]:
    if not PRESETS_DIR.exists():
        return []
    presets: list[Preset] = []
    for path in sorted(PRESETS_DIR.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            name = str(raw.get("name") or path.stem)
            values = raw.get("values") or {}
            if not isinstance(values, dict):
                continue
            clean = {
                str(k): float(v)
                for k, v in values.items()
                if isinstance(v, (int, float))
            }
            presets.append(Preset(name=name, values=clean, is_builtin=False))
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            # Ignore malformed preset files rather than bailing the whole load.
            continue
    return presets


# `save_user_preset` / `delete_user_preset` lived here with zero callers in
# `src/` or `tests/` — write surface for a Tk Delete button that no longer
# exists (change session-settings-owner). Deleted; each is eight lines if the
# button returns. Presets are still read: drop a JSON file in ~/.skinny/presets/
# and `load_user_presets` picks it up.
