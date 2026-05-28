"""Persistent on-disk settings + user presets, stored under ~/.skinny/.

Layout::

    ~/.skinny/
        settings.json          # window geometry + parameter snapshot
        presets/
            <name>.json        # one user-saved preset per file

`settings.json` is rewritten atomically on exit via tmp-file + replace. Every
field is optional — a missing file, missing key, or out-of-range value falls
back to the in-code default so a corrupted/partial settings file can never
brick startup.
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
    ensure_dirs()
    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
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


def save_user_preset(name: str, values: dict[str, float]) -> Path:
    ensure_dirs()
    safe = _sanitize_filename(name)
    path = PRESETS_DIR / f"{safe}.json"
    payload = {
        "name": name,
        "values": {k: float(v) for k, v in values.items()},
    }
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def delete_user_preset(name: str) -> bool:
    safe = _sanitize_filename(name)
    path = PRESETS_DIR / f"{safe}.json"
    if path.exists():
        try:
            path.unlink()
            return True
        except OSError:
            return False
    return False
