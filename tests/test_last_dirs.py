"""Last-used-directory registry in skinny.settings."""

from __future__ import annotations

import json

import pytest

from skinny import settings


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Redirect settings.json to a tmp dir and reset the in-memory cache."""
    monkeypatch.setattr(settings, "SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(settings, "PRESETS_DIR", tmp_path / "presets")
    monkeypatch.setattr(settings, "MESH_CACHE_DIR", tmp_path / "mesh_cache")
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(settings, "_last_dirs_cache", None)
    return tmp_path


def test_default_when_unset(tmp_settings):
    # Defaults point at real repo asset dirs, which exist.
    assert settings.get_last_dir("ibl") == str(settings.LAST_DIR_DEFAULTS["ibl"])
    assert settings.get_last_dir("model") == str(settings.LAST_DIR_DEFAULTS["model"])


def test_unknown_category_returns_empty(tmp_settings):
    assert settings.get_last_dir("nope") == ""


def test_record_then_get_roundtrip(tmp_settings, tmp_path):
    target = tmp_path / "work"
    target.mkdir()
    settings.record_last_dir("model", target)
    assert settings.get_last_dir("model") == str(target)


def test_missing_remembered_falls_back_to_default(tmp_settings, tmp_path):
    settings.record_last_dir("lens", tmp_path / "gone")  # never created
    assert settings.get_last_dir("lens") == str(settings.LAST_DIR_DEFAULTS["lens"])


def test_record_persists_across_cache_reset(tmp_settings, tmp_path, monkeypatch):
    target = tmp_path / "ibls"
    target.mkdir()
    settings.record_last_dir("ibl", target)
    # Simulate a fresh launch: drop the cache, reload from disk.
    monkeypatch.setattr(settings, "_last_dirs_cache", None)
    assert settings.get_last_dir("ibl") == str(target)


def test_write_through_preserves_other_keys(tmp_settings, tmp_path):
    settings.save_settings({"params": {"a": 1.0}, "camera": {"fov": 50}})
    settings.record_last_dir("model", tmp_path)
    data = json.loads((tmp_settings / "settings.json").read_text())
    assert data["params"] == {"a": 1.0}
    assert data["camera"] == {"fov": 50}
    assert data["last_dirs"]["model"] == str(tmp_path)


def test_snapshot_survives_full_save(tmp_settings, tmp_path, monkeypatch):
    """A full settings overwrite (Qt closeEvent) must not drop last_dirs when
    it includes last_dirs_snapshot()."""
    target = tmp_path / "scenes"
    target.mkdir()
    settings.record_last_dir("model", target)
    # Qt close path: rewrite the whole dict, including the snapshot.
    snapshot = {"params": {}, "last_dirs": settings.last_dirs_snapshot()}
    settings.save_settings(snapshot)
    monkeypatch.setattr(settings, "_last_dirs_cache", None)
    assert settings.get_last_dir("model") == str(target)
