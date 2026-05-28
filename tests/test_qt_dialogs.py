"""macOS native dialogs ignore the initial directory, so loaders must force the
non-native dialog there. Guard that platform gating."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog  # noqa: E402

from skinny.ui.qt import dialogs  # noqa: E402


def test_macos_forces_non_native(monkeypatch):
    monkeypatch.setattr(dialogs.sys, "platform", "darwin")
    assert dialogs._open_options() == QFileDialog.Option.DontUseNativeDialog


def test_other_platforms_keep_native(monkeypatch):
    for plat in ("linux", "win32"):
        monkeypatch.setattr(dialogs.sys, "platform", plat)
        assert dialogs._open_options() == QFileDialog.Option(0)
