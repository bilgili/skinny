"""Shared Qt file-dialog helpers.

On macOS the native file dialog ignores the initial-directory (and filter)
arguments — it always reopens at its own last-visited location. To honour the
last-used-directory registry we must force Qt's non-native dialog there. Other
platforms keep their native dialogs, which already respect the initial directory.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QFileDialog, QWidget


def _open_options() -> QFileDialog.Option:
    if sys.platform == "darwin":
        return QFileDialog.Option.DontUseNativeDialog
    return QFileDialog.Option(0)


def get_open_file_name(
    parent: QWidget | None, caption: str, directory: str, filt: str,
) -> str:
    """``QFileDialog.getOpenFileName`` that honours ``directory`` on every
    platform (forces the non-native dialog on macOS, where the native one
    ignores it). Returns the chosen path, or an empty string if cancelled.
    """
    path, _ = QFileDialog.getOpenFileName(
        parent, caption, directory, filt, options=_open_options(),
    )
    return path
