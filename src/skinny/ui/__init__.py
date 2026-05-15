"""Backend-agnostic UI description for Skinny.

`spec` defines a programmatic widget-tree builder + dataclass nodes. Two
backends consume the tree:

- ``skinny.ui.qt`` — PySide6 desktop (`skinny-gui` entry).
- ``skinny.ui.panel`` — HoloViz Panel for the browser (`skinny-web` entry).

`build_app_ui.build_main_ui(renderer)` is the single source of truth: any
new control is added there once and shows up in both backends.
"""
