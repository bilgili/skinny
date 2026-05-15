"""PySide6 desktop backend for the Skinny UI spec.

Entry point: ``skinny.ui.qt.app:main`` (registered in pyproject as
``skinny-gui``). Walks the tree returned by ``build_main_ui`` and
instantiates Qt widgets; renders the scene to a ``QImage``-backed
``RenderViewport`` driven by a background ``QThread``.
"""
