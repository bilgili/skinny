"""Render-thread-safety guard for the Material Graph dock (Phase 4).

Every MaterialX-doc read/mutation runs on the render worker (the doc lives on the
renderer): build_view, topology edits, and the material preview all dispatch via
`renderer.request(...)` and marshal results to the GUI thread. Source-level,
mirroring `test_qt_gizmo_viewport.py` — the doc-edit path itself needs the
MaterialX-enabled py3.13 env + interactive GPU to fully exercise.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from skinny.ui.qt.windows import material_graph as mg  # noqa: E402

Dock = mg.MaterialGraphDock


def test_dock_defines_marshaller_signal() -> None:
    assert hasattr(Dock, "_run_on_gui")


def test_doc_helpers_are_worker_side_module_functions() -> None:
    # The doc helpers take the real renderer/doc, not `self` — so they can run
    # inside worker closures, never touching the GUI-side proxy's None library.
    assert callable(mg._worker_doc)
    assert callable(mg._worker_mtlx_node)
    assert callable(mg._set_mtlx_input)
    assert callable(mg._build_view_on_worker)


def test_topology_edits_run_on_the_worker() -> None:
    runner = inspect.getsource(Dock._run_edit)
    assert "self.renderer.request(" in runner
    assert "_worker_doc(r)" in runner
    assert "build_view(" in runner
    # Every edit method funnels through _run_edit (or the flat fast path).
    for method in ("_apply_value_edit", "_apply_connect", "_apply_disconnect",
                   "_apply_delete_node", "_apply_add_node"):
        body = inspect.getsource(getattr(Dock, method))
        assert ("self._run_edit(" in body) or ("apply_material_override(" in body), method


def test_build_view_runs_on_the_worker() -> None:
    picked = inspect.getsource(Dock._on_material_picked)
    assert "self.renderer.request(" in picked
    assert "_build_view_on_worker" in picked


def test_preview_runs_on_the_worker() -> None:
    src = inspect.getsource(Dock._render_preview)
    assert "self.renderer.render_material_preview(" in src
    assert "self._resolve_to_gui(" in src
    # `.result()` runs only inside the worker-side done-callback in
    # `_resolve_to_gui`, never on the GUI thread.
    assert inspect.getsource(Dock._resolve_to_gui).count(".result()") == 1
    assert inspect.getsource(mg).count(".result()") == 1


def test_poll_refreshes_scene_state_on_the_worker() -> None:
    poll = inspect.getsource(Dock._poll_scene_swap)
    assert "self.renderer.refresh_scene_state()" in poll
