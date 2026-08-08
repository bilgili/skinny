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


# ── Behavioural: input rows route edits through _apply_value_edit ──────
#
# The per-type input-row switch moved to the shared `graph_input_to_node` mapper
# (change ui-spec-scene-properties); these exercise the migrated `_refresh_side`
# without a live MaterialX doc by driving a hand-built view. Bypass the heavy
# dock `__init__` with `__new__` and supply only what `_refresh_side` reads.


def _bare_dock(view, selected):
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    dock = Dock.__new__(Dock)
    dock._input_builder = None
    dock._view = view
    dock._selected_node = selected
    dock._side_title = QLabel()
    host = QWidget()
    dock._side_host = host
    dock._side_form = QVBoxLayout(host)
    dock._side_form.addStretch(1)  # `_add_side_widget` inserts before this
    dock._edits = []
    dock._apply_value_edit = lambda node, port, value: dock._edits.append(
        (node.name, port.name, value)
    )
    return dock


def _view_with(*ports):
    from skinny.mtlx_graph_view import NodeGraphView, NodeView

    node = NodeView(name="N", category="noise", inputs=list(ports))
    return NodeGraphView(
        material_id=0, material_name="M", target_name="T", nodes=[node],
        flat=False, structural_signature="sig",
    ), node


def test_material_input_float_routes_edit_through_apply_value_edit() -> None:
    from PySide6.QtWidgets import QApplication, QDoubleSpinBox

    from skinny.mtlx_graph_view import PortView

    app = QApplication.instance() or QApplication([])
    view, _node = _view_with(PortView(name="amp", type_name="float", value=0.25))
    dock = _bare_dock(view, "N")
    try:
        dock._refresh_side()
        host = dock._input_builder.parent
        spins = host.findChildren(QDoubleSpinBox)
        assert spins  # an editable float control, not a read-only label
        spins[0].setValue(0.5)
        assert dock._edits and dock._edits[-1] == ("N", "amp", 0.5)
    finally:
        if dock._input_builder is not None:
            dock._input_builder._timer.stop()
        app.processEvents()


def test_material_connected_input_is_readonly_no_edit() -> None:
    from PySide6.QtWidgets import QApplication, QDoubleSpinBox

    from skinny.mtlx_graph_view import PortView

    app = QApplication.instance() or QApplication([])
    view, _node = _view_with(
        PortView(name="in", type_name="float", value=0.0,
                 connected_from=("Up", "out")),
    )
    dock = _bare_dock(view, "N")
    try:
        dock._refresh_side()
        host = dock._input_builder.parent
        # A wired input shows a read-only label, no editable spinbox.
        assert not host.findChildren(QDoubleSpinBox)
    finally:
        if dock._input_builder is not None:
            dock._input_builder._timer.stop()
        app.processEvents()
