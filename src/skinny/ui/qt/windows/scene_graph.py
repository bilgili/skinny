"""Qt port of ``scene_graph_window.SceneGraphWindow``.

Tree view (left) + property editor (right) for the USD scene graph.
Selecting a node rebuilds the right pane with typed widgets for the
node's editable properties; edits route through the renderer's existing
``apply_*`` API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDockWidget, QHBoxLayout, QLabel, QMenu, QPushButton, QScrollArea, QSplitter, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from skinny.scene_graph import (
    SceneGraphNode, SceneGraphProperty, find_node_by_path, type_icon,
)
from skinny.settings import get_last_dir, record_last_dir
from skinny.ui import spec
from skinny.ui.qt.backend import QtTreeBuilder
from skinny.ui.qt.dialogs import get_open_file_name
from skinny.ui.scene_edit_actions import (
    SUPPORTED_LIGHT_TYPES,
    add_parent_for_node,
    apply_scene_property,
    has_editable_stage,
    is_deletable,
)
from skinny.ui.scene_property_nodes import scene_property_to_node

_USD_PICKER_FILTER = "USD (*.usda *.usdc *.usdz);;All files (*)"


class SceneGraphDock(QDockWidget):
    """Non-modal dock with a tree view + property editor. Mirrors the
    behaviour of the legacy Tk window.
    """

    TICK_MS = 200

    # Marshals a callable emitted from a render-worker future-callback onto the
    # GUI thread (Qt delivers a queued cross-thread signal). All async results
    # (scene-state refresh, add/save/delete/texture/lens) route through it.
    _run_on_gui = Signal(object)

    def __init__(
        self, renderer,
        parent: QWidget | None = None,
        *,
        on_open_python_material: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__("Scene Graph", parent)
        self.renderer = renderer
        self._on_open_python_material = on_open_python_material
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self._run_on_gui.connect(self._invoke_on_gui)
        self._state_inflight = False

        self._last_graph_id: int = -1
        self._last_graph_version: int = -1
        self._selected_path: str | None = None
        # Live "pull" callbacks for the active property widgets — refresh
        # them from the camera each tick so external orbit/zoom shows up.
        self._pulls: list[Callable[[], None]] = []
        # The embedded backend builder that renders the selected node's property
        # nodes; owns its own pull timer, disposed on the next selection.
        self._prop_builder: QtTreeBuilder | None = None

        # Container: an editing toolbar across the top, then a vertical
        # splitter (tree above, property editor below).
        container = QWidget()
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setWidget(container)

        # ── Editing toolbar ──
        toolbar = QWidget()
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(4, 4, 4, 2)
        tb_layout.setSpacing(4)
        self._add_btn = QPushButton("Add model…")
        self._add_btn.setToolTip("Reference a USD file under the selected group (or /World)")
        self._add_btn.clicked.connect(self._on_add_model)
        self._add_light_btn = QToolButton()
        self._add_light_btn.setText("Add light")
        self._add_light_btn.setToolTip(
            "Author a USD light under the selected group (or /World)"
        )
        self._add_light_btn.setPopupMode(QToolButton.InstantPopup)
        light_menu = QMenu(self._add_light_btn)
        for light_type in SUPPORTED_LIGHT_TYPES:
            action = light_menu.addAction(f"Add {light_type}")
            action.setData(light_type)
            action.triggered.connect(
                lambda _checked=False, lt=light_type: self._on_add_light(lt)
            )
        self._add_light_btn.setMenu(light_menu)
        self._save_btn = QPushButton("Save edits…")
        self._save_btn.setToolTip("Write the runtime edits to a USD layer")
        self._save_btn.clicked.connect(self._on_save_edits)
        tb_layout.addWidget(self._add_btn)
        tb_layout.addWidget(self._add_light_btn)
        tb_layout.addWidget(self._save_btn)
        tb_layout.addStretch(1)
        root_layout.addWidget(toolbar)
        has_stage = getattr(self.renderer, "_usd_stage", None) is not None
        self._add_btn.setEnabled(has_stage)
        self._add_light_btn.setEnabled(has_editable_stage(self.renderer))

        splitter = QSplitter(Qt.Vertical)
        root_layout.addWidget(splitter, 1)

        # ── Tree ──
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type"])
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree.setColumnWidth(0, 220)
        self.tree.itemSelectionChanged.connect(self._on_select)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        # Right-click "Delete", and the Delete key, remove the selected node.
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        del_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self.tree)
        del_shortcut.activated.connect(self._on_delete_selected)
        splitter.addWidget(self.tree)

        # ── Properties ──
        props_outer = QWidget()
        outer_layout = QVBoxLayout(props_outer)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        header = QLabel("Properties")
        f = header.font(); f.setBold(True); header.setFont(f)
        outer_layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._props_host = QWidget()
        self._props_layout = QVBoxLayout(self._props_host)
        self._props_layout.setContentsMargins(2, 2, 2, 2)
        self._props_layout.setSpacing(4)
        self._props_layout.addStretch(1)
        scroll.setWidget(self._props_host)
        outer_layout.addWidget(scroll)
        splitter.addWidget(props_outer)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        self._populate_tree()

        # Periodic refresh — picks up scene-graph regenerations and live
        # camera state changes (orbit/zoom from the viewport).
        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ── Tree population ───────────────────────────────────────────

    def _populate_tree(self) -> None:
        graph = self.renderer.scene_graph
        self.tree.clear()
        if graph is None:
            return
        # Record the LIVE tree's identity (from the snapshot), not the copy's —
        # see the poll-side change check.
        self._last_graph_id = getattr(self.renderer, "_scene_graph_id", 0)
        self._last_graph_version = getattr(self.renderer, "_scene_graph_version", 0)
        root_item = self._insert_node(None, graph)
        if root_item is not None and 0 < len(graph.children) <= 8:
            root_item.setExpanded(True)

    def _insert_node(
        self, parent: QTreeWidgetItem | None, node: SceneGraphNode,
    ) -> QTreeWidgetItem:
        icon = type_icon(node.type_name)
        display = f"{icon} {node.name}"
        item = QTreeWidgetItem([display, node.type_name])
        item.setData(0, Qt.UserRole, node.path)
        if parent is None:
            self.tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        for child in node.children:
            child_item = self._insert_node(item, child)
            if 0 < len(child.children) <= 8:
                child_item.setExpanded(True)
        return item

    # ── Selection / property build ────────────────────────────────

    def _on_double_click(self, item: QTreeWidgetItem, _col: int) -> None:
        """Double-click on a material node bound to a Python slangpile
        module routes it to the Python Material Editor.
        """
        if self._on_open_python_material is None:
            return
        path = item.data(0, Qt.UserRole)
        if not isinstance(path, str):
            return
        graph = self.renderer.scene_graph
        if graph is None:
            return
        node = find_node_by_path(graph, path)
        if node is None or node.renderer_ref is None:
            return
        if node.renderer_ref.kind != "material":
            return
        idx = node.renderer_ref.index
        # Scene-graph `RendererRef.index` for materials is built from
        # `_usd_scene.materials` (the authored list), not the per-frame
        # `self.scene.materials` placeholder.
        usd_scene = getattr(self.renderer, "_usd_scene", None)
        source = usd_scene if usd_scene is not None else self.renderer.scene
        materials = getattr(source, "materials", None) or []
        if not 0 <= idx < len(materials):
            return
        mod = getattr(materials[idx], "python_module", None)
        if not mod:
            return
        self._on_open_python_material(mod)

    def _on_select(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        path = items[0].data(0, Qt.UserRole)
        if path == self._selected_path:
            return
        self._selected_path = path

        graph = self.renderer.scene_graph
        if graph is None:
            return
        node = find_node_by_path(graph, path)
        if node is None:
            return

        # Auto-target the rotate gizmo when a mesh instance is selected.
        ref = node.renderer_ref
        if hasattr(self.renderer, "set_gizmo_target"):
            if ref is not None and ref.kind == "instance":
                self.renderer.set_gizmo_target(ref.index)
            else:
                self.renderer.set_gizmo_target(-1)
        self._build_properties(node)

    # ── Editing actions (add / delete / save) ─────────────────────

    def _selected_node(self) -> SceneGraphNode | None:
        items = self.tree.selectedItems()
        graph = self.renderer.scene_graph
        if not items or graph is None:
            return None
        path = items[0].data(0, Qt.UserRole)
        return find_node_by_path(graph, path) if isinstance(path, str) else None

    def _status(self, msg: str) -> None:
        """Surface a transient, non-modal message (status bar if available)."""
        print(f"[skinny] {msg}")
        try:
            self.window().statusBar().showMessage(msg, 4000)
        except Exception:  # noqa: BLE001 — no status bar in this host
            pass

    # ── Render-worker round-trips ─────────────────────────────────────────

    def _invoke_on_gui(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except RuntimeError:
            # A widget the callback closed over may have been torn down.
            pass

    def _await(
        self, fut, on_ok: Callable[[Any], None], fail_prefix: str,
    ) -> None:
        """Resolve a worker `Future` off-thread; run the GUI update on the GUI
        thread. Renderer edits that report a result (add/save/delete/texture/
        lens) run on the render worker and must not block the GUI thread."""
        def done(f) -> None:
            try:
                result = f.result()
            except Exception as exc:  # noqa: BLE001
                self._run_on_gui.emit(
                    lambda exc=exc: self._status(f"{fail_prefix}: {exc}"),
                )
                return
            self._run_on_gui.emit(lambda result=result: on_ok(result))
        fut.add_done_callback(done)

    def _on_add_model(self) -> None:
        r = self.renderer
        if getattr(r, "_usd_stage", None) is None:
            self._status("Load a USD scene before adding a model.")
            return
        start = str(get_last_dir("model") or "")
        path = get_open_file_name(self, "Add model", start, _USD_PICKER_FILTER)
        if not path:
            return
        record_last_dir("model", Path(path).parent)
        parent = add_parent_for_node(self._selected_node())
        self._await(
            r.add_model(path, parent_prim_path=parent),
            lambda new_path: self._status(f"Added {new_path}"),
            "Add model failed",
        )

    def _on_add_light(self, light_type: str) -> None:
        r = self.renderer
        if not has_editable_stage(r):
            self._status("Load an editable USD scene before adding a light.")
            return
        parent = add_parent_for_node(self._selected_node())
        self._await(
            r.add_light(light_type, parent_prim_path=parent),
            lambda new_path: self._status(f"Added {new_path}"),
            f"Add {light_type} failed",
        )

    def _on_save_edits(self) -> None:
        r = self.renderer
        if getattr(r, "_usd_edit_layer", None) is None:
            self._status("No edits to save (no USD scene loaded).")
            return
        self._await(
            r.save_edits(),
            lambda written: self._status(f"Saved edits to {written}"),
            "Save edits failed",
        )

    def _on_tree_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        item.setSelected(True)
        node = self._selected_node()
        if not is_deletable(node):
            return
        menu = QMenu(self.tree)
        act = menu.addAction("Delete")
        act.triggered.connect(self._on_delete_selected)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _on_delete_selected(self) -> None:
        node = self._selected_node()
        if node is None:
            return
        if not is_deletable(node):
            self._status(f"{node.path} cannot be deleted.")
            return
        self._await(
            self.renderer.remove_node(node.path),
            lambda _res, p=node.path: self._status(f"Deleted {p}"),
            "Delete failed",
        )

    def _build_properties(self, node: SceneGraphNode) -> None:
        # Tear down old widgets + pulls.
        self._clear_props()
        self._pulls.clear()

        # Header.
        header = QLabel(f"{type_icon(node.type_name)} {node.name}")
        f = header.font(); f.setBold(True); header.setFont(f)
        self._add_prop_widget(header)

        path_lbl = QLabel(node.path)
        path_lbl.setStyleSheet("color: gray;")
        self._add_prop_widget(path_lbl)

        if not node.properties:
            self._add_prop_widget(QLabel("(no properties)"))
            return

        # Build spec nodes from the shared mapper and render them through the
        # Qt backend walker (change ui-spec-scene-properties). The per-prop-type
        # switch and the eight `_add_*` helpers that used to live here are gone;
        # the front-end supplies only the commit transport and the live-value
        # read. `commit` sets `prop.value` before routing so a following edit of
        # a sibling transform component reads the fresh value, then calls the
        # shared dispatcher (the same routing the old helpers used).
        def commit(prop: SceneGraphProperty, value: Any, _node=node) -> None:
            prop.value = value
            # File loads (dome texture / camera lens) are the one edit whose
            # renderer verb returns a *result* — and the Qt proxy returns it as a
            # Future, not the bool `apply_scene_property`'s `is False` guard
            # expects. Route these through `_await` so a failed load reports its
            # reason off-thread (the dock owns the async file-load call, as the
            # dispatcher's docstring states); every other edit goes through the
            # shared dispatcher.
            t = prop.type_name
            if t == "texture_file":
                ref = _node.renderer_ref
                if ref is None or ref.kind != "light_env":
                    self._status(f"{prop.name!r} is not a dome-light texture")
                    return
                self._await(
                    self.renderer.apply_dome_light_texture(ref.index, value),
                    lambda ok, v=value: None if ok is not False
                    else self._status(f"could not load environment texture {v!r}"),
                    "could not load environment texture",
                )
                return
            if t == "lens_file":
                self._await(
                    self.renderer.apply_camera_lens_file(value),
                    lambda ok, v=value: None if ok is not False
                    else self._status(f"could not load lens file {v!r}"),
                    "could not load lens file",
                )
                return
            reason = apply_scene_property(
                self.renderer, _node, prop, value,
                graph=self.renderer.scene_graph,
            )
            if reason:
                self._status(reason)

        def get_live(prop: SceneGraphProperty, _node=node):
            ref = _node.renderer_ref
            if ref is not None and ref.kind == "renderer_camera":
                live = _read_camera_param(self.renderer.camera, prop.name)
                if live is not None:
                    return live
            return prop.value

        root = spec.Section(title="")
        for prop in node.properties:
            root.children.append(
                scene_property_to_node(prop, commit=commit, get_live=get_live)
            )
        host = QWidget()
        self._prop_builder = QtTreeBuilder(root, host)
        self._add_prop_widget(host)

    def _clear_props(self) -> None:
        # Stop the embedded property builder's pull timer before its host widget
        # is deleted, so a stale timer can't tick against a dead panel.
        if self._prop_builder is not None:
            self._prop_builder.stop()
            self._prop_builder = None
        while self._props_layout.count() > 1:  # keep trailing stretch
            item = self._props_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _add_prop_widget(self, widget: QWidget) -> None:
        # Insert before the trailing stretch.
        self._props_layout.insertWidget(self._props_layout.count() - 1, widget)

    # ── Per-tick refresh ──────────────────────────────────────────

    def _tick(self) -> None:
        # Pull a fresh scene-state projection from the render worker, then apply
        # it + refresh the UI on the GUI thread. Skip if one is already in flight
        # so slow frames can't pile up requests.
        if self._state_inflight:
            return
        self._state_inflight = True
        fut = self.renderer.refresh_scene_state()
        fut.add_done_callback(self._on_scene_state_future)

    def _on_scene_state_future(self, fut) -> None:
        # Worker thread: marshal the applied refresh onto the GUI thread.
        try:
            state = fut.result()
        except Exception:  # noqa: BLE001
            state = None
        self._run_on_gui.emit(
            lambda state=state: self._apply_scene_state_tick(state),
        )

    def _apply_scene_state_tick(self, state) -> None:
        self._state_inflight = False
        if state is not None:
            self.renderer.apply_scene_state(state)

        # Toolbar enablement tracks loaded-scene / edit-layer state.
        has_stage = getattr(self.renderer, "_usd_stage", None) is not None
        self._add_btn.setEnabled(has_stage)
        self._add_light_btn.setEnabled(has_editable_stage(self.renderer))
        self._save_btn.setEnabled(getattr(self.renderer, "_usd_edit_layer", None) is not None)

        graph = self.renderer.scene_graph
        version = getattr(self.renderer, "_scene_graph_version", 0)
        # `graph` is a fresh detached copy every refresh (copy_scene_graph), so
        # `id(graph)` would trip every poll — compare the LIVE tree's identity the
        # snapshot carries (`_scene_graph_id`) instead, plus the version.
        graph_id = getattr(self.renderer, "_scene_graph_id", 0)
        if graph is not None and (
            graph_id != self._last_graph_id
            or version != self._last_graph_version
        ):
            self._populate_tree()
            self._last_graph_version = version
            # The selected node may have been removed by an edit; clear the
            # stale property panel so it doesn't reference a gone prim.
            if self._selected_path is not None and find_node_by_path(
                graph, self._selected_path
            ) is None:
                self._selected_path = None
                self._clear_props()
                self._pulls.clear()
        for pull in self._pulls:
            try:
                pull()
            except RuntimeError:
                continue


# ── Helpers ────────────────────────────────────────────────────────


def _b(c: float) -> int:
    return max(0, min(255, int(round(c * 255.0))))


def _read_camera_param(cam, name: str):
    """Mirror of keys recognised by ``Renderer.apply_camera_param``."""
    if cam is None:
        return None
    if name == "fov":             return float(getattr(cam, "fov", 0.0))
    if name == "near":            return float(getattr(cam, "near", 0.0))
    if name == "far":             return float(getattr(cam, "far", 0.0))
    if name == "fstop":           return float(getattr(cam, "fstop", 0.0))
    if name == "focus_distance":  return float(getattr(cam, "focus_distance", 0.0))
    if name == "yaw":             return float(getattr(cam, "yaw", 0.0))
    if name == "pitch":           return float(getattr(cam, "pitch", 0.0))
    if name == "distance" and hasattr(cam, "distance"):
        return float(getattr(cam, "distance", 0.0))
    if name == "lens_enabled":
        lens = getattr(cam, "lens", None)
        return bool(lens.enabled) if lens is not None else None
    return None
