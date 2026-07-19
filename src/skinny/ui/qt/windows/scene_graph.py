"""Qt port of ``scene_graph_window.SceneGraphWindow``.

Tree view (left) + property editor (right) for the USD scene graph.
Selecting a node rebuilds the right pane with typed widgets for the
node's editable properties; edits route through the renderer's existing
``apply_*`` API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QSignalBlocker, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QDockWidget, QDoubleSpinBox,
    QHBoxLayout, QLabel, QMenu, QPushButton, QScrollArea, QSlider, QSplitter,
    QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from skinny.scene_graph import (
    SceneGraphNode, SceneGraphProperty, find_node_by_path, type_icon,
)
from skinny.settings import get_last_dir, record_last_dir
from skinny.ui.qt.dialogs import get_open_file_name
from skinny.ui.scene_edit_actions import (
    SUPPORTED_LIGHT_TYPES,
    add_parent_for_node,
    has_editable_stage,
    is_deletable,
    trs_to_matrix,
)

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

        for prop in node.properties:
            row = self._build_property_widget(node, prop)
            if row is not None:
                self._add_prop_widget(row)

    def _clear_props(self) -> None:
        while self._props_layout.count() > 1:  # keep trailing stretch
            item = self._props_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _add_prop_widget(self, widget: QWidget) -> None:
        # Insert before the trailing stretch.
        self._props_layout.insertWidget(self._props_layout.count() - 1, widget)

    # ── Property rows ─────────────────────────────────────────────

    def _build_property_widget(
        self, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> QWidget | None:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label = QLabel(prop.display_name)
        label.setMinimumWidth(120)
        layout.addWidget(label)

        if prop.type_name == "bool" and prop.editable:
            self._add_bool(layout, node, prop)
        elif prop.type_name == "float" and prop.editable:
            self._add_float(layout, node, prop)
        elif prop.type_name == "color3f" and prop.editable:
            self._add_color(layout, node, prop)
        elif prop.type_name == "vec3f" and prop.editable:
            self._add_vec3(layout, node, prop)
        elif prop.type_name == "vec3f":
            v = prop.value
            layout.addWidget(QLabel(f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"))
            layout.addStretch(1)
        elif prop.type_name == "color3f":
            self._add_color_readonly(layout, prop)
        elif prop.type_name in ("float", "int"):
            v = prop.value
            txt = f"{v:.4f}" if isinstance(v, float) else str(v)
            lbl = QLabel(txt)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(lbl, stretch=1)
        elif prop.type_name == "rel":
            lbl = QLabel(f"→ {prop.value}")
            lbl.setStyleSheet("color: steelblue;")
            layout.addWidget(lbl, stretch=1)
        elif prop.type_name == "asset":
            lbl = QLabel(str(prop.value))
            lbl.setStyleSheet("color: gray;")
            layout.addWidget(lbl, stretch=1)
        elif prop.type_name == "lens_file" and prop.editable:
            self._add_lens_file(layout, node, prop)
        elif prop.type_name == "texture_file" and prop.editable:
            self._add_texture_file(layout, node, prop)
        else:
            layout.addWidget(QLabel(str(prop.value)), stretch=1)
        return row

    def _add_bool(
        self, layout: QHBoxLayout, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> None:
        cb = QCheckBox()
        cb.setChecked(bool(prop.value))

        def on_toggle(checked: bool) -> None:
            value = bool(checked)
            prop.value = value
            toggle = prop.metadata.get("toggle", "node")
            if toggle == "subtree":
                self.renderer.apply_subtree_enabled(node.path, value)
                return
            ref = node.renderer_ref
            if ref is None:
                return
            # Camera bool params (lens_enabled, ...) are camera scalars,
            # not enable-flags on a scene record.
            if ref.kind == "renderer_camera":
                self.renderer.apply_camera_param(prop.name, value)
                return
            self.renderer.apply_node_enabled(node.path, value)

        cb.toggled.connect(on_toggle)
        layout.addWidget(cb)
        layout.addStretch(1)

        # Live pull for camera-bound bool props.
        if node.renderer_ref is not None and node.renderer_ref.kind == "renderer_camera":
            def pull() -> None:
                live = _read_camera_param(self.renderer.camera, prop.name)
                if live is None:
                    return
                if cb.isChecked() != bool(live):
                    with QSignalBlocker(cb):
                        cb.setChecked(bool(live))
                    prop.value = bool(live)
            self._pulls.append(pull)

    def _add_float(
        self, layout: QHBoxLayout, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> None:
        lo = float(prop.metadata.get("min", 0.0))
        hi = float(prop.metadata.get("max", 1.0))
        cur = float(prop.value)
        growable = bool(prop.metadata.get("growable"))

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)
        spin = QDoubleSpinBox()
        spin.setRange(lo, 1e9 if growable else hi)
        spin.setDecimals(3)
        # Mutable mapping bounds so a growable range can be rescaled in place
        # without rebuilding the widget (preserves the slider grab mid-drag).
        rng = {"hi": hi, "span": max(hi - lo, 1e-9)}

        def to_int(v: float) -> int:
            return int(round((v - lo) / rng["span"] * 1000.0))

        def from_int(i: int) -> float:
            return lo + (i / 1000.0) * rng["span"]

        with QSignalBlocker(slider), QSignalBlocker(spin):
            slider.setValue(to_int(cur))
            spin.setValue(cur)

        def apply(v: float) -> None:
            prop.value = float(v)
            self._apply_property(node, prop, float(v))

        slider.valueChanged.connect(lambda i: (spin.setValue(from_int(i))))
        spin.valueChanged.connect(lambda v: (
            self._update_slider_from_spin(slider, to_int, v),
            apply(v),
        ))

        layout.addWidget(slider, stretch=1)
        layout.addWidget(spin)

        if node.renderer_ref is not None and node.renderer_ref.kind == "renderer_camera":
            def pull() -> None:
                cam = self.renderer.camera
                live = _read_camera_param(cam, prop.name)
                if live is None:
                    return
                if growable:
                    live_max = float(getattr(cam, "max_distance", rng["hi"]))
                    if abs(live_max - rng["hi"]) > 1e-4:
                        rng["hi"] = live_max
                        rng["span"] = max(live_max - lo, 1e-9)
                        with QSignalBlocker(slider):
                            slider.setValue(to_int(float(live)))
                if abs(spin.value() - float(live)) > 1e-4:
                    with QSignalBlocker(slider), QSignalBlocker(spin):
                        spin.setValue(float(live))
                        slider.setValue(to_int(float(live)))
                    prop.value = float(live)
            self._pulls.append(pull)

    @staticmethod
    def _update_slider_from_spin(
        slider: QSlider, to_int: Callable[[float], int], v: float,
    ) -> None:
        with QSignalBlocker(slider):
            slider.setValue(to_int(v))

    def _add_color(
        self, layout: QHBoxLayout, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> None:
        v = prop.value
        r, g, b = float(v[0]), float(v[1]), float(v[2])
        swatch = QPushButton()
        swatch.setFixedWidth(36); swatch.setFixedHeight(20)

        def paint_swatch(rr: float, gg: float, bb: float) -> None:
            swatch.setStyleSheet(
                f"background-color: rgb({_b(rr)}, {_b(gg)}, {_b(bb)}); "
                f"border: 1px solid #555;"
            )

        paint_swatch(r, g, b)

        def on_click() -> None:
            cur = prop.value
            init = QColor(_b(cur[0]), _b(cur[1]), _b(cur[2]))
            chosen = QColorDialog.getColor(init, self, prop.display_name)
            if not chosen.isValid():
                return
            new_color = (chosen.redF(), chosen.greenF(), chosen.blueF())
            prop.value = new_color
            paint_swatch(*new_color)
            self._apply_property(node, prop, new_color)

        swatch.clicked.connect(on_click)
        layout.addWidget(swatch)
        layout.addStretch(1)

    def _add_color_readonly(
        self, layout: QHBoxLayout, prop: SceneGraphProperty,
    ) -> None:
        v = prop.value
        r, g, b = float(v[0]), float(v[1]), float(v[2])
        swatch = QLabel()
        swatch.setFixedSize(36, 20)
        swatch.setStyleSheet(
            f"background-color: rgb({_b(r)}, {_b(g)}, {_b(b)}); border: 1px solid #555;"
        )
        layout.addWidget(swatch)
        layout.addWidget(QLabel(f"({r:.2f}, {g:.2f}, {b:.2f})"), stretch=1)

    def _add_vec3(
        self, layout: QHBoxLayout, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> None:
        v = prop.value
        spins: list[QDoubleSpinBox] = []
        for i, axis in enumerate(("X", "Y", "Z")):
            layout.addWidget(QLabel(axis))
            s = QDoubleSpinBox()
            s.setRange(-1e6, 1e6); s.setDecimals(4); s.setSingleStep(0.05)
            with QSignalBlocker(s):
                s.setValue(float(v[i]))
            spins.append(s)
            layout.addWidget(s)

        def commit() -> None:
            try:
                values = tuple(float(s.value()) for s in spins)
            except (TypeError, ValueError):
                return
            prop.value = values
            self._apply_vec3_property(node, prop, values)

        for s in spins:
            s.editingFinished.connect(commit)

    def _add_lens_file(
        self, layout: QHBoxLayout, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> None:
        cur_label = QLabel(str(prop.value))
        cur_label.setStyleSheet("color: steelblue;")
        layout.addWidget(cur_label, stretch=1)
        btn = QPushButton("Load…")

        def on_pick() -> None:
            path = get_open_file_name(
                self, "Load lens (.usda)",
                get_last_dir("lens"),
                "USDA lens (*.usda);;All files (*.*)",
            )
            if not path:
                return
            if not hasattr(self.renderer, "apply_camera_lens_file"):
                return

            def on_ok(ok: bool, path=path) -> None:
                if not ok:
                    return
                record_last_dir("lens", Path(path).parent)
                name = Path(path).name
                cur_label.setText(name)
                prop.value = name

            self._await(
                self.renderer.apply_camera_lens_file(path), on_ok,
                "Load lens failed",
            )

        btn.clicked.connect(on_pick)
        layout.addWidget(btn)

    def _add_texture_file(
        self, layout: QHBoxLayout, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> None:
        cur = str(prop.value or "")
        label_text = Path(cur).name if cur else "(none)"
        cur_label = QLabel(label_text)
        cur_label.setStyleSheet("color: steelblue;")
        layout.addWidget(cur_label, stretch=1)
        btn = QPushButton("Load…")

        def on_pick() -> None:
            ref = node.renderer_ref
            if ref is None or ref.kind != "light_env":
                return
            path = get_open_file_name(
                self, "Load HDR",
                get_last_dir("ibl"),
                "HDR images (*.hdr *.exr *.pfm);;All files (*.*)",
            )
            if not path:
                return
            if not hasattr(self.renderer, "apply_dome_light_texture"):
                return

            def on_ok(ok: bool, path=path) -> None:
                if not ok:
                    return
                record_last_dir("ibl", Path(path).parent)
                cur_label.setText(Path(path).name)
                prop.value = path

            self._await(
                self.renderer.apply_dome_light_texture(ref.index, path), on_ok,
                "Load HDR failed",
            )

        btn.clicked.connect(on_pick)
        layout.addWidget(btn)

    # ── Apply edits ───────────────────────────────────────────────

    def _apply_property(
        self, node: SceneGraphNode, prop: SceneGraphProperty, value: Any,
    ) -> None:
        ref = node.renderer_ref
        if ref is None:
            ref = self._find_shader_material_ref(node)
            if ref is None:
                return
        if ref.kind == "material":
            self.renderer.apply_material_override(ref.index, prop.name, value)
        elif ref.kind in ("light_dir", "light_sphere", "light_env"):
            light_type = {
                "light_dir": "dir",
                "light_sphere": "sphere",
                "light_env": "env",
            }[ref.kind]
            self.renderer.apply_light_override(light_type, ref.index, prop.name, value)
        elif ref.kind == "renderer_camera":
            self.renderer.apply_camera_param(prop.name, value)

    def _apply_vec3_property(
        self, node: SceneGraphNode, prop: SceneGraphProperty,
        values: tuple[float, float, float],
    ) -> None:
        ref = node.renderer_ref
        if ref is not None and ref.kind == "renderer_camera":
            axis_kind = prop.metadata.get("camera_axis", "")
            if axis_kind == "target":
                keys = ("target_x", "target_y", "target_z")
            elif axis_kind == "position":
                keys = ("position_x", "position_y", "position_z")
            else:
                return
            for k, v in zip(keys, values):
                self.renderer.apply_camera_param(k, float(v))
            return
        is_authored_light = node.type_name in SUPPORTED_LIGHT_TYPES
        if ref is None and not is_authored_light:
            return
        if ref is not None and ref.kind != "instance" and not is_authored_light:
            return

        # TRS for an instance needs all three vectors. Walk the active
        # property panel to pick up the live values for the other two.
        translate = scale = (0.0, 0.0, 0.0)
        rotate = (0.0, 0.0, 0.0)
        for p in node.properties:
            if p.name == "translate":
                translate = values if p is prop else p.value
            elif p.name == "rotate":
                rotate = values if p is prop else p.value
            elif p.name == "scale":
                scale = values if p is prop else p.value
        # Author to the stage (edit layer) so the move persists and is captured
        # by "Save edits"; falls back to the runtime path if no stage is loaded.
        if getattr(self.renderer, "_usd_stage", None) is not None:
            self.renderer.set_transform(node.path, trs_to_matrix(translate, rotate, scale))
        else:
            self.renderer.apply_instance_transform(node.path, translate, rotate, scale)

    def _find_shader_material_ref(self, node: SceneGraphNode):
        graph = self.renderer.scene_graph
        if graph is None:
            return None
        parts = node.path.rstrip("/").split("/")
        for i in range(len(parts) - 1, 0, -1):
            parent_path = "/".join(parts[:i]) or "/"
            parent = find_node_by_path(graph, parent_path)
            if parent is not None and parent.renderer_ref is not None:
                if parent.renderer_ref.kind == "material":
                    return parent.renderer_ref
        return None

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
