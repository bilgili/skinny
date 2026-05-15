"""Qt port of ``scene_graph_window.SceneGraphWindow``.

Tree view (left) + property editor (right) for the USD scene graph.
Selecting a node rebuilds the right pane with typed widgets for the
node's editable properties; edits route through the renderer's existing
``apply_*`` API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QSignalBlocker, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QDockWidget, QDoubleSpinBox, QFileDialog,
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QSlider, QSplitter,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from skinny.scene_graph import (
    SceneGraphNode, SceneGraphProperty, find_node_by_path, type_icon,
)


class SceneGraphDock(QDockWidget):
    """Non-modal dock with a tree view + property editor. Mirrors the
    behaviour of the legacy Tk window.
    """

    TICK_MS = 200

    def __init__(self, renderer, parent: QWidget | None = None) -> None:
        super().__init__("Scene Graph", parent)
        self.renderer = renderer
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        self._last_graph_id: int = -1
        self._last_graph_version: int = -1
        self._selected_path: str | None = None
        # Live "pull" callbacks for the active property widgets — refresh
        # them from the camera each tick so external orbit/zoom shows up.
        self._pulls: list[Callable[[], None]] = []

        # Vertical splitter: tree above, property editor below. Matches
        # the user's request to have the properties laid out below the
        # tree rather than to the side.
        splitter = QSplitter(Qt.Vertical)
        self.setWidget(splitter)

        # ── Tree ──
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type"])
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree.setColumnWidth(0, 220)
        self.tree.itemSelectionChanged.connect(self._on_select)
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
        self._last_graph_id = id(graph)
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
            self.renderer.apply_node_enabled(ref.kind, ref.index, value)

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

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(3)
        span = max(hi - lo, 1e-9)

        def to_int(v: float) -> int:
            return int(round((v - lo) / span * 1000.0))
        def from_int(i: int) -> float:
            return lo + (i / 1000.0) * span

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
                live = _read_camera_param(self.renderer.camera, prop.name)
                if live is None:
                    return
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
            path, _ = QFileDialog.getOpenFileName(
                self, "Load lens (.usda)",
                str(Path(__file__).resolve().parents[4] / "lenses"),
                "USDA lens (*.usda);;All files (*.*)",
            )
            if not path:
                return
            ok = False
            if hasattr(self.renderer, "apply_camera_lens_file"):
                ok = self.renderer.apply_camera_lens_file(path)
            if ok:
                name = Path(path).name
                cur_label.setText(name)
                prop.value = name

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
            start_dir = ""
            if cur:
                start_dir = str(Path(cur).resolve().parent)
            if not start_dir:
                start_dir = str(
                    Path(__file__).resolve().parents[4] / "hdrs",
                )
            path, _ = QFileDialog.getOpenFileName(
                self, "Load HDR",
                start_dir,
                "HDR images (*.hdr *.exr *.pfm);;All files (*.*)",
            )
            if not path:
                return
            ok = False
            if hasattr(self.renderer, "apply_dome_light_texture"):
                ok = self.renderer.apply_dome_light_texture(ref.index, path)
            if ok:
                cur_label.setText(Path(path).name)
                prop.value = path

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
        if ref is None:
            return
        if ref.kind == "renderer_camera":
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
        if ref.kind != "instance":
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
        self.renderer.apply_instance_transform(ref.index, translate, rotate, scale)

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
        graph = self.renderer.scene_graph
        version = getattr(self.renderer, "_scene_graph_version", 0)
        if graph is not None and (
            id(graph) != self._last_graph_id
            or version != self._last_graph_version
        ):
            self._populate_tree()
            self._last_graph_version = version
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
