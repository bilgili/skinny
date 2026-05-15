"""Qt port of ``material_graph_editor.MaterialGraphEditor``.

QGraphicsView-based node-graph editor for the picked scene material's
MaterialX nodegraph. View-model helpers (NodeView, PortView, build_view,
_layout, _ADDABLE_CATEGORIES) come from the legacy module unchanged —
they're pure data + MaterialX bindings, no Tk.

Edits route through the same renderer fast-path used by the legacy Tk
editor: ``apply_material_override`` for flat-path values,
``_gen_scene_materials`` + ``_upload_graph_param_buffers`` for topology
changes inside a nodegraph.
"""

from __future__ import annotations

from typing import Optional

import MaterialX as mx
import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QAction, QBrush, QColor, QFont, QImage, QMouseEvent, QPainter,
    QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDockWidget,
    QDoubleSpinBox, QFileDialog, QGraphicsItem, QGraphicsPathItem,
    QGraphicsScene, QGraphicsSceneMouseEvent, QGraphicsView, QGroupBox,
    QHBoxLayout, QLabel, QMenu, QPushButton, QScrollArea, QSlider, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from skinny.mtlx_graph_view import (
    _ADDABLE_CATEGORIES, NodeGraphView, NodeView, PortView, _layout,
    _val_to_py, build_view,
)


# ── Node graphics item ────────────────────────────────────────────


NODE_W = 180
HEADER_H = 22
PORT_H = 22


def _node_height(n: NodeView) -> float:
    return HEADER_H + max(1, len(n.inputs) + len(n.outputs)) * PORT_H + 6


def _in_port_pos(n: NodeView, i: int) -> QPointF:
    return QPointF(0.0, HEADER_H + i * PORT_H + PORT_H / 2)


def _out_port_pos(n: NodeView, i: int) -> QPointF:
    return QPointF(
        NODE_W, HEADER_H + (len(n.inputs) + i) * PORT_H + PORT_H / 2,
    )


class _NodeItem(QGraphicsItem):
    """One node in the graph. Header + body + port circles + labels drawn
    in ``paint``; hit-testing for ports happens via ``port_at(scenePos)``
    which the scene queries during wire-drag.
    """

    def __init__(self, node: NodeView, parent_widget) -> None:
        super().__init__()
        self.node = node
        self.parent_widget = parent_widget
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setPos(node.x, node.y)
        self._h = _node_height(node)

    def boundingRect(self) -> QRectF:
        return QRectF(-6, -6, NODE_W + 12, self._h + 12)

    def paint(self, painter: QPainter, _opt, _widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        selected = self.isSelected()
        body_rect = QRectF(0, 0, NODE_W, self._h)
        outline = QColor(0xFF, 0xE1, 0x88) if selected else QColor(0x44, 0x44, 0x66)
        painter.setPen(QPen(outline, 3 if selected else 1))
        painter.setBrush(QBrush(QColor(0x2C, 0x2C, 0x3C)))
        painter.drawRect(body_rect)

        header = QRectF(0, 0, NODE_W, HEADER_H)
        painter.setBrush(QBrush(
            QColor(0x5A, 0x3A, 0x8E) if self.node.is_output else QColor(0x3A, 0x3A, 0x5A)
        ))
        painter.setPen(QPen(outline, 1))
        painter.drawRect(header)

        # Title
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(9)
        painter.setFont(title_font)
        painter.setPen(QColor(0xE8, 0xE8, 0xF0))
        painter.drawText(
            QRectF(6, 0, NODE_W - 12, HEADER_H), Qt.AlignVCenter | Qt.AlignLeft,
            f"{self.node.category}  ({self.node.name})",
        )

        # Ports + labels
        port_font = QFont()
        port_font.setPointSize(8)
        painter.setFont(port_font)
        for i, inp in enumerate(self.node.inputs):
            p = _in_port_pos(self.node, i)
            fill = QColor(0xFF, 0xD0, 0x60) if inp.connected_from else QColor(0x80, 0xC0, 0xFF)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(QColor(0x11, 0x11, 0x1A), 1))
            painter.drawEllipse(p, 5, 5)
            painter.setPen(QColor(0xC8, 0xC8, 0xD8))
            painter.drawText(
                QRectF(p.x() + 10, p.y() - PORT_H / 2, NODE_W - 24, PORT_H),
                Qt.AlignVCenter | Qt.AlignLeft, inp.name,
            )
        for i, op in enumerate(self.node.outputs):
            p = _out_port_pos(self.node, i)
            painter.setBrush(QBrush(QColor(0x80, 0xFF, 0x90)))
            painter.setPen(QPen(QColor(0x11, 0x11, 0x1A), 1))
            painter.drawEllipse(p, 5, 5)
            painter.setPen(QColor(0xC8, 0xC8, 0xD8))
            painter.drawText(
                QRectF(0, p.y() - PORT_H / 2, NODE_W - 14, PORT_H),
                Qt.AlignVCenter | Qt.AlignRight, op.name,
            )

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.node.x = float(value.x())
            self.node.y = float(value.y())
            self.parent_widget._refresh_wires()
        return super().itemChange(change, value)

    def port_at(self, scene_pos: QPointF) -> Optional[tuple[str, str]]:
        """Return ``(kind, port_name)`` if ``scene_pos`` hits a port on
        this node, else ``None``. ``kind`` is ``"in"`` or ``"out"``.
        """
        local = self.mapFromScene(scene_pos)
        for i, inp in enumerate(self.node.inputs):
            p = _in_port_pos(self.node, i)
            if (local - p).manhattanLength() < 12:
                return ("in", inp.name)
        for i, op in enumerate(self.node.outputs):
            p = _out_port_pos(self.node, i)
            if (local - p).manhattanLength() < 12:
                return ("out", op.name)
        return None


# ── Graph scene ──────────────────────────────────────────────────


class _GraphScene(QGraphicsScene):
    """Owns nodes + wires + temporary drag wire. Forwards port-press
    events to the parent dock so it can run the wire-drag flow.
    """

    def __init__(self, parent_widget) -> None:
        super().__init__()
        self.parent_widget = parent_widget
        self.setBackgroundBrush(QBrush(QColor(0x1A, 0x1A, 0x22)))
        self._drag_wire: Optional[QGraphicsPathItem] = None
        self._drag_from: Optional[tuple[str, str, QPointF]] = None

    def find_port_at(self, scene_pos: QPointF) -> Optional[tuple[_NodeItem, str, str]]:
        for item in self.items(scene_pos):
            if isinstance(item, _NodeItem):
                hit = item.port_at(scene_pos)
                if hit is not None:
                    return (item, hit[0], hit[1])
        return None

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            hit = self.find_port_at(event.scenePos())
            if hit is not None:
                node_item, kind, port_name = hit
                if kind == "out":
                    start = node_item.mapToScene(_out_port_pos(node_item.node, 0))
                    self._drag_from = (node_item.node.name, port_name, start)
                    self._drag_wire = QGraphicsPathItem()
                    pen = QPen(QColor(0xFF, 0xE1, 0x88), 2)
                    pen.setStyle(Qt.DashLine)
                    self._drag_wire.setPen(pen)
                    self.addItem(self._drag_wire)
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._drag_wire is not None and self._drag_from is not None:
            _, _, start = self._drag_from
            path = QPainterPath(start)
            path.lineTo(event.scenePos())
            self._drag_wire.setPath(path)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._drag_wire is not None:
            self.removeItem(self._drag_wire)
            self._drag_wire = None
            if self._drag_from is not None and event.button() == Qt.LeftButton:
                hit = self.find_port_at(event.scenePos())
                if hit is not None and hit[1] == "in":
                    src_node, src_port, _ = self._drag_from
                    self.parent_widget._apply_connect(
                        src_node, src_port, hit[0].node.name, hit[2],
                    )
            self._drag_from = None
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        self.parent_widget._show_context_menu(event.scenePos(), event.screenPos())


# ── Dock widget ──────────────────────────────────────────────────


class MaterialGraphDock(QDockWidget):
    PREVIEW_SIZE = 256
    PREVIEW_DEBOUNCE_MS = 120

    def __init__(self, renderer, parent: QWidget | None = None) -> None:
        super().__init__("Material Graph Editor", parent)
        self.renderer = renderer
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        self._view: Optional[NodeGraphView] = None
        self._materials: list[tuple[int, str, str]] = []
        self._selected_node: Optional[str] = None
        self._node_items: dict[str, _NodeItem] = {}
        self._wire_items: list[QGraphicsPathItem] = []
        self._last_scene_id: int = -1

        self._build_widgets()
        self._refresh_material_combo()
        self._refresh_env_combo()

        self._scene_poll = QTimer(self)
        self._scene_poll.setInterval(500)
        self._scene_poll.timeout.connect(self._poll_scene_swap)
        self._scene_poll.start()

    # ── Layout ────────────────────────────────────────────────────

    def _build_widgets(self) -> None:
        host = QWidget()
        outer = QVBoxLayout(host)
        outer.setContentsMargins(6, 6, 6, 6)

        # Top bar.
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Material:"))
        self._material_combo = QComboBox()
        self._material_combo.currentIndexChanged.connect(self._on_material_picked)
        bar.addWidget(self._material_combo, stretch=1)
        reset_btn = QPushButton("Reset layout")
        reset_btn.clicked.connect(self._on_relayout)
        bar.addWidget(reset_btn)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._refresh_material_combo)
        bar.addWidget(reload_btn)
        outer.addLayout(bar)

        # Splitter: graph view left, side panel right.
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, stretch=1)

        self._scene = _GraphScene(self)
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)
        self._gview = QGraphicsView(self._scene)
        self._gview.setRenderHint(QPainter.Antialiasing, True)
        self._gview.setDragMode(QGraphicsView.RubberBandDrag)
        self._gview.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._gview.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        splitter.addWidget(self._gview)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(4, 4, 4, 4)

        prim_row = QHBoxLayout()
        prim_row.addWidget(QLabel("Primitive:"))
        self._prim_combo = QComboBox()
        self._prim_combo.addItems(["sphere", "cube", "plane"])
        self._prim_combo.currentIndexChanged.connect(self._schedule_preview)
        prim_row.addWidget(self._prim_combo)
        prim_row.addStretch(1)
        side_layout.addLayout(prim_row)

        env_row = QHBoxLayout()
        env_row.addWidget(QLabel("Env light:"))
        self._env_combo = QComboBox()
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)
        env_row.addWidget(self._env_combo, stretch=1)
        side_layout.addLayout(env_row)

        self._preview_label = QLabel()
        self._preview_label.setFixedSize(self.PREVIEW_SIZE, self.PREVIEW_SIZE)
        self._preview_label.setStyleSheet(
            "background-color: #11111a; border: 1px solid #333;"
        )
        side_layout.addWidget(self._preview_label, alignment=Qt.AlignCenter)

        self._side_title = QLabel("Pick a node to edit its inputs.")
        f = self._side_title.font(); f.setBold(True); self._side_title.setFont(f)
        side_layout.addWidget(self._side_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._side_host = QWidget()
        self._side_form = QVBoxLayout(self._side_host)
        self._side_form.setContentsMargins(2, 2, 2, 2)
        self._side_form.setSpacing(4)
        self._side_form.addStretch(1)
        scroll.setWidget(self._side_host)
        side_layout.addWidget(scroll, stretch=1)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)

        self._status = QLabel("")
        outer.addWidget(self._status)

        self.setWidget(host)

    # ── Material combo ────────────────────────────────────────────

    def _scene_materials(self) -> list:
        scene = getattr(self.renderer, "_usd_scene", None)
        if scene is None:
            return []
        return list(getattr(scene, "materials", []) or [])

    def _refresh_material_combo(self) -> None:
        mats = self._scene_materials()
        cm_map = getattr(self.renderer, "_mtlx_scene_materials", {}) or {}
        opts: list[tuple[int, str, str]] = []
        for i, mat in enumerate(mats):
            if i == 0:
                continue
            target = getattr(mat, "mtlx_target_name", None)
            if not target:
                cm = cm_map.get(i)
                target = getattr(cm, "target_name", None) if cm else None
            if target:
                opts.append((i, mat.name, target))
        self._materials = opts
        labels = [f"#{i}  {name}  ({target})" for i, name, target in opts]
        prev_block = self._material_combo.blockSignals(True)
        self._material_combo.clear()
        self._material_combo.addItems(labels)
        self._material_combo.blockSignals(prev_block)
        if not opts:
            self._view = None
            self._clear_scene()
            self._refresh_side()
            self._status.setText("No MaterialX materials in scene.")
            return
        sel_idx = 0
        if self._view is not None:
            for k, (mid, _, _) in enumerate(opts):
                if mid == self._view.material_id:
                    sel_idx = k
                    break
        self._material_combo.setCurrentIndex(sel_idx)

    def _on_material_picked(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._materials):
            return
        mid, name, target = self._materials[idx]
        lib = getattr(self.renderer, "_mtlx_library", None)
        if lib is None:
            self._status.setText("MaterialX library not loaded.")
            return
        try:
            view = build_view(lib.document, mid, name, target)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"build_view error: {exc}")
            return
        if view is None:
            self._status.setText(f"Could not resolve target '{target}'.")
            self._view = None
            self._clear_scene()
            self._refresh_side()
            return
        self._view = view
        self._selected_node = None
        self._rebuild_scene()
        self._refresh_side()
        suffix = "  [flat std_surface]" if view.flat else ""
        self._status.setText(f"{len(view.nodes)} node(s){suffix}")
        self._schedule_preview()

    def _on_relayout(self) -> None:
        if self._view is None:
            return
        _layout(self._view.nodes)
        self._rebuild_scene()

    def _poll_scene_swap(self) -> None:
        cur = id(getattr(self.renderer, "_usd_scene", None))
        if cur != self._last_scene_id:
            self._last_scene_id = cur
            self._refresh_material_combo()
            self._refresh_env_combo()

    # ── Graphics scene ────────────────────────────────────────────

    def _clear_scene(self) -> None:
        self._scene.clear()
        self._node_items.clear()
        self._wire_items.clear()

    def _rebuild_scene(self) -> None:
        self._clear_scene()
        if self._view is None:
            return
        for n in self._view.nodes:
            item = _NodeItem(n, self)
            self._scene.addItem(item)
            self._node_items[n.name] = item
        self._refresh_wires()
        # Fit scene rect to node extents + margin.
        xs = [n.x for n in self._view.nodes]
        ys = [n.y for n in self._view.nodes]
        if xs and ys:
            self._scene.setSceneRect(
                min(xs) - 60, min(ys) - 60,
                (max(xs) + NODE_W + 120) - (min(xs) - 60),
                (max(ys) + max(_node_height(n) for n in self._view.nodes) + 120)
                - (min(ys) - 60),
            )

    def _refresh_wires(self) -> None:
        for w in self._wire_items:
            self._scene.removeItem(w)
        self._wire_items.clear()
        if self._view is None:
            return
        wire_pen = QPen(QColor(0xA0, 0xA0, 0xC0), 2)
        for n in self._view.nodes:
            for i, inp in enumerate(n.inputs):
                if not inp.connected_from:
                    continue
                up_name, _up_port = inp.connected_from
                up = next((nn for nn in self._view.nodes if nn.name == up_name), None)
                if up is None:
                    continue
                src_item = self._node_items.get(up.name)
                dst_item = self._node_items.get(n.name)
                if src_item is None or dst_item is None:
                    continue
                p1 = src_item.mapToScene(_out_port_pos(up, 0))
                p2 = dst_item.mapToScene(_in_port_pos(n, i))
                path = QPainterPath(p1)
                path.lineTo(p2)
                wire = QGraphicsPathItem(path)
                wire.setPen(wire_pen)
                wire.setZValue(-1)  # behind nodes
                self._scene.addItem(wire)
                self._wire_items.append(wire)

    def _show_context_menu(self, scene_pos: QPointF, screen_pos) -> None:
        menu = QMenu(self)
        hit_port = self._scene.find_port_at(scene_pos)
        node_under = None
        for item in self._scene.items(scene_pos):
            if isinstance(item, _NodeItem):
                node_under = item.node
                break

        if hit_port is None and node_under is None:
            add_menu = menu.addMenu("Add node…")
            for cat, _t in _ADDABLE_CATEGORIES:
                act = QAction(cat, self)
                act.triggered.connect(
                    lambda _c=False, cc=cat, sp=scene_pos:
                        self._apply_add_node(cc, sp.x(), sp.y()),
                )
                add_menu.addAction(act)
        elif hit_port is not None and hit_port[1] == "in":
            node_item = hit_port[0]
            port_name = hit_port[2]
            act = QAction("Disconnect", self)
            act.triggered.connect(
                lambda _c=False, nn=node_item.node.name, pp=port_name:
                    self._apply_disconnect(nn, pp),
            )
            menu.addAction(act)
        elif node_under is not None:
            act = QAction("Delete node", self)
            act.triggered.connect(
                lambda _c=False, nn=node_under.name:
                    self._apply_delete_node(nn),
            )
            menu.addAction(act)
        else:
            return
        menu.exec(screen_pos)

    # ── Side editor ───────────────────────────────────────────────

    def _refresh_side(self) -> None:
        while self._side_form.count() > 1:
            item = self._side_form.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        if self._view is None or self._selected_node is None:
            self._side_title.setText("Pick a node to edit its inputs.")
            return
        node = next(
            (n for n in self._view.nodes if n.name == self._selected_node), None,
        )
        if node is None:
            return
        self._side_title.setText(f"{node.category}  /  {node.name}")
        if not node.inputs:
            empty = QLabel("(no inputs)")
            empty.setStyleSheet("color: #808090;")
            self._add_side_widget(empty)
            return
        for inp in node.inputs:
            row = self._build_input_row(node, inp)
            self._add_side_widget(row)

    def _add_side_widget(self, widget: QWidget) -> None:
        self._side_form.insertWidget(self._side_form.count() - 1, widget)

    def _build_input_row(self, node: NodeView, port: PortView) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QLabel(port.name)
        label.setMinimumWidth(110)
        layout.addWidget(label)

        if port.connected_from:
            up, op = port.connected_from
            lbl = QLabel(f"← {up}.{op}")
            lbl.setStyleSheet("color: #a0a0c0;")
            layout.addWidget(lbl, stretch=1)
            return row

        t = port.type_name
        if t == "float":
            self._add_float_row(layout, node, port)
        elif t in ("color3", "vector3"):
            self._add_vec3_row(layout, node, port, is_color=(t == "color3"))
        elif t == "vector2":
            self._add_vec2_row(layout, node, port)
        elif t == "integer":
            self._add_int_row(layout, node, port)
        elif t == "boolean":
            self._add_bool_row(layout, node, port)
        elif t == "filename":
            self._add_filename_row(layout, node, port)
        else:
            lbl = QLabel(f"(type {t} not editable)")
            lbl.setStyleSheet("color: #808090;")
            layout.addWidget(lbl, stretch=1)
        return row

    def _add_float_row(
        self, layout: QHBoxLayout, node: NodeView, port: PortView,
    ) -> None:
        v = float(port.value if port.value is not None else 0.0)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)
        slider.setValue(int(round(v * 1000)))
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0); spin.setDecimals(3); spin.setSingleStep(0.01)
        spin.setValue(v)
        slider.valueChanged.connect(lambda i: spin.setValue(i / 1000.0))

        def on_spin(value: float) -> None:
            slider.blockSignals(True)
            slider.setValue(int(round(value * 1000)))
            slider.blockSignals(False)
            self._apply_value_edit(node, port, float(value))

        spin.valueChanged.connect(on_spin)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(spin)

    def _add_vec3_row(
        self, layout: QHBoxLayout, node: NodeView, port: PortView,
        is_color: bool,
    ) -> None:
        vals = port.value
        if not isinstance(vals, (list, tuple)) or len(vals) < 3:
            vals = (0.0, 0.0, 0.0)
        spins: list[QDoubleSpinBox] = []
        for i, ch in enumerate("rgb" if is_color else "xyz"):
            layout.addWidget(QLabel(ch))
            s = QDoubleSpinBox()
            s.setRange(0.0, 1.0); s.setDecimals(3); s.setSingleStep(0.01)
            s.setValue(float(vals[i]))
            layout.addWidget(s)
            spins.append(s)

        def push() -> None:
            self._apply_value_edit(
                node, port, tuple(float(s.value()) for s in spins),
            )

        for s in spins:
            s.valueChanged.connect(lambda _v: push())

        if is_color:
            btn = QPushButton("…")
            btn.setFixedWidth(24)
            layout.addWidget(btn)

            def pick() -> None:
                rgb = tuple(s.value() for s in spins)
                init = QColor(
                    max(0, min(255, int(round(rgb[0] * 255)))),
                    max(0, min(255, int(round(rgb[1] * 255)))),
                    max(0, min(255, int(round(rgb[2] * 255)))),
                )
                chosen = QColorDialog.getColor(init, self, port.name)
                if not chosen.isValid():
                    return
                vs = (chosen.redF(), chosen.greenF(), chosen.blueF())
                for s, v in zip(spins, vs):
                    s.blockSignals(True); s.setValue(v); s.blockSignals(False)
                self._apply_value_edit(node, port, vs)

            btn.clicked.connect(pick)

    def _add_vec2_row(
        self, layout: QHBoxLayout, node: NodeView, port: PortView,
    ) -> None:
        vals = port.value
        if not isinstance(vals, (list, tuple)) or len(vals) < 2:
            vals = (0.0, 0.0)
        spins: list[QDoubleSpinBox] = []
        for i, ch in enumerate("xy"):
            layout.addWidget(QLabel(ch))
            s = QDoubleSpinBox()
            s.setRange(-1e6, 1e6); s.setDecimals(3); s.setSingleStep(0.05)
            s.setValue(float(vals[i]))
            layout.addWidget(s)
            spins.append(s)

        def push() -> None:
            self._apply_value_edit(
                node, port, tuple(float(s.value()) for s in spins),
            )

        for s in spins:
            s.valueChanged.connect(lambda _v: push())

    def _add_int_row(
        self, layout: QHBoxLayout, node: NodeView, port: PortView,
    ) -> None:
        s = QSpinBox()
        s.setRange(-1024, 1024)
        try:
            s.setValue(int(port.value or 0))
        except (TypeError, ValueError):
            s.setValue(0)
        s.valueChanged.connect(
            lambda v: self._apply_value_edit(node, port, int(v)),
        )
        layout.addWidget(s)
        layout.addStretch(1)

    def _add_bool_row(
        self, layout: QHBoxLayout, node: NodeView, port: PortView,
    ) -> None:
        cb = QCheckBox()
        cb.setChecked(bool(port.value))
        cb.toggled.connect(
            lambda v: self._apply_value_edit(node, port, bool(v)),
        )
        layout.addWidget(cb)
        layout.addStretch(1)

    def _add_filename_row(
        self, layout: QHBoxLayout, node: NodeView, port: PortView,
    ) -> None:
        lbl = QLabel(str(port.value or ""))
        lbl.setStyleSheet("color: steelblue;")
        layout.addWidget(lbl, stretch=1)
        btn = QPushButton("…")
        btn.setFixedWidth(24)

        def browse() -> None:
            path, _ = QFileDialog.getOpenFileName(self, port.name)
            if not path:
                return
            lbl.setText(path)
            self._apply_value_edit(node, port, path)

        btn.clicked.connect(browse)
        layout.addWidget(btn)

    # ── Selection ────────────────────────────────────────────────

    def _select_node(self, name: str) -> None:
        if name == self._selected_node:
            return
        self._selected_node = name
        for item_name, item in self._node_items.items():
            item.setSelected(item_name == name)
        self._refresh_side()

    def _on_scene_selection_changed(self) -> None:
        selected = self._scene.selectedItems()
        if not selected:
            return
        for item in selected:
            if isinstance(item, _NodeItem):
                self._select_node(item.node.name)
                return

    # ── Edit application (MaterialX) ─────────────────────────────

    def _doc(self):
        lib = getattr(self.renderer, "_mtlx_library", None)
        return lib.document if lib is not None else None

    def _mtlx_node(self, node_name: str):
        doc = self._doc()
        if doc is None or self._view is None:
            return None
        target = doc.getChild(self._view.target_name)
        if target is not None:
            try:
                ss_input = target.getInput("surfaceshader")
                if ss_input is not None:
                    ss = ss_input.getConnectedNode()
                    if ss is not None and ss.getName() == node_name:
                        return ss
            except Exception:
                pass
        if self._view.nodegraph_name:
            ng = doc.getNodeGraph(self._view.nodegraph_name)
            if ng is not None:
                node = ng.getNode(node_name)
                if node is not None:
                    return node
        return None

    def _set_input_value(self, inp, type_name: str, value) -> None:
        if type_name == "float":
            inp.setValue(float(value))
        elif type_name == "integer":
            inp.setValue(int(value))
        elif type_name == "boolean":
            inp.setValue(bool(value))
        elif type_name == "color3":
            r, g, b = (float(x) for x in value)
            inp.setValue(mx.Color3(r, g, b))
        elif type_name == "vector3":
            x, y, z = (float(v) for v in value)
            inp.setValue(mx.Vector3(x, y, z))
        elif type_name == "color4":
            r, g, b, a = (float(x) for x in value)
            inp.setValue(mx.Color4(r, g, b, a))
        elif type_name == "vector4":
            x, y, z, w = (float(v) for v in value)
            inp.setValue(mx.Vector4(x, y, z, w))
        elif type_name == "vector2":
            x, y = (float(v) for v in value)
            inp.setValue(mx.Vector2(x, y))
        elif type_name == "filename":
            inp.setValueString(str(value))
        elif type_name == "string":
            inp.setValueString(str(value))
        else:
            raise ValueError(f"unsupported input type: {type_name}")

    def _apply_value_edit(self, node: NodeView, port: PortView, value) -> None:
        if self._view is None:
            return
        mx_node = self._mtlx_node(node.name)
        if mx_node is None:
            self._status.setText(f"node '{node.name}' missing in doc")
            return
        inp = mx_node.getInput(port.name)
        if inp is None:
            try:
                inp = mx_node.addInput(port.name, port.type_name)
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"addInput fail: {exc}")
                return
        try:
            self._set_input_value(inp, port.type_name, value)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"setValue fail: {exc}")
            return
        port.value = value

        # Fast path: flat or std_surface direct → apply_material_override.
        # Graph-internal: regen needed.
        if self._view.flat or node.is_output:
            self.renderer.apply_material_override(
                self._view.material_id, port.name, value,
            )
            self._status.setText(f"{node.name}.{port.name} updated (flat path)")
            self._schedule_preview()
            return
        try:
            self.renderer._gen_scene_materials()
            self.renderer._upload_graph_param_buffers()
            self.renderer._material_version += 1
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"renderer update fail: {exc}")
            return
        self._status.setText(f"{node.name}.{port.name} updated (graph path)")
        self._schedule_preview()

    def _apply_connect(
        self, src_node: str, src_port: str, dst_node: str, dst_port: str,
    ) -> None:
        if self._view is None:
            return
        src_mx = self._mtlx_node(src_node)
        dst_mx = self._mtlx_node(dst_node)
        if src_mx is None or dst_mx is None:
            self._status.setText("connect: node missing")
            return
        inp = dst_mx.getInput(dst_port)
        if inp is None:
            try:
                type_name = src_mx.getType() or "float"
                inp = dst_mx.addInput(dst_port, type_name)
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"addInput fail: {exc}")
                return
        try:
            src_type = src_mx.getType()
            if src_type and inp.getType() != src_type:
                self._status.setText(
                    f"type mismatch {src_type} → {inp.getType()}"
                )
                return
        except Exception:
            pass
        try:
            inp.removeAttribute("value")
        except Exception:
            pass
        try:
            inp.setNodeName(src_node)
            if src_port and src_port != "out":
                inp.setOutputString(src_port)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"connect fail: {exc}")
            return
        self._post_topology_edit("connected")

    def _apply_disconnect(self, node_name: str, port_name: str) -> None:
        mx_node = self._mtlx_node(node_name)
        if mx_node is None:
            return
        inp = mx_node.getInput(port_name)
        if inp is None:
            return
        for attr in ("nodename", "nodegraph", "output"):
            try:
                inp.removeAttribute(attr)
            except Exception:
                pass
        self._post_topology_edit("disconnected")

    def _apply_delete_node(self, node_name: str) -> None:
        if self._view is None:
            return
        out_name = next(
            (n.name for n in self._view.nodes if n.is_output), None,
        )
        if node_name == out_name:
            self._status.setText("cannot delete the output node")
            return
        doc = self._doc()
        if doc is None or self._view.nodegraph_name is None:
            return
        ng = doc.getNodeGraph(self._view.nodegraph_name)
        if ng is None:
            return
        # Strip incoming references first.
        for n in self._view.nodes:
            for inp in n.inputs:
                if inp.connected_from and inp.connected_from[0] == node_name:
                    mx_n = self._mtlx_node(n.name)
                    if mx_n is None:
                        continue
                    mx_in = mx_n.getInput(inp.name)
                    if mx_in is None:
                        continue
                    for attr in ("nodename", "nodegraph", "output"):
                        try:
                            mx_in.removeAttribute(attr)
                        except Exception:
                            pass
        try:
            ng.removeChild(node_name)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"delete fail: {exc}")
            return
        self._post_topology_edit("deleted")

    def _apply_add_node(self, category: str, x: float, y: float) -> None:
        if self._view is None:
            return
        doc = self._doc()
        if doc is None:
            return
        if self._view.nodegraph_name is None:
            self._status.setText(
                "flat material — connect an input to a nodegraph first"
            )
            return
        ng = doc.getNodeGraph(self._view.nodegraph_name)
        if ng is None:
            return
        out_type = next((t for c, t in _ADDABLE_CATEGORIES if c == category), "float")
        i = 0
        while True:
            cand = f"{category}_{i}"
            if ng.getChild(cand) is None:
                break
            i += 1
        try:
            node = ng.addNode(category, cand, out_type)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"addNode fail: {exc}")
            return
        nv = NodeView(
            name=cand, category=category, inputs=[],
            outputs=[PortView(name="out", type_name=out_type)],
            x=x, y=y,
        )
        try:
            nd = node.getNodeDef()
        except Exception:
            nd = None
        if nd is not None:
            for nd_in in nd.getInputs():
                nv.inputs.append(PortView(
                    name=nd_in.getName(), type_name=nd_in.getType(),
                    value=_val_to_py(nd_in.getValue()),
                ))
        self._view.nodes.append(nv)
        self._post_topology_edit(f"added {category}")

    def _post_topology_edit(self, what: str) -> None:
        if self._view is None:
            return
        lib = getattr(self.renderer, "_mtlx_library", None)
        if lib is None:
            return
        new_view = build_view(
            lib.document, self._view.material_id,
            self._view.material_name, self._view.target_name,
        )
        if new_view is not None:
            old_xy = {n.name: (n.x, n.y) for n in self._view.nodes}
            for n in new_view.nodes:
                if n.name in old_xy:
                    n.x, n.y = old_xy[n.name]
            self._view = new_view
        try:
            valid, msg = lib.document.validate()
        except Exception:
            valid, msg = True, ""
        if not valid:
            self._status.setText(f"mtlx validation: {msg[:200]}")
        try:
            self.renderer._gen_scene_materials()
            self.renderer._upload_graph_param_buffers()
            self.renderer._material_version += 1
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"renderer rebuild fail: {exc}")
            self._rebuild_scene()
            self._refresh_side()
            return
        self._rebuild_scene()
        self._refresh_side()
        if valid:
            self._status.setText(f"topology updated ({what})")
        self._schedule_preview()

    # ── Preview viewport ─────────────────────────────────────────

    _PRIM_KIND = {"sphere": 0, "cube": 1, "plane": 2}

    def _refresh_env_combo(self) -> None:
        envs = getattr(self.renderer, "environments", None) or []
        names = [getattr(e, "name", f"env#{i}") for i, e in enumerate(envs)]
        prev = self._env_combo.blockSignals(True)
        self._env_combo.clear()
        self._env_combo.addItems(names)
        if names:
            idx = int(getattr(self.renderer, "env_index", 0) or 0)
            idx = max(0, min(idx, len(names) - 1))
            self._env_combo.setCurrentIndex(idx)
        self._env_combo.blockSignals(prev)

    def _on_env_changed(self, idx: int) -> None:
        envs = getattr(self.renderer, "environments", None) or []
        if not (0 <= idx < len(envs)):
            return
        self.renderer.env_index = idx
        try:
            self.renderer._ensure_env_uploaded()
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"env upload fail: {exc}")
            return
        self.renderer._material_version += 1
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        QTimer.singleShot(self.PREVIEW_DEBOUNCE_MS, self._render_preview)

    def _render_preview(self) -> None:
        if self._view is None:
            return
        prim = self._PRIM_KIND.get(self._prim_combo.currentText(), 0)
        try:
            result = self.renderer.render_material_preview(
                self._view.material_id, prim, size=self.PREVIEW_SIZE,
            )
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"preview render fail: {exc}")
            return
        if result is None:
            self._status.setText("preview unavailable")
            return
        rgba_bytes, sz = result
        arr = np.frombuffer(rgba_bytes, dtype=np.float32).reshape(sz, sz, 4)
        rgb = np.clip(arr[..., :3], 0.0, 1.0)
        rgb8 = (rgb * 255.0 + 0.5).astype(np.uint8)
        # PIL → bytes → QImage. The .copy() detaches from the numpy buffer
        # so later overwrites don't corrupt the displayed pixmap.
        img = Image.fromarray(rgb8, mode="RGB")
        data = img.tobytes("raw", "RGB")
        qimg = QImage(
            data, img.width, img.height, 3 * img.width, QImage.Format_RGB888,
        ).copy()
        self._preview_label.setPixmap(QPixmap.fromImage(qimg))

    # ── Click-to-select on scene ──────────────────────────────────

    def eventFilter(self, _obj, _event):
        return False  # not currently used
