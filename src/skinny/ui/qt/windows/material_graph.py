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
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction, QBrush, QColor, QFont, QImage, QPainter,
    QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDockWidget,
    QDoubleSpinBox, QFileDialog, QGraphicsItem, QGraphicsPathItem,
    QGraphicsScene, QGraphicsSceneMouseEvent, QGraphicsView, QHBoxLayout, QLabel, QMenu, QPushButton, QScrollArea, QSlider, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from skinny.mtlx_graph_view import (
    _ADDABLE_CATEGORIES, NodeGraphView, NodeView, PortView, _layout,
    build_view,
)


# ── Worker-side MaterialX helpers ─────────────────────────────────
# The MaterialX document lives on the renderer, which the render worker owns.
# These run inside `renderer.request(...)`/`post(...)` closures on the worker,
# taking the *real* renderer/doc (never the GUI-side proxy).

def _worker_doc(renderer):
    lib = getattr(renderer, "_mtlx_library", None)
    return lib.document if lib is not None else None


def _worker_mtlx_node(doc, view, node_name):
    if doc is None or view is None:
        return None
    target = doc.getChild(view.target_name)
    if target is not None:
        try:
            ss_input = target.getInput("surfaceshader")
            if ss_input is not None:
                ss = ss_input.getConnectedNode()
                if ss is not None and ss.getName() == node_name:
                    return ss
        except Exception:  # noqa: BLE001
            pass
    if view.nodegraph_name:
        ng = doc.getNodeGraph(view.nodegraph_name)
        if ng is not None:
            node = ng.getNode(node_name)
            if node is not None:
                return node
    return None


def _set_mtlx_input(inp, type_name: str, value) -> None:
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


def _build_view_on_worker(renderer, mid, name, target):
    doc = _worker_doc(renderer)
    if doc is None:
        return ("no_lib", None)
    return ("ok", build_view(doc, mid, name, target))


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

    # Marshal a worker-thread callback (scene-state refresh, build_view, topology
    # edit result, preview render) onto the GUI thread.
    _run_on_gui = Signal(object)

    def __init__(self, renderer, parent: QWidget | None = None) -> None:
        super().__init__("Material Graph Editor", parent)
        self.renderer = renderer
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self._run_on_gui.connect(self._invoke_on_gui)
        self._state_inflight = False

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

    def _invoke_on_gui(self, fn) -> None:
        try:
            fn()
        except RuntimeError:
            pass

    def _resolve_to_gui(self, fut, handler) -> None:
        """Resolve a worker `Future` off-thread and hand ``("ok", value)`` or
        ``("exc", exception)`` to `handler` on the GUI thread — the future is
        never awaited synchronously on the GUI thread."""
        def done(f) -> None:
            try:
                data = ("ok", f.result())
            except Exception as exc:  # noqa: BLE001
                data = ("exc", exc)
            self._run_on_gui.emit(lambda data=data: handler(data))
        fut.add_done_callback(done)

    def _refresh_material_combo(self) -> None:
        mats = self._scene_materials()
        opts: list[tuple[int, str, str]] = []
        for i, mat in enumerate(mats):
            if i == 0:
                continue
            # The material projection carries both the authored target and the
            # `_mtlx_scene_materials` fallback target.
            target = (
                getattr(mat, "mtlx_target_name", None)
                or getattr(mat, "mtlx_scene_target", None)
            )
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
        # build_view reads the live MaterialX document — build it on the worker.
        fut = self.renderer.request(
            lambda r, mid=mid, name=name, target=target:
                _build_view_on_worker(r, mid, name, target),
        )
        self._resolve_to_gui(
            fut, lambda data, target=target: self._apply_picked_view(data, target),
        )

    def _apply_picked_view(self, data, target: str) -> None:
        kind, payload = data
        if kind == "exc":
            self._status.setText(f"build_view error: {payload}")
            return
        status, view = payload
        if status == "no_lib":
            self._status.setText("MaterialX library not loaded.")
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
        # Pull a fresh scene projection from the render worker, then compare on
        # the GUI thread (stable projected id, not a per-tick projection object).
        if self._state_inflight:
            return
        self._state_inflight = True
        self._resolve_to_gui(
            self.renderer.refresh_scene_state(), self._apply_state_poll,
        )

    def _apply_state_poll(self, data) -> None:
        self._state_inflight = False
        kind, payload = data
        state = payload if kind == "ok" else None
        if state is not None:
            self.renderer.apply_scene_state(state)
        cur = getattr(self.renderer, "_usd_scene_id", 0)
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

    def _run_edit(self, mutate, what: str, *, rebuild_view: bool = True,
                  place_xy=None) -> None:
        """Run a MaterialX doc mutation on the render worker, then rebuild the
        view / regenerate materials / re-upload — all atomically on the worker —
        and marshal the result back to the GUI thread.

        `mutate(renderer) -> None | error_str` performs the doc edit.
        """
        view = self._view
        if view is None:
            return
        mid, name, target = (
            view.material_id, view.material_name, view.target_name,
        )

        def worker(r, mutate=mutate, mid=mid, name=name, target=target,
                   rebuild_view=rebuild_view):
            err = mutate(r)
            if err is not None:
                return ("error", err, None, True, "")
            doc = _worker_doc(r)
            new_view = None
            valid, msg = True, ""
            if rebuild_view and doc is not None:
                new_view = build_view(doc, mid, name, target)
                try:
                    valid, msg = doc.validate()
                except Exception:  # noqa: BLE001
                    valid, msg = True, ""
            try:
                r._gen_scene_materials()
                r._upload_graph_param_buffers()
                r._material_version += 1
            except Exception as exc:  # noqa: BLE001
                return ("rebuild_error", repr(exc), new_view, valid, msg)
            return ("ok", "", new_view, valid, msg)

        self._resolve_to_gui(
            self.renderer.request(worker),
            lambda data, what=what, place_xy=place_xy:
                self._on_edit_result(data, what, place_xy),
        )

    def _on_edit_result(self, data, what: str, place_xy) -> None:
        kind, payload = data
        if kind == "exc":
            self._status.setText(f"edit failed: {payload}")
            return
        status, detail, new_view, valid, msg = payload
        if status == "error":
            self._status.setText(detail)
            return
        if new_view is not None:
            old_xy = (
                {n.name: (n.x, n.y) for n in self._view.nodes}
                if self._view is not None else {}
            )
            for n in new_view.nodes:
                if n.name in old_xy:
                    n.x, n.y = old_xy[n.name]
                elif place_xy is not None:
                    n.x, n.y = place_xy
            self._view = new_view
            self._rebuild_scene()
            self._refresh_side()
        if status == "rebuild_error":
            self._status.setText(f"renderer rebuild fail: {detail}")
            return
        if not valid:
            self._status.setText(f"mtlx validation: {msg[:200]}")
        else:
            self._status.setText(f"updated ({what})")
        self._schedule_preview()

    def _apply_value_edit(self, node: NodeView, port: PortView, value) -> None:
        if self._view is None:
            return
        port.value = value  # optimistic GUI-side update
        view = self._view
        node_name, port_name, type_name = node.name, port.name, port.type_name

        # Fast path: flat or std_surface direct → apply_material_override.
        if view.flat or node.is_output:
            self.renderer.apply_material_override(
                view.material_id, port_name, value,
            )
            self._status.setText(f"{node_name}.{port_name} updated (flat path)")
            self._schedule_preview()
            return

        # Graph-internal: mutate the doc input on the worker, then regen.
        def mutate(r, view=view, node_name=node_name, port_name=port_name,
                   type_name=type_name, value=value):
            doc = _worker_doc(r)
            mx_node = _worker_mtlx_node(doc, view, node_name)
            if mx_node is None:
                return f"node '{node_name}' missing in doc"
            inp = mx_node.getInput(port_name)
            if inp is None:
                try:
                    inp = mx_node.addInput(port_name, type_name)
                except Exception as exc:  # noqa: BLE001
                    return f"addInput fail: {exc}"
            try:
                _set_mtlx_input(inp, type_name, value)
            except Exception as exc:  # noqa: BLE001
                return f"setValue fail: {exc}"
            return None

        self._run_edit(
            mutate, f"{node_name}.{port_name} (graph path)", rebuild_view=False,
        )

    def _apply_connect(
        self, src_node: str, src_port: str, dst_node: str, dst_port: str,
    ) -> None:
        if self._view is None:
            return

        def mutate(r, view=self._view, src_node=src_node, src_port=src_port,
                   dst_node=dst_node, dst_port=dst_port):
            doc = _worker_doc(r)
            src_mx = _worker_mtlx_node(doc, view, src_node)
            dst_mx = _worker_mtlx_node(doc, view, dst_node)
            if src_mx is None or dst_mx is None:
                return "connect: node missing"
            inp = dst_mx.getInput(dst_port)
            if inp is None:
                try:
                    type_name = src_mx.getType() or "float"
                    inp = dst_mx.addInput(dst_port, type_name)
                except Exception as exc:  # noqa: BLE001
                    return f"addInput fail: {exc}"
            try:
                src_type = src_mx.getType()
                if src_type and inp.getType() != src_type:
                    return f"type mismatch {src_type} → {inp.getType()}"
            except Exception:  # noqa: BLE001
                pass
            try:
                inp.removeAttribute("value")
            except Exception:  # noqa: BLE001
                pass
            try:
                inp.setNodeName(src_node)
                if src_port and src_port != "out":
                    inp.setOutputString(src_port)
            except Exception as exc:  # noqa: BLE001
                return f"connect fail: {exc}"
            return None

        self._run_edit(mutate, "connected")

    def _apply_disconnect(self, node_name: str, port_name: str) -> None:
        if self._view is None:
            return

        def mutate(r, view=self._view, node_name=node_name, port_name=port_name):
            doc = _worker_doc(r)
            mx_node = _worker_mtlx_node(doc, view, node_name)
            if mx_node is None:
                return None
            inp = mx_node.getInput(port_name)
            if inp is None:
                return None
            for attr in ("nodename", "nodegraph", "output"):
                try:
                    inp.removeAttribute(attr)
                except Exception:  # noqa: BLE001
                    pass
            return None

        self._run_edit(mutate, "disconnected")

    def _apply_delete_node(self, node_name: str) -> None:
        if self._view is None:
            return
        view = self._view
        out_name = next((n.name for n in view.nodes if n.is_output), None)
        if node_name == out_name:
            self._status.setText("cannot delete the output node")
            return
        # Capture incoming references from the GUI-side view for the worker.
        incoming = [
            (n.name, inp.name)
            for n in view.nodes for inp in n.inputs
            if inp.connected_from and inp.connected_from[0] == node_name
        ]

        def mutate(r, view=view, node_name=node_name, incoming=incoming):
            doc = _worker_doc(r)
            if doc is None or view.nodegraph_name is None:
                return None
            ng = doc.getNodeGraph(view.nodegraph_name)
            if ng is None:
                return None
            for (n_name, inp_name) in incoming:
                mx_n = _worker_mtlx_node(doc, view, n_name)
                if mx_n is None:
                    continue
                mx_in = mx_n.getInput(inp_name)
                if mx_in is None:
                    continue
                for attr in ("nodename", "nodegraph", "output"):
                    try:
                        mx_in.removeAttribute(attr)
                    except Exception:  # noqa: BLE001
                        pass
            try:
                ng.removeChild(node_name)
            except Exception as exc:  # noqa: BLE001
                return f"delete fail: {exc}"
            return None

        self._run_edit(mutate, "deleted")

    def _apply_add_node(self, category: str, x: float, y: float) -> None:
        if self._view is None:
            return
        view = self._view
        if view.nodegraph_name is None:
            self._status.setText(
                "flat material — connect an input to a nodegraph first"
            )
            return

        def mutate(r, view=view, category=category):
            doc = _worker_doc(r)
            if doc is None:
                return None
            ng = doc.getNodeGraph(view.nodegraph_name)
            if ng is None:
                return "nodegraph missing"
            out_type = next(
                (t for c, t in _ADDABLE_CATEGORIES if c == category), "float",
            )
            i = 0
            while True:
                cand = f"{category}_{i}"
                if ng.getChild(cand) is None:
                    break
                i += 1
            try:
                ng.addNode(category, cand, out_type)
            except Exception as exc:  # noqa: BLE001
                return f"addNode fail: {exc}"
            return None

        # build_view (worker rebuild) picks up the new node + its nodedef inputs;
        # place_xy drops the new node at the requested position.
        self._run_edit(mutate, f"added {category}", place_xy=(x, y))

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
        self.renderer.env_index = idx  # proxy posts the attr set
        self.renderer.ensure_env_uploaded()  # worker: upload + version bump
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        QTimer.singleShot(self.PREVIEW_DEBOUNCE_MS, self._render_preview)

    def _render_preview(self) -> None:
        if self._view is None:
            return
        prim = self._PRIM_KIND.get(self._prim_combo.currentText(), 0)
        # render_material_preview is GPU work — run it on the worker and blit the
        # returned pixels on the GUI thread.
        self._resolve_to_gui(
            self.renderer.render_material_preview(
                self._view.material_id, prim, self.PREVIEW_SIZE,
            ),
            self._apply_preview,
        )

    def _apply_preview(self, data) -> None:
        kind, payload = data
        if kind == "exc":
            self._status.setText(f"preview render fail: {payload}")
            return
        result = payload
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
