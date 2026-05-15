"""MaterialX node-graph editor — separate Tk Toplevel window.

Renders a real boxes-and-wires view of the picked scene material's
MaterialX nodegraph (or a synthetic single-node view for flat
standard_surface materials), and lets the user edit parameter values
and topology. Value edits route through the renderer's fast-path SSBO
re-upload; topology edits trigger `_gen_scene_materials()` which lets
the existing `_graph_set_signature()` check decide whether to rebuild
the pipeline.

Lifecycle mirrors `bxdf_visualizer.BXDFVisualizer`: instantiated from
the control panel, ticked from `ControlPanel.tick()`, destroyed on
window close.
"""

from __future__ import annotations

import hashlib
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import colorchooser, filedialog, ttk
from typing import Optional

import MaterialX as mx
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError as _exc:  # pragma: no cover — pillow is a hard dep
    raise RuntimeError(
        "MaterialGraphEditor preview requires Pillow; install with "
        "`pip install Pillow`."
    ) from _exc


# ── View model ─────────────────────────────────────────────────────


@dataclass
class PortView:
    name: str
    type_name: str
    value: object | None = None
    connected_from: Optional[tuple[str, str]] = None  # (upstream_node, output_name)


@dataclass
class NodeView:
    name: str
    category: str
    inputs: list[PortView] = field(default_factory=list)
    outputs: list[PortView] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    is_output: bool = False  # the wrapping standard_surface node


@dataclass
class NodeGraphView:
    material_id: int
    material_name: str
    target_name: str
    nodes: list[NodeView]
    flat: bool
    structural_signature: str
    nodegraph_name: Optional[str] = None


# ── MaterialX → view-model construction ─────────────────────────────


def _val_to_py(val) -> object | None:
    if val is None:
        return None
    try:
        return val.getData()
    except Exception:
        return None


def _input_connected_from(inp) -> Optional[tuple[str, str]]:
    """Return (node_name, output_name) if this input is wired, else None."""
    node_name = ""
    out_name = ""
    try:
        node_name = inp.getNodeName() or ""
    except Exception:
        pass
    try:
        out_name = inp.getOutputString() or ""
    except Exception:
        pass
    if node_name:
        return (node_name, out_name or "out")
    return None


def _build_node_view(mx_node) -> NodeView:
    ports: list[PortView] = []
    for inp in mx_node.getInputs():
        pv = PortView(
            name=inp.getName(),
            type_name=inp.getType(),
            value=_val_to_py(inp.getValue()),
            connected_from=_input_connected_from(inp),
        )
        ports.append(pv)
    out_type = ""
    try:
        out_type = mx_node.getType() or ""
    except Exception:
        pass
    return NodeView(
        name=mx_node.getName(),
        category=mx_node.getCategory(),
        inputs=ports,
        outputs=[PortView(name="out", type_name=out_type)],
    )


def _structural_signature(nodes: list[NodeView]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for n in sorted(nodes, key=lambda n: n.name):
        h.update(n.name.encode() + b"|" + n.category.encode() + b"|")
        for inp in n.inputs:
            if inp.connected_from:
                up, port = inp.connected_from
                h.update(b"e:" + inp.name.encode() + b"<-"
                         + up.encode() + b"." + port.encode() + b";")
            else:
                h.update(b"v:" + inp.name.encode() + b"=" + inp.type_name.encode() + b";")
    return h.hexdigest()


def _layout(nodes: list[NodeView]) -> None:
    """Topological-depth column layout. Output node on the right."""
    by_name = {n.name: n for n in nodes}
    depth: dict[str, int] = {}

    def visit(name: str, seen: set[str]) -> int:
        if name in depth:
            return depth[name]
        if name in seen:
            return 0
        seen.add(name)
        n = by_name.get(name)
        if n is None:
            return 0
        d = 0
        for inp in n.inputs:
            if inp.connected_from:
                d = max(d, visit(inp.connected_from[0], seen) + 1)
        seen.discard(name)
        depth[name] = d
        return d

    for n in nodes:
        visit(n.name, set())

    max_d = max(depth.values(), default=0)
    cols: dict[int, list[NodeView]] = {}
    for n in nodes:
        col = max_d - depth.get(n.name, 0)
        cols.setdefault(col, []).append(n)

    NODE_W, COL_GAP, ROW_GAP = 180, 60, 28
    x0, y0 = 40, 40
    for col in sorted(cols):
        ns = cols[col]
        x = x0 + col * (NODE_W + COL_GAP)
        y = y0
        for n in ns:
            n.x = x
            n.y = y
            port_count = max(1, len(n.inputs) + len(n.outputs))
            y += 22 + port_count * 22 + 6 + ROW_GAP


def build_view(
    doc, material_id: int, material_name: str, target_name: str,
) -> Optional[NodeGraphView]:
    """Snapshot one surfacematerial as a NodeGraphView. None if unresolvable."""
    target = doc.getChild(target_name)
    if target is None:
        return None

    ss_node = None
    try:
        ss_input = target.getInput("surfaceshader")
        if ss_input is not None:
            ss_node = ss_input.getConnectedNode()
    except Exception:
        ss_node = None
    if ss_node is None:
        return None

    nodegraph_name: Optional[str] = None
    ss_inputs: list[PortView] = []
    for inp in ss_node.getInputs():
        type_name = inp.getType()
        pv = PortView(name=inp.getName(), type_name=type_name,
                      value=_val_to_py(inp.getValue()))
        # nodegraph-routed input?
        ng_name = ""
        out_name = ""
        try:
            ng_name = inp.getNodeGraphString() or ""
            out_name = inp.getOutputString() or ""
        except Exception:
            pass
        if ng_name:
            ng = doc.getNodeGraph(ng_name)
            if ng is not None and out_name:
                try:
                    out_elem = ng.getOutput(out_name)
                except Exception:
                    out_elem = None
                if out_elem is not None:
                    src = ""
                    try:
                        src = out_elem.getNodeName() or ""
                    except Exception:
                        pass
                    if src:
                        pv.connected_from = (src, "out")
                        nodegraph_name = ng.getName()
        else:
            pv.connected_from = _input_connected_from(inp)
        ss_inputs.append(pv)

    nodes_by_name: dict[str, NodeView] = {}
    if nodegraph_name:
        ng = doc.getNodeGraph(nodegraph_name)
        if ng is not None:
            for node in ng.getNodes():
                nodes_by_name[node.getName()] = _build_node_view(node)

    ss_view = NodeView(
        name=ss_node.getName(),
        category=ss_node.getCategory(),
        inputs=ss_inputs,
        outputs=[PortView(name="out", type_name="surfaceshader")],
        is_output=True,
    )
    nodes_by_name[ss_view.name] = ss_view

    flat = nodegraph_name is None
    nodes_list = list(nodes_by_name.values())
    _layout(nodes_list)
    return NodeGraphView(
        material_id=material_id,
        material_name=material_name,
        target_name=target_name,
        nodes=nodes_list,
        flat=flat,
        structural_signature=_structural_signature(nodes_list),
        nodegraph_name=nodegraph_name,
    )


# ── Toplevel window ─────────────────────────────────────────────────


_ADDABLE_CATEGORIES = (
    ("constant", "float"),
    ("multiply", "float"),
    ("add", "float"),
    ("subtract", "float"),
    ("mix", "color3"),
    ("noise3d", "float"),
    ("image", "color3"),
    ("texcoord", "vector2"),
    ("position", "vector3"),
    ("normal", "vector3"),
)


class MaterialGraphEditor:
    """Non-modal Tk Toplevel hosting the MaterialX nodegraph editor."""

    NODE_W = 180
    HEADER_H = 22
    PORT_H = 22

    def __init__(self, panel) -> None:
        self.panel = panel
        self.renderer = panel.renderer
        self._alive = True

        top = tk.Toplevel(panel.root)
        top.title("Material Graph Editor")
        top.geometry("1200x720")
        top.protocol("WM_DELETE_WINDOW", self.close)
        self.top = top

        bar = ttk.Frame(top)
        bar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(bar, text="Material:").pack(side="left", padx=(0, 4))
        self._material_var = tk.StringVar()
        self._material_combo = ttk.Combobox(
            bar, textvariable=self._material_var, state="readonly", width=50,
        )
        self._material_combo.pack(side="left")
        self._material_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_material_picked(),
        )
        ttk.Button(bar, text="Reset layout",
                   command=self._on_relayout).pack(side="left", padx=8)
        ttk.Button(bar, text="Reload",
                   command=self._refresh_material_combo).pack(side="left")

        body = ttk.Panedwindow(top, orient="horizontal")
        body.pack(fill="both", expand=True, padx=6, pady=6)

        canvas_frame = ttk.Frame(body)
        body.add(canvas_frame, weight=4)
        self._canvas = tk.Canvas(
            canvas_frame, bg="#1a1a22", highlightthickness=0,
        )
        hbar = ttk.Scrollbar(canvas_frame, orient="horizontal",
                             command=self._canvas.xview)
        vbar = ttk.Scrollbar(canvas_frame, orient="vertical",
                             command=self._canvas.yview)
        self._canvas.configure(
            xscrollcommand=hbar.set, yscrollcommand=vbar.set,
            scrollregion=(0, 0, 3000, 3000),
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="we")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        side_frame = ttk.Frame(body, width=360)
        body.add(side_frame, weight=2)

        # ── Preview viewport ────────────────────────────────────
        # Top of the side pane: primitive picker + a small GPU-rendered
        # image of the picked material on the chosen primitive. The image
        # comes from `renderer.render_material_preview` which dispatches
        # a dedicated compute pipeline that reuses the main pipeline's
        # descriptor set 0, so MaterialX procedural graphs evaluate
        # identically to the main viewport.
        prim_row = ttk.Frame(side_frame)
        prim_row.pack(fill="x", padx=8, pady=(8, 2))
        ttk.Label(prim_row, text="Primitive:").pack(side="left")
        self._preview_prim_var = tk.StringVar(value="sphere")
        prim_combo = ttk.Combobox(
            prim_row, textvariable=self._preview_prim_var,
            state="readonly", width=12,
            values=("sphere", "cube", "plane"),
        )
        prim_combo.pack(side="left", padx=4)
        prim_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._schedule_preview(),
        )

        env_row = ttk.Frame(side_frame)
        env_row.pack(fill="x", padx=8, pady=(2, 2))
        ttk.Label(env_row, text="Env light:").pack(side="left")
        self._env_var = tk.StringVar()
        self._env_combo = ttk.Combobox(
            env_row, textvariable=self._env_var,
            state="readonly", width=22,
        )
        self._env_combo.pack(side="left", padx=4, fill="x", expand=True)
        self._env_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_env_changed(),
        )

        self.PREVIEW_SIZE = 256
        self._preview_canvas = tk.Canvas(
            side_frame, width=self.PREVIEW_SIZE, height=self.PREVIEW_SIZE,
            bg="#11111a", highlightthickness=0,
        )
        self._preview_canvas.pack(padx=8, pady=(2, 4))
        self._preview_photo = None
        self._preview_image_item = self._preview_canvas.create_image(
            self.PREVIEW_SIZE // 2, self.PREVIEW_SIZE // 2,
        )
        self._preview_after_id: Optional[str] = None

        self._side_label = ttk.Label(
            side_frame, text="Pick a node to edit its inputs.",
            font=("TkDefaultFont", 10, "bold"),
        )
        self._side_label.pack(anchor="nw", padx=8, pady=(8, 4))
        self._side_container = ttk.Frame(side_frame)
        self._side_container.pack(fill="both", expand=True, padx=8)

        self._status = ttk.Label(top, text="")
        self._status.pack(fill="x", padx=6, pady=(0, 6))

        self._view: Optional[NodeGraphView] = None
        self._selected_node: Optional[str] = None
        self._materials: list[tuple[int, str, str]] = []
        self._last_scene_id: int = -1
        self._drag: Optional[dict] = None

        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<ButtonPress-3>", self._on_right)

        self._refresh_material_combo()
        self._refresh_env_combo()

    # ── Lifecycle ────────────────────────────────────────────

    def is_open(self) -> bool:
        return self._alive

    def focus(self) -> None:
        try:
            self.top.lift()
            self.top.focus_force()
        except tk.TclError:
            pass

    def close(self) -> None:
        if not self._alive:
            return
        self._alive = False
        try:
            self.top.destroy()
        except tk.TclError:
            pass

    def tick(self) -> None:
        if not self._alive:
            return
        cur = id(getattr(self.renderer, "_usd_scene", None))
        if cur != self._last_scene_id:
            self._last_scene_id = cur
            self._refresh_material_combo()
            self._refresh_env_combo()

    # ── Material picker ──────────────────────────────────────

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
        self._material_combo.configure(values=labels)
        if not opts:
            self._material_combo.set("")
            self._view = None
            self._clear_canvas()
            self._refresh_side()
            self._status.configure(text="No MaterialX materials in scene.")
            return
        sel_idx = 0
        if self._view is not None:
            for k, (mid, _, _) in enumerate(opts):
                if mid == self._view.material_id:
                    sel_idx = k
                    break
        self._material_combo.current(sel_idx)
        self._on_material_picked()

    def _on_material_picked(self) -> None:
        idx = self._material_combo.current()
        if idx < 0 or idx >= len(self._materials):
            return
        mid, name, target = self._materials[idx]
        lib = getattr(self.renderer, "_mtlx_library", None)
        if lib is None:
            self._status.configure(text="MaterialX library not loaded.")
            return
        try:
            view = build_view(lib.document, mid, name, target)
        except Exception as e:  # noqa: BLE001
            self._status.configure(text=f"build_view error: {e}")
            return
        if view is None:
            self._status.configure(text=f"Could not resolve target '{target}'.")
            self._view = None
            self._clear_canvas()
            self._refresh_side()
            return
        self._view = view
        self._selected_node = None
        self._draw()
        self._refresh_side()
        suffix = "  [flat std_surface]" if view.flat else ""
        self._status.configure(text=f"{len(view.nodes)} node(s){suffix}")
        self._schedule_preview()

    def _on_relayout(self) -> None:
        if self._view is None:
            return
        _layout(self._view.nodes)
        self._draw()

    # ── Canvas drawing ───────────────────────────────────────

    def _clear_canvas(self) -> None:
        self._canvas.delete("all")

    def _node_height(self, n: NodeView) -> int:
        return self.HEADER_H + max(1, len(n.inputs) + len(n.outputs)) * self.PORT_H + 6

    def _in_port_pos(self, n: NodeView, i: int) -> tuple[float, float]:
        return (n.x, n.y + self.HEADER_H + i * self.PORT_H + self.PORT_H / 2)

    def _out_port_pos(self, n: NodeView, i: int) -> tuple[float, float]:
        return (n.x + self.NODE_W,
                n.y + self.HEADER_H + (len(n.inputs) + i) * self.PORT_H + self.PORT_H / 2)

    def _draw(self) -> None:
        self._clear_canvas()
        if self._view is None:
            return
        c = self._canvas
        by_name = {n.name: n for n in self._view.nodes}

        # Wires (drawn first → under nodes).
        for n in self._view.nodes:
            for i, inp in enumerate(n.inputs):
                if not inp.connected_from:
                    continue
                up_name, up_port = inp.connected_from
                up = by_name.get(up_name)
                if up is None:
                    continue
                op_idx = 0
                for k, op in enumerate(up.outputs):
                    if op.name == up_port:
                        op_idx = k
                        break
                x1, y1 = self._out_port_pos(up, op_idx)
                x2, y2 = self._in_port_pos(n, i)
                c.create_line(x1, y1, x2, y2, fill="#a0a0c0", width=2)

        for n in self._view.nodes:
            self._draw_node(n)

        if self._view.nodes:
            xs = [n.x for n in self._view.nodes] + \
                 [n.x + self.NODE_W for n in self._view.nodes]
            ys = [n.y for n in self._view.nodes] + \
                 [n.y + self._node_height(n) for n in self._view.nodes]
            c.configure(scrollregion=(min(xs) - 60, min(ys) - 60,
                                      max(xs) + 60, max(ys) + 60))

    def _draw_node(self, n: NodeView) -> None:
        c = self._canvas
        h = self._node_height(n)
        sel = (n.name == self._selected_node)
        outline = "#ffe188" if sel else "#444466"
        c.create_rectangle(
            n.x, n.y, n.x + self.NODE_W, n.y + h,
            fill="#2c2c3c", outline=outline, width=(3 if sel else 1),
            tags=("node", f"node:{n.name}"),
        )
        header_fill = "#5a3a8e" if n.is_output else "#3a3a5a"
        c.create_rectangle(
            n.x, n.y, n.x + self.NODE_W, n.y + self.HEADER_H,
            fill=header_fill, outline=outline,
            tags=("node_header", f"node:{n.name}"),
        )
        c.create_text(
            n.x + 6, n.y + self.HEADER_H / 2,
            text=f"{n.category}  ({n.name})", anchor="w",
            fill="#e8e8f0", font=("TkDefaultFont", 9, "bold"),
            tags=("node_label", f"node:{n.name}"),
        )
        for i, inp in enumerate(n.inputs):
            px, py = self._in_port_pos(n, i)
            fill = "#ffd060" if inp.connected_from else "#80c0ff"
            c.create_oval(
                px - 5, py - 5, px + 5, py + 5,
                fill=fill, outline="#11111a",
                tags=("port_in", f"port_in:{n.name}:{inp.name}"),
            )
            c.create_text(
                px + 10, py, text=inp.name, anchor="w",
                fill="#c8c8d8", font=("TkDefaultFont", 8),
                tags=("node_label", f"node:{n.name}"),
            )
        for i, op in enumerate(n.outputs):
            px, py = self._out_port_pos(n, i)
            c.create_oval(
                px - 5, py - 5, px + 5, py + 5,
                fill="#80ff90", outline="#11111a",
                tags=("port_out", f"port_out:{n.name}:{op.name}"),
            )
            c.create_text(
                px - 10, py, text=op.name, anchor="e",
                fill="#c8c8d8", font=("TkDefaultFont", 8),
                tags=("node_label", f"node:{n.name}"),
            )

    # ── Hit-test / mouse handling ────────────────────────────

    def _canvas_xy(self, e) -> tuple[float, float]:
        return (self._canvas.canvasx(e.x), self._canvas.canvasy(e.y))

    def _hit(self, x: float, y: float) -> Optional[tuple[str, str, str]]:
        c = self._canvas
        ids = c.find_overlapping(x - 1, y - 1, x + 1, y + 1)
        for cid in reversed(ids):
            tags = c.gettags(cid)
            for t in tags:
                if t.startswith("port_in:"):
                    _, nn, pp = t.split(":", 2)
                    return ("port_in", nn, pp)
                if t.startswith("port_out:"):
                    _, nn, pp = t.split(":", 2)
                    return ("port_out", nn, pp)
            for t in tags:
                if t.startswith("node:"):
                    return ("node", t.split(":", 1)[1], "")
        return None

    def _on_press(self, e) -> None:
        if self._view is None:
            return
        x, y = self._canvas_xy(e)
        hit = self._hit(x, y)
        if hit is None:
            self._drag = None
            return
        kind, node, port = hit
        if kind == "port_out":
            self._drag = {
                "kind": "wire",
                "from_node": node, "from_port": port,
                "line": self._canvas.create_line(
                    x, y, x, y, fill="#ffe188", width=2, dash=(4, 2),
                ),
            }
            return
        self._select_node(node)
        n = next((nn for nn in self._view.nodes if nn.name == node), None)
        if n is not None:
            self._drag = {"kind": "node", "node": node,
                          "dx": x - n.x, "dy": y - n.y}

    def _on_drag(self, e) -> None:
        if not self._drag or self._view is None:
            return
        x, y = self._canvas_xy(e)
        if self._drag["kind"] == "node":
            n = next((nn for nn in self._view.nodes
                      if nn.name == self._drag["node"]), None)
            if n is not None:
                n.x = x - self._drag["dx"]
                n.y = y - self._drag["dy"]
                self._draw()
        elif self._drag["kind"] == "wire":
            n = next((nn for nn in self._view.nodes
                      if nn.name == self._drag["from_node"]), None)
            if n is not None:
                fx, fy = self._out_port_pos(n, 0)
                self._canvas.coords(self._drag["line"], fx, fy, x, y)

    def _on_release(self, e) -> None:
        if not self._drag or self._view is None:
            self._drag = None
            return
        x, y = self._canvas_xy(e)
        if self._drag["kind"] == "wire":
            self._canvas.delete(self._drag["line"])
            hit = self._hit(x, y)
            if hit and hit[0] == "port_in":
                self._apply_connect(
                    self._drag["from_node"], self._drag["from_port"],
                    hit[1], hit[2],
                )
        self._drag = None

    def _on_right(self, e) -> None:
        if self._view is None:
            return
        x, y = self._canvas_xy(e)
        hit = self._hit(x, y)
        menu = tk.Menu(self.top, tearoff=0)
        if hit is None:
            add_menu = tk.Menu(menu, tearoff=0)
            for cat, _t in _ADDABLE_CATEGORIES:
                add_menu.add_command(
                    label=cat,
                    command=lambda c=cat, sx=x, sy=y: self._apply_add_node(c, sx, sy),
                )
            menu.add_cascade(label="Add node…", menu=add_menu)
        elif hit[0] == "node":
            menu.add_command(
                label="Delete node",
                command=lambda nn=hit[1]: self._apply_delete_node(nn),
            )
        elif hit[0] == "port_in":
            menu.add_command(
                label="Disconnect",
                command=lambda nn=hit[1], pp=hit[2]: self._apply_disconnect(nn, pp),
            )
        try:
            menu.tk_popup(e.x_root, e.y_root)
        finally:
            menu.grab_release()

    def _select_node(self, name: str) -> None:
        if name == self._selected_node:
            return
        self._selected_node = name
        self._draw()
        self._refresh_side()

    # ── Side-pane parameter editor ───────────────────────────

    def _refresh_side(self) -> None:
        for w in self._side_container.winfo_children():
            w.destroy()
        if self._view is None or self._selected_node is None:
            self._side_label.configure(text="Pick a node to edit its inputs.")
            return
        node = next((n for n in self._view.nodes
                     if n.name == self._selected_node), None)
        if node is None:
            return
        self._side_label.configure(text=f"{node.category}  /  {node.name}")
        if not node.inputs:
            ttk.Label(self._side_container, text="(no inputs)",
                      foreground="#808090").pack(anchor="w", pady=4)
            return
        for inp in node.inputs:
            self._build_input_row(self._side_container, node, inp)

    def _build_input_row(self, parent, node: NodeView, port: PortView) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=port.name, width=14, anchor="w").pack(side="left")
        if port.connected_from:
            up, op = port.connected_from
            ttk.Label(row, text=f"← {up}.{op}",
                      foreground="#a0a0c0").pack(side="left", padx=4)
            return

        t = port.type_name
        if t == "float":
            v = float(port.value if port.value is not None else 0.0)
            var = tk.DoubleVar(value=v)
            lbl = ttk.Label(row, text=f"{v:.3f}", width=8)

            def on_change(_v, vv=var, n=node, p=port, lbl=lbl):
                val = float(vv.get())
                lbl.configure(text=f"{val:.3f}")
                self._apply_value_edit(n, p, val)
            scale = ttk.Scale(row, from_=0.0, to=1.0, orient="horizontal",
                              variable=var, length=180, command=on_change)
            scale.pack(side="left", fill="x", expand=True, padx=4)
            lbl.pack(side="left")
        elif t in ("color3", "vector3"):
            vals = port.value
            if not isinstance(vals, (list, tuple)) or len(vals) < 3:
                vals = (0.0, 0.0, 0.0)
            rv = tk.DoubleVar(value=float(vals[0]))
            gv = tk.DoubleVar(value=float(vals[1]))
            bv = tk.DoubleVar(value=float(vals[2]))

            def push(_v=None, n=node, p=port, r=rv, g=gv, b=bv):
                self._apply_value_edit(
                    n, p, (float(r.get()), float(g.get()), float(b.get())),
                )
            for var, ch in ((rv, "r"), (gv, "g"), (bv, "b")):
                f = ttk.Frame(row)
                f.pack(side="left", padx=2)
                ttk.Label(f, text=ch).pack(side="left")
                ttk.Scale(f, from_=0.0, to=1.0, variable=var, length=70,
                          command=push).pack(side="left")
            if t == "color3":
                def pick(r=rv, g=gv, b=bv, p=push):
                    init = "#{0:02x}{1:02x}{2:02x}".format(
                        int(max(0, min(1, r.get())) * 255),
                        int(max(0, min(1, g.get())) * 255),
                        int(max(0, min(1, b.get())) * 255),
                    )
                    rgb, _hex = colorchooser.askcolor(
                        initialcolor=init, parent=self.top,
                    )
                    if rgb is None:
                        return
                    r.set(rgb[0] / 255.0)
                    g.set(rgb[1] / 255.0)
                    b.set(rgb[2] / 255.0)
                    p()
                ttk.Button(row, text="…", width=2,
                           command=pick).pack(side="left", padx=2)
        elif t == "vector2":
            vals = port.value
            if not isinstance(vals, (list, tuple)) or len(vals) < 2:
                vals = (0.0, 0.0)
            xv = tk.DoubleVar(value=float(vals[0]))
            yv = tk.DoubleVar(value=float(vals[1]))

            def push2(_v=None, n=node, p=port, x=xv, y=yv):
                self._apply_value_edit(n, p, (float(x.get()), float(y.get())))
            ttk.Scale(row, from_=0.0, to=1.0, variable=xv, length=100,
                      command=push2).pack(side="left")
            ttk.Scale(row, from_=0.0, to=1.0, variable=yv, length=100,
                      command=push2).pack(side="left")
        elif t == "integer":
            iv = tk.IntVar(value=int(port.value or 0))
            sp = ttk.Spinbox(row, from_=-1024, to=1024,
                             textvariable=iv, width=8,
                             command=lambda n=node, p=port, v=iv:
                                 self._apply_value_edit(n, p, int(v.get())))
            sp.pack(side="left")
        elif t == "boolean":
            bvar = tk.BooleanVar(value=bool(port.value))
            ttk.Checkbutton(
                row, variable=bvar,
                command=lambda n=node, p=port, v=bvar:
                    self._apply_value_edit(n, p, bool(v.get())),
            ).pack(side="left")
        elif t == "filename":
            sv = tk.StringVar(value=str(port.value or ""))
            ent = ttk.Entry(row, textvariable=sv, width=30)
            ent.pack(side="left", fill="x", expand=True)

            def browse(v=sv, n=node, p=port):
                path = filedialog.askopenfilename(parent=self.top)
                if not path:
                    return
                v.set(path)
                self._apply_value_edit(n, p, path)
            ttk.Button(row, text="…", width=2,
                       command=browse).pack(side="left", padx=2)
        else:
            ttk.Label(row, text=f"(type {t} not editable)",
                      foreground="#808090").pack(side="left", padx=4)

    # ── Edit application ─────────────────────────────────────

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
            self._status.configure(text=f"node '{node.name}' missing in doc")
            return
        inp = mx_node.getInput(port.name)
        if inp is None:
            try:
                inp = mx_node.addInput(port.name, port.type_name)
            except Exception as e:  # noqa: BLE001
                self._status.configure(text=f"addInput fail: {e}")
                return
        try:
            self._set_input_value(inp, port.type_name, value)
        except Exception as e:  # noqa: BLE001
            self._status.configure(text=f"setValue fail: {e}")
            return
        port.value = value

        # Fast path:
        #  - flat material → use renderer.apply_material_override (StdSurfaceParams).
        #  - std_surface direct input on a graph material → same path (also mirrors
        #    into _material_graph_overrides via renderer.py:3099).
        #  - input on a graph-internal node → regen needed (uniforms live in the
        #    per-graph SSBO with gen-mangled names; cheapest correct path is to
        #    re-run _gen_scene_materials and re-upload).
        if self._view.flat or node.is_output:
            self.renderer.apply_material_override(
                self._view.material_id, port.name, value,
            )
            self._status.configure(
                text=f"{node.name}.{port.name} updated (flat path)",
            )
            self._schedule_preview()
            return

        try:
            self.renderer._gen_scene_materials()
            self.renderer._upload_graph_param_buffers()
            self.renderer._material_version += 1
        except Exception as e:  # noqa: BLE001
            self._status.configure(text=f"renderer update fail: {e}")
            return
        self._status.configure(
            text=f"{node.name}.{port.name} updated (graph path)",
        )
        self._schedule_preview()

    def _apply_connect(
        self, src_node: str, src_port: str, dst_node: str, dst_port: str,
    ) -> None:
        if self._view is None:
            return
        src_mx = self._mtlx_node(src_node)
        dst_mx = self._mtlx_node(dst_node)
        if src_mx is None or dst_mx is None:
            self._status.configure(text="connect: node missing")
            return
        inp = dst_mx.getInput(dst_port)
        if inp is None:
            try:
                type_name = src_mx.getType() or "float"
                inp = dst_mx.addInput(dst_port, type_name)
            except Exception as e:  # noqa: BLE001
                self._status.configure(text=f"addInput fail: {e}")
                return
        try:
            src_type = src_mx.getType()
            if src_type and inp.getType() != src_type:
                self._status.configure(
                    text=f"type mismatch {src_type} → {inp.getType()}",
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
        except Exception as e:  # noqa: BLE001
            self._status.configure(text=f"connect fail: {e}")
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
            self._status.configure(text="cannot delete the output node")
            return
        doc = self._doc()
        if doc is None or self._view.nodegraph_name is None:
            return
        ng = doc.getNodeGraph(self._view.nodegraph_name)
        if ng is None:
            return
        # Strip any incoming references to this node first.
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
        except Exception as e:  # noqa: BLE001
            self._status.configure(text=f"delete fail: {e}")
            return
        self._post_topology_edit("deleted")

    def _apply_add_node(self, category: str, x: float, y: float) -> None:
        if self._view is None:
            return
        doc = self._doc()
        if doc is None:
            return
        if self._view.nodegraph_name is None:
            self._status.configure(
                text="flat material — connect an input to a nodegraph first",
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
        except Exception as e:  # noqa: BLE001
            self._status.configure(text=f"addNode fail: {e}")
            return
        nv = NodeView(
            name=cand, category=category,
            inputs=[],
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
                    name=nd_in.getName(),
                    type_name=nd_in.getType(),
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
            self._status.configure(text=f"mtlx validation: {msg[:200]}")
        try:
            self.renderer._gen_scene_materials()
            self.renderer._upload_graph_param_buffers()
            self.renderer._material_version += 1
        except Exception as e:  # noqa: BLE001
            self._status.configure(text=f"renderer rebuild fail: {e}")
            self._draw()
            self._refresh_side()
            return
        self._draw()
        self._refresh_side()
        if valid:
            self._status.configure(text=f"topology updated ({what})")
        self._schedule_preview()

    # ── Preview viewport ─────────────────────────────────────

    _PRIM_KIND = {"sphere": 0, "cube": 1, "plane": 2}

    def _refresh_env_combo(self) -> None:
        envs = getattr(self.renderer, "environments", None) or []
        names = [getattr(e, "name", f"env#{i}") for i, e in enumerate(envs)]
        self._env_combo.configure(values=names)
        if not names:
            self._env_var.set("")
            return
        idx = int(getattr(self.renderer, "env_index", 0) or 0)
        idx = max(0, min(idx, len(names) - 1))
        self._env_var.set(names[idx])

    def _on_env_changed(self) -> None:
        envs = getattr(self.renderer, "environments", None) or []
        target = self._env_var.get()
        for i, e in enumerate(envs):
            if getattr(e, "name", "") == target:
                self.renderer.env_index = i
                try:
                    self.renderer._ensure_env_uploaded()
                except Exception as exc:  # noqa: BLE001
                    self._status.configure(text=f"env upload fail: {exc}")
                    return
                self.renderer._material_version += 1
                self._schedule_preview()
                return

    def _schedule_preview(self) -> None:
        """Debounced trigger for re-rendering the preview canvas."""
        if not self._alive:
            return
        try:
            if self._preview_after_id is not None:
                self.top.after_cancel(self._preview_after_id)
        except tk.TclError:
            pass
        try:
            self._preview_after_id = self.top.after(
                120, self._render_preview,
            )
        except tk.TclError:
            self._preview_after_id = None

    def _render_preview(self) -> None:
        self._preview_after_id = None
        if not self._alive or self._view is None:
            return
        prim = self._PRIM_KIND.get(self._preview_prim_var.get(), 0)
        try:
            result = self.renderer.render_material_preview(
                self._view.material_id, prim,
                size=self.PREVIEW_SIZE,
            )
        except Exception as e:  # noqa: BLE001
            self._status.configure(text=f"preview render fail: {e}")
            return
        if result is None:
            self._status.configure(text="preview unavailable")
            return
        rgba_bytes, sz = result
        arr = np.frombuffer(rgba_bytes, dtype=np.float32).reshape(sz, sz, 4)
        # Shader already applied Reinhard + gamma 2.2, so just clamp + scale.
        rgb8 = np.clip(arr[..., :3], 0.0, 1.0)
        rgb8 = (rgb8 * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(rgb8, mode="RGB")
        self._preview_photo = ImageTk.PhotoImage(img)
        try:
            self._preview_canvas.itemconfigure(
                self._preview_image_item, image=self._preview_photo,
            )
        except tk.TclError:
            pass
