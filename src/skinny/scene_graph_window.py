"""Separate Toplevel window showing the USD scene graph as a tree view
with an editable property panel for the selected node.

Opened from the control panel via a "Scene Graph" button. The tree
mirrors the USD prim hierarchy; selecting a node populates the right-
hand properties panel with typed widgets (sliders, color pickers,
vec3 entries) whose edits flow through the renderer's existing
``apply_material_override`` / ``apply_light_override`` /
``apply_instance_transform`` methods.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import colorchooser, ttk
from typing import Optional

import numpy as np

from skinny.scene_graph import (
    SceneGraphNode,
    SceneGraphProperty,
    find_node_by_path,
    type_icon,
)


class SceneGraphWindow:
    """Non-modal Toplevel with a tree view (left) and property editor (right)."""

    TREE_WIDTH = 320
    PROPS_WIDTH = 380
    WIN_HEIGHT = 560

    def __init__(self, panel) -> None:
        self.panel = panel
        self.renderer = panel.renderer
        self._alive = True
        self._suppress_cb = False
        self._last_graph_id: int = -1

        top = tk.Toplevel(panel.root)
        top.title("Scene Graph")
        top.geometry(f"{self.TREE_WIDTH + self.PROPS_WIDTH}x{self.WIN_HEIGHT}")
        top.protocol("WM_DELETE_WINDOW", self.close)
        self.top = top

        # ── Paned layout ─────────────────────────────────────────
        paned = ttk.PanedWindow(top, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        # Left: tree
        tree_frame = ttk.Frame(paned)
        paned.add(tree_frame, weight=1)

        self.tree = ttk.Treeview(
            tree_frame, columns=("type",), show="tree headings",
            selectmode="browse",
        )
        self.tree.heading("#0", text="Name", anchor="w")
        self.tree.heading("type", text="Type", anchor="w")
        self.tree.column("#0", width=200, stretch=True)
        self.tree.column("type", width=100, stretch=False)

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Right: properties (scrollable)
        props_outer = ttk.Frame(paned)
        paned.add(props_outer, weight=1)

        props_label = ttk.Label(props_outer, text="Properties", font=("", 10, "bold"))
        props_label.pack(fill="x", padx=4, pady=(4, 2))

        self._props_canvas = tk.Canvas(props_outer, highlightthickness=0)
        props_scroll = ttk.Scrollbar(props_outer, orient="vertical",
                                      command=self._props_canvas.yview)
        self._props_canvas.configure(yscrollcommand=props_scroll.set)
        props_scroll.pack(side="right", fill="y")
        self._props_canvas.pack(side="left", fill="both", expand=True)

        self._props_frame = ttk.Frame(self._props_canvas)
        self._props_canvas.create_window((0, 0), window=self._props_frame,
                                          anchor="nw")
        self._props_frame.bind("<Configure>", lambda _e:
            self._props_canvas.configure(scrollregion=self._props_canvas.bbox("all")))

        self._selected_path: str | None = None
        self._prop_widgets: list = []  # keep refs alive

        self._populate_tree()

    # ── Tree population ──────────────────────────────────────────

    def _populate_tree(self) -> None:
        graph = self.renderer.scene_graph
        if graph is None:
            return
        self._last_graph_id = id(graph)
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._insert_node("", graph)

    def _insert_node(self, parent_iid: str, node: SceneGraphNode) -> None:
        icon = type_icon(node.type_name)
        display_name = f"{icon} {node.name}"
        self.tree.insert(
            parent_iid, "end",
            iid=node.path,
            text=display_name,
            values=(node.type_name,),
            open=(len(node.children) > 0 and len(node.children) <= 8),
        )
        for child in node.children:
            self._insert_node(node.path, child)

    # ── Node selection ───────────────────────────────────────────

    def _on_select(self, _event) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        path = selection[0]
        if path == self._selected_path:
            return
        self._selected_path = path

        graph = self.renderer.scene_graph
        if graph is None:
            return
        node = find_node_by_path(graph, path)
        if node is None:
            return
        self._build_properties(node)

    # ── Property panel ───────────────────────────────────────────

    def _build_properties(self, node: SceneGraphNode) -> None:
        # Clear previous
        for w in self._props_frame.winfo_children():
            w.destroy()
        self._prop_widgets.clear()

        # Header
        header = ttk.Label(
            self._props_frame,
            text=f"{type_icon(node.type_name)} {node.name}",
            font=("", 10, "bold"),
        )
        header.pack(fill="x", padx=4, pady=(4, 2))

        path_lbl = ttk.Label(
            self._props_frame, text=node.path,
            foreground="gray",
        )
        path_lbl.pack(fill="x", padx=4, pady=(0, 8))

        if not node.properties:
            ttk.Label(self._props_frame, text="(no properties)").pack(padx=4, pady=4)
            return

        for prop in node.properties:
            self._build_property_widget(node, prop)

    def _build_property_widget(
        self, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> None:
        row = ttk.Frame(self._props_frame)
        row.pack(fill="x", padx=4, pady=2)

        label = ttk.Label(row, text=prop.display_name, width=16, anchor="w")
        label.pack(side="left")

        if prop.type_name == "float" and prop.editable:
            self._build_float_slider(row, node, prop)
        elif prop.type_name == "color3f" and prop.editable:
            self._build_color_picker(row, node, prop)
        elif prop.type_name == "vec3f" and prop.editable:
            self._build_vec3_editor(row, node, prop)
        elif prop.type_name == "vec3f":
            self._build_vec3_readonly(row, prop)
        elif prop.type_name == "color3f":
            self._build_color_readonly(row, prop)
        elif prop.type_name == "float" or prop.type_name == "int":
            val_text = f"{prop.value}"
            if isinstance(prop.value, float):
                val_text = f"{prop.value:.4f}"
            ttk.Label(row, text=val_text, anchor="e").pack(side="left", fill="x", expand=True)
        elif prop.type_name == "rel":
            ttk.Label(row, text=f"→ {prop.value}", foreground="steelblue").pack(
                side="left", fill="x", expand=True)
        elif prop.type_name == "asset":
            ttk.Label(row, text=str(prop.value), foreground="gray").pack(
                side="left", fill="x", expand=True)
        else:
            ttk.Label(row, text=str(prop.value)).pack(side="left", fill="x", expand=True)

    def _build_float_slider(
        self, parent: ttk.Frame, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> None:
        lo = prop.metadata.get("min", 0.0)
        hi = prop.metadata.get("max", 1.0)
        var = tk.DoubleVar(value=float(prop.value))

        scale = ttk.Scale(
            parent, from_=lo, to=hi, variable=var, orient="horizontal",
            command=lambda v, n=node, p=prop: self._on_float_changed(n, p, float(v)),
        )
        scale.pack(side="left", fill="x", expand=True, padx=(0, 4))

        val_lbl = ttk.Label(parent, width=7, anchor="e", text=f"{prop.value:.3f}")
        val_lbl.pack(side="left")
        self._prop_widgets.append((var, scale, val_lbl, node, prop))

    def _build_color_picker(
        self, parent: ttk.Frame, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> None:
        color = prop.value
        r, g, b = float(color[0]), float(color[1]), float(color[2])

        canvas = tk.Canvas(parent, width=36, height=18, bd=1, relief="sunken",
                           highlightthickness=0)
        canvas.pack(side="left", padx=(0, 4))
        hex_color = _rgb_to_hex(r, g, b)
        rect_id = canvas.create_rectangle(0, 0, 36, 18, fill=hex_color, outline="")

        def on_pick():
            init = (
                max(0, min(255, int(round(r * 255)))),
                max(0, min(255, int(round(g * 255)))),
                max(0, min(255, int(round(b * 255)))),
            )
            result = colorchooser.askcolor(
                color="#%02x%02x%02x" % init,
                title=f"{prop.display_name}",
            )
            if result is None or result[0] is None:
                return
            rr, gg, bb = result[0]
            new_color = (rr / 255.0, gg / 255.0, bb / 255.0)
            try:
                canvas.itemconfig(rect_id, fill=_rgb_to_hex(*new_color))
            except tk.TclError:
                pass
            self._apply_property(node, prop, new_color)

        ttk.Button(parent, text="Pick...", width=6, command=on_pick).pack(side="left")
        self._prop_widgets.append((canvas, rect_id))

    def _build_vec3_editor(
        self, parent: ttk.Frame, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> None:
        val = prop.value
        vars_ = []
        for i, axis in enumerate(("X", "Y", "Z")):
            ttk.Label(parent, text=axis, width=2).pack(side="left")
            var = tk.StringVar(value=f"{val[i]:.4f}")
            entry = ttk.Entry(parent, textvariable=var, width=8)
            entry.pack(side="left", padx=(0, 2))
            entry.bind("<Return>", lambda _e, n=node, p=prop: self._on_vec3_commit(n, p))
            entry.bind("<FocusOut>", lambda _e, n=node, p=prop: self._on_vec3_commit(n, p))
            vars_.append(var)
        self._prop_widgets.append(("vec3", vars_, node, prop))

    def _build_vec3_readonly(self, parent: ttk.Frame, prop: SceneGraphProperty) -> None:
        val = prop.value
        text = f"({val[0]:.3f}, {val[1]:.3f}, {val[2]:.3f})"
        ttk.Label(parent, text=text).pack(side="left", fill="x", expand=True)

    def _build_color_readonly(self, parent: ttk.Frame, prop: SceneGraphProperty) -> None:
        color = prop.value
        r, g, b = float(color[0]), float(color[1]), float(color[2])
        canvas = tk.Canvas(parent, width=36, height=18, bd=1, relief="sunken",
                           highlightthickness=0)
        canvas.pack(side="left", padx=(0, 4))
        canvas.create_rectangle(0, 0, 36, 18, fill=_rgb_to_hex(r, g, b), outline="")
        ttk.Label(parent, text=f"({r:.2f}, {g:.2f}, {b:.2f})").pack(
            side="left", fill="x", expand=True)

    # ── Edit callbacks ───────────────────────────────────────────

    def _on_float_changed(
        self, node: SceneGraphNode, prop: SceneGraphProperty, value: float,
    ) -> None:
        if self._suppress_cb:
            return
        # Update value label
        for entry in self._prop_widgets:
            if len(entry) >= 5 and entry[4] is prop:
                entry[2].configure(text=f"{value:.3f}")
                break
        self._apply_property(node, prop, value)

    def _on_vec3_commit(
        self, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> None:
        if self._suppress_cb:
            return
        # Find the matching vec3 widget entry
        for entry in self._prop_widgets:
            if len(entry) >= 4 and entry[0] == "vec3" and entry[3] is prop:
                vars_ = entry[1]
                try:
                    values = tuple(float(v.get()) for v in vars_)
                except (ValueError, TypeError):
                    return
                self._apply_vec3_property(node, prop, values)
                return

    def _apply_property(
        self, node: SceneGraphNode, prop: SceneGraphProperty, value: object,
    ) -> None:
        ref = node.renderer_ref
        if ref is None:
            # Check if this is a shader property — find ancestor material ref
            ref = self._find_shader_material_ref(node)
            if ref is None:
                return

        if ref.kind == "material":
            self.renderer.apply_material_override(ref.index, prop.name, value)
        elif ref.kind in ("light_dir", "light_sphere"):
            light_type = "dir" if ref.kind == "light_dir" else "sphere"
            self.renderer.apply_light_override(light_type, ref.index, prop.name, value)

    def _apply_vec3_property(
        self, node: SceneGraphNode, prop: SceneGraphProperty,
        value: tuple[float, float, float],
    ) -> None:
        ref = node.renderer_ref
        if ref is None or ref.kind != "instance":
            return

        # Collect all TRS values from current widgets
        translate = scale = (0.0, 0.0, 0.0)
        rotate = (0.0, 0.0, 0.0)
        for entry in self._prop_widgets:
            if len(entry) < 4 or entry[0] != "vec3":
                continue
            p = entry[3]
            vars_ = entry[1]
            try:
                vals = tuple(float(v.get()) for v in vars_)
            except (ValueError, TypeError):
                vals = (0.0, 0.0, 0.0)
            if p.name == "translate":
                translate = vals
            elif p.name == "rotate":
                rotate = vals
            elif p.name == "scale":
                scale = vals

        self.renderer.apply_instance_transform(
            ref.index, translate, rotate, scale,
        )

    def _find_shader_material_ref(self, node: SceneGraphNode):
        """Walk up the tree to find an ancestor Material node's RendererRef."""
        graph = self.renderer.scene_graph
        if graph is None:
            return None
        # Walk parent path
        parts = node.path.rstrip("/").split("/")
        for i in range(len(parts) - 1, 0, -1):
            parent_path = "/".join(parts[:i]) or "/"
            parent = find_node_by_path(graph, parent_path)
            if parent is not None and parent.renderer_ref is not None:
                if parent.renderer_ref.kind == "material":
                    return parent.renderer_ref
        return None

    # ── Per-frame update ─────────────────────────────────────────

    def tick(self) -> None:
        if not self._alive:
            return
        graph = self.renderer.scene_graph
        if graph is not None and id(graph) != self._last_graph_id:
            self._populate_tree()
        try:
            self.top.update()
        except tk.TclError:
            self._alive = False

    # ── Lifecycle ────────────────────────────────────────────────

    def is_open(self) -> bool:
        if not self._alive:
            return False
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    def focus(self) -> None:
        try:
            self.top.lift()
            self.top.focus_set()
        except tk.TclError:
            pass

    def close(self) -> None:
        self._alive = False
        try:
            self.top.destroy()
        except tk.TclError:
            pass


# ── Helpers ─────────────────────────────────────────────────────────


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    rr = max(0, min(255, int(round(r * 255.0))))
    gg = max(0, min(255, int(round(g * 255.0))))
    bb = max(0, min(255, int(round(b * 255.0))))
    return f"#{rr:02x}{gg:02x}{bb:02x}"
