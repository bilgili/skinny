"""Pure view-model for MaterialX node graphs.

Extracted from the legacy Tk ``material_graph_editor`` so the Qt port
can reuse the view-model + MaterialX bindings without dragging in Tk.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


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


# Categories the editor's "Add node…" menu exposes. ``(category, out_type)``.
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


# ── MaterialX → view-model ─────────────────────────────────────────


def _val_to_py(val) -> object | None:
    if val is None:
        return None
    try:
        return val.getData()
    except Exception:
        return None


def _input_connected_from(inp) -> Optional[tuple[str, str]]:
    """Return ``(node_name, output_name)`` if this input is wired, else ``None``."""
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
    """Snapshot one ``surfacematerial`` as a ``NodeGraphView``.

    ``None`` when ``target_name`` cannot be resolved or its surfaceshader
    input is unwired.
    """
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
